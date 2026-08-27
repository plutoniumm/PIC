"""Train the dynamically-pruned MLP DIRECTLY FROM HARDWARE, dense-first then pruned.

  input  = [working-heaters^2] + [laser drift: dbm, bfm, diode_temp, mA]
           (+ dead-PD refs on PIC A only)
  output = the learnable photodiodes (--pds; default all 14 on PIC B)

Working heaters = the drivable channels (elec=="ok") from pic_data/pic_b_config.json in
--pic-b mode -- loaded at runtime, NOT hardcoded. Linear heaters dropped: thermo-optic
phase ~ V^2, so V^2 carries the signal and raw V only adds noise dims (confirmed, lowers
held-out R^2).

Drift correction: on PIC A the dead PDs (0,2,7,11) track global laser/thermal drift but not
the heaters, so their live reading is fed in; on PIC B there are NO dead PDs, so the drift
reference is the laser telemetry (bfm/diode_temp/mA) alone. Either way the drift signals are
inputs the net factors the drift out through.

Schedule: the first --dense-rounds rounds train the FULL-WIDTH net with NO pruning; the next
round establishes the pruned width (train_pruned from the dense weights), and every later
round fixed-width finetunes it. Each round lights the chip (laser on only during collection,
off during training), samples random (heater, dbm) configs on the 0..4 V grid, measures all
14 PDs, appends to a growing buffer, (re)trains, and CHECKPOINTS weights+norm+buffer.
Ctrl-C saves and exits. Resumable with --resume (reloads buffer + architecture + schedule).

  python scripts/train_hw_dpnn.py --pic-b --rounds 8 --dense-rounds 3 --n-per-round 200   # hw
  python scripts/train_hw_dpnn.py --mock --pic-b --rounds 5 --dense-rounds 3 --n-per-round 60
  python scripts/train_hw_dpnn.py --resume --rounds 4                                     # continue
"""
from __future__ import annotations
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from template import open_devices, laser_session, parse_channels
from src.pic.config import NUM_DAC, NUM_ADC_RAW, DAMAGED_PDS, LIVE_PDS, FIRMWARE_VMAX
from src.prune import DynamicPrunedMLP, train_pruned, finetune_fixed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEAD = list(DAMAGED_PDS)          # PIC-A drift-reference PDs (empty on PIC B)
LIVE = list(LIVE_PDS)             # output PDs (overridable with --pds)
N_DAC = NUM_DAC                   # heater vector length (128 on the PIC-B board)
FEAT = list(range(N_DAC))         # heater columns fed as features (the working heaters)
GRID = np.arange(0.0, 4.5, 0.5)   # host heater grid 0..4 V (firmware clamp is now 4 V)
# live PDs with real single-channel signal in the 2026-07-21 influence sweep (>15 mV swing);
# pd4/6/13 read ~noise now, so the honest learnable map is these 7 ("reduced map").
RESPONSIVE = (1, 3, 5, 8, 9, 10, 12)
# PIC B (post-rewire): 128-ch board, NO dead PDs -> targets default to all 14, drift signal =
# laser telemetry only. Working heaters + default sampling come from the config's elec=="ok"
# set at runtime (the old hardcoded census is stale).
B_RESPONSIVE = (3, 5, 6, 7, 8, 9, 10, 12)   # strong tier (swing >=25 mV from >=2 channels)
PIC_B_CONFIG = "pic_data/pic_b_config.json"
CKPT = "runs/dpnn_hw"


def load_drivable_channels(path=None):
    """(sorted DAC channels with elec=='ok', config dict) from the PIC-B config -- the
    working (drivable) heaters, replacing the stale hardcoded census list."""
    cfg = json.load(open(path or os.path.join(ROOT, PIC_B_CONFIG)))
    return sorted(c["dac"] for c in cfg["channels"] if c.get("elec") == "ok"), cfg


def make_features(H, dbm, tel, Ypd):
    """[working-heaters^2(|FEAT|), dbm(1), bfm(1), diode_temp(1), mA(1), dead-PDs(|DEAD|)]
    -> features, output PDs -> targets. Only the working-heater columns (`FEAT`) enter as
    features; linear V dropped (phase ~ V^2). Drift signals = laser telemetry bfm/temp/current
    (+ dead PDs on PIC A). `tel` is [n, 3]."""
    H = np.asarray(H, float); Ypd = np.asarray(Ypd, float)
    dbm = np.asarray(dbm, float).reshape(-1, 1)
    tel = np.asarray(tel, float).reshape(-1, 3)
    F = np.hstack([H[:, FEAT] ** 2, dbm, tel, Ypd[:, DEAD]])
    return F, Ypd[:, LIVE]


def compute_norm(F, Y, n_heaters):
    """Feature/target normalization. Inputs whose ranges we KNOW get a fixed analytic scale
    -- heaters^2 on the actual sampling grid (`GRID`), dbm on its span, and the laser
    telemetry bfm/temp/mA on their physical spans -- so every data regime shares ONE scaling and
    none degenerates to std~0 (the
    old frozen-norm blow-up). Dead-PD drift ref + PD targets are data-driven, floored. Column
    layout: `n_heaters` heaters^2, then dbm, bfm, diode_temp, mA, then any dead-PD columns."""
    n = F.shape[1]
    nh = n_heaters
    g2 = GRID ** 2
    fm = np.zeros(n); fs = np.ones(n)
    fm[:nh] = g2.mean(); fs[:nh] = g2.std() + 1e-6  # V^2 on the 0..4 V grid
    fm[nh] = 1.5;  fs[nh] = 3.5               # dbm span [-2, 5]
    fm[nh + 1] = 0.8;  fs[nh + 1] = 0.6       # bfm optical power ~[0.2, 1.4]
    fm[nh + 2] = 25.0; fs[nh + 2] = 2.0       # diode temperature ~[24, 26] C
    fm[nh + 3] = 14.0; fs[nh + 3] = 13.0      # measured current ~[1.6, 27] mA
    if n > nh + 4:                            # dead-PD drift refs (none in --pic-b mode)
        fm[nh + 4:] = F[:, nh + 4:].mean(0)
        fs[nh + 4:] = F[:, nh + 4:].std(0) + 1e-3
    return [fm, fs, Y.mean(0), Y.std(0) + 1e-3]


def read14(pic, v, settle_s, repeats):
    pic.measure_raw(v)                        # apply + prime
    if settle_s > 0:
        time.sleep(settle_s)                  # local thermo-optic settle (<0.1s; 0.2 is safe)
    ys = [np.asarray(pic.measure_raw(v), float) for _ in range(max(1, repeats))]
    return np.mean(ys, 0)                      # 14 raw PDs, host-averaged over `repeats`


def collect_round(laser, pic, dbm_grid, per_level, settle_s, repeats, rng, channels):
    """One lit session that SWEEPS the dbm grid: per_level random heater configs at each
    dbm level, so every round covers the whole power axis. Only `channels` are varied
    (others held at 0) -- restrict to the heaters that actually move PDs so per-PD variance
    is real signal, not noise. Returns (H, dbm_col, tel, Ypd14). The session opens at the
    top of the grid (solid latch) and steps power per level."""
    dbm_grid = list(dbm_grid)
    channels = np.asarray(channels, int)
    n = per_level * len(dbm_grid)
    Hs, Ds, Ts, Ys = [], [], [], []
    dur = n * (settle_s + repeats * 0.12 + 0.05) * 1.4 + len(dbm_grid) * 1.5 + 15.0
    with laser_session(laser, duration_s=dur, power_dbm=max(dbm_grid)) as ls:
        for dbm in dbm_grid:
            if ls.expired():
                print("    watchdog reached mid-round -- keeping partial round.")
                break
            ls.set_power(dbm)
            ls.keepalive(settle_s)            # let the PDs track the new power level
            tel = ls.telemetry()              # measured laser state (bfm, temp, mA) at level
            for _ in range(per_level):
                if ls.expired():
                    break
                v = np.zeros(N_DAC)
                v[channels] = rng.choice(GRID, size=channels.size)
                Ys.append(read14(pic, v, settle_s, repeats))
                Hs.append(v); Ds.append(float(dbm)); Ts.append(tel)
                ls.keepalive()                # hold the laser lit at the current level
    return np.asarray(Hs), np.asarray(Ds), np.asarray(Ts), np.asarray(Ys)


def r2_vec(y, yh):
    y = np.asarray(y); yh = np.asarray(yh)
    return 1 - ((y - yh) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def good_r2(r2):
    """Mean held-out R^2 over the RESPONSIVE PDs that are actually in the output set LIVE
    (which --pds may restrict); nan if none overlap."""
    resp = [p for p in RESPONSIVE if p in LIVE]
    return float(np.mean([r2[LIVE.index(p)] for p in resp])) if resp else float("nan")


def build_model(din, dout, hidden, act, min_neurons):
    return DynamicPrunedMLP(din, dout, hidden=hidden, activation=act, min_neurons=min_neurons)


def train_round(model, F, Y, norm, mode, epochs, seed, verbose=False):
    """mode: 'dense' = full-width train, NO pruning (warmup_pct=100 leaves the prune schedule
    empty) -- used for the first --dense-rounds hardware rounds; 'prune' =
    progressive pruning, run ONCE after the dense rounds to establish the pruned width from the
    dense weights; 'finetune' = warm-start at the fixed pruned width (every later round)."""
    fm, fs, ym, ys = norm
    n = len(F)
    rng = np.random.default_rng(seed)
    vi = rng.choice(n, max(1, int(0.15 * n)), replace=False)
    tr = np.ones(n, bool); tr[vi] = False
    Xn = ((F - fm) / fs).astype(np.float32)
    Yn = ((Y - ym) / ys).astype(np.float32)
    if mode == "finetune":
        model, _ = finetune_fixed(model, Xn[tr], Yn[tr], Xn[~tr], Y[~tr], ym, ys, epochs=epochs)
    else:
        warmup = 100 if mode == "dense" else 20  # dense -> no prune epochs scheduled
        model, _ = train_pruned(model, Xn[tr], Yn[tr], Xn[~tr], Y[~tr], ym, ys,
                                total_epochs=epochs, warmup_pct=warmup, prune_frac=0.15,
                                verbose=verbose)
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xn[~tr])).numpy() * ys + ym
    return model, r2_vec(Y[~tr], pv)


def save_ckpt(path, model, norm, buf, meta):
    os.makedirs(path, exist_ok=True)
    fm, fs, ym, ys = norm
    torch.save({"state_dict": model.state_dict(), "widths": model.widths(),
                "din": model.din, "dout": model.dout, "meta": meta,
                "norm": [fm.tolist(), fs.tolist(), ym.tolist(), ys.tolist()]},
               os.path.join(path, "ckpt.pt"))
    np.savez(os.path.join(path, "buffer.npz"), H=buf["H"], dbm=buf["dbm"],
             tel=buf["tel"], Ypd=buf["Ypd"])
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=2)


def load_ckpt(path, act, min_neurons):
    ck = torch.load(os.path.join(path, "ckpt.pt"), weights_only=False)
    model = build_model(ck["din"], ck["dout"], tuple(ck["widths"]), act, min_neurons)
    model.load_state_dict(ck["state_dict"])
    norm = [np.array(x, float) for x in ck["norm"]]
    b = np.load(os.path.join(path, "buffer.npz"))
    buf = {"H": b["H"], "dbm": b["dbm"], "tel": b["tel"], "Ypd": b["Ypd"]}
    return model, norm, buf, ck["meta"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--dense-rounds", type=int, default=3,
                    help="train full-width with NO pruning for this many rounds, then prune")
    ap.add_argument("--per-level", type=int, default=8,
                    help="heater samples per dbm level per round (each round sweeps the grid)")
    ap.add_argument("--n-per-round", type=int, default=None,
                    help="total heater samples per round; overrides --per-level "
                         "(split evenly across the dbm grid)")
    ap.add_argument("--dbm-min", type=float, default=-2.0)
    ap.add_argument("--dbm-max", type=float, default=5.0, help="PD-safe ceiling is +5 dBm")
    ap.add_argument("--dbm-step", type=float, default=0.5)
    ap.add_argument("--sample-channels", default="all",
                    help="heaters to vary, others held 0 (e.g. all / 35,37,46 / 24-63); "
                    "restrict to the influential ones from the sweep for real signal")
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=5,
                    help="host reads averaged per config to beat down white noise")
    ap.add_argument("--epochs", type=int, default=120,
                    help="epochs for the dense rounds and the single prune round")
    ap.add_argument("--finetune-epochs", type=int, default=60)
    ap.add_argument("--hidden", default="128,64,32")
    ap.add_argument("--min-neurons", type=int, default=8)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--out", default=CKPT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--pds", default=None,
                    help="comma list of output PD indices to learn, 0-13 (a PD-learnability "
                         "test picks the final set); default all 14 on --pic-b, else the 10 live")
    ap.add_argument("--pic-b", action="store_true",
                    help="PIC B / 128-ch board: default targets = all 14 PDs (none dead), "
                         "telemetry-only drift, working heaters + default sampling = the config's "
                         "elec=='ok' set")
    a = ap.parse_args(argv)

    global DEAD, LIVE, RESPONSIVE, N_DAC, FEAT
    pds = None
    if a.pds:
        pds = [int(x) for x in a.pds.replace(",", " ").split()]
        bad = [p for p in pds if not 0 <= p < NUM_ADC_RAW]
        if bad:
            ap.error(f"--pds out of 0..{NUM_ADC_RAW - 1}: {bad}")
    drivable = None
    if a.pic_b:
        drivable, cfg = load_drivable_channels()
        N_DAC = int(cfg["firmware"]["num_dac"])
        DEAD, RESPONSIVE = [], B_RESPONSIVE
        LIVE = pds if pds is not None else list(range(NUM_ADC_RAW))
        # PIC B couples ~13x low, so it operates at +15 dBm; the PIC-A -2..+5 grid would be
        # near-noise. Default to a high band (telemetry still supplies drift variation).
        if (a.dbm_min, a.dbm_max, a.dbm_step) == (-2.0, 5.0, 0.5):
            a.dbm_min, a.dbm_max, a.dbm_step = 12.0, 15.0, 1.0
    elif pds is not None:
        LIVE = pds

    dbm_grid = np.round(np.arange(a.dbm_min, a.dbm_max + 1e-9, a.dbm_step), 2).tolist()
    per_level = a.per_level if a.n_per_round is None else max(1, a.n_per_round // len(dbm_grid))
    if a.pic_b and a.sample_channels.strip() in ("all", "*"):
        channels = list(drivable)             # working heaters (elec=='ok') from the config
    else:
        channels = parse_channels(a.sample_channels, N_DAC)
    FEAT = list(channels) if a.pic_b else list(range(N_DAC))  # working-heater feature columns
    ceiling = 15.0 if a.pic_b else 5.0   # PIC B operates at +15; above it risks the PDs
    if max(dbm_grid) > ceiling + 1e-9 and not a.mock:
        ap.error(f"dbm grid reaches {max(dbm_grid):+.1f} dBm, above the +{ceiling:.0f} dBm "
                 f"ceiling (PD damage) -- lower --dbm-max")
    print(f"dbm grid: {len(dbm_grid)} levels {dbm_grid[0]:+.1f}..{dbm_grid[-1]:+.1f} "
          f"step {a.dbm_step}; {per_level}/level -> {per_level * len(dbm_grid)} samples/round")
    print(f"sampling {len(channels)} working heater channel(s)"
          f"{'' if len(channels) == N_DAC else ' ' + str(channels)}")
    print(f"outputs: {len(LIVE)} PD(s) {LIVE}; dense-first {a.dense_rounds} round(s) then prune")
    hidden = tuple(int(x) for x in a.hidden.split(","))
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    model = None; norm = None; meta = {"rounds_done": 0, "act": a.activation, "pic_b": a.pic_b}
    buf = {"H": np.zeros((0, N_DAC)), "dbm": np.zeros(0), "tel": np.zeros((0, 3)),
           "Ypd": np.zeros((0, NUM_ADC_RAW))}
    if a.resume and os.path.exists(os.path.join(a.out, "ckpt.pt")):
        model, norm, buf, meta = load_ckpt(a.out, a.activation, a.min_neurons)
        print(f"resumed: {buf['H'].shape[0]} samples, widths {model.widths()}, "
              f"{meta['rounds_done']} rounds done")

    if a.pic_b:
        if a.mock:
            from src.pic import MockPIC, mock_fringe_forward
            from template import _FakeLaser
            laser = _FakeLaser().open()
            pic = MockPIC(mock_fringe_forward(num_dac=N_DAC), noise=3e-4, num_dac=N_DAC).open()
        else:
            from laser.laser import Laser
            from src.pic import PIC
            from template import _resolve_pic_port
            laser = Laser(port=a.laser_port).open()
            pic = PIC(port=_resolve_pic_port(a.pic_port, laser.dev.port), num_dac=N_DAC).open()
    else:
        laser, pic = open_devices(a.mock, a.laser_port, a.pic_port)
    r0 = meta["rounds_done"]
    hl = 8 if 8 in LIVE else LIVE[0]          # a representative PD to print each round
    try:
        for r in range(r0, r0 + a.rounds):
            print(f"\n[round {r + 1}] sweep dbm grid, {per_level}/level ...")
            t0 = time.time()
            H, D, T, Y = collect_round(laser, pic, dbm_grid, per_level, a.settle,
                                       a.repeats, rng, channels)
            buf = {"H": np.vstack([buf["H"], H]), "dbm": np.concatenate([buf["dbm"], D]),
                   "tel": np.vstack([buf["tel"], T]), "Ypd": np.vstack([buf["Ypd"], Y])}
            F, Yl = make_features(buf["H"], buf["dbm"], buf["tel"], buf["Ypd"])
            if norm is None:                     # freeze once; analytic inputs can't degenerate
                norm = compute_norm(F, Yl, len(FEAT))
            if model is None:
                model = build_model(F.shape[1], len(LIVE), hidden, a.activation, a.min_neurons)
            # dense-first: no pruning for the first --dense-rounds rounds, then ONE prune round
            # fixes the width, fixed-width finetune after. `pruned` persists for --resume, and
            # `r` (== rounds already done) drives the dense->prune switch.
            if meta.get("pruned"):
                mode = "finetune"
            elif r < a.dense_rounds:
                mode = "dense"
            else:
                mode = "prune"
            ep = a.finetune_epochs if mode == "finetune" else a.epochs
            print(f"    collected {len(H)} ({time.time()-t0:.0f}s); buffer={len(F)}; {mode} ...")
            model, r2 = train_round(model, F, Yl, norm, mode, ep, a.seed, verbose=(mode == "prune"))
            meta.update(rounds_done=r + 1, pruned=bool(meta.get("pruned")) or mode == "prune",
                        dense_rounds=a.dense_rounds, n_samples=int(len(F)),
                        widths=model.widths(), n_params=model.n_params(), hidden=list(hidden))
            save_ckpt(a.out, model, norm, buf, meta)
            print(f"    R2 mean {r2.mean():+.3f} | good-PDs {good_r2(r2):+.3f} | pd{hl} "
                  f"{r2[LIVE.index(hl)]:+.3f} | widths {model.widths()} "
                  f"({model.n_params()}p) [{mode}] -> saved {a.out}/ckpt.pt")
    except KeyboardInterrupt:
        print("\ninterrupted -- last checkpoint already saved; exiting.")
    finally:
        laser.close(); pic.close()
    print(f"\ndone: {meta['rounds_done']} rounds, {meta.get('n_samples', 0)} samples, "
          f"checkpoint in {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
