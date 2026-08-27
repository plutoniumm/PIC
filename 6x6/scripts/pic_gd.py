"""Gradient-descent voltage optimiser: optimise heater voltages on the trained DPNN
surrogate (a DIFFERENTIABLE torch model), then verify/step on real hardware.

Per-heater fringe characterisation plateaus for deep multi-path heaters, so instead we
treat the whole PIC as a black box and optimise the voltage vector directly. The DPNN maps
[working-heaters^2, laser telemetry] -> 14 PD volts; backprop gives d(objective)/d(volts)
analytically (no hardware finite-differencing). Optimise on the surrogate, then verify.

  python scripts/pic_gd.py --mock                       # full flow, no hardware
  python scripts/pic_gd.py --mock --objective max-light --pd 8
  python scripts/pic_gd.py --dbm 10                      # HARDWARE: max-light, +10 dBm
  python scripts/pic_gd.py --objective target --target-file tgt.npy --dbm 10

Surrogate-power caveat: the DPNN was trained at +12..+15 dBm. Optical power mostly scales
PD MAGNITUDE, so the argmax CONFIG is ~power-independent -- we optimise the surrogate at
its training power (--surrogate-dbm, default the buffer median) but run hardware at a LOWER,
PD-safe power (--dbm, default +10) and verify. max-light deliberately drives PDs UP, so the
PD-abort guard zeroes everything if any PD exceeds --pd-limit.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import scripts.train_hw_dpnn as T
from src.inverse import GradientInverse
from src.pic.config import NUM_DAC, NUM_ADC_RAW

CKPT = "runs/dpnn_hw"
DBM_CEILING = 15.0  # hard cap -- above this risks the PDs


def load_surrogate(ckpt=CKPT, act="relu", min_neurons=8):
    """Load the frozen DPNN with the exact training feature pipeline. Sets the module
    globals make_features depends on (all 14 PDs, no dead-PD refs on PIC B, the 112
    working-heater feature columns) BEFORE anything touches them, or dims mismatch."""
    T.LIVE = list(range(NUM_ADC_RAW))
    T.DEAD = []
    T.FEAT, cfg = T.load_drivable_channels()
    model, norm, buf, meta = T.load_ckpt(ckpt, act, min_neurons)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, norm, buf, meta, list(T.FEAT), cfg


def operating_telemetry(buf, dbm, k=128):
    """Mean laser telemetry (bfm, temp, mA) over the buffer rows whose dbm is nearest the
    chosen operating point -- the surrogate saw these values, so holding them fixed keeps
    the telemetry inputs in-distribution (NOMINAL_TEL is PIC-A and wildly off this chip)."""
    d = np.abs(np.asarray(buf["dbm"], float) - dbm)
    idx = np.argsort(d)[: min(k, d.size)]
    return np.asarray(buf["tel"], float)[idx].mean(0)


def make_predict(model, norm, feat, surrogate_dbm, tel):
    """Differentiable v_work -> 14 PD volts. v_work is the (leaf) voltage vector for the
    working heaters; features = [v_work^2, dbm, bfm, temp, mA] normalised exactly as in
    training, telemetry HELD FIXED at the operating point. Backprop flows into v_work."""
    fm, fs, ym, ys = (torch.tensor(np.asarray(x), dtype=torch.float32) for x in norm)
    nh = len(feat)
    tel_vec = torch.tensor([surrogate_dbm, *np.asarray(tel, float)], dtype=torch.float32)
    tel_n = (tel_vec - fm[nh:nh + 4]) / fs[nh:nh + 4]

    def predict(v_work):
        xh = (v_work ** 2 - fm[:nh]) / fs[:nh]
        xn = torch.cat([xh, tel_n]).unsqueeze(0)
        return (model(xn) * ys + ym)[0]

    return predict


def make_objective(predict, mode, pd=None, target=None):
    """Scalar objective to MAXIMISE. max-light: sum of the 14 predicted PDs, or one PD.
    target: negative Euclidean distance to a target PD vector (the Ax=B frame)."""
    if mode == "target":
        tgt = torch.tensor(np.asarray(target, float), dtype=torch.float32)
        return lambda v: -torch.linalg.norm(predict(v) - tgt)
    if pd is not None:
        return lambda v: predict(v)[pd]
    return lambda v: predict(v).sum()


def objective_np(y14, mode, pd=None, target=None):
    """Same objective evaluated on measured raw PD volts (numpy) -- for hardware scoring."""
    y = np.asarray(y14, float)
    if mode == "target":
        return float(-np.linalg.norm(y - np.asarray(target, float)))
    return float(y[pd]) if pd is not None else float(y.sum())


def to_full(v_work, feat, num_dac):
    v = np.zeros(num_dac)
    v[np.asarray(feat, int)] = v_work
    return v


def guarded_read(pic, v, pd_limit, label=""):
    """Apply v, read raw PDs, flag if any PD exceeds the abort limit. Returns (raw, ok)."""
    raw = np.asarray(pic.measure_raw(v), float)
    ok = float(raw.max()) <= pd_limit
    if not ok:
        i = int(np.argmax(raw))
        print(f"  ** PD-ABORT{(' ' + label) if label else ''}: pd{i}={raw[i]:.3f} V "
              f"> limit {pd_limit:.2f} V -- zeroing heaters + laser off **")
    return raw, ok


def report_pred_vs_actual(pred14, raw14, mode, pd, target):
    print("    PD   predicted   actual   (V)")
    for i in range(NUM_ADC_RAW):
        mark = "  <-- pd" if (pd is not None and i == pd) else ""
        print(f"    {i:2d}   {pred14[i]:8.4f}   {raw14[i]:7.4f}{mark}")
    print(f"  objective: surrogate {objective_np(pred14, mode, pd, target):+.4f}  "
          f"hardware {objective_np(raw14, mode, pd, target):+.4f}")


def open_devices(mock, laser_port, pic_port):
    """(laser, pic), both open -- PIC-B 128-ch board. Mock exercises the same safety."""
    if mock:
        from src.pic import MockPIC, mock_fringe_forward
        from template import _FakeLaser
        laser = _FakeLaser().open()
        pic = MockPIC(mock_fringe_forward(num_dac=NUM_DAC), noise=3e-4, num_dac=NUM_DAC).open()
        return laser, pic
    from laser.laser import Laser
    from src.pic import PIC
    from template import _resolve_pic_port
    laser = Laser(port=laser_port).open()
    pic = PIC(port=_resolve_pic_port(pic_port, laser.dev.port), num_dac=NUM_DAC).open()
    return laser, pic


def hw_verify(laser, pic, v_full, pred14, *, dbm, pd_limit, duration, mode, pd, target):
    """Apply the optimised voltages on hardware and report predicted-vs-actual. The laser
    is only ever on inside laser_session (watchdog + lock + forced off on every exit).
    Reads a zero-heater baseline first, then the optimised config, aborting on either if a
    PD exceeds pd_limit -- a constructive peak must not cook a photodiode."""
    from template import laser_session
    zeros = np.zeros_like(v_full)
    out = {"aborted": False, "raw": None}
    with laser_session(laser, duration_s=duration, power_dbm=dbm) as ls:
        print(f"  laser {dbm:+.1f} dBm  monitor {ls.bfm_off:.3g}->{ls.bfm_on:.3g}"
              + ("" if ls.emitted else "  ** no rise -- emission suspect (task #9) **"))
        base, ok = guarded_read(pic, zeros, pd_limit, "baseline")
        if not ok:
            pic.set_zero(); out["aborted"] = True; return out
        raw, ok = guarded_read(pic, v_full, pd_limit, "optimised")
        out["raw"] = raw
        if not ok:
            pic.set_zero(); out["aborted"] = True; return out
        report_pred_vs_actual(pred14, raw, mode, pd, target)
    pic.set_zero()
    return out


def hil_optimize(laser, pic, predict, objective, v_work0, feat, *, dbm, pd_limit, duration,
                 rounds, iters, lr, vmax, mode, pd, target):
    """Hardware-in-the-loop: alternate surrogate-gradient proposals with hardware line
    searches. Each round runs a short surrogate GD from the current best config, then
    (on hardware, guarded) evaluates a few points along current->proposal and keeps the
    one with the best MEASURED objective. One laser_session wraps all rounds."""
    from template import laser_session
    gi = GradientInverse(objective, 0.0, vmax)
    best_v = np.asarray(v_work0, float)
    with laser_session(laser, duration_s=duration, power_dbm=dbm) as ls:
        if not ls.emitted:
            print(f"  ** monitor PD did not rise ({ls.bfm_off:.3g}->{ls.bfm_on:.3g}) "
                  "-- emission suspect (task #9); HIL scores may be meaningless **")
        raw, ok = guarded_read(pic, to_full(best_v, feat, NUM_DAC), pd_limit, "hil-start")
        if not ok:
            pic.set_zero(); return best_v, None
        best_true = objective_np(raw, mode, pd, target)
        print(f"  [hil 0] measured objective {best_true:+.4f}")
        for r in range(rounds):
            if ls.expired():
                print("  watchdog reached -- stopping HIL."); break
            prop = gi.run(best_v, iters=iters, lr=lr, verbose=False)["v"]
            for a in (1.0, 0.5, 0.25):
                ls.keepalive()
                cand = np.clip(best_v + a * (prop - best_v), 0.0, vmax)
                raw, ok = guarded_read(pic, to_full(cand, feat, NUM_DAC), pd_limit,
                                       f"hil r{r + 1} a={a}")
                if not ok:
                    pic.set_zero(); return best_v, best_true
                true = objective_np(raw, mode, pd, target)
                if true > best_true:
                    best_true, best_v = true, cand
                    print(f"  [hil {r + 1}] a={a:.2f} accepted -> objective {best_true:+.4f}")
                    break
            else:
                print(f"  [hil {r + 1}] no line-search point improved (best {best_true:+.4f})")
    pic.set_zero()
    return best_v, best_true


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objective", choices=["max-light", "target"], default="max-light")
    ap.add_argument("--pd", type=int, default=None, help="max-light one PD (0-13) not the sum")
    ap.add_argument("--target-file", default=None, help="npy of 14 target PD volts (target mode)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--vmax", type=float, default=4.0, help="per-heater voltage bound [0, vmax]")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surrogate-dbm", type=float, default=None,
                    help="power the surrogate is optimised at (default = buffer median, in "
                         "the +12..+15 training band where the argmax config is trustworthy)")
    ap.add_argument("--dbm", type=float, default=10.0, help="HARDWARE laser power [dBm], PD-safe")
    ap.add_argument("--pd-limit", type=float, default=3.0,
                    help="abort (zero heaters + laser off) if any PD exceeds this [V]")
    ap.add_argument("--duration", type=float, default=45.0, help="laser watchdog [s]")
    ap.add_argument("--hil", type=int, default=0, help="hardware-in-the-loop rounds (0 = off)")
    ap.add_argument("--hil-iters", type=int, default=100, help="surrogate GD iters per HIL round")
    ap.add_argument("--skip-verify", action="store_true", help="surrogate optimise only, no hw")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    a = ap.parse_args(argv)

    if a.pd is not None and not 0 <= a.pd < NUM_ADC_RAW:
        ap.error(f"--pd out of 0..{NUM_ADC_RAW - 1}")
    if a.dbm > DBM_CEILING and not a.mock:
        ap.error(f"--dbm {a.dbm:+.1f} above the +{DBM_CEILING:.0f} dBm ceiling (PD damage)")
    target = None
    if a.objective == "target":
        if not a.target_file:
            ap.error("--objective target needs --target-file")
        target = np.load(a.target_file).ravel().astype(float)
        if target.size != NUM_ADC_RAW:
            ap.error(f"--target-file must hold {NUM_ADC_RAW} values, got {target.size}")
        a.pd = None

    model, norm, buf, meta, feat, cfg = load_surrogate(a.ckpt)
    sdbm = a.surrogate_dbm if a.surrogate_dbm is not None else float(np.median(buf["dbm"]))
    tel = operating_telemetry(buf, sdbm)
    print(f"surrogate: widths {meta.get('widths')} ({meta.get('n_params')}p), "
          f"{len(feat)} working heaters, {NUM_ADC_RAW} PDs; "
          f"optimise @ {sdbm:+.1f} dBm (tel bfm/temp/mA={tel[0]:.2f}/{tel[1]:.2f}/{tel[2]:.1f})")
    print(f"objective: {a.objective}" + (f" pd{a.pd}" if a.pd is not None else "")
          + f"; bounds [0, {a.vmax}] V")

    predict = make_predict(model, norm, feat, sdbm, tel)
    objective = make_objective(predict, a.objective, a.pd, target)

    rng = np.random.default_rng(a.seed)
    v0 = rng.uniform(0.0, a.vmax, len(feat))
    print(f"\nsurrogate gradient ascent: {a.iters} iters, lr {a.lr} ...")
    res = GradientInverse(objective, 0.0, a.vmax).run(
        v0, iters=a.iters, lr=a.lr, log_every=max(1, a.iters // 8), verbose=True)
    v_work = res["v"]
    pred14 = predict(torch.tensor(v_work, dtype=torch.float32)).detach().numpy()
    print(f"objective: {res['obj0']:+.5f} (init) -> {res['obj']:+.5f} (final), "
          f"improved {res['obj'] - res['obj0']:+.5f}")

    if a.skip_verify:
        v_full = to_full(v_work, feat, NUM_DAC)
        print(f"\nskip-verify: optimised config on {int((v_full > 0).sum())} heaters, "
              f"predicted objective {objective_np(pred14, a.objective, a.pd, target):+.4f}")
        return 0

    laser, pic = open_devices(a.mock, a.laser_port, a.pic_port)
    try:
        if a.hil > 0:
            print(f"\nhardware-in-the-loop: {a.hil} round(s) @ {a.dbm:+.1f} dBm ...")
            v_work, best_true = hil_optimize(
                laser, pic, predict, objective, v_work, feat, dbm=a.dbm,
                pd_limit=a.pd_limit, duration=a.duration, rounds=a.hil, iters=a.hil_iters,
                lr=a.lr, vmax=a.vmax, mode=a.objective, pd=a.pd, target=target)
            pred14 = predict(torch.tensor(v_work, dtype=torch.float32)).detach().numpy()
            print(f"HIL best measured objective: "
                  f"{'aborted' if best_true is None else f'{best_true:+.4f}'}")
        else:
            print(f"\nhardware verify @ {a.dbm:+.1f} dBm (pd-limit {a.pd_limit} V) ...")
            out = hw_verify(laser, pic, to_full(v_work, feat, NUM_DAC), pred14, dbm=a.dbm,
                            pd_limit=a.pd_limit, duration=a.duration, mode=a.objective,
                            pd=a.pd, target=target)
            if out["aborted"]:
                print("  verify ABORTED by PD guard -- heaters zeroed, laser off.")
    finally:
        laser.close(); pic.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
