"""Hardware Ising on PIC B via the POWER route -- the cheapest real compute demo, no homodyne.

|M s|^2 = s^T Re(M^H M) s, so the chip only has to host the target coupling J as its Gram
matrix. Fit the Gram on the characterized heaters, launch each spin config as 0/pi input
phases at full amplitude, and the summed monitor-PD power IS the Ising energy plus a
constant (`theory/ising.py`). No LO, no phase calibration.

Two runners share the encode/program/power-route plumbing:

* ``main`` -- the single-config Ising run (`scripts/ising_hw.py`). Modes:
  ``--gate`` programs one known config and compares the 14 measured PDs to the twin's
  prediction (the twin-vs-hardware sanity gate); the default run fits the Gram for a random
  SK instance, sweeps all 2^n spin configs, and scores which config the chip picks as GS.
* ``main_ab`` -- the paired baseline-vs-corrected A/B run (`scripts/ising_ab.py`). The Gram
  fit is config-independent, so BASE and CORR share target phases; only the phase->volt map
  (phi0, set by each heater's V0) differs. Every spin config is programmed through BOTH
  configs back-to-back so each pair is ~1 s apart and thermal drift cancels in the pair.

Backends: ``--mock`` uses `theory/twin.py` as stand-in hardware (validates the plumbing:
encode -> volts -> DAC map -> extract -> score); otherwise the real PIC + laser. The laser is
driven ONLY through `pic.session.laser_session` (watchdog hard-off); power defaults to the
+12..+15 dBm base regime (coupling ~11 dB down post-rewire, so <+12 gives no PD signal).

    PYTHONPATH=. python -m pic ising --mock                    # offline pipeline check
    PYTHONPATH=. python -m pic ising --gate --dbm 13 ...       # step-0 on hardware
    PYTHONPATH=. python -m pic ising --n 6 --dbm 13 \
        --pic-port /dev/cu.usbserial-1110 --laser-port /dev/cu.usbserial-AU05XLI8
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from theory.hw import Hardware, adc_noise
from theory.twin import Twin, NH
from theory.gram import fit_gram, hosted_J
from theory.ising import random_ising, brute_force, calibrate_scale
from theory.program import encode_phases, inner_rails

MON_PDS = [2, 4, 6, 7, 9, 11]   # schematic monitor taps, one per signal rail -- UNVERIFIED on B
DBM_CEIL = 15.0                 # hard PD-damage ceiling, never exceed
MOCK_GAIN = 40.0                # cosmetic: lift twin optical power into a volt-like range


# --- shared plumbing (encode / program / power route) -------------------------------------
def build_problem(rng, n):
    """Random SK instance embedded on the n innermost signal rails. Returns
    (J6, rails, cfgs_n, E_true) with cfgs/energies enumerated over the n live spins."""
    rails = inner_rails(n)
    Jn = random_ising(rng, n=n)
    J6 = np.zeros((6, 6))
    J6[np.ix_(rails, rails)] = Jn
    cfgs, E = brute_force(Jn, n=n)
    return J6, rails, cfgs, E


def spin_phases_full(mesh_phases, s6):
    """Full 120-phase vector: input encoding (H0..H23) for spin vector s6, mesh program
    (H24..H119) held fixed. Spins ride in as full-amplitude 0/pi input phases."""
    ph = mesh_phases.clone() if torch.is_tensor(mesh_phases) else torch.as_tensor(
        np.asarray(mesh_phases), dtype=torch.float32)
    ph = ph.clone()
    enc = torch.as_tensor(encode_phases(np.asarray(s6, complex)), dtype=torch.float32)
    ph[:24] = enc
    return ph


def phases_to_dac(hw, phases, ncol):
    """120 heater phases -> ncol-wide DAC voltage vector. Uncommandable heaters hold 0 V."""
    v = hw.volts(phases if not torch.is_tensor(phases) else phases.numpy())
    dac = np.zeros(ncol)
    for h in range(NH):
        d = hw.dac_of[h]
        if 0 <= d < ncol and np.isfinite(v[h]):
            dac[d] = v[h]
    return dac


def total_power(raw):
    return float(np.sum([raw[pd] for pd in MON_PDS]))


def hw_quantized(hw, phases):
    ph, _ = hw.quantize(phases if not torch.is_tensor(phases) else phases.numpy())
    return torch.as_tensor(ph, dtype=torch.float32)


# --- single-config runner (scripts/ising_hw.py) -------------------------------------------
def fit_program(J6, hw, seed, throughput, restarts=12, steps=2000):
    """Mesh phase program hosting J6 as its Gram, using only characterized heaters."""
    return fit_gram(J6, trainable=hw.known, seed=seed, throughput=throughput,
                    restarts=restarts, steps=steps)


def mock_read(hw, twin, phases, rng):
    """Twin as stand-in hardware: quantize known heaters through the volt round-trip,
    forward, and pack the 6 monitor powers into a 14-PD vector at MON_PDS (so the same
    extraction code runs on mock and real reads). adc_noise models the ADC floor."""
    ph, _ = hw.quantize(phases if not torch.is_tensor(phases) else phases.numpy())
    out = twin.forward(torch.as_tensor(ph, dtype=torch.float32))
    mon = out["mon"].detach().numpy() * MOCK_GAIN
    raw = np.zeros(14)
    for k, pd in enumerate(MON_PDS):
        raw[pd] = mon[k]
    return adc_noise(raw, rng)


def hw_read(pic, hw, phases, repeats, settle):
    """Program the real chip, let the heaters settle, return host-averaged 14 raw PDs."""
    dac = phases_to_dac(hw, phases, pic.cfg.num_dac)
    pic.measure_raw(dac)          # apply, then settle before trusting the read
    if settle > 0:
        time.sleep(settle)
    reads = [pic.measure_raw(dac) for _ in range(repeats)]
    return np.mean(reads, axis=0)


def run_gate(read_fn, hw, mesh_phases, rng):
    """Step 0: program one config, compare measured 14-PD pattern to the twin prediction."""
    twin = Twin()
    s = np.ones(6)                       # all spins +1: every signal rail lit, phase 0
    ph = spin_phases_full(mesh_phases, s)
    pred = twin.forward(hw_quantized(hw, ph))["mon"].detach().numpy()
    predv = np.zeros(14)
    for k, pd in enumerate(MON_PDS):
        predv[pd] = pred[k]
    meas = read_fn(ph)
    live = [pd for pd in MON_PDS]
    r = float(np.corrcoef([predv[i] for i in live], [meas[i] for i in live])[0, 1])
    return {"pred_mon": [float(predv[i]) for i in MON_PDS],
            "meas_mon": [float(meas[i]) for i in MON_PDS],
            "pearson": r}


def run_ising(read_fn, hw, J6, rails, cfgs, E_true, mesh_phases, keepalive=None):
    """Sweep every spin config, read total monitor power, score against brute force."""
    raw_pow, rows = [], []
    for i, s_n in enumerate(cfgs):
        s6 = np.zeros(6)
        s6[rails] = s_n
        ph = spin_phases_full(mesh_phases, s6)
        meas = read_fn(ph)
        p = total_power(meas)
        raw_pow.append(p)
        rows.append({"cfg": s_n.tolist(), "power": p, "E_true": float(E_true[i]),
                     "raw14": [float(x) for x in meas]})
        if keepalive:
            keepalive()
    raw_pow = np.asarray(raw_pow)
    k, b = calibrate_scale(-2.0 * raw_pow, E_true)   # E = -0.5 s^T J s; power ~ +s^T(J+cI)s
    E_chip = k * (-2.0 * raw_pow) + b
    gs_true = set(np.flatnonzero(np.isclose(E_true, E_true.min())))
    gs_chip = int(np.argmin(E_chip))
    gap = (E_true[gs_chip] - E_true.min()) / (E_true.max() - E_true.min() + 1e-18)
    corr = float(np.corrcoef(E_chip, E_true)[0, 1]) if len(E_true) > 2 else float("nan")
    return {"found_gs": bool(gs_chip in gs_true), "excess": float(gap),
            "pearson": corr, "gs_chip": gs_chip,
            "gs_true": sorted(int(x) for x in gs_true), "rows": rows}


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pic ising")
    ap.add_argument("--n", type=int, default=6, help="spins (<=6); placed on inner rails")
    ap.add_argument("--gate", action="store_true", help="step-0 twin-vs-hardware check only")
    ap.add_argument("--throughput", type=float, default=0.05, help="fit_gram amplitude weight")
    ap.add_argument("--dbm", type=float, default=13.0)
    ap.add_argument("--repeats", type=int, default=8, help="host averages per config")
    ap.add_argument("--settle", type=float, default=0.3)
    ap.add_argument("--pd-limit", type=float, default=4.5, help="abort if any PD exceeds (V)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instances", type=int, default=1,
                    help=">1 sweeps N random SK instances (seeds seed..seed+N-1) for a GS-found rate")
    ap.add_argument("--fit-restarts", type=int, default=12)
    ap.add_argument("--fit-steps", type=int, default=2000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--out", default="runs/ising/run.json")
    a = ap.parse_args(argv)
    if a.dbm > DBM_CEIL:
        raise SystemExit(f"--dbm {a.dbm} exceeds the {DBM_CEIL} dBm PD-damage ceiling")

    hw = Hardware()
    twin = Twin()

    def build(i):
        rng = np.random.default_rng(a.seed + i)
        J6, rails, cfgs, E = build_problem(rng, a.n)
        fit = fit_program(J6, hw, a.seed + i, a.throughput, a.fit_restarts, a.fit_steps)
        off = ~np.eye(6, dtype=bool)
        c = np.corrcoef(hosted_J(fit["M"])[off], J6[off])[0, 1]
        print(f"  inst {i}: n={a.n} rails={rails} gram {fit['err']:.3f} "
              f"amp {fit['scale']:.3f} hostedJ-corr {c:+.2f}")
        return {"J6": J6, "rails": rails, "cfgs": cfgs, "E": E, "phases": fit["phases"],
                "gram_err": float(fit["err"]), "gram_amp": float(fit["scale"])}

    progs = [build(i) for i in range(a.instances)]
    meta = {"n": a.n, "rails": progs[0]["rails"], "dbm": a.dbm, "mock": a.mock,
            "mon_pds": MON_PDS, "seed": a.seed, "instances": a.instances,
            "known_heaters": int(hw.known.sum())}

    def sweep(read_fn, keepalive=None):
        results = [run_ising(read_fn, hw, p["J6"], p["rails"], p["cfgs"], p["E"],
                             p["phases"], keepalive=keepalive) for p in progs]
        gs = float(np.mean([r["found_gs"] for r in results]))
        exc = float(np.mean([r["excess"] for r in results]))
        cs = [r["pearson"] for r in results if np.isfinite(r["pearson"])]
        ec = float(np.mean(cs)) if cs else float("nan")
        print(f"n={a.n}: {len(results)} instances | GS-found {gs:.0%} | "
              f"mean-excess {exc:.3f} | mean E-corr {ec:.3f}")
        return {"gs_rate": gs, "mean_excess": exc, "mean_ecorr": ec,
                "per_instance": [{"found_gs": r["found_gs"], "excess": r["excess"],
                                  "pearson": r["pearson"], "gs_chip": r["gs_chip"],
                                  "gs_true": r["gs_true"], "gram_err": p["gram_err"]}
                                 for p, r in zip(progs, results)],
                "rows": [r["rows"] for r in results]}

    if a.mock:
        read_fn = lambda ph: mock_read(hw, twin, ph, np.random.default_rng(a.seed))
        if a.gate:
            res = run_gate(read_fn, hw, progs[0]["phases"], None)
            print(f"[gate/mock] pred vs meas Pearson r = {res['pearson']:.3f}")
        else:
            res = sweep(read_fn)
        _save(a.out, {**meta, "result": res})
        return 0

    from pic.session import open_devices, laser_session
    laser, pic = open_devices(mock=False, laser_port=a.laser_port, pic_port=a.pic_port)
    n_cfg = 1 if a.gate else sum(len(p["cfgs"]) for p in progs)
    dur = 40 + n_cfg * (a.repeats * 0.12 + a.settle + 0.3)
    print(f"laser ON {a.dbm} dBm for up to {dur:.0f}s "
          f"({n_cfg} configs, {a.instances} instances)")
    try:
        with laser_session(laser, duration_s=dur, power_dbm=a.dbm) as sess:
            if not sess.emitted:
                print("WARNING: no emission detected (bfm did not rise) -- results suspect")

            def guarded_read(ph):
                raw = hw_read(pic, hw, ph, a.repeats, a.settle)
                if np.max(raw) > a.pd_limit:
                    raise SystemExit(f"PD over limit ({np.max(raw):.2f} V) -- aborting")
                sess.keepalive()
                return raw

            tel = sess.telemetry()
            meta["telemetry"] = {"bfm": tel[0], "diode_temp": tel[1], "mA": tel[2],
                                 "emitted": bool(sess.emitted)}
            if a.gate:
                res = run_gate(guarded_read, hw, progs[0]["phases"], None)
                print(f"[gate] pred vs meas Pearson r = {res['pearson']:.3f}")
                print(f"  pred {np.round(res['pred_mon'],3)}\n  meas {np.round(res['meas_mon'],3)}")
            else:
                res = sweep(guarded_read, sess.keepalive)
    finally:
        try:
            pic.set_zero(); pic.close()
        except Exception:
            pass
        try:
            laser.off(); laser.close()
        except Exception:
            pass
    _save(a.out, {**meta, "result": res})
    return 0


# --- paired baseline-vs-corrected A/B runner (scripts/ising_ab.py) -------------------------
def score(raw_pow, E_true):
    raw_pow = np.asarray(raw_pow)
    k, b = calibrate_scale(-2.0 * raw_pow, E_true)
    E_chip = k * (-2.0 * raw_pow) + b
    gs_true = set(np.flatnonzero(np.isclose(E_true, E_true.min())))
    gs_chip = int(np.argmin(E_chip))
    gap = (E_true[gs_chip] - E_true.min()) / (E_true.max() - E_true.min() + 1e-18)
    corr = float(np.corrcoef(E_chip, E_true)[0, 1]) if len(E_true) > 2 else float("nan")
    return {"found_gs": bool(gs_chip in gs_true), "excess": float(gap), "pearson": corr}


def main_ab(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pic ising --ab")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--throughput", type=float, default=0.05)
    ap.add_argument("--fit-restarts", type=int, default=12)
    ap.add_argument("--fit-steps", type=int, default=2000)
    ap.add_argument("--dbm", type=float, default=13.0)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--settle", type=float, default=0.3)
    ap.add_argument("--pd-limit", type=float, default=4.5)
    ap.add_argument("--config-base", required=True)
    ap.add_argument("--config-corr", default=None,
                    help="corrected config; omit to use config-base + --dphi")
    ap.add_argument("--dphi", default=None,
                    help="npy of per-heater phi0 offset (backprop drift correction) for CORR")
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/ising/ab.json")
    a = ap.parse_args(argv)
    if a.dbm > DBM_CEIL:
        raise SystemExit(f"--dbm {a.dbm} exceeds the {DBM_CEIL} dBm ceiling")

    hw_base = Hardware(a.config_base)
    dphi = np.load(a.dphi) if a.dphi else None
    corr_cfg = a.config_corr or a.config_base
    hw_corr = Hardware(corr_cfg, phi0_offset=dphi)
    twin = Twin()
    off = ~np.eye(6, dtype=bool)

    # target programs (config-independent): fit once on the known mask (identical in both)
    progs = []
    for i in range(a.instances):
        rng = np.random.default_rng(a.seed + i)
        J6, rails, cfgs, E = build_problem(rng, a.n)
        fit = fit_gram(J6, trainable=hw_base.known, seed=a.seed + i,
                       throughput=a.throughput, restarts=a.fit_restarts, steps=a.fit_steps)
        c = np.corrcoef(hosted_J(fit["M"])[off], J6[off])[0, 1]
        print(f"  inst {i}: rails={rails} gram {fit['err']:.3f} hostedJ-corr {c:+.2f}", flush=True)
        progs.append({"J6": J6, "rails": rails, "cfgs": cfgs, "E": E,
                      "phases": fit["phases"], "gram_err": float(fit["err"])})

    n_cfg = sum(len(p["cfgs"]) for p in progs)
    meta = {"n": a.n, "instances": a.instances, "seed": a.seed, "dbm": a.dbm,
            "mon_pds": MON_PDS, "config_base": a.config_base, "config_corr": corr_cfg,
            "dphi": a.dphi, "known_heaters": int(hw_base.known.sum()), "paired": True}

    def run_paired(read_base, read_corr, keepalive=None):
        rows = []
        for pi, p in enumerate(progs):
            pb, pc = [], []
            for s_n in p["cfgs"]:
                s6 = np.zeros(6); s6[p["rails"]] = s_n
                ph = spin_phases_full(p["phases"], s6)
                # read CORR then BASE back-to-back -> pair separated by ~1 s
                mc = read_corr(ph); mb = read_base(ph)
                pc.append(total_power(mc)); pb.append(total_power(mb))
                if keepalive:
                    keepalive()
            sb = score(pb, p["E"]); sc = score(pc, p["E"])
            rows.append({"inst": pi, "gram_err": p["gram_err"], "base": sb, "corr": sc})
            print(f"  inst {pi}: BASE gs={int(sb['found_gs'])} exc={sb['excess']:.3f} "
                  f"ec={sb['pearson']:+.2f}  |  CORR gs={int(sc['found_gs'])} "
                  f"exc={sc['excess']:.3f} ec={sc['pearson']:+.2f}", flush=True)
        return rows

    if a.mock:
        rng = np.random.default_rng(a.seed)

        def mk_reader(hw):
            def rd(ph):
                phq, _ = hw.quantize(ph if not torch.is_tensor(ph) else ph.numpy())
                out = twin.forward(torch.as_tensor(phq, dtype=torch.float32))
                mon = out["mon"].detach().numpy() * MOCK_GAIN
                raw = np.zeros(14)
                for k, pd in enumerate(MON_PDS):
                    raw[pd] = mon[k]
                return adc_noise(raw, rng)
            return rd
        rows = run_paired(mk_reader(hw_base), mk_reader(hw_corr))
    else:
        from pic.session import open_devices, laser_session
        laser, pic = open_devices(mock=False, laser_port=a.laser_port, pic_port=a.pic_port)
        ncol = pic.cfg.num_dac
        dur = 40 + 2 * n_cfg * (a.repeats * 0.12 + a.settle + 0.3)
        print(f"laser ON {a.dbm} dBm up to {dur:.0f}s ({2*n_cfg} paired reads)", flush=True)

        def mk_reader(hw, sess):
            def rd(ph):
                dac = phases_to_dac(hw, ph, ncol)
                pic.measure_raw(dac)
                if a.settle > 0:
                    time.sleep(a.settle)
                raw = np.mean([pic.measure_raw(dac) for _ in range(a.repeats)], axis=0)
                if np.max(raw) > a.pd_limit:
                    raise SystemExit(f"PD over limit ({np.max(raw):.2f} V)")
                sess.keepalive()
                return raw
            return rd
        try:
            with laser_session(laser, duration_s=dur, power_dbm=a.dbm) as sess:
                if not sess.emitted:
                    print("WARNING: no emission (bfm flat) -- results suspect", flush=True)
                tel = sess.telemetry()
                meta["telemetry"] = {"bfm": tel[0], "diode_temp": tel[1], "mA": tel[2],
                                     "emitted": bool(sess.emitted)}
                rows = run_paired(mk_reader(hw_base, sess), mk_reader(hw_corr, sess),
                                  sess.keepalive)
        finally:
            try:
                pic.set_zero(); pic.close()
            except Exception:
                pass
            try:
                laser.off(); laser.close()
            except Exception:
                pass

    # paired summary
    d_exc = np.array([r["corr"]["excess"] - r["base"]["excess"] for r in rows])
    d_ec = np.array([r["corr"]["pearson"] - r["base"]["pearson"] for r in rows
                     if np.isfinite(r["corr"]["pearson"]) and np.isfinite(r["base"]["pearson"])])
    gs_b = np.mean([r["base"]["found_gs"] for r in rows])
    gs_c = np.mean([r["corr"]["found_gs"] for r in rows])
    print(f"\nn={a.n}  {len(rows)} paired instances")
    print(f"  GS-found   base {gs_b:.0%}  corr {gs_c:.0%}")
    print(f"  excess     base {np.mean([r['base']['excess'] for r in rows]):.3f}  "
          f"corr {np.mean([r['corr']['excess'] for r in rows]):.3f}  "
          f"| paired d(corr-base) mean {d_exc.mean():+.3f} (neg=better)  "
          f"wins {int((d_exc<0).sum())}/{len(d_exc)}")
    print(f"  E-corr     base {np.mean([r['base']['pearson'] for r in rows]):.3f}  "
          f"corr {np.mean([r['corr']['pearson'] for r in rows]):.3f}  "
          f"| paired d mean {d_ec.mean():+.3f} (pos=better)  "
          f"wins {int((d_ec>0).sum())}/{len(d_ec)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({**meta, "rows": rows,
               "summary": {"gs_base": float(gs_b), "gs_corr": float(gs_c),
                           "d_excess_mean": float(d_exc.mean()),
                           "d_ecorr_mean": float(d_ec.mean()) if len(d_ec) else None}},
              open(a.out, "w"), indent=2)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    main()
