"""Signed matrix-vector product on PIC B hardware, INTENSITY-only -- no homodyne.

Programs |M|^2 = B+ and B- on the mesh (`theory.intensity_matvec.fit_abs2_hw`, reachability-
constrained), then for each signed input x runs the 4-pass differential + phase-dithered
readout on the monitor PDs {2,4,6,7,9,11} (`theory/intensity_matvec.py` for the physics).

Everything downstream of the mesh works in **DAC-volt space** (not optical phase): the encode
amplitude is set from an EMPIRICAL per-rail calibration, avoiding the config-Vpi errors and
2pi-wrap pitfalls that corrupt a phase-space encoder. Two calibrations, both intensity-only:

* **Encoder amplitude (`calibrate_encoder`)** -- per signal rail, sweep the amplitude-MZI arm
  heater and fit the output-power fringe; V0 (max) = full transmission, Vnull (min) =
  extinction. Command amplitude `a` (transmission a^2) by interpolating in V^2 between them.
  Phases only need to be uniformly random for the dither (the 6 phase shifters span 2pi), so
  no phase calibration is needed.
* **Hosted matrix (`measure_A`, real-time)** -- one-hot input on rail j -> output power =
  column j of A=|M|^2, exact. Re-run every `--recal-every` matvecs to track mesh drift and
  refresh the per-program scales sp/sm.

    python -m pic matvec --mock --n 4
    python -m pic matvec --n 4 --n-dither 48 --dbm 13 \
        --pic-port /dev/cu.usbserial-1110 --laser-port /dev/cu.usbserial-AU05XLI8
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from theory.hw import Hardware
from theory.twin import Twin, NH
from theory.intensity_matvec import fit_abs2_hw
from theory.program import inner_rails
from src.census import robust_fit
from src.sweep_analysis import fringe_extrema
from pic.compute.ising import MON_PDS, DBM_CEIL, adc_noise, MOCK_GAIN

HW = None          # Hardware(), set in main
CAL = None         # {twin_rail r: (V0, Vnull)} encoder amplitude calibration
_VMAX = 4.0
_2PI = 2 * np.pi


# --- volt <-> phase (clean, no 2pi wrap: we always go volt->phase for the twin) -----------
def phases_from_volts_vec(hw, dac):
    """128 DAC volts -> 120 optical phases for the twin. phi = phi0 + pi (v/Vpi)^2 for
    characterised heaters; uncharacterised held at phase 0 (matches the fit's assumption)."""
    ph = np.zeros(NH)
    for h in range(NH):
        d = hw.dac_of[h]
        if d >= 0 and hw.known[h]:
            ph[h] = hw.phi0[h] + np.pi * (dac[d] / hw.vpi[h]) ** 2
    return ph


def mesh_dac_from_phases(hw, mesh_phases, ncol):
    """Mesh heaters (24..119) only: fit phase phi0+pi(v/Vpi)^2 -> v (clean inverse)."""
    ph = mesh_phases.numpy() if torch.is_tensor(mesh_phases) else np.asarray(mesh_phases)
    dac = np.zeros(ncol)
    for h in range(24, NH):
        d = hw.dac_of[h]
        if d >= 0 and d < ncol and hw.known[h]:
            u = (ph[h] - hw.phi0[h]) / np.pi
            dac[d] = float(np.clip(hw.vpi[h] * np.sqrt(max(u, 0.0)), 0, _VMAX))
    return dac


def phase_shifter_volt(hw, h, theta):
    """DAC volt for optical phase theta on phase-shifter heater h (spans 2pi on B)."""
    u = ((theta - hw.phi0[h]) % _2PI) / np.pi
    return float(np.clip(hw.vpi[h] * np.sqrt(max(u, 0.0)), 0, _VMAX))


def amp_to_volt(a, V0, Vnull):
    """Amplitude a in [0,1] -> amplitude-arm DAC volt, from the fringe endpoints.
    Transmission a^2 = cos^2(theta/2), theta linear in V^2, theta=0 at V0, pi at Vnull."""
    a = min(max(float(a), 0.0), 1.0)
    v2 = V0 ** 2 + (2 * np.arccos(a) / np.pi) * (Vnull ** 2 - V0 ** 2)
    return float(np.sqrt(max(v2, 0.0)))


# --- input encoding (DAC space, calibrated amplitude) -------------------------------------
def input_dac(hw, mesh_dac, x):
    """Full 128 DAC vector: complex input x encoded on the signal rails' amplitude arms +
    phase shifters (calibrated), mesh program held."""
    v = mesh_dac.copy()
    x = np.asarray(x, complex)
    for r in range(1, 7):
        a = abs(x[r - 1]); theta = float(np.angle(x[r - 1]))
        h1, h2, hp = 2 * r, 2 * r + 1, 16 + r
        V0, Vnull = CAL[r]
        v[hw.dac_of[h1]] = amp_to_volt(a, V0, Vnull)
        v[hw.dac_of[h2]] = 0.0
        v[hw.dac_of[hp]] = phase_shifter_volt(hw, hp, theta)
    return v


def read_rails(read_dac_fn, mesh_dac, x):
    raw = read_dac_fn(input_dac(HW, mesh_dac, x))
    return np.array([raw[pd] for pd in MON_PDS])


def measure_A(read_dac_fn, mesh_dac, rails):
    A = np.zeros((6, 6))
    for j in rails:
        e = np.zeros(6, complex); e[j] = 1.0
        A[:, j] = read_rails(read_dac_fn, mesh_dac, e)
    return A


def dithered(read_dac_fn, mesh_dac, p, n_dither, rng, rails):
    acc = np.zeros(6)
    for _ in range(n_dither):
        x = np.zeros(6, complex)
        for r in rails:
            x[r] = np.sqrt(max(p[r], 0.0)) * np.exp(1j * rng.uniform(0, _2PI))
        acc += read_rails(read_dac_fn, mesh_dac, x)
    return acc / n_dither


def signed(read_dac_fn, dp, dm, x, n_dither, rng, rails, sp, sm):
    x = np.asarray(x, float)
    c = max(float(np.max(np.abs(x))), 1e-12)
    xn = x / c
    xp, xm = np.clip(xn, 0, None), np.clip(-xn, 0, None)
    a = dithered(read_dac_fn, dp, xp, n_dither, rng, rails) / sp
    b = dithered(read_dac_fn, dm, xm, n_dither, rng, rails) / sm
    p = dithered(read_dac_fn, dp, xm, n_dither, rng, rails) / sp
    q = dithered(read_dac_fn, dm, xp, n_dither, rng, rails) / sm
    return c * ((a + b) - (p + q))


# --- calibrations -------------------------------------------------------------------------
def calibrate_encoder(read_dac_fn, mesh_dac, rails, levels, keepalive=None):
    """Per signal rail: sweep the amplitude-arm heater, fit the output-power fringe -> the
    (V0, Vnull) endpoints the amplitude encoder interpolates between. Intensity-only.
    All OTHER rails are held near extinction (arm1 ~ config Vpi -> arm difference ~pi) so the
    swept rail's fringe isn't swamped by other rails at 0 V (= full transmission)."""
    dark = mesh_dac.copy()
    for rr in range(1, 7):
        dark[HW.dac_of[2 * rr]] = float(min(HW.vpi[2 * rr] if np.isfinite(HW.vpi[2 * rr]) else 2.8, _VMAX))
    cal = {}
    for k in rails:
        r = k + 1; h1 = 2 * r
        dac1 = HW.dac_of[h1]
        base = dark.copy()
        P = []
        for V in levels:
            v = base.copy(); v[dac1] = float(V)
            raw = read_dac_fn(v)
            P.append(float(np.sum([raw[pd] for pd in MON_PDS])))
            if keepalive:
                keepalive()
        fit = robust_fit(np.asarray(levels, float), np.asarray(P))
        if fit is None:
            cal[r] = (0.0, _VMAX)
            print(f"  rail {r} (H{h1}): no fringe -> default (0,{_VMAX})")
            continue
        ex = fringe_extrema(fit, vmax=_VMAX)
        V0 = ex["v_at_max"] if ex["v_at_max"] is not None else 0.0
        Vnull = ex["v_at_min"] if ex["v_at_min"] is not None else _VMAX
        cal[r] = (float(V0), float(Vnull))
        print(f"  rail {r} (H{h1}): V0={V0:.2f} Vnull={Vnull:.2f} swing={1000*(max(P)-min(P)):.0f}mV")
    return cal


def scale_of(A, T, idx):
    Ah, Tt = A[np.ix_(idx, idx)], T[np.ix_(idx, idx)]
    return float((Ah * Tt).sum() / max((Tt * Tt).sum(), 1e-18))


def recalibrate(read_dac_fn, dp, dm, Bp, Bm, rails, idx):
    Ap, Am = measure_A(read_dac_fn, dp, rails), measure_A(read_dac_fn, dm, rails)
    sp, sm = scale_of(Ap, Bp, idx), scale_of(Am, Bm, idx)
    def drift(A, T, s):
        R = (A - s * T)[np.ix_(idx, idx)]
        return float(np.linalg.norm(R) / max(np.linalg.norm(s * T[np.ix_(idx, idx)]), 1e-18))
    B_eff = (Ap / sp - Am / sm)[np.ix_(idx, idx)]
    return sp, sm, drift(Ap, Bp, sp), drift(Am, Bm, sm), B_eff


def run_batch(read_dac_fn, B, dp, dm, rails, n_dither, matvecs, recal_every, rng, keepalive=None):
    idx = np.array(rails)
    Bp = np.zeros((6, 6)); Bm = np.zeros((6, 6))
    Bp[np.ix_(idx, idx)] = np.clip(B, 0, None); Bm[np.ix_(idx, idx)] = np.clip(-B, 0, None)
    sp = sm = 1.0; B_eff = B
    rows = []
    for i in range(matvecs):
        if i % recal_every == 0:
            sp, sm, ddp, ddm, B_eff = recalibrate(read_dac_fn, dp, dm, Bp, Bm, rails, idx)
            print(f"  [recal @{i}] sp={sp:.3f} sm={sm:.3f} | hosted drift +{ddp:.2f} -{ddm:.2f}")
            if keepalive:
                keepalive()
        x6 = np.zeros(6); x6[idx] = rng.normal(size=len(rails))
        y = signed(read_dac_fn, dp, dm, x6, n_dither, rng, rails, sp, sm)
        y_hat, y_true, y_eff = np.array([y[r] for r in idx]), B @ x6[idx], B_eff @ x6[idx]
        err = float(np.linalg.norm(y_hat - y_true) / max(np.linalg.norm(y_true), 1e-18))
        err_eff = float(np.linalg.norm(y_hat - y_eff) / max(np.linalg.norm(y_eff), 1e-18))
        sign = float(np.mean(np.sign(y_hat) == np.sign(y_true)))
        rows.append({"x": x6[idx].tolist(), "y_true": y_true.tolist(), "y_hat": y_hat.tolist(),
                     "err": err, "err_eff": err_eff, "sign": sign})
        print(f"  matvec {i}: vs target err {err:.3f} sign {sign:.0%} | vs measured-op err {err_eff:.3f}")
        if keepalive:
            keepalive()
    ve = float(np.mean([r["err"] for r in rows])); vee = float(np.mean([r["err_eff"] for r in rows]))
    sa = float(np.mean([r["sign"] for r in rows]))
    print(f"n={len(rails)}: {matvecs} matvecs | vs TARGET err {ve:.3f} sign {sa:.0%} | vs MEASURED-OP err {vee:.3f}")
    return {"vec_err": ve, "vec_err_eff": vee, "sign_acc": sa, "rows": rows}


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pic matvec")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n-dither", type=int, default=48)
    ap.add_argument("--matvecs", type=int, default=3)
    ap.add_argument("--recal-every", type=int, default=1)
    ap.add_argument("--cal-levels", default="0:4.0:0.25", help="amplitude-cal sweep grid")
    ap.add_argument("--dbm", type=float, default=13.0)
    ap.add_argument("--settle", type=float, default=0.12)
    ap.add_argument("--pd-limit", type=float, default=4.5)
    ap.add_argument("--fit-restarts", type=int, default=8)
    ap.add_argument("--fit-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--out", default="runs/matvec/run.json")
    a = ap.parse_args(argv)
    if a.dbm > DBM_CEIL:
        raise SystemExit(f"--dbm {a.dbm} exceeds the {DBM_CEIL} dBm ceiling")

    global HW, CAL
    HW = Hardware(); twin = Twin()
    rng = np.random.default_rng(a.seed)
    rails = inner_rails(a.n); idx = np.array(rails)
    cal_levels = [round(x, 4) for x in np.arange(*[float(t) for t in a.cal_levels.split(":")[:2]] + [float(a.cal_levels.split(":")[2])])]
    B = rng.normal(size=(a.n, a.n))
    Bp = np.zeros((6, 6)); Bm = np.zeros((6, 6))
    Bp[np.ix_(idx, idx)] = np.clip(B, 0, None); Bm[np.ix_(idx, idx)] = np.clip(-B, 0, None)
    print(f"fitting B+/B- (n={a.n} rails={rails}) ...")
    fp = fit_abs2_hw(Bp, HW, twin, a.seed + 1, a.fit_steps, restarts=a.fit_restarts)
    fm = fit_abs2_hw(Bm, HW, twin, a.seed + 2, a.fit_steps, restarts=a.fit_restarts)
    print(f"  fit+ err {fp['err']:.3f}  fit- err {fm['err']:.3f}")
    dp = mesh_dac_from_phases(HW, fp["phases"], 128)
    dm = mesh_dac_from_phases(HW, fm["phases"], 128)
    meta = {"n": a.n, "rails": rails, "n_dither": a.n_dither, "matvecs": a.matvecs,
            "recal_every": a.recal_every, "dbm": a.dbm, "mock": a.mock, "seed": a.seed,
            "fit_plus": fp["err"], "fit_minus": fm["err"]}

    if a.mock:
        def read_dac_fn(dac):
            out = twin.forward(torch.as_tensor(phases_from_volts_vec(HW, dac), dtype=torch.float32))
            raw = np.zeros(14)
            for k, pd in enumerate(MON_PDS):
                raw[pd] = float(out["mon"][k]) * MOCK_GAIN
            return adc_noise(raw, rng)
        print("calibrating encoder amplitude (mock):")
        CAL = calibrate_encoder(read_dac_fn, dp, list(range(6)), cal_levels)
        res = run_batch(read_dac_fn, B, dp, dm, rails, a.n_dither, a.matvecs, a.recal_every, rng)
        _save(a.out, {**meta, "cal": {str(k): v for k, v in CAL.items()}, "result": res})
        return 0

    from pic.session import open_devices, laser_session
    laser, pic = open_devices(mock=False, laser_port=a.laser_port, pic_port=a.pic_port)
    reads = len(rails) * len(cal_levels) + a.matvecs * 4 * a.n_dither + \
        (a.matvecs // a.recal_every + 1) * 2 * a.n
    dur = min(90 + reads * (a.settle + 0.35), 540)  # ~0.35 s/read (2 measure_raw), margin
    print(f"laser ON {a.dbm} dBm ~{reads} reads, watchdog {dur:.0f}s")
    try:
        with laser_session(laser, duration_s=dur, power_dbm=a.dbm) as sess:
            if not sess.emitted:
                print("WARNING: no emission -- results suspect")

            def read_dac_fn(dac):
                if np.max(dac) > _VMAX + 1e-6:
                    dac = np.clip(dac, 0, _VMAX)
                pic.measure_raw(dac); time.sleep(a.settle)
                raw = np.asarray(pic.measure_raw(dac), float)
                if np.max(raw) > a.pd_limit:
                    raise SystemExit(f"PD over limit ({np.max(raw):.2f} V)")
                sess.keepalive()
                return raw

            tel = sess.telemetry()
            meta["telemetry"] = {"bfm": tel[0], "diode_temp": tel[1], "mA": tel[2],
                                 "emitted": bool(sess.emitted)}
            print("calibrating encoder amplitude:")
            CAL = calibrate_encoder(read_dac_fn, dp, list(range(6)), cal_levels, keepalive=sess.keepalive)
            if sess.expired():
                print("session expired during calibration -- aborting matvec")
            else:
                res = run_batch(read_dac_fn, B, dp, dm, rails, a.n_dither, a.matvecs,
                                a.recal_every, rng, keepalive=sess.keepalive)
                meta["result"] = res
    finally:
        try:
            pic.set_zero(); pic.close()
        except Exception:
            pass
        try:
            laser.off(); laser.close()
        except Exception:
            pass
    _save(a.out, {**meta, "cal": {str(k): v for k, v in (CAL or {}).items()}})
    return 0


if __name__ == "__main__":
    main()
