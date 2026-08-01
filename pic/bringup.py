"""PIC hardware bring-up: dark/warming/PD13/SNR tests + thermal-settling characterization.

Two independent CLIs, both landing here from `python -m pic bringup ...`:

* bring-up tests (`hwtests_main`, was `scripts/hw_tests.py`) -- warming / dark / pd13 / snr /
  compare. Run one subcommand at a time while you adjust the rig (TEC switch, laser block,
  laser power). All reads are the raw 14 photodiodes, so PD13 (bypass) and the dead PDs stay
  visible. Add ``--mock`` (a global flag, before the subcommand) to dry-run with the simulator.

* settling (`settling_main`, was `scripts/settling.py`) -- thermal step-response fit (heat-up /
  cool-down tau, t99, recommended host settle_s) and a round-trip loop-speed budget. The
  ``step --selftest`` path fits a synthetic exponential and needs no hardware.

    python -m pic bringup --mock dark --samples 300
    python -m pic bringup warming --minutes 10 --tag tec_on
    python -m pic bringup step --selftest
    python -m pic bringup step --channel 5 --vhi 1.5
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

from pic import (
    PIC,
    MockPIC,
    PICError,
    find_port,
    config,
    config as C,
    NUM_DAC,
    NUM_ADC_RAW,
    DAMAGED_PDS,
    LIVE_PDS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of pic/)

# ---- bring-up tests (warming / dark / pd13 / snr / compare) ----

VMAX = config.FIRMWARE_VMAX  # 2.0 V — the range the firmware actually actuates
LSB_MV = config.DAC_REF_V / 2**10 * 1000  # ~4.9 mV Arduino ADC step
MIDDLE_PDS = [
    i for i in LIVE_PDS if i != 13
]  # compute-path PDs (exclude the PD13 bypass)
OUT = os.path.join(ROOT, "runs", "hwtests")


def _mock_forward():
    """Simulator: smooth 64->14 map, PD13 high (bypass) and attenuated by ch63,
    slow upward baseline drift so the warming/SNR analysis has something to chew on."""
    rng = np.random.default_rng(1)
    W = rng.normal(0.0, 0.10, (NUM_ADC_RAW, NUM_DAC))
    b = rng.uniform(0.0, 2 * np.pi, NUM_ADC_RAW)
    dead = np.array([i in DAMAGED_PDS for i in range(NUM_ADC_RAW)])
    t0 = time.time()

    def f(v):
        v = np.clip(np.asarray(v, float), 0, VMAX)
        y = 0.02 + 0.04 * (1 + np.sin(W @ (v / VMAX) + b))
        y[13] = max(0.005, 0.185 - 0.09 * v[63])  # bypass, attenuated by ch63
        y += 3e-4 * (time.time() - t0) / 60  # ~0.3 mV/min creep
        y[dead] = 0.006
        return y

    return f


def open_rig(args):
    os.makedirs(OUT, exist_ok=True)
    if getattr(args, "mock", False):
        print("rig: MOCK (no hardware)")
        return MockPIC(_mock_forward(), noise=8e-4).open()
    port, cands = find_port(getattr(args, "port", None))
    if not port:
        sys.exit(
            f"no serial device found (looked at {cands or 'the usual /dev names'}); "
            f"pass --port or --mock"
        )
    print(f"rig: {port}")
    return PIC(port=port).open()


def save_matrix(path, t, Y, note=""):
    hdr = (
        (f"# {note}\n" if note else "")
        + "t_s,"
        + ",".join(f"pd{i}" for i in range(Y.shape[1]))
    )
    np.savetxt(path, np.column_stack([t, Y]), delimiter=",", header=hdr, comments="# ")
    return path


def load_run(path):
    M = np.loadtxt(path, delimiter=",", comments="#")
    return M[:, 0], M[:, 1:]


def pd_tag(i):
    return "dead" if i in DAMAGED_PDS else ("PD13*" if i == 13 else "")


def slopes_mv_min(t, Y):
    m = t / 60
    return np.array(
        [
            np.polyfit(m, Y[:, i], 1)[0] * 1000 if len(t) >= 2 else np.nan
            for i in range(Y.shape[1])
        ]
    )


def _sweep_raw(pic, channel, vals, base, settle=0.0, repeats=5):
    """Sweep one channel (monotone), return (vals, mean14, std14) from raw reads."""
    Y, S = [], []
    for x in np.sort(np.asarray(vals, float)):
        v = base.copy()
        v[channel] = x
        pic.measure_raw(v)  # apply
        if settle:
            time.sleep(settle)
        ys = np.asarray([pic.measure_raw(v) for _ in range(repeats)])
        Y.append(ys.mean(0))
        S.append(ys.std(0))
    return np.sort(np.asarray(vals, float)), np.asarray(Y), np.asarray(S)


def _crossing(x, y, target):
    """First x where y(x) crosses target (linear interp), or None."""
    for k in range(1, len(y)):
        a, b = y[k - 1], y[k]
        if (a - target) * (b - target) <= 0 and a != b:
            return float(x[k - 1] + (target - a) / (b - a) * (x[k] - x[k - 1]))
    return None


# ---- test 1: thermal drift / "global warming" ----


def cmd_warming(args, pic):
    v = np.zeros(NUM_DAC)
    chans = args.channels or []
    for c in chans:
        v[c] = args.drive
    print(
        f"warming [{args.tag}]: {args.minutes:.1f} min, ~{args.period}s/sample, "
        f"drive={args.drive}V on {chans or 'none (baseline)'}.  Ctrl-C to stop early.\n"
    )
    t0 = time.time()
    T, Y = [], []
    try:
        while (t := time.time() - t0) <= args.minutes * 60:
            y = pic.measure_raw(v)
            T.append(t)
            Y.append(y)
            print(
                f"  {t:6.1f}s  live=" + " ".join(f"{y[i]*1000:5.1f}" for i in LIVE_PDS),
                end="\r",
            )
            time.sleep(max(0.0, args.period))
    except KeyboardInterrupt:
        print("\nstopped early")
    T, Y = np.asarray(T), np.asarray(Y)
    if len(T) < 2:
        sys.exit("\nnot enough samples")
    _warming_report(T, Y, args.tag)
    p = save_matrix(
        os.path.join(OUT, f"warming_{args.tag}.csv"),
        T,
        Y,
        note=f"warming tag={args.tag} drive={args.drive} channels={chans}",
    )
    _plot_drift(
        [(args.tag, T, Y)],
        os.path.join(OUT, f"warming_{args.tag}.png"),
        f"thermal drift — {args.tag}",
    )
    print(f"\nsaved {p}")


def _warming_report(T, Y, tag):
    sl = slopes_mv_min(T, Y)
    print(f"\n[{tag}] {T[-1]/60:.1f} min, n={len(T)}   (Δ = end − start)")
    print("  PD          slope(mV/min)  start(mV)   end(mV)   Δ(mV)")
    for i in range(Y.shape[1]):
        print(
            f"  {i:2d} {pd_tag(i):5s}  {sl[i]:12.2f}  {Y[0,i]*1000:9.2f} {Y[-1,i]*1000:9.2f} {(Y[-1,i]-Y[0,i])*1000:8.2f}"
        )
    base = np.nanmean([sl[i] for i in DAMAGED_PDS])
    print(
        f"  → dead-PD baseline drift (no light ⇒ thermal/electronic): {base:+.2f} mV/min"
    )


# ---- test 1b: compare two warming runs (TEC on vs off) ----


def cmd_compare(args, _pic=None):
    Ta, Ya = load_run(args.a)
    Tb, Yb = load_run(args.b)
    sa, sb = slopes_mv_min(Ta, Ya), slopes_mv_min(Tb, Yb)
    na, nb = os.path.basename(args.a), os.path.basename(args.b)
    print(f"drift comparison   A={na}   B={nb}      (slope mV/min)")
    print("  PD          A        B      B−A")
    for i in range(len(sa)):
        print(f"  {i:2d} {pd_tag(i):5s} {sa[i]:8.2f} {sb[i]:8.2f} {sb[i]-sa[i]:8.2f}")
    da = np.nanmean([sa[i] for i in DAMAGED_PDS])
    db = np.nanmean([sb[i] for i in DAMAGED_PDS])
    print(f"\n  dead-PD baseline drift:  A={da:+.2f}   B={db:+.2f}   mV/min")
    worse = "B" if abs(db) > abs(da) else "A"
    print(
        f"  → {worse} warms {abs(db-da):.2f} mV/min faster — the run with more baseline drift is TEC-off."
    )
    os.makedirs(OUT, exist_ok=True)
    _plot_drift(
        [(na, Ta, Ya), (nb, Tb, Yb)],
        os.path.join(OUT, "warming_compare.png"),
        "thermal drift: A vs B",
        dead_only=True,
    )
    print(f"  saved {os.path.join(OUT, 'warming_compare.png')}")


# ---- test 2: dark counts / zero-input noise floor ----


def cmd_dark(args, pic):
    v = np.zeros(NUM_DAC)
    print(
        f"dark [{args.tag}]: {args.samples} samples at 0 V. "
        f"For TRUE dark counts, block or switch off the laser first.\n"
    )
    Y = []
    for k in range(args.samples):
        Y.append(pic.measure_raw(v))
        if args.period:
            time.sleep(args.period)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{args.samples}", end="\r")
    Y = np.asarray(Y)
    mean, std = Y.mean(0) * 1000, Y.std(0) * 1000
    print(f"\n\n[{args.tag}] per-PD dark level and read noise (mV):")
    print("  PD          mean      σ(noise)   note")
    for i in range(NUM_ADC_RAW):
        print(
            f"  {i:2d} {pd_tag(i):5s} {mean[i]:9.2f} {std[i]:9.2f}    {pd_tag(i) or 'live'}"
        )
    print(
        f"\n  ADC LSB ≈ {LSB_MV:.1f} mV  → σ near this floor is quantisation/read noise."
    )
    print(
        f"  live-PD median noise σ ≈ {np.median([std[i] for i in MIDDLE_PDS]):.2f} mV "
        f"(this is the SNR=1 floor for test 4)."
    )
    p = save_matrix(
        os.path.join(OUT, f"dark_{args.tag}.csv"),
        np.arange(len(Y)),
        Y,
        note=f"dark tag={args.tag} samples={args.samples}",
    )
    print(f"  saved {p}")


# ---- test 3: PD13 bypass — trim its power to the middle-PD level ----


def cmd_pd13(args, pic):
    base = np.zeros(NUM_DAC)
    y0 = pic.measure_raw(base)
    target = (
        args.target if args.target is not None else float(np.median(y0[MIDDLE_PDS]))
    )
    print(
        f"PD13 bypass calibration.  base: PD13={y0[13]*1000:.1f} mV, "
        f"middle-PD median={target*1000:.1f} mV (target).\n"
    )

    if args.channel is None:
        print(f"discovering which channel attenuates PD13 (each → {args.vmax} V)…")
        d0 = pic.measure_raw(base)[13]
        eff = []
        for c in range(NUM_DAC):
            v = base.copy()
            v[c] = args.vmax
            pic.measure_raw(v)
            eff.append((c, pic.measure_raw(v)[13] - d0))
        eff.sort(key=lambda t: abs(t[1]), reverse=True)
        print("  top channels by |ΔPD13| (negative = attenuates):")
        for c, d in eff[:8]:
            print(f"    ch{c:2d}:  ΔPD13 = {d*1000:+7.1f} mV")
        best = eff[0][0]
        print(
            f"\n  → likely bypass attenuator: ch{best}. "
            f"Re-run:  python -m pic bringup pd13 --channel {best}"
            + (" --mock" if args.mock else "")
        )
        return

    vals = np.arange(0, args.vmax + 1e-9, args.step)
    vv, Y, S = _sweep_raw(
        pic, args.channel, vals, base, settle=args.settle, repeats=args.repeats
    )
    pd13 = Y[:, 13]
    print(f"sweep ch{args.channel}:  V → PD13")
    for x, p13 in zip(vv, pd13):
        bar = "#" * int(pd13.max() and p13 / pd13.max() * 30)
        print(f"  {x:4.2f} V  {p13*1000:7.1f} mV  {bar}")
    vc = _crossing(vv, pd13, target)
    if vc is None:
        print(
            f"\n  target {target*1000:.0f} mV not reached; PD13 spans "
            f"{pd13.min()*1000:.0f}–{pd13.max()*1000:.0f} mV over 0–{args.vmax} V."
        )
    else:
        print(
            f"\n  → set ch{args.channel} ≈ {vc:.2f} V to bring PD13 to the "
            f"middle-PD level (~{target*1000:.0f} mV)."
        )
    _plot_pd13(
        vv, pd13, target, args.channel, os.path.join(OUT, f"pd13_ch{args.channel}.png")
    )


# ---- test 4: drive level for SNR = 1 ----


def cmd_snr(args, pic):
    if args.dark:
        _, D = load_run(args.dark)
        dark_mean, dark_std = D.mean(0), D.std(0)
        print(f"dark from {os.path.basename(args.dark)} ({len(D)} samples)")
    else:
        v0 = np.zeros(NUM_DAC)
        D = np.asarray([pic.measure_raw(v0) for _ in range(args.samples)])
        dark_mean, dark_std = D.mean(0), D.std(0)
        print(f"dark measured inline ({args.samples} samples at 0 V)")

    base = np.zeros(NUM_DAC)
    vals = np.arange(0, args.vmax + 1e-9, args.step)
    vv, Y, S = _sweep_raw(
        pic, args.channel, vals, base, settle=args.settle, repeats=args.repeats
    )

    if args.pd is not None:
        pd = args.pd
    else:  # the live PD (not the bypass) this channel drives hardest
        resp = [(Y[-1, i] - dark_mean[i]) / max(dark_std[i], 1e-9) for i in MIDDLE_PDS]
        pd = MIDDLE_PDS[int(np.argmax(resp))]

    sig = Y[:, pd] - dark_mean[pd]
    noise = np.maximum(S[:, pd], dark_std[pd])  # measured σ, floored at dark read noise
    snr = sig / noise
    print(
        f"\nSNR on PD{pd}, drive ch{args.channel}.  "
        f"dark: mean={dark_mean[pd]*1000:.1f} mV, σ={dark_std[pd]*1000:.2f} mV"
    )
    print("  drive(V)  PD(mV)  signal(mV)   σ(mV)     SNR")
    for x, y, s, n, r in zip(vv, Y[:, pd] * 1000, sig * 1000, noise * 1000, snr):
        print(f"  {x:6.2f}  {y:7.1f}  {s:9.2f}  {n:7.2f}  {r:7.2f}")
    vc = _crossing(vv, snr, 1.0)
    if vc is None:
        rng = f"{snr.min():.2f}–{snr.max():.2f}"
        print(
            f"\n  SNR never crosses 1 over 0–{args.vmax} V (range {rng}); widen the sweep "
            f"or raise laser power."
        )
    else:
        sc = np.interp(vc, vv, sig) * 1000
        print(
            f"\n  → SNR=1 at drive ≈ {vc:.2f} V  (signal ≈ {sc:.2f} mV above dark, "
            f"≈ the σ={dark_std[pd]*1000:.2f} mV noise floor)."
        )
        print(f"    'power for SNR=1' in PD terms = {sc:.2f} mV above dark on PD{pd}.")
    _plot_snr(
        vv, snr, args.channel, pd, os.path.join(OUT, f"snr_ch{args.channel}_pd{pd}.png")
    )


# ---- plots ----


def _agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_drift(runs, path, title, dead_only=False):
    plt = _agg()
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ["-", "--"]
    for k, (name, T, Y) in enumerate(runs):
        m = T / 60
        pds = DAMAGED_PDS if dead_only else range(Y.shape[1])
        for i in pds:
            d = (Y[:, i] - Y[0, i]) * 1000
            lbl = (
                f"{name}·PD{i}"
                if len(runs) > 1
                else f"PD{i}{'*' if i in DAMAGED_PDS else ''}"
            )
            ax.plot(
                m,
                d,
                styles[k % 2],
                lw=1.4 if i in DAMAGED_PDS else 0.9,
                alpha=0.9 if i in DAMAGED_PDS else 0.6,
                label=lbl if (dead_only or len(runs) == 1) else None,
            )
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("minutes")
    ax.set_ylabel("Δ from start (mV)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_pd13(vv, pd13, target, ch, path):
    plt = _agg()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vv, pd13 * 1000, "o-", color="#2563eb")
    ax.axhline(
        target * 1000,
        color="#dc2626",
        ls="--",
        label=f"middle-PD level {target*1000:.0f} mV",
    )
    ax.set_xlabel(f"ch{ch} (V)")
    ax.set_ylabel("PD13 (mV)")
    ax.set_title(f"PD13 bypass attenuation via ch{ch}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_snr(vv, snr, ch, pd, path):
    plt = _agg()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vv, snr, "o-", color="#16a34a")
    ax.axhline(1.0, color="#dc2626", ls="--", label="SNR = 1")
    ax.set_xlabel(f"ch{ch} (V)")
    ax.set_ylabel(f"SNR on PD{pd}")
    ax.set_title(f"SNR vs drive — ch{ch} → PD{pd}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def hwtests_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m pic bringup", description="PIC hardware bring-up tests"
    )
    ap.add_argument(
        "--mock", action="store_true", help="use the simulator, no hardware"
    )
    ap.add_argument("--port", help="serial device override (else auto-detect)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser(
        "warming", help="thermal drift over time (run TEC-on and TEC-off)"
    )
    w.add_argument("--minutes", type=float, default=10.0)
    w.add_argument("--period", type=float, default=2.0, help="seconds between samples")
    w.add_argument(
        "--drive", type=float, default=0.0, help="fixed voltage on --channels"
    )
    w.add_argument(
        "--channels",
        type=lambda s: [int(x) for x in s.split(",") if x != ""],
        default=[],
    )
    w.add_argument("--tag", default="tec_on")

    c = sub.add_parser("compare", help="compare two warming CSVs")
    c.add_argument("a")
    c.add_argument("b")

    d = sub.add_parser("dark", help="dark / zero-input noise floor")
    d.add_argument("--samples", type=int, default=300)
    d.add_argument("--period", type=float, default=0.0)
    d.add_argument("--tag", default="laser_off")

    p = sub.add_parser("pd13", help="trim the PD13 bypass to the middle-PD level")
    p.add_argument(
        "--channel",
        type=int,
        default=None,
        help="attenuator channel (omit to discover)",
    )
    p.add_argument(
        "--target",
        type=float,
        default=None,
        help="target PD13 volts (default: middle-PD median)",
    )
    p.add_argument("--vmax", type=float, default=VMAX)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--settle", type=float, default=0.0)
    p.add_argument("--repeats", type=int, default=5)

    s = sub.add_parser("snr", help="find the drive level where SNR = 1")
    s.add_argument("--channel", type=int, required=True, help="drive channel to sweep")
    s.add_argument(
        "--pd", type=int, default=None, help="target PD (default: most-driven live PD)"
    )
    s.add_argument(
        "--dark",
        default=None,
        help="dark CSV from the dark test (else measured inline)",
    )
    s.add_argument(
        "--samples", type=int, default=200, help="inline dark samples if --dark omitted"
    )
    s.add_argument("--vmax", type=float, default=VMAX)
    s.add_argument("--step", type=float, default=0.1)
    s.add_argument("--settle", type=float, default=0.0)
    s.add_argument("--repeats", type=int, default=20)

    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args)

    pic = open_rig(args)
    try:
        {"warming": cmd_warming, "dark": cmd_dark, "pd13": cmd_pd13, "snr": cmd_snr}[
            args.cmd
        ](args, pic)
    finally:
        pic.close()  # zeros the DACs
    return 0


# ---- thermal settling-time + loop-speed budget ----

NOISE_V = 0.005  # 5 mV ADC/dark noise floor (measured ~4.9 mV)
BOLD, RST = "\033[1m", "\033[0m"


def _poll(pic, v, seconds):
    """Hold DACs at v and read as fast as the firmware replies; returns (t_mid, raw14) pairs."""
    out, t0 = [], time.perf_counter()
    while True:
        ta = time.perf_counter()
        raw = pic.measure_raw(v)
        tb = time.perf_counter()
        out.append(((ta + tb) / 2 - t0, raw))
        if tb - t0 >= seconds:
            return out


def _hold(pic, v, seconds):
    """Drive ``v`` for ``seconds`` and return the mean of the last few reads."""
    tail = _poll(pic, v, seconds)
    return np.mean([r for _, r in tail[-3:]], axis=0)


def _pick_channel(pic, vhi):
    """Find the (DAC channel, live PD) pair with the largest response to a step."""
    base = pic.measure_raw(np.zeros(C.NUM_DAC))
    live = [i for i in range(C.NUM_ADC_RAW) if i not in C.DAMAGED_PDS]
    best = (-1.0, 0, live[0])
    for ch in range(C.NUM_DAC):
        v = np.zeros(C.NUM_DAC)
        v[ch] = vhi
        d = np.abs(pic.measure_raw(v) - base)
        pd = max(live, key=lambda i: d[i])
        if d[pd] > best[0]:
            best = (float(d[pd]), ch, pd)
    return best[1], best[2], best[0]


def _fit_edge(t, y):
    """Fit y(t)=y_inf+(y0-y_inf)exp(-t/tau). Returns dict or None if degenerate."""
    from scipy.optimize import curve_fit

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y0, yinf = y[0], np.mean(y[-3:])
    swing = abs(yinf - y0)
    if swing < NOISE_V:  # step smaller than the noise floor
        return {
            "tau": 0.0,
            "t99": 0.0,
            "t_floor": 0.0,
            "y0": y0,
            "yinf": yinf,
            "swing": swing,
            "resolved": False,
            "note": "swing < noise floor",
        }

    def model(tt, yi, amp, tau):
        return yi - amp * np.exp(-tt / tau)

    p0 = [yinf, yinf - y0, max(t[-1] / 5, 1e-3)]
    try:
        popt, _ = curve_fit(
            model,
            t,
            y,
            p0=p0,
            maxfev=20000,
            bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, t[-1] * 10]),
        )
    except Exception:
        return None
    yi, amp, tau = popt
    t_floor = max(0.0, tau * np.log(abs(amp) / NOISE_V)) if abs(amp) > NOISE_V else 0.0
    # if the very first sample is already within the noise floor, settling is
    # faster than one ADC window and this method cannot resolve it.
    resolved = abs(y[0] - yinf) > NOISE_V and t[0] < 0.5 * t_floor if t_floor else False
    return {
        "tau": float(tau),
        "t99": float(4.6 * tau),
        "t_floor": float(t_floor),
        "y0": float(y0),
        "yinf": float(yi),
        "swing": float(swing),
        "resolved": bool(resolved),
        "note": "",
    }


def _report_edge(name, fit, n, span):
    if fit is None:
        print(f"  {name:8s}  fit failed (noisy / non-exponential trace)")
        return
    print(
        f"  {name:8s}  swing {fit['swing']*1000:6.1f} mV   "
        f"tau {fit['tau']*1000:7.1f} ms   t99 {fit['t99']*1000:7.1f} ms   "
        f"t_floor {fit['t_floor']*1000:7.1f} ms"
    )
    if not fit["resolved"] and fit["swing"] >= NOISE_V:
        print(
            f"            note: settles within the first read (~{span/n*1000:.0f} ms "
            f"sample spacing) -- faster than the firmware can resolve; "
            f"reduce the ADC window to measure it directly."
        )
    elif fit["note"]:
        print(f"            note: {fit['note']}")


def cmd_step(pic, args):
    vhi = args.vhi
    print(f"{BOLD}[settling] step response, edge captured for {args.capture:g}s each{RST}")
    if args.all:
        ch_desc = "all 64 heaters together"
        vlo_vec = np.zeros(C.NUM_DAC)
        vhi_vec = np.full(C.NUM_DAC, vhi)
        # track the PD that moves most under the full-chip load
        base = pic.measure_raw(vlo_vec)
        d = np.abs(pic.measure_raw(vhi_vec) - base)
        live = [i for i in range(C.NUM_ADC_RAW) if i not in C.DAMAGED_PDS]
        pd = max(live, key=lambda i: d[i])
    else:
        ch = args.channel
        if ch is None:
            print("  scanning 64 channels for the strongest heater->PD coupling ...")
            ch, pd, sw = _pick_channel(pic, vhi)
            print(f"  using channel {ch} -> PD{pd}  (swing {sw*1000:.1f} mV)")
        else:
            base = pic.measure_raw(np.zeros(C.NUM_DAC))
            v = np.zeros(C.NUM_DAC)
            v[ch] = vhi
            d = np.abs(pic.measure_raw(v) - base)
            live = [i for i in range(C.NUM_ADC_RAW) if i not in C.DAMAGED_PDS]
            pd = max(live, key=lambda i: d[i])
        ch_desc = f"channel {ch} -> PD{pd}"
        vlo_vec = np.zeros(C.NUM_DAC)
        vhi_vec = np.zeros(C.NUM_DAC)
        vhi_vec[ch] = vhi
    print(f"  driving {ch_desc}, low=0 V high={vhi:g} V\n")

    # settle low, step UP, then (already high) step DOWN
    _hold(pic, vlo_vec, args.presettle)
    up = _poll(pic, vhi_vec, args.capture)
    _hold(pic, vhi_vec, args.presettle)
    down = _poll(pic, vlo_vec, args.capture)

    tu = [t for t, _ in up]
    yu = [r[pd] for _, r in up]
    td = [t for t, _ in down]
    yd = [r[pd] for _, r in down]
    fu, fd = _fit_edge(tu, yu), _fit_edge(td, yd)

    print(f"{BOLD}results (PD{pd}, {len(up)} up / {len(down)} down samples):{RST}")
    _report_edge("heat up", fu, len(up), args.capture)
    _report_edge("cool dn", fd, len(down), args.capture)

    floors = [f["t_floor"] for f in (fu, fd) if f and f["resolved"]]
    if floors:
        rec = max(floors)
        print(
            f"\n  -> recommended host dwell (settle_s): {rec:.3f} s "
            f"(slower edge's time-to-noise-floor)"
        )
    else:
        print(
            f"\n  -> both edges settle within one ADC window ({C.ADC_AVG_MS} ms): "
            f"no host dwell needed; the loop is window/serial-bound, not settling-bound.\n"
            f"     to measure the true tau, reflash with a shorter ADC window and re-run."
        )

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["edge", "t_s", f"PD{pd}_V"])
            for t, y in zip(tu, yu):
                w.writerow(["up", f"{t:.4f}", f"{y:.4f}"])
            for t, y in zip(td, yd):
                w.writerow(["down", f"{t:.4f}", f"{y:.4f}"])
        print(f"  wrote traces -> {args.csv}")
    return 0


def cmd_bench(pic, args):
    v = np.full(C.NUM_DAC, args.vhi)
    line_bytes = len(",".join(f"{x:.1f}" for x in v) + "\n")
    rx_bytes = C.NUM_ADC_RAW * 6  # ~"1.234," per value
    print(f"{BOLD}[bench] {args.n} round trips at {v[0]:g} V on all channels{RST}")
    pic.measure_raw(v)  # warm up
    t0 = time.perf_counter()
    for _ in range(args.n):
        pic.measure_raw(v)
    dt = (time.perf_counter() - t0) / args.n

    baud = C.BAUD_RATE
    tx_ms = line_bytes * 10 / baud * 1000  # 8N1 = 10 bits/byte
    rx_ms = rx_bytes * 10 / baud * 1000
    win_ms = C.ADC_AVG_MS
    overhead_ms = dt * 1000 - tx_ms - rx_ms - win_ms

    print(f"\n  measured round trip   {dt*1000:6.1f} ms   ({1/dt:5.1f} reads/s)")
    print(f"  {BOLD}budget (estimated split){RST}")
    print(f"    serial TX  {tx_ms:6.1f} ms   ({line_bytes} B in  @ {baud} baud)")
    print(f"    serial RX  {rx_ms:6.1f} ms   (~{rx_bytes} B back)")
    print(f"    ADC window {win_ms:6.1f} ms   (firmware fixed average)")
    print(
        f"    overhead   {overhead_ms:6.1f} ms   (firmware parse + analogRead loop + host)"
    )

    # project the two cheap levers
    def proj(baud2, win2):
        return (
            ((line_bytes + rx_bytes) * 10 / baud2 * 1000) + win2 + max(overhead_ms, 0)
        )

    print(f"\n  {BOLD}projected round trip if we ...{RST}")
    print(
        f"    baud 115200 -> 500000            {proj(500000, win_ms):6.1f} ms  ({1000/proj(500000, win_ms):5.1f}/s)"
    )
    print(
        f"    ADC window 31 -> 10 ms           {proj(baud, 10):6.1f} ms  ({1000/proj(baud, 10):5.1f}/s)"
    )
    print(
        f"    both                             {proj(500000, 10):6.1f} ms  ({1000/proj(500000, 10):5.1f}/s)"
    )
    print(f"    (shrinking the window is gated by the `step` result + the noise floor)")
    return 0


def _selftest(args):
    """No hardware: synth an exponential trace and confirm the fit recovers tau."""
    print(f"{BOLD}[selftest] synthetic edge, true tau = 0.120 s{RST}")
    t = np.linspace(0, 2.0, 50)
    rng = np.random.default_rng(0)
    y = 0.80 - (0.80 - 0.20) * np.exp(-t / 0.120) + rng.normal(0, 0.003, t.size)
    f = _fit_edge(t, y)
    _report_edge("synth", f, t.size, 2.0)
    ok = f and abs(f["tau"] - 0.120) < 0.02
    print(f"  fit tau = {f['tau']*1000:.1f} ms  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def settling_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pic bringup",
        description="PIC settling-time + loop-speed characterization",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("step", help="thermal step response (up + down)")
    sp.add_argument(
        "port", nargs="?", help="serial port (else auto-detect / $PIC_PORT)"
    )
    sp.add_argument(
        "--channel",
        type=int,
        default=None,
        help="DAC channel to step (default: auto-pick strongest)",
    )
    sp.add_argument(
        "--all",
        action="store_true",
        help="step all 64 heaters together (full chip load)",
    )
    sp.add_argument(
        "--vhi", type=float, default=C.VPI_NOMINAL, help="high voltage of the step"
    )
    sp.add_argument(
        "--capture", type=float, default=5.0, help="seconds to track each edge"
    )
    sp.add_argument(
        "--presettle",
        type=float,
        default=3.0,
        help="seconds to settle before each step",
    )
    sp.add_argument("--csv", help="write the raw traces to this CSV")
    sp.add_argument(
        "--selftest",
        action="store_true",
        help="no hardware: verify the fit on a synthetic trace",
    )
    bp = sub.add_parser("bench", help="round-trip time budget + speedup projection")
    bp.add_argument(
        "port", nargs="?", help="serial port (else auto-detect / $PIC_PORT)"
    )
    bp.add_argument("--n", type=int, default=50, help="round trips to time")
    bp.add_argument(
        "--vhi",
        type=float,
        default=C.VPI_NOMINAL,
        help="drive voltage during the bench",
    )
    a = ap.parse_args(argv)

    if getattr(a, "selftest", False):
        return _selftest(a)

    port, cands = find_port(getattr(a, "port", None))
    if not port:
        print(
            "no Arduino serial device found. plug it in or pass a port / set $PIC_PORT."
        )
        return 1
    print(
        f"using {port}"
        + (f"  (candidates: {', '.join(cands)})" if len(cands) > 1 else "")
    )
    pic = PIC(port=port)
    try:
        pic.open()
    except Exception as e:
        print(f"could not open {port}: {type(e).__name__}: {e}")
        return 2
    try:
        return cmd_step(pic, a) if a.cmd == "step" else cmd_bench(pic, a)
    except PICError as e:
        print(f"firmware/link error: {e}")
        return 3
    finally:
        pic.close()  # drives DACs back to 0 V


# ---- unified `python -m pic bringup` dispatcher ----

_SETTLING_CMDS = ("step", "bench")


def main(argv=None):
    """Dispatch to the settling CLI for `step`/`bench`, else the bring-up-tests CLI.
    The two share no subcommand names, so the first token disambiguates cleanly."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _SETTLING_CMDS:
        return settling_main(argv)
    return hwtests_main(argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
