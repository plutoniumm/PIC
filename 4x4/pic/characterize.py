"""Bootstrap calibration: sweep each heater, fit its fringe, get (Vpi, phi0).

    y(V) = A + B cos( pi (V/Vpi)^2 + phi0 )

This is the cheap first pass. It gives every heater an independent estimate good enough
to program with, and a starting point for `learn.unitary_fit`, which then fits all 18
heaters jointly through the twin using data the sweeps already collected.

Two heaters in three are invisible with the mesh at 0 V, and that is physics rather than a
fault. An external phase only shows up where its interferometer actually splits light, so a
phi heater whose MZI happens to sit at bar or cross modulates nothing; sweeping from a few
random base biases instead of a single all-zero one puts every MZI somewhere useful in at
least one of them. The output phase trimmers are a harder case: a diagonal phase screen at
the output cannot change |U x|^2 at all, so intensity detection can never see them. They
come back rejected here, correctly, and they stay unidentifiable in the joint fit too --
which costs nothing, because anything that cannot be measured also cannot affect a result
read out in intensity.

Sign convention, stated once because getting it wrong is silent: B is forced non-negative
and phi0 absorbs the flip. If you take abs(B) without adding pi to phi0 you reflect the
fringe, the fitted maximum and minimum swap places, and the residual still looks fine --
that exact bug put a third of the 6x6 channels' operating points on the wrong extremum.
"""

from __future__ import annotations

import numpy as np

from .config import VOLTAGE_MAX, VPI_NOMINAL
from theory.clements import NMODE

# The band a thermo-optic shifter on this process can plausibly sit in. A fit outside it is
# a fit to noise dressed up as a fringe, and it will read back with a fine residual.
#
# The upper edge is measured, not chosen: the six wired channels fit Vpi = 2.3 - 4.6 V
# against 8000 rows of `mrunal/Combined_Project_Data (1).xlsx` (see `pic.sim.BENCH_VPI`),
# so the 4.0 V this used to sit at excluded a real channel and pushed its fit onto an alias.
VPI_MIN, VPI_MAX = 0.4, 6.0

SAMPLES_PER_FRINGE = 4  # the sampling floor a fitted fringe has to clear to be believed


def resolvable_span(n_levels: int, per_fringe: int = SAMPLES_PER_FRINGE) -> float:
    """Widest phase span, in units of pi, a sweep of `n_levels` can honestly resolve.

    On a uniform-V^2 grid the fringe is a pure sinusoid in V^2, so this is Nyquist with
    headroom. It matters because an aliased fit does not look wrong: a heater whose true
    span was 11 pi was seen here fitting a 84 pi alias at r2 = 1.0000 and a residual better
    than most of the honest fits. Only the sample count catches that."""
    return 2.0 * (n_levels - 1) / per_fringe


def fringe(v, A, B, vpi, phi0):
    return A + B * np.cos(np.pi * (np.asarray(v, float) / vpi) ** 2 + phi0)


def seed_vpi(v, y, vmax: float = VOLTAGE_MAX):
    """Estimate Vpi straight off the data, by FFT, with no search.

    Phase is linear in V^2, so against u = V^2 the trace is a pure sinusoid:
    y(u) = A + B cos(pi u / Vpi^2 + phi0). Resample onto a uniform u grid, take the
    spectrum, and the dominant bin *is* the heater's Vpi -- frequency f cycles per unit u
    means Vpi = 1/sqrt(2f). This replaces a multi-start search over Vpi with one transform,
    and it is what makes a full 18-heater characterization affordable.

    Candidates that the sweep cannot resolve are dropped here rather than fitted: see
    `resolvable_span`. Returns the best few, strongest first, so the least-squares fit that
    follows starts from a short list rather than a grid."""
    u = np.asarray(v, float).ravel() ** 2
    y = np.asarray(y, float).ravel()
    n = max(256, 8 * u.size)
    uu = np.linspace(u.min(), u.max(), n)
    yy = np.interp(uu, u, y[np.argsort(u)] if not np.all(np.diff(u) >= 0) else y)
    spec = np.abs(np.fft.rfft(yy - yy.mean()))
    freq = np.fft.rfftfreq(n, d=(uu[1] - uu[0]))
    order = np.argsort(spec)[::-1]
    span_max = resolvable_span(u.size)
    out = []
    for k in order:
        if freq[k] <= 0:
            continue
        vpi = 1.0 / np.sqrt(2 * freq[k])
        if VPI_MIN <= vpi <= VPI_MAX and (vmax / vpi) ** 2 <= span_max:
            out.append(float(vpi))
        if len(out) >= 3:
            break
    return out or [VPI_NOMINAL]


def fit_fringe(v, y, vpi_seeds=None, vmax: float = VOLTAGE_MAX):
    """Least-squares fit of one heater's fringe, seeded by `seed_vpi`.

    Returns a dict with A, B, vpi, phi0, rmse, r2 and `visibility` = B / A, the contrast
    the channel actually produced. A fit with low visibility is a fit to noise however good
    its residual looks, so callers gate on visibility, not rmse. A fit whose Vpi runs to the
    edge of the physical band is rejected outright for the same reason."""
    from scipy.optimize import curve_fit

    v = np.asarray(v, float).ravel()
    y = np.asarray(y, float).ravel()
    if v.size != y.size or v.size < 8:
        raise ValueError(f"need >=8 paired samples, got {v.size}/{y.size}")

    seeds = seed_vpi(v, y, vmax) if vpi_seeds is None else list(vpi_seeds)
    A0, B0 = float(y.mean()), float((y.max() - y.min()) / 2)
    best = None
    for vpi0 in seeds:
        for phi00 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            try:
                p, _ = curve_fit(fringe, v, y, p0=[A0, B0, vpi0, phi00], maxfev=4000,
                                 bounds=([-np.inf, 0.0, VPI_MIN, -4 * np.pi],
                                         [np.inf, np.inf, VPI_MAX, 4 * np.pi]))
            except (RuntimeError, ValueError):
                continue
            r = float(np.sqrt(np.mean((fringe(v, *p) - y) ** 2)))
            if best is None or r < best[0]:
                best = (r, p)
    if best is None:
        raise RuntimeError("no fringe fit converged")

    rmse, (A, B, vpi, phi0) = best
    if not VPI_MIN * 1.01 <= vpi <= VPI_MAX * 0.99:  # ran to a bound: not a real fringe
        raise RuntimeError(f"fitted Vpi {vpi:.2f} V ran to the edge of the physical band")
    span = (vmax / vpi) ** 2
    span_max = resolvable_span(v.size)
    if span > span_max:  # an alias, however good the residual looks
        raise RuntimeError(f"fitted span {span:.0f} pi exceeds what {v.size} levels resolve "
                           f"({span_max:.0f} pi); sweep more levels")
    if B < 0:  # keep B >= 0 and let phi0 carry the flip; see the module docstring
        B, phi0 = -B, phi0 + np.pi
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "A": float(A), "B": float(B), "vpi": float(vpi),
        "phi0": float(np.mod(phi0, 2 * np.pi)), "rmse": rmse,
        "r2": float(1 - np.sum((fringe(v, A, B, vpi, phi0) - y) ** 2) / ss) if ss > 0 else 0.0,
        "visibility": float(B / A) if A > 0 else 0.0,
        # the fringe in volts, which is what decides whether it is above the readout noise;
        # `visibility` is the same number relative to a mean that may itself be noise
        "amplitude": float(B),
        "span_pi": float((vmax / vpi) ** 2),
        # residual relative to the fringe amplitude: how well the cosine actually describes
        # this trace, independent of how big the trace was
        "score": float(rmse / max(B, 1e-9)),
    }


NO_FIT = {"visibility": 0.0, "amplitude": 0.0, "rmse": float("nan"), "r2": float("nan"),
          "score": float("inf"), "pd": None}

# Fringe amplitude, in photodiode volts, a fit has to clear to be a fringe at all.
#
# Visibility alone cannot do this job, and the bench proves it. `pic.sim` reads out at a
# measured 0.6-5.7 mV of noise per channel (`Drift_data.xlsx` Experiment 2), and several
# outputs sit near a null at under 1 mV -- port 3's PD2 averages 0.4 mV. On a trace like
# that B/A is a ratio of noise to noise and comes back at 0.5, so a run against the bench
# simulator reported all twelve *unwired* channels as characterized, with fitted Vpi between
# 0.7 and 4.9 V. Three times the worst measured read noise rejects every one of them and
# keeps all six real channels.
MIN_AMPLITUDE_V = 0.017


def better(a, b, min_visibility: float = 0.05, min_amplitude: float = MIN_AMPLITUDE_V,
           min_r2: float = 0.5) -> dict:
    """Pick the more trustworthy of two fringe fits.

    Contrast is a gate, not a ranking. A big fringe fitted badly is worse than a small one
    fitted cleanly, and ranking by size alone picks up traces where two heaters moved at
    once -- measured here as a 0.77 V Vpi error on channels whose clean fit was exact.

    The gate is two-sided: relative contrast (`visibility`) catches a heater that modulates
    a bright output only slightly, absolute amplitude catches noise on a dark one, and r2
    catches the remaining case -- a fit that grew a large B by riding the near-linear part of
    a cosine it never turns over. Against `pic.sim` all three are needed: contrast alone
    passes 12 channels of 12 that have no heater on them, contrast plus amplitude still
    passes one at r2 = 0.19."""
    def ok(f):
        return (f.get("visibility", 0) >= min_visibility
                and f.get("amplitude", 0) >= min_amplitude
                and (f.get("r2") or 0) >= min_r2)

    ok_a, ok_b = ok(a), ok(b)
    if ok_a != ok_b:
        return a if ok_a else b
    if ok_a:
        return a if a["score"] <= b["score"] else b
    return a if a.get("amplitude", 0) >= b.get("amplitude", 0) else b


def best_fringe(levels, curves, pds=None, min_visibility: float = 0.05,
                min_amplitude: float = MIN_AMPLITUDE_V, min_r2: float = 0.5) -> dict:
    """Fit one heater's sweep against every detector it was recorded on, keep the best.
    `curves` is (levels, n_detectors)."""
    curves = np.atleast_2d(np.asarray(curves, float))
    if curves.shape[0] != len(levels):
        curves = curves.T
    pds = list(range(curves.shape[1])) if pds is None else list(pds)
    best = dict(NO_FIT)
    for k, p in enumerate(pds):
        try:
            f = fit_fringe(levels, curves[:, k])
        except (RuntimeError, ValueError):
            continue
        f["pd"] = int(p)
        best = better(f, best, min_visibility, min_amplitude, min_r2)
    return best


def probe_inputs(nmode: int = None, paired: bool = True):
    """The set of input fields a full characterization has to be run through.

    The four single ports come free: the 1x4 switch selects them under program control, so
    they are a loop index rather than four fibre re-plugs.

    The two adjacent pairs do not. An external phase on a first-column MZI's input arm is a
    global phase when only that port is lit, so the heater modulates nothing however long
    you sweep it; lighting two ports at once makes it a relative phase and it appears. A 1x4
    switch cannot do that, so the pairs need an external splitter into two fibres of the
    array. Measured on the mock instrument: single ports alone identify 10 of the 12 mesh
    heaters, adding the pairs identifies all 12.

    `paired=False` returns only what the switch can reach on its own."""
    import numpy as _np

    n = NMODE if nmode is None else int(nmode)
    E = _np.eye(n, dtype=complex)
    single = [E[p] for p in range(n)]
    if not paired:
        return single
    return single + [(E[0] + E[1]) / 2**0.5, (E[2] + E[3]) / 2**0.5]


def random_bases(n: int, rng=None, channels=None, vmax: float = VOLTAGE_MAX):
    """Base biases to sweep from. The first is all-zero, the rest random, so a heater that
    is dark in one mesh state has other chances to be seen."""
    import numpy as _np

    from .layout import ACTIVE_DACS, N_HEATERS

    rng = _np.random.default_rng(0) if rng is None else rng
    channels = ACTIVE_DACS if channels is None else _np.asarray(channels, int)
    out = [_np.zeros(N_HEATERS)]
    for _ in range(max(0, n - 1)):
        v = _np.zeros(N_HEATERS)
        v[channels] = rng.uniform(0, vmax, channels.size)
        out.append(v)
    return out


def characterize(pic, session, *, pd=None, levels=None, channels=None, bases=None,
                 switch=None, ports=None, settle_s: float = 0.5, repeats: int = 5,
                 min_visibility: float = 0.05, min_amplitude: float = MIN_AMPLITUDE_V,
                 min_r2: float = 0.5, verbose: bool = True):
    """Sweep every active heater and fit its fringe.

    With `pd=None` each heater is fitted against all four outputs and the one it modulates
    hardest is kept; with `bases=None` the sweep is repeated from three base biases; and with
    a `switch`, from every input port too. Which detector sees a given heater, and whether
    any detector sees it at all, depends on where it sits in the mesh, on what the rest of
    the mesh is doing, and on which port is lit -- so pinning one of each leaves most of the
    chip uncharacterized, and an uncharacterized Vpi poisons the joint fit that follows.

    `session` is a live `laser_session` handle; the laser must be on and emitting or every
    heater modulates nothing and every fit is noise. Returns (results, calibration) where
    `results[dac]` is the fit dict plus an `ok` flag."""
    from theory.calib import Calibration

    from .acquisition import grid, settled_read
    from .config import OUT_PDS
    from .layout import ACTIVE_DACS, LABEL_OF_DAC, N_HEATERS

    pds = list(OUT_PDS) if pd is None else [int(pd)]

    levels = grid() if levels is None else np.asarray(levels, float)
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    bases = random_bases(3, channels=channels) if bases is None else list(bases)
    ports = ([None] if switch is None
             else list(range(NMODE)) if ports is None else list(ports))
    role = LABEL_OF_DAC

    vpi = np.full(N_HEATERS, VPI_NOMINAL)
    phi0 = np.zeros(N_HEATERS)
    results = {}
    for c in channels:
        if session.expired():
            if verbose:
                print("  watchdog reached; keeping the partial characterization.")
            break
        f = dict(NO_FIT)
        for port in ports:
            if session.expired():
                break
            if port is not None:
                switch.select(port)
            for bi, base in enumerate(bases):
                if session.expired():
                    break
                v = np.asarray(base, float).copy()
                ys = []
                for lv in levels:
                    v[c] = lv
                    ys.append(settled_read(pic, v, settle_s, repeats)[pds])
                    session.keepalive()
                g = best_fringe(levels, np.asarray(ys), pds, min_visibility, min_amplitude,
                                min_r2)
                g["base"], g["port"] = bi, port
                f = better(g, f, min_visibility, min_amplitude, min_r2)
        f["ok"] = bool(f["visibility"] >= min_visibility
                       and f.get("amplitude", 0) >= min_amplitude
                       and (f.get("r2") or 0) >= min_r2)
        f["dac"], f["label"] = int(c), role[int(c)]
        if f["ok"]:
            vpi[c], phi0[c] = f["vpi"], f["phi0"]
        results[int(c)] = f
        if verbose:
            print(f"  {role[int(c)]:<14} PD{f.get('pd')} port{f.get('port')} base{f.get('base')}  "
                  f"vis {f['visibility']:.3f}  amp {1e3 * f.get('amplitude', 0):5.1f}mV  "
                  f"r2 {f.get('r2', 0):+.3f}  "
                  f"Vpi {f.get('vpi', float('nan')):.2f}  "
                  f"{'ok' if f['ok'] else 'REJECTED (invisible in intensity)'}")

    n_ok = sum(r["ok"] for r in results.values())
    calib = Calibration(vpi, phi0, meta={
        "source": "pic.characterize.characterize",
        "pds": pds, "n_bases": len(bases), "ports": ports,
        "min_visibility": min_visibility, "min_amplitude": min_amplitude, "min_r2": min_r2,
        "n_ok": n_ok, "n_swept": len(results),
        "chip_c": getattr(session, "chip_c", None),
    })
    return results, calib


def digest(results) -> str:
    rows = [f"{'heater':<14} {'Vpi':>6} {'phi0/pi':>8} {'vis':>6} {'amp/mV':>7} {'r2':>7} "
            f"{'span/pi':>8}  ok"]
    for _, r in sorted(results.items()):
        rows.append(f"{r['label']:<14} {r.get('vpi', float('nan')):>6.2f} "
                    f"{r.get('phi0', float('nan')) / np.pi:>8.2f} {r['visibility']:>6.3f} "
                    f"{1e3 * r.get('amplitude', 0):>7.1f} "
                    f"{r.get('r2', float('nan')):>+7.3f} {r.get('span_pi', float('nan')):>8.1f}"
                    f"  {'y' if r['ok'] else 'n'}")
    n_ok = sum(r["ok"] for r in results.values())
    rows.append(f"\n{n_ok}/{len(results)} heaters characterized")
    return "\n".join(rows)
