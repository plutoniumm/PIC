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

from .config import VOLTAGE_MAX_CH, VPI_NOMINAL
from .session import WatchdogTripped, check_active
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


def seed_vpi(v, y, vmax: float = None):
    """Estimate Vpi straight off the data, by FFT, with no search.

    `vmax` is the top of the range these samples cover, and it defaults to exactly that --
    `max(v)` -- not to the global `VOLTAGE_MAX`. It only ever appears inside the alias
    gate, where the quantity that matters is the phase span the *data* covers: assuming a
    3.0 V sweep of a channel swept to 4.55 understates the span by 2.3x and lets an alias
    through, and assuming it of a channel swept to 2.25 overstates it and rejects an honest
    fit. Callers that know the channel's ceiling pass it and get the same number.

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
    vmax = float(np.sqrt(u.max())) if vmax is None else float(vmax)
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


def fit_fringe(v, y, vpi_seeds=None, vmax: float = None):
    """Least-squares fit of one heater's fringe, seeded by `seed_vpi`.

    Returns a dict with A, B, vpi, phi0, rmse, r2 and `visibility` = B / A, the contrast
    the channel actually produced. A fit with low visibility is a fit to noise however good
    its residual looks, so callers gate on visibility, not rmse. A fit whose Vpi runs to the
    edge of the physical band is rejected outright for the same reason.

    `vmax` sets both the alias gate and the reported `span_pi`, and defaults to the top of
    `v` -- the range these samples actually cover. It used to default to the global
    `VOLTAGE_MAX` of 3.0 V, which was right only while every channel was swept to 3.0 V. It
    is not any more: the per-channel ceilings run 1.50 to 4.75 V, so a fixed 3.0 both
    understated the span on the wide channels (loosening the alias gate that
    `resolvable_span` exists to be) and overstated it on the narrow ones, which is why
    stored `span_pi` disagreed with the sweep it came from."""
    from scipy.optimize import curve_fit

    v = np.asarray(v, float).ravel()
    y = np.asarray(y, float).ravel()
    if v.size != y.size or v.size < 8:
        raise ValueError(f"need >=8 paired samples, got {v.size}/{y.size}")
    vmax = float(np.max(np.abs(v))) if vmax is None else float(vmax)

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
# 0.7 and 4.9 V. 2-sigma of the worst per-PD read noise rejects every one of them and keeps
# all six real channels: PD0's 8.0 mV repeat rms is a difference of two reads, so per-read
# sigma is 5.7 mV and 2-sigma is 11 mV. The accept set is unchanged for any margin from
# 1.5-sigma to 3-sigma, so the exact multiplier is not load-bearing.
MIN_AMPLITUDE_V = 0.011


MIN_SNR = 6.0   # a fringe must stand this far above its own detector's read noise


def read_noise(pic, n: int = 16, settle_s: float = 1.0) -> np.ndarray:
    """Per-detector read-noise standard deviation, measured at rest.

    A single absolute amplitude threshold assumes every photodiode is equally quiet. On
    this bench PD0's dark scatter is ~170x PD1's, so an amplitude gate that PD1 passes on
    signal, PD0 passes on noise -- and since `better` fell back to ranking by amplitude,
    the noisiest detector won every channel that had no real fringe. Ranking on SNR needs
    a per-detector noise floor, and the only trustworthy one is measured.

    Measured over the same timescale the sweep runs at, which is the whole difficulty. Taken
    back-to-back this reports PD0 at 0.2 mV; spread over seconds the same detector reports
    ~44 mV, because its noise is slow rather than white. A sweep point is seconds from its
    neighbours, so the slow figure is the one that decides whether a fringe is real, and a
    fast sample flatters the noisiest detector by two orders of magnitude."""
    # 16 reads at a second each, and it goes straight to the board rather than through
    # `settled_read`, so it needs its own guard: a floor measured on a dark chip is small,
    # and a small floor makes the SNR gate pass everything it should reject.
    check_active()
    z = np.zeros(pic.cfg.num_dac)
    a = np.array([pic.measure(z, settle_s=settle_s) for _ in range(n)])
    check_active()
    return a.std(axis=0)


STRONG_SNR, STRONG_R2 = 12.0, 0.95


def passes(f, min_visibility: float = 0.10, min_amplitude: float = MIN_AMPLITUDE_V,
           min_r2: float = 0.5) -> bool:
    """Is this fit trustworthy? The single definition -- `better` ranks with it and
    `characterize` records with it, so a channel accepted while sweeping cannot come out of
    the file rejected. Two copies of this rule is exactly the bug that lost dac9."""
    snr = f.get("snr", np.inf)
    r2 = f.get("r2") or 0
    if f.get("amplitude", 0) < min_amplitude or snr < MIN_SNR or r2 < min_r2:
        return False
    # Visibility is amplitude/mean, so it collapses whenever a real fringe rides a bright
    # DC pedestal -- which is what a heater modulating a bright output looks like. It cost
    # H13 a 38 mV fringe at r2 0.986 and SNR 50. Measured against each detector's own noise,
    # SNR says the same thing without the blind spot, so a clearly strong and clearly
    # sinusoidal fit overrides it. The 0.10 floor is the same constant on both devices:
    # the highest that keeps every heater this chip confirms real (H13 itself reaches
    # visibility 0.10), and on the 6x6 the lowest whose wrong-fit rate matches its old 0.15.
    if snr >= STRONG_SNR and r2 >= STRONG_R2:
        return True
    return f.get("visibility", 0) >= min_visibility


def better(a, b, min_visibility: float = 0.10, min_amplitude: float = MIN_AMPLITUDE_V,
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
        return passes(f, min_visibility, min_amplitude, min_r2)

    ok_a, ok_b = ok(a), ok(b)
    if ok_a != ok_b:
        return a if ok_a else b
    if ok_a:
        return a if a["score"] <= b["score"] else b
    # Neither is trustworthy. Rank by SNR, never by raw amplitude: the loudest trace on a
    # noisy detector is not a better candidate than a quiet one on a clean detector.
    return a if a.get("snr", 0) >= b.get("snr", 0) else b


def best_fringe(levels, curves, pds=None, min_visibility: float = 0.10,
                min_amplitude: float = MIN_AMPLITUDE_V, min_r2: float = 0.5,
                noise=None, vmax: float = None) -> dict:
    """Fit one heater's sweep against every detector it was recorded on, keep the best.
    `curves` is (levels, n_detectors). `vmax` is the channel's own voltage ceiling and is
    passed straight to `fit_fringe`; leaving it None uses the swept top, which is the same
    number whenever the caller sized the grid per channel."""
    curves = np.atleast_2d(np.asarray(curves, float))
    if curves.shape[0] != len(levels):
        curves = curves.T
    pds = list(range(curves.shape[1])) if pds is None else list(pds)
    best = dict(NO_FIT)
    for k, p in enumerate(pds):
        try:
            f = fit_fringe(levels, curves[:, k], vmax=vmax)
        except (RuntimeError, ValueError):
            continue
        f["pd"] = int(p)
        nz = 1e-6 if noise is None else max(float(np.ravel(noise)[k]), 1e-6)
        f["snr"] = float(f.get("amplitude", 0.0)) / nz
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


def random_bases(n: int, rng=None, channels=None, vmax=None):
    """Base biases to sweep from. The first is all-zero, the rest random, so a heater that
    is dark in one mesh state has other chances to be seen.

    Drawn per channel from `VOLTAGE_MAX_CH`, not from a flat global ceiling. A flat 3.0 V
    draw is clipped by the driver on every channel rated below it, so a third of each base
    would be the same clipped value at a state the host believed was random -- the host and
    the chip in different states with nothing reporting it."""
    import numpy as _np

    from .layout import ACTIVE_DACS, N_HEATERS

    rng = _np.random.default_rng(0) if rng is None else rng
    channels = ACTIVE_DACS if channels is None else _np.asarray(channels, int)
    hi = (_np.asarray(VOLTAGE_MAX_CH, float)[channels] if vmax is None
          else _np.full(channels.size, float(vmax)))
    out = [_np.zeros(N_HEATERS)]
    for _ in range(max(0, n - 1)):
        v = _np.zeros(N_HEATERS)
        v[channels] = rng.uniform(0, hi, channels.size)
        out.append(v)
    return out


def sensitive_base(dac, calib, fitted, tries=3000, seed=0):
    """Bias the known heaters so detector response to `dac` is as large as possible.

    Brightness is the wrong objective. A heater needs two things to be measurable: light has
    to arrive on its arm, and the phase it imposes has to survive the rest of the mesh and
    reach a detector. A heater downstream of it can null that modulation before any PD sees
    it, so a bias can be bright at every output and still leave the target invisible.

    So maximise |d T[pd, port] / d phase_target| over all (pd, port) -- the largest response
    any detector has to moving this heater -- rather than the light level. Only `fitted`
    channels are moved; unknown phases stay at 0 V rather than being guessed at.

    Returns (base_volts, predicted_sensitivity)."""
    import numpy as _np
    import torch as _torch

    from theory.layout import N_HEATERS
    from theory.twin import Twin

    twin = Twin()
    rng = _np.random.default_rng(seed)
    movable = [int(c) for c in fitted
               if VOLTAGE_MAX_CH[int(c)] > 0 and int(c) != int(dac)]
    if not movable:
        return _np.zeros(N_HEATERS), 0.0
    hi = _np.asarray([VOLTAGE_MAX_CH[c] for c in movable], float)

    def sens(v):
        ph = _torch.tensor(calib.phases(v), dtype=_torch.float64, requires_grad=True)
        T = twin.matrix(ph).abs() ** 2
        # the strongest single response, so a heater only needs ONE good detector
        g = _torch.autograd.grad(T.abs().max(), ph, retain_graph=False)[0]
        return float(abs(g[int(dac)]))

    best, best_s = _np.zeros(N_HEATERS), -1.0
    for _ in range(tries):
        v = _np.zeros(N_HEATERS)
        v[movable] = rng.uniform(0.0, hi)
        sc = sens(v)
        if sc > best_s:
            best, best_s = v, sc
    return best, best_s


def open_base(calib, fitted, target=None, tries=4000, seed=0):
    """Bias the *characterized* heaters so every output carries light.

    Random bases find a heater only when they happen to leave light on its arm, and a heater
    whose detectors are all dark cannot be fitted however hard it is driven. Once some
    heaters are known, that is no longer a lottery: the twin predicts |U x|^2 for any bias,
    so we can pick one that opens the mesh up rather than waiting to get lucky.

    The objective is the *worst* output over all four input ports, not the total -- a bias
    that dumps everything into one bright detector is what leaves the others blocked, which
    is the failure this exists to avoid. Only `fitted` channels are moved; the rest stay at
    0 V because their phase is unknown and setting them would be guessing."""
    import numpy as _np

    from theory.layout import N_HEATERS
    from theory.twin import Twin

    twin = Twin()
    rng = _np.random.default_rng(seed)
    fitted = [int(c) for c in fitted if VOLTAGE_MAX_CH[int(c)] > 0]
    if not fitted:
        return _np.zeros(N_HEATERS)

    hi = _np.asarray([VOLTAGE_MAX_CH[c] for c in fitted], float)
    best, best_score = _np.zeros(N_HEATERS), -_np.inf
    for _ in range(tries):
        v = _np.zeros(N_HEATERS)
        v[fitted] = rng.uniform(0.0, hi)
        T = twin.transfer(calib.phases(v)) if hasattr(twin, "transfer") else None
        if T is None:
            U = twin.matrix(torch.as_tensor(calib.phases(v)))
            T = (U.abs() ** 2).detach().numpy()
        score = float(_np.min(T))          # worst (output, port) pair
        if score > best_score:
            best, best_score = v, score
    return best


def transparent_base(dac, calib, vmax=None):
    """A base bias that opens the path *up to* `dac` and leaves everything after it dark.

    Random bases find a heater only when they happen to leave light on its arm, which is why
    an all-zero sweep identifies the first column and little else. The 6x6's answer is to
    walk the mesh outside-in: hold every heater upstream of the frontier at its most
    transparent setting so the frontier heater sees as much light as the mesh can deliver,
    and leave everything downstream at 0 V so it cannot re-route what comes back out.

    "Transparent" is the reachable voltage whose phase lands nearest a bar state (a multiple
    of 2 pi). On this chip no heater spans a full period, so that is a nearest-approach, not
    an exact null -- which is worth knowing when a frontier heater still comes back weak."""
    from theory.layout import HEATERS, N_HEATERS

    vmax = VOLTAGE_MAX_CH if vmax is None else vmax
    col = HEATERS[int(dac)].column
    v = np.zeros(N_HEATERS)
    for h in HEATERS:
        if h.role != "theta" or h.column >= col or vmax[h.h] <= 0:
            continue
        ph = calib.phi0[h.h]
        target = 2 * np.pi * np.round(ph / (2 * np.pi))     # nearest bar state
        cand = np.linspace(0.0, vmax[h.h], 64)
        got = np.pi * (cand / calib.vpi[h.h]) ** 2 + ph
        v[h.h] = cand[int(np.argmin(np.abs(got - target)))]
    return v


def characterize(pic, session, *, pd=None, levels=None, channels=None, bases=None,
                 switch=None, ports=None, settle_s: float = 0.5, repeats: int = 5,
                 min_visibility: float = 0.10, min_amplitude: float = MIN_AMPLITUDE_V,
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

    # Per channel, because the ceiling is per channel. A single commanded grid gets clipped
    # by the driver on the 60R and dark channels, but the fit is handed the *commanded*
    # voltages -- so above 1.5 V the fringe flattens while the axis keeps going, and Vpi
    # comes out biased high. On a dark channel it is worse: nothing moves at all and the fit
    # ascribes whatever drifted to a voltage axis that never existed.
    base_levels = grid() if levels is None else np.asarray(levels, float)

    def levels_for(c):
        # Scaled to the channel's ceiling in BOTH directions, so the swept top IS
        # `VOLTAGE_MAX_CH[c]` and the alias gate can be evaluated against it. Down-scaling
        # was always here; up-scaling is new and matters since the 40 mA limit opened the
        # ~118 ohm channels to 4.55-4.75 V. A grid stopping at the old 3.0 V global threw
        # away (4.55/3.0)^2 = 2.3x of phase span -- which was the entire point of raising
        # the current -- and left `span_pi` describing a range the sweep never reached.
        # The grid stays uniform in V^2 under any scaling, so the sampling argument holds.
        vmax = VOLTAGE_MAX_CH[int(c)]
        if vmax <= 0:
            return None                       # nothing to sweep; do not invent an axis
        top = float(base_levels.max())
        return base_levels if top <= 0 else base_levels * (vmax / top)

    levels = base_levels
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    bases = random_bases(3, channels=channels) if bases is None else list(bases)
    ports = ([None] if switch is None
             else list(range(NMODE)) if ports is None else list(ports))
    role = LABEL_OF_DAC

    vpi = np.full(N_HEATERS, VPI_NOMINAL)
    phi0 = np.zeros(N_HEATERS)
    noise = read_noise(pic)[pds]
    if verbose:
        print("  read noise per detector (mV): "
              + "  ".join(f"PD{p}:{1e3 * n:.1f}" for p, n in zip(pds, noise)))
    results, raw = {}, {}
    truncated = False
    # A watchdog trip aborts mid-channel rather than finishing it: `settled_read` raises
    # once the laser has been hard-offed, so the sweep in flight is dropped whole instead
    # of being fitted from a trace that goes dark part way through. Everything already in
    # `results` was measured before the trip -- each channel is recorded only after its
    # last complete trace -- so the partial that survives is provably lit data.
    try:
        for c in channels:
            if session.expired():
                # Asked before the sweep, so nothing here was read after the trip: the
                # channels already in `results` are all fully lit.
                truncated = "watchdog deadline reached between channels"
                if verbose:
                    print("  watchdog reached; keeping the partial characterization "
                          "(every channel below was swept before the deadline).")
                break
            levels = levels_for(c)
            if levels is None:
                f = dict(NO_FIT)
                f["dac"], f["label"] = int(c), role[int(c)]
                f["ok"], f["fingerprint"] = False, []
                f["note"] = "held at 0 V (resistance unconfirmed); not swept"
                results[int(c)] = f
                if verbose:
                    print(f"  {role[int(c)]:<14} held at 0 V, not swept")
                continue
            f = dict(NO_FIT)
            # Modulation depth on every (port, detector), not just the winning one. `better`
            # keeps a single best fit because that is what Vpi needs, but the map from DAC to
            # mesh position is carried by the *pattern* -- a first-column MZI moves a different
            # set of outputs from a third-column one, and that distinction is invisible in the
            # one number the fit reports. The sweep already measures it; only the keeping is new.
            fp = np.full((len(ports), len(pds)), np.nan)
            raw_c = np.full((len(ports), len(bases), len(levels), len(pds)), np.nan)
            for pi, port in enumerate(ports):
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
                    ys = np.asarray(ys)
                    raw_c[pi, bi] = ys
                    depth = ys.max(axis=0) - ys.min(axis=0)
                    fp[pi] = depth if np.isnan(fp[pi]).all() else np.fmax(fp[pi], depth)
                    g = best_fringe(levels, ys, pds, min_visibility, min_amplitude,
                                    min_r2, noise=noise, vmax=VOLTAGE_MAX_CH[int(c)])
                    g["base"], g["port"] = bi, port
                    f = better(g, f, min_visibility, min_amplitude, min_r2)
            # One gate, not two. This used to re-derive `ok` with its own copy of the rule, so
            # the SNR override in `better` applied when ranking candidates and then vanished
            # when the winner was recorded -- a channel accepted during the sweep came out of
            # the file rejected. Ask `better` instead, by comparing the fit against a reject.
            f["ok"] = passes(f, min_visibility, min_amplitude, min_r2)
            f["dac"], f["label"] = int(c), role[int(c)]
            f["fingerprint"] = fp.tolist()
            raw[int(c)] = raw_c
            # Keep every trace, not just the winning one. All of a heater's (port, base) sweeps
            # share the same Vpi and phi0 -- only A and B change with the upstream state -- so
            # fitting them together turns ~11 points into ~88 AND breaks the degeneracy that
            # makes (A, B, Vpi) inseparable over a sub-pi segment, because the degenerate
            # direction differs in each trace. `better` keeps one fit and discards the rest,
            # which is the right call for ranking and the wrong one for estimating Vpi.
            f["levels"] = np.asarray(levels, float).tolist()
            f["curves"] = raw_c.tolist()
            if f["ok"]:
                vpi[c], phi0[c] = f["vpi"], f["phi0"]
            results[int(c)] = f
            if verbose:
                print(f"  {role[int(c)]:<14} PD{f.get('pd')} port{f.get('port')} base{f.get('base')}  "
                      f"vis {f['visibility']:.3f}  amp {1e3 * f.get('amplitude', 0):5.1f}mV  "
                      f"r2 {f.get('r2', 0):+.3f}  "
                      f"Vpi {f.get('vpi', float('nan')):.2f}  "
                      f"{'ok' if f['ok'] else 'REJECTED (invisible in intensity)'}")
    except WatchdogTripped as e:
        truncated = "watchdog tripped mid-sweep"
        if verbose:
            print(f"\n  {e}")
            print(f"  keeping the {len(results)} channel(s) completed before the trip; "
                  f"the channel in flight is discarded.")

    n_ok = sum(r["ok"] for r in results.values())
    calib = Calibration(vpi, phi0, meta={
        "source": "pic.characterize.characterize",
        "pds": pds, "n_bases": len(bases), "ports": ports,
        "min_visibility": min_visibility, "min_amplitude": min_amplitude, "min_r2": min_r2,
        "n_ok": n_ok, "n_swept": len(results),
        "chip_c": getattr(session, "chip_c", None),
        # provenance, not decoration: a calibration merged from a truncated run covers
        # fewer channels than its `ports`/`n_bases` fields imply
        "truncated": truncated,
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
