"""Acquisition routines: settled reads, channel sweeps, dataset collection, homodyne."""

from __future__ import annotations
import time
import numpy as np


def measure_averaged(
    pic, voltages, repeats: int = 1, settle_s=None, dwell_s: float = 0.0
):
    """Mean and std over ``repeats`` settled reads (a local noise-floor estimate)."""
    ys = []
    for r in range(repeats):
        ys.append(pic.measure(voltages, settle_s=settle_s if r == 0 else 0.0))
        if dwell_s:
            time.sleep(dwell_s)
    Y = np.asarray(ys)
    return Y.mean(0), Y.std(0)


def sweep_channel(
    pic, channel: int, values, base=None, settle_s=None, repeats: int = 1
):
    """Sweep one DAC channel over ``values`` (visited monotonically), others at ``base``.

    Returns ``(values_sorted, Y[n_vals x n_live], Ystd)``.
    """
    base = np.zeros(pic.cfg.num_dac) if base is None else np.asarray(base, float).copy()
    vals = np.sort(np.asarray(values, float))
    Y, S = [], []
    for x in vals:
        v = base.copy()
        v[channel] = x
        m, s = measure_averaged(pic, v, repeats=repeats, settle_s=settle_s)
        Y.append(m)
        S.append(s)
    return vals, np.asarray(Y), np.asarray(S)


def maximize_pd(
    pic,
    target_pd: int,
    channels,
    values,
    base=None,
    rounds: int = 2,
    settle_s=None,
    repeats: int = 1,
):
    """Coordinate-ascent peak-up: walk each channel over ``values``, keep the level that
    maximises raw PD ``target_pd``, repeat ``rounds`` times. This is how a PD's *global*
    max (the renormalisation full-scale) is found, vs. the marginal max a one-at-a-time
    sweep gives -- useful precisely for PDs on off-paths that stay dark until light is
    routed to them. Returns ``(best_vec, best_value, trace[(channel, level, value)])``.
    """
    v = (np.zeros(pic.cfg.num_dac) if base is None else np.asarray(base, float)).copy()
    vals = np.asarray(values, float)
    best = float(pic.measure_raw(v)[target_pd])
    trace = []
    for _ in range(rounds):
        for c in channels:
            ys = []
            for x in vals:
                v[c] = x
                m, _ = measure_averaged(pic, v, repeats=repeats, settle_s=settle_s)
                ys.append(float(m[target_pd]))
            k = int(np.argmax(ys))
            v[c] = vals[k]
            best = ys[k]
            trace.append((int(c), float(vals[k]), best))
    return v.copy(), best, trace


def rolling_mean(Y, window: int = 10) -> np.ndarray:
    """Trailing rolling mean over axis 0: row ``i`` is the mean of rows
    ``[max(0, i-window+1) .. i]`` (early rows average fewer samples). Accepts a 1-D
    trace or a ``[n, k]`` block; returns the same shape. This is the "average the last
    few rounds to smooth it out" the readout needs; ``window=10`` is the default."""
    Y = np.asarray(Y, float)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    n, k = Y.shape
    pad = np.concatenate([np.zeros((1, k)), np.cumsum(Y, axis=0)], axis=0)
    idx = np.arange(1, n + 1)
    lo = np.maximum(0, idx - window)
    out = (pad[idx] - pad[lo]) / (idx - lo)[:, None]
    return out[:, 0] if squeeze else out


def poll_pds(pic, v, seconds: float, on_tick=None, tick_s: float = 0.5,
             min_reads: int = 1):
    """Hold the DACs at ``v`` and read the 14 raw PDs as fast as the firmware replies
    for ``seconds``. ``t`` is each read's mid-time from the first read. ``on_tick`` (if
    given) is called about every ``tick_s`` s during the poll -- used to keepalive the
    laser, which self-disables on its own serial idle even while the PIC is being read.
    Returns ``(t[n], raw[n, NUM_ADC_RAW])``."""
    v = np.asarray(v, float)
    ts, ys, t0, next_tick = [], [], None, tick_s
    while True:
        ta = time.perf_counter()
        raw = np.asarray(pic.measure_raw(v), float)
        tb = time.perf_counter()
        if t0 is None:
            t0 = ta
        ts.append((ta + tb) / 2 - t0)
        ys.append(raw)
        elapsed = tb - t0
        if on_tick is not None and elapsed >= next_tick:
            on_tick()
            next_tick = elapsed + tick_s
        if elapsed >= seconds and len(ys) >= min_reads:
            return np.asarray(ts), np.asarray(ys)


def settling_time(t, Y, window: int = 10, band=None, rel_band: float = 0.05,
                  floor: float = 0.005) -> dict:
    """Per-channel settling time from a polled step trace, measured on the *smoothed*
    (rolling-mean) trace so a stray sample doesn't reset the clock. Final value is the
    last smoothed sample; a channel is "settled" once its smoothed trace enters and
    stays within ``+-band`` of final for the rest of the capture. ``band`` defaults to
    ``max(floor, rel_band * |final - initial|)`` per channel (an absolute-floor OR
    fraction-of-swing tolerance). Returns t_settle/final/swing/band/smoothed.
    ``t_settle`` is ``nan`` for a channel that never settles inside the window."""
    t = np.asarray(t, float)
    sm = rolling_mean(np.asarray(Y, float), window)
    final, initial = sm[-1], sm[0]
    swing = np.abs(final - initial)
    b = np.maximum(floor, rel_band * swing) if band is None else np.full_like(final, float(band))
    n, k = sm.shape
    t_settle = np.full(k, np.nan)
    for j in range(k):
        outside = np.where(np.abs(sm[:, j] - final[j]) > b[j])[0]
        if outside.size == 0:
            t_settle[j] = t[0]
        elif outside[-1] + 1 < n:
            t_settle[j] = t[outside[-1] + 1]
        # else: still outside the band at the last sample -> never settled -> nan
    return {"t_settle": t_settle, "final": final, "swing": swing, "band": b,
            "smoothed": sm, "t": t}


def step_response(pic, v_step, seconds: float, base=None, window: int = 10,
                  presettle_s: float = 0.0, on_tick=None, tick_s: float = 0.5,
                  **st_kw):
    """Apply a step (``base`` -> ``v_step``) and poll the PDs for ``seconds``, then
    compute per-PD settling from the rolling mean. Chip must already be lit (this is the
    laser-free core; :func:`template.run_step_response` wraps it with the laser). Returns
    ``(t, raw, settling_dict)``."""
    base = np.zeros(pic.cfg.num_dac) if base is None else np.asarray(base, float)
    if presettle_s > 0:
        pic.measure_raw(base)
        time.sleep(presettle_s)
    t, raw = poll_pds(pic, v_step, seconds, on_tick=on_tick, tick_s=tick_s)
    return t, raw, settling_time(t, raw, window=window, **st_kw)


def collect_dataset(pic, n: int, sampler=None, settle_s=None, progress: bool = False):
    """Collect ``n`` (input, live-output) pairs. ``sampler() -> 64-vector`` (default:
    uniform on the 0.5 V grid). Returns ``X[n x 64], Y[n x n_live]``."""
    rng = np.random.default_rng(0)
    grid = np.arange(pic.cfg.voltage_min, pic.cfg.voltage_max + 1e-9, 0.5)
    sampler = sampler or (lambda: rng.choice(grid, size=pic.cfg.num_dac))
    X, Y = [], []
    for i in range(n):
        v = np.asarray(sampler(), float)
        X.append(v)
        Y.append(pic.measure(v, settle_s=settle_s))
        if progress and (i + 1) % 100 == 0:
            print(f"  collected {i + 1}/{n}")
    return np.asarray(X), np.asarray(Y)


def homodyne_sweep(
    pic, ref_channel: int, ref_values, base=None, settle_s=None, repeats: int = 1
):
    """Step a reference-arm heater through ``ref_values``, record all live PDs.

    Returns ``(ref_values_sorted, Y, Ystd)``; feed a column of ``Y`` to :func:`fit_homodyne`.
    """
    return sweep_channel(
        pic, ref_channel, ref_values, base=base, settle_s=settle_s, repeats=repeats
    )


def fit_homodyne(ref_values, pd_trace, vpi: float = 1.5):
    """Fit ``P(v) = A + B*cos(a*v^2 + psi)`` to one PD's reference-phase sweep.

    Returns dict(A, B, a, psi, visibility, rel_phase), or ``None`` if the fit fails.
    """
    from scipy.optimize import curve_fit

    v = np.asarray(ref_values, float)
    P = np.asarray(pd_trace, float)

    def model(v, A, B, a, psi):
        return A + B * np.cos(a * v**2 + psi)

    p0 = [P.mean(), (P.max() - P.min()) / 2 or 1e-3, np.pi / vpi**2, 0.0]
    try:
        popt, _ = curve_fit(model, v, P, p0=p0, maxfev=20000)
    except Exception:
        return None
    A, B, a, psi = popt
    return {
        "A": float(A),
        "B": float(abs(B)),
        "a": float(a),
        "psi": float(psi % (2 * np.pi)),
        "visibility": float(abs(B) / A) if A else float("nan"),
        "rel_phase": float(psi % (2 * np.pi)),
    }
