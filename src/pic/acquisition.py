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
