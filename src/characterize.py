"""Per-heater characterization from single-channel sweeps: fit the fringe
``P(v) = A + B*cos(phi2*v^2 + phi0)`` to recover (phi2, phi0); Vpi = sqrt(pi/phi2)."""

from __future__ import annotations
import numpy as np


def fit_fringe(v, p, vpi0: float = 1.5):
    """Fit one heater's fringe. Returns dict(A, B, phi2, phi0, Vpi, visibility, rmse)
    or ``None`` if the fit fails."""
    from scipy.optimize import curve_fit

    v = np.asarray(v, float)
    p = np.asarray(p, float)

    def model(v, A, B, phi2, phi0):
        return A + B * np.cos(phi2 * v**2 + phi0)

    p0 = [p.mean(), (p.max() - p.min()) / 2 or 1e-3, np.pi / vpi0**2, 0.0]
    try:
        popt, _ = curve_fit(model, v, p, p0=p0, maxfev=20000)
    except Exception:
        return None
    A, B, phi2, phi0 = popt
    # Canonicalize so the returned (A,B,phi2,phi0) reconstruct the SAME curve the rmse and
    # fringe_extrema read off. curve_fit may return phi2<0 or B<0; naively storing abs(B)
    # without shifting phi0 reflects the curve about A and swaps its max/min (which then
    # swaps V0/Vnull in fringe_extrema). Fold the signs into phi0 instead.
    if phi2 < 0:
        phi2, phi0 = -phi2, -phi0
    if B < 0:
        B, phi0 = -B, phi0 + np.pi
    phi0 = phi0 % (2 * np.pi)
    resid = p - model(v, A, B, phi2, phi0)
    return {
        "A": float(A),
        "B": float(B),
        "phi2": float(phi2),
        "phi0": float(phi0),
        "Vpi": float(np.sqrt(np.pi / phi2)) if phi2 else float("nan"),
        "visibility": float(B / A) if A else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid**2))),
    }


def characterize_channels(
    pic,
    channels,
    values,
    base=None,
    settle_s=None,
    repeats: int = 1,
    target_pd: int | None = None,
):
    """Sweep each channel and fit its fringe on the most-responsive live PD
    (or ``target_pd``). Returns ``{channel: fit_dict}``."""
    from .pic.acquisition import sweep_channel

    out = {}
    for ch in channels:
        vals, Y, _ = sweep_channel(
            pic, ch, values, base=base, settle_s=settle_s, repeats=repeats
        )
        k = target_pd if target_pd is not None else int(np.argmax(Y.std(axis=0)))
        out[ch] = fit_fringe(vals, Y[:, k])
        if out[ch] is not None:
            out[ch]["pd"] = k
    return out
