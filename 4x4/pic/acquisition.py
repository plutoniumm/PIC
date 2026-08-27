"""Read primitives and sweep plans.

Every measurement here carries its instrument state with it -- laser telemetry and chip
temperature -- because a photodiode volt on its own is not a measurement of the chip, it
is a measurement of the chip plus whatever the rest of the rig was doing at the time.
"""

from __future__ import annotations

import time

import numpy as np

from .config import NUM_DAC, VOLTAGE_MAX
from .layout import ACTIVE_DACS, N_HEATERS, sweep_columns


def settled_read(pic, v, settle_s: float = 0.2, repeats: int = 5) -> np.ndarray:
    """Apply `v`, wait out the local thermo-optic transient, average `repeats` reads."""
    pic.measure_raw(v)  # apply and prime
    if settle_s > 0:
        time.sleep(settle_s)
    ys = [np.asarray(pic.measure_raw(v), float) for _ in range(max(1, repeats))]
    return np.mean(ys, axis=0)


def rolling_mean(x, window: int = 10) -> np.ndarray:
    """Trailing-window smoother, used to read a settling time off a noisy transient."""
    x = np.asarray(x, float)
    if x.size < window:
        return np.cumsum(x) / np.arange(1, x.size + 1)
    k = np.ones(window) / window
    head = np.cumsum(x[: window - 1]) / np.arange(1, window)
    return np.concatenate([head, np.convolve(x, k, mode="valid")])


def sweep_vectors(channels, levels, base=None, include_base: bool = True):
    """One-at-a-time sweep: for each channel, each level, with the others held at `base`.

    Yields (channel, level, vector) so a caller can log what it set without re-deriving it."""
    base = np.zeros(N_HEATERS) if base is None else np.asarray(base, float).copy()
    if include_base:
        yield (-1, float("nan"), base.copy())
    for c in np.atleast_1d(np.asarray(channels, int)):
        for lv in np.atleast_1d(np.asarray(levels, float)):
            v = base.copy()
            v[c] = lv
            yield (int(c), float(lv), v)


def column_sweep_vectors(levels, base=None):
    """Sweep a whole mesh column at once instead of one heater at a time.

    Heaters in different columns never share an interferometer, so driving one column
    together cannot confound the fits within it -- what a column-mate does is invisible at
    the outputs of the others' interferometers. On the 6x6 this needed an interference-graph
    colouring computed from measured influence; at 4x4 the column index is the colouring,
    which cuts a full characterization pass from 16 channel-sweeps to 5 column-sweeps."""
    base = np.zeros(N_HEATERS) if base is None else np.asarray(base, float).copy()
    for gi, group in enumerate(sweep_columns()):
        for lv in np.atleast_1d(np.asarray(levels, float)):
            v = base.copy()
            v[group] = lv
            yield (gi, tuple(group.tolist()), float(lv), v)


def grid(n: int = 96, vmax: float = VOLTAGE_MAX) -> np.ndarray:
    """The voltage grid a sweep walks: uniform in V^2, so the phase step is constant.

    Phase goes as V^2, so a grid uniform in V crowds samples where the fringe is slow and
    starves them where it is fast. Over 0..5 V a heater with Vpi near 1.5 sweeps about
    11 pi, and a uniform-in-V grid aliases near the top of the range badly enough that the
    fitted Vpi comes back wrong by a third.

    `n` is set by the smallest Vpi on the chip, not by how smooth you want the plot:
    `pic.characterize.resolvable_span` is the span this many levels can resolve, and a fit
    beyond it is rejected. The default clears roughly 47 pi, against the 8-17 pi these
    heaters reach over 0..5 V."""
    return np.sqrt(np.linspace(0.0, vmax**2, n))


def estimate_seconds(n_vectors: int, settle_s: float, repeats: int, read_s: float = 0.12,
                     keepalive_s: float = 0.4) -> float:
    """Wall clock a sweep will take, with headroom -- the number a laser session's
    watchdog duration should be set from."""
    return n_vectors * (settle_s + repeats * read_s + keepalive_s) * 1.3 + 5.0


def random_vectors(n: int, rng=None, channels=None, vmax: float = VOLTAGE_MAX):
    """Random operating points over the active heaters, for surrogate training data."""
    rng = np.random.default_rng() if rng is None else rng
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    V = np.zeros((n, NUM_DAC))
    V[:, channels] = rng.uniform(0.0, vmax, (n, channels.size))
    return V
