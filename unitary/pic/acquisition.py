"""Read primitives and sweep plans.

Every measurement here carries its instrument state with it -- laser telemetry and chip
temperature -- because a photodiode volt on its own is not a measurement of the chip, it
is a measurement of the chip plus whatever the rest of the rig was doing at the time.
"""

from __future__ import annotations

import time

import numpy as np

from theory.clements import NMODE

from .config import (
    ADC_FRAME_S,
    DEFAULT_SETTLE_S,
    NUM_DAC,
    SERIAL_RTT_S,
    SWITCH_SETTLE_S,
    VOLTAGE_MAX,
    VOLTAGE_MAX_CH,
)
from .layout import ACTIVE_DACS, N_HEATERS, sweep_columns
from .session import check_active

READ_S = 0.12  # one firmware round trip: ADC_AVG_N sweeps plus the serial reply
SWITCH_MOVE_S = SWITCH_SETTLE_S + 0.05  # mechanical settle plus the command round trip
# The same mirror move with no host round trip around it: the firmware sends `SET n` on
# Serial1, reads the reply and waits SWITCH_SETTLE_MS, all inside one command.
SWITCH_FW_S = SWITCH_SETTLE_S + 0.01


def settled_read(pic, v, settle_s: float = 0.2, repeats: int = 5) -> np.ndarray:
    """Apply `v`, wait out the local thermo-optic transient, average `repeats` reads."""
    check_active()  # after a watchdog trip this would return dark current, silently
    pic.measure_raw(v)  # apply and prime
    if settle_s > 0:
        time.sleep(settle_s)
    ys = [np.asarray(pic.measure_raw(v), float) for _ in range(max(1, repeats))]
    check_active()  # again: a trip during the reads would leave this average half dark
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


def estimate_seconds(
    n_vectors: int, settle_s: float, repeats: int, read_s: float = READ_S, keepalive_s: float = 0.4
) -> float:
    """Wall clock a sweep will take, with headroom -- the number a laser session's
    watchdog duration should be set from."""
    return n_vectors * (settle_s + repeats * read_s + keepalive_s) * 1.3 + 5.0


def batch_sweep_seconds(
    repeats: int = 1, cycles: int = 1, settle_s: float = DEFAULT_SETTLE_S, read_s: float = READ_S
) -> float:
    """The same sweep with the port cycle inside the firmware (`PIC.sweep_raw`).

    One round trip for the whole thing, `cycles` complete visits to the four ports, and
    `ceil(repeats/cycles)` averaged frames at each visit. The three terms are the three
    costs and they are wildly unequal, which is the point:

        round trip      0.12 s   once, not once per read
        mirror move     0.19 s   4*cycles of them, and the mirror is the expensive part
        averaged frame  0.007 s  4*cycles*reads of them -- AVG_N conversions, nothing else

    So repeats are nearly free once the round trips are gone. That is what turns "faster"
    and "quieter" into the same change instead of opposed ones: at cycles=1 the whole span
    from 6 to 32 repeats costs 1.54 s to 2.36 s, against 8.9 s and 45.4 s unbatched."""
    n = max(1, int(repeats))
    c = min(n, max(1, int(cycles)))
    reads = -(-n // c)
    return settle_s + read_s + c * NMODE * (SWITCH_FW_S + reads * ADC_FRAME_S)


def sweep_seconds(
    repeats: int = 1,
    settle_s: float = DEFAULT_SETTLE_S,
    read_s: float = READ_S,
    switch_s: float = SWITCH_MOVE_S,
    batched: bool = False,
    cycles: int = 1,
) -> float:
    """One heater program measured through all four input ports (`pic.normalise.sweep`).

    Unbatched, the switch and the serial round trip dominate and the ADC does not: four
    mirror moves and four reads *per repeat*, at 0.19 and 0.12 s, against 0.007 s of actual
    conversion inside each read. A state costs ~5 s at repeats=3 and 8.9 s at repeats=6, and
    sizing a matvec watchdog off the read count alone is how a job got a 120 s budget for
    twenty minutes of work.

    `batched=True` is the firmware doing the cycle itself; see `batch_sweep_seconds`. Pass
    it whenever the board reports `sweep=1`, and pass it honestly -- this number sizes the
    laser watchdog, and a watchdog cut short leaves the run measuring in the dark."""
    if batched:
        return batch_sweep_seconds(repeats, cycles, settle_s, read_s)
    return settle_s + max(1, int(repeats)) * NMODE * (switch_s + read_s)


def estimate_sweep_job(
    n_states: int,
    repeats: int = 1,
    settle_s: float = DEFAULT_SETTLE_S,
    overhead_s: float = 15.0,
    margin: float = 1.3,
    batched: bool = False,
    cycles: int = 1,
) -> float:
    """Watchdog duration for a job measured in whole port cycles rather than in reads.

    Same shape as `estimate_seconds` -- cost times a margin plus a fixed overhead -- with
    the port cycle as the unit. The overhead covers the laser ramp and the first TEC poll,
    which are paid once and are not small next to a short job."""
    return (
        max(0, int(n_states))
        * sweep_seconds(repeats, settle_s, batched=batched, cycles=cycles)
        * margin
        + overhead_s
    )


def random_vectors(n: int, rng=None, channels=None, vmax: float = VOLTAGE_MAX):
    """Random operating points over the active heaters, for surrogate training data.

    `vmax` is the ceiling of whichever variable the caller works in: pass
    `pic.config.DRIVE_MAX_V` to draw uniform *drive* commands instead of volts."""
    rng = np.random.default_rng() if rng is None else rng
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    V = np.zeros((n, NUM_DAC))
    # Per-channel when the caller works in volts and passes the default: a flat 3 V draw
    # saturates the four 60R channels at 1.5 V, so a third of every random vector would be
    # the same clipped value and the surrogate would see a constant where it expects a
    # sample. A caller working in drive units passes DRIVE_MAX_V and gets a flat draw,
    # which is correct there because the scale has already equalised the channels.
    hi = (
        np.asarray(VOLTAGE_MAX_CH, float)[channels]
        if vmax == VOLTAGE_MAX
        else np.full(channels.size, float(vmax))
    )
    V[:, channels] = rng.uniform(0.0, hi, (n, channels.size))
    return V
