"""Parallel-sweep scheduler: drive several heaters at once without confounding them.

Ported from the 6x6's `scripts/parallel_groups.py`. Sweeping one heater at a time costs
n_channels x n_levels x n_bases x n_ports reads and most of that is thermal settling that
several heaters could have shared. Two heaters can be swept in lockstep provided each can be
read on a detector the other does not move -- so this is list-colouring an interference
graph, with the read-PD as a decision variable rather than fixed.

Making the read-PD a variable is the part that matters. The 6x6 found that assigning each
heater its *brightest* PD and then colouring capped the speedup at ~2x, because one bright
detector dominated for most heaters and formed a large same-PD clique. Ranking each heater's
usable detectors and giving each round member a distinct one breaks that clique.

Multiplexing (Hadamard / group testing) does not apply: the fringe is nonlinear and two
co-driven heaters interfere on a shared detector, so their responses cannot be separated
linearly afterwards. Spatial demux plus scheduling is the way.
"""

from __future__ import annotations

import numpy as np

from .config import VOLTAGE_MAX_CH
from theory.layout import HEATERS, N_HEATERS

MIN_READ_SNR = 4.0  # a detector is usable for a heater only above this
FOOTPRINT_FRAC = 0.25  # PDs a heater moves by >this fraction of its best are "occupied"


def read_sets(fingerprint, noise, min_snr: float = MIN_READ_SNR):
    """PDs a heater can be cleanly read on, best first, and the PDs it disturbs.

    `fingerprint` is the (port, pd) modulation-depth matrix `characterize` records; `noise`
    is the per-detector read noise. A heater is read on the (port, pd) where it is loudest
    relative to that detector's own noise -- not where it is loudest in millivolts, which is
    how the noisiest detector wins everything."""
    fp = np.asarray(fingerprint, float)
    if fp.size == 0 or not np.isfinite(fp).any():
        return [], set()
    snr = fp / np.maximum(np.asarray(noise, float)[None, :], 1e-9)
    best = np.nanmax(snr, axis=0)  # over ports, per detector
    usable = [int(p) for p in np.argsort(-best) if best[p] >= min_snr]
    peak = np.nanmax(fp)
    footprint = {int(p) for p in range(fp.shape[1]) if np.nanmax(fp[:, p]) >= FOOTPRINT_FRAC * peak}
    return usable, footprint


def schedule(channels, fingerprints, noise, n_pd: int = 4):
    """Partition `channels` into rounds. Each round is {dac: read_pd}, all distinct.

    Greedy: take channels hardest to place first (fewest usable detectors), and add a channel
    to a round only if some usable detector of its own is still free AND no member of that
    round disturbs it, nor it them. A channel with no usable detector is returned separately
    -- it has to be swept alone, and pretending otherwise would silently corrupt it."""
    info = {}
    for c in channels:
        u, f = read_sets(fingerprints.get(int(c), []), noise)
        info[int(c)] = (u, f)

    solo = [c for c in channels if not info[int(c)][0]]
    todo = sorted(
        (c for c in channels if info[int(c)][0]), key=lambda c: (len(info[int(c)][0]), -int(c))
    )

    rounds = []
    for c in todo:
        usable, foot = info[c]
        for rnd, taken in rounds:
            if len(rnd) >= n_pd:
                continue
            free = [p for p in usable if p not in taken]
            if not free:
                continue
            # nobody in the round may sit on a detector this heater disturbs, or vice versa
            clash = any(
                rnd[o] in foot or o_pd in foot or info[o][1] & {free[0]}
                for o, o_pd in rnd.items()
                for rnd_pd in [o_pd]
                for o_pd in [rnd_pd]
            )
            if clash:
                continue
            rnd[c] = free[0]
            taken.add(free[0])
            break
        else:
            rounds.append(({c: usable[0]}, {usable[0]}))
    return [r for r, _ in rounds], solo


def summary(rounds, solo, n_channels):
    n_par = sum(len(r) for r in rounds)
    cost = len(rounds) + len(solo)
    return (
        f"{n_channels} channels -> {len(rounds)} parallel round(s) covering {n_par}, "
        f"{len(solo)} solo; {cost} sweeps instead of {n_channels} "
        f"({n_channels / max(cost, 1):.1f}x)"
    )
