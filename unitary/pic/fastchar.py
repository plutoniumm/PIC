"""Characterize every heater in parallel rounds instead of one at a time.

`characterize` sweeps one channel at a time over every port and every base bias, which is
12 x 4 x 3 = 144 full sweeps for a mesh that has 12 modelled phases. Most of that measures
nothing: a heater is visible on a couple of (port, detector) pairs and invisible on the
rest, and which ones is a property of the chip that one coarse pass can find.

Three steps, and the middle one is `pic.schedule`, which was written for this and never
wired up:

  1. PRESCAN. Three levels per channel per port -- enough to see whether a channel moves a
     detector at all, not enough to fit anything. Costs about a minute and yields the
     (port, pd) modulation-depth fingerprint that step 2 needs.
  2. SCHEDULE. List-colour the interference graph: two heaters may sweep together if each
     can be read on a detector the other does not disturb. Read-PD is a decision variable,
     which is what takes the speedup past 2x (see `pic.schedule`).
  3. SWEEP. Drive a whole round in lockstep at one port, one full grid, and fit each member
     on its own detector. Heaters share the thermal settle, which is the entire cost.

The fit is `characterize`'s, unchanged -- same `best_fringe`, same gates, same records. What
changes is only which measurements get taken.
"""

from __future__ import annotations

import numpy as np

from .acquisition import settled_read
from .config import ADC_BITS, ADC_REF_V, VOLTAGE_MAX_CH
from .characterize import NO_FIT, best_fringe, fringe, read_noise
from .config import mirror_pairs
from .layout import ACTIVE_DACS, LABEL_OF_DAC, N_HEATERS
from .log import ev
from .schedule import schedule, summary
from theory.clements import NMODE
from theory.stat import accepts, p_amplitude, p_shape

# Three, now that the ceiling is 4.95 V and every fitted Vpi is 3.9-7.5: the widest span on
# this mesh is under one pi, and three samples cannot alias a fringe that never repeats. Five
# was insurance against a 4 pi channel, and no such channel exists here. The prescan is 60%
# of the run, so this is where the time is.
PRESCAN_LEVELS = 3


def _dark_noise(pic, switch, n: int = 12, settle_s: float = 0.5):
    """Per-detector noise with the optical path OPEN, so it is DETECTOR noise, not RIN.

    Measured lit, the brightest detector reports the largest "noise" -- it carries the most
    laser intensity noise, which is multiplicative. Ranking a heater's absolute modulation
    depth against that number then throws away the best detector: PD0 measured 22.95 mV lit
    against PD3's 0.04, so a 68 mV fringe on PD0 scored SNR 2.9 and was cut by the SNR
    floor, while 3 mV of nothing on PD3 scored 75 and won. Every strong channel was then
    read on a detector that could not see it.

    With the switch dumped (`P0` opens the path) what is left is detector and TIA noise:
    additive, comparable between detectors, and the right denominator for a depth in volts.
    """
    if switch is not None:
        switch.dark()
    z = np.zeros(N_HEATERS)
    a = np.array([settled_read(pic, z, settle_s, 1) for _ in range(n)])
    return a.std(axis=0)


def _evidence(levels, y, fit, sigma):
    """(p_amplitude, p_shape) for one fitted fringe. Both must be small; see theory.stat."""
    if not np.isfinite(fit.get("vpi", np.nan)):
        return 1.0, 1.0
    model = fringe(levels, fit["A"], fit["B"], fit["vpi"], fit["phi0"])
    return p_amplitude(np.size(y), fit.get("B", 0.0), sigma), p_shape(y, model)


def _levels(dac, n):
    """Uniform in V^2, which is uniform in phase, up to this channel's own ceiling."""
    return np.sqrt(np.linspace(0.0, VOLTAGE_MAX_CH[int(dac)] ** 2, n))


def prescan(
    pic, switch, session, channels, *, ports=None, bases=None, settle_s=0.2, repeats=2, verbose=True
):
    """(port, base, pd) modulation depth per channel. Coarse and cheap; the schedule's input.

    Bases are not optional. An external phase only shows where its interferometer actually
    splits light, so a phi heater whose MZI sits at bar or cross modulates nothing however
    long it is swept -- `characterize` puts it at two in three channels with the mesh at
    0 V. Prescanning from one base would find those channels dark and schedule them as
    unreadable, which is the same wrong answer the slow path was built to avoid."""
    ports = list(range(NMODE)) if ports is None else list(ports)
    bases = [np.zeros(N_HEATERS)] if bases is None else [np.asarray(b, float) for b in bases]
    fps = {}
    for ci, c in enumerate(channels):
        fp = np.zeros((len(ports), len(bases), NMODE))
        for pi, port in enumerate(ports):
            if switch is not None:
                switch.select(port)
            for bi, base in enumerate(bases):
                ys = []
                for lv in _levels(c, PRESCAN_LEVELS):
                    v = base.copy()
                    v[int(c)] = lv
                    mirror_pairs(v)
                    ys.append(settled_read(pic, v, settle_s, repeats))
                    session.keepalive()
                ys = np.asarray(ys)
                fp[pi, bi] = ys.max(axis=0) - ys.min(axis=0)
        fps[int(c)] = fp
        if verbose:
            pi, bi, pd = np.unravel_index(int(np.argmax(fp)), fp.shape)
            top = fp.reshape(-1, NMODE).max(axis=0)  # in PD order, NOT sorted
            ev(
                "heater",
                "prescan",
                f"prescan {ci + 1:>2}/{len(channels)} {LABEL_OF_DAC[int(c)]:<14} "
                f"best PD{pd} port{ports[pi]} base{bi} depth {1e3 * fp[pi, bi, pd]:7.2f}mV"
                f"   per-PD {np.round(1e3 * top, 1)}",
                k=ci + 1,
                n=len(channels),
                pad=_pad(c),
                dac=int(c),
                pd=int(pd),
                port=int(ports[pi]),
                depth_mv=1e3 * fp[pi, bi, pd],
                per_pd_mv=1e3 * top,
            )
    return fps, ports, bases


def _pad(c):
    return LABEL_OF_DAC[int(c)].split(":")[0]


def _best_setting(members, fps, noise, n_ports, n_bases):
    """The one (port, base) that serves a whole round best: maximise its WORST SNR.

    One switch position and one background state have to serve every member at once, so the
    binding constraint is the member that sees least, not the total."""
    best, score = (0, 0), -np.inf
    for pi in range(n_ports):
        for bi in range(n_bases):
            worst = min(fps[c][pi, bi, pd] / max(noise[pd], 1e-9) for c, pd in members.items())
            if worst > score:
                best, score = (pi, bi), worst
    return best, score


def run(
    pic,
    session,
    *,
    switch=None,
    channels=None,
    levels_n=21,
    settle_s=0.2,
    repeats=5,
    ports=None,
    bases=None,
    noise=None,
    pd=None,
    verbose=True,
    **gates,
):
    """Prescan, schedule, then sweep each round in lockstep. Returns characterize's records."""
    from .interface import need_outputs

    need_outputs(pic, "heater characterization")
    channels = [int(c) for c in (ACTIVE_DACS if channels is None else channels)]
    # Lit, so RIN is in it: that is correct, because RIN is exactly what limits a fringe on
    # a bright detector. PD0 carries ~23 mV of it and its traces are pure scatter, so a rank
    # that ignored it read every heater on a detector that could not resolve one.
    #
    # Floored at one ADC LSB, because a detector cannot be quieter than the quantiser. PD3
    # measures 0.00 mV dark, and without the floor its ratio goes to 3e6 and it wins every
    # channel while seeing nothing.
    noise = read_noise(pic) if noise is None else np.asarray(noise, float)
    noise = np.maximum(noise, ADC_REF_V / (2**ADC_BITS - 1))
    if verbose:
        ev("pd", "noise", f"read noise per PD (mV): {np.round(1e3 * noise, 2)}", mv=1e3 * noise)

    fps, ports, bases = prescan(
        pic, switch, session, channels, ports=ports, bases=bases, settle_s=settle_s, verbose=verbose
    )

    # Largest measured modulation wins. Nothing else.
    #
    # This used to rank depth/noise through `pic.schedule` and it was wrong twice in a row.
    # Lit, the brightest detector reports the most noise because laser RIN is multiplicative,
    # so PD0's 68 mV fringe scored SNR 2.9 against PD3's 3 mV of nothing at 75, and every
    # strong channel was read where it could not be seen. Measured dark instead, PD3's floor
    # came out 0.00 mV and the ratio went to 3e6, which is the same mistake with the sign
    # flipped. A depth in volts already IS the signal; dividing it by a number that is either
    # RIN or zero only destroys the ranking.
    best = {}
    for c in channels:
        snr = fps[c] / noise[None, None, :]
        pi, bi, pd = np.unravel_index(int(np.argmax(snr)), snr.shape)
        best[c] = (int(pi), int(bi), int(pd))
        if verbose:
            d = fps[c].reshape(-1, NMODE).max(axis=0)
            ev(
                "heater",
                "queued",
                f"{LABEL_OF_DAC[c]:<14} depth {np.round(1e3 * d, 1)} -> "
                f"PD{pd} port{ports[pi]} base{bi}",
                pad=_pad(c),
                dac=c,
                pd=int(pd),
                port=int(ports[pi]),
            )

    results = {}
    for c in channels:
        pi, bi, pd = best[c]
        members = {c: pd}
        if switch is not None:
            switch.select(ports[pi])

        if verbose:
            ev("heater", "sweeping", f"sweeping {LABEL_OF_DAC[c]}", pad=_pad(c), dac=c)
        grid = {c: _levels(c, levels_n) for c in members}
        ys = []
        for i in range(levels_n):
            v = bases[bi].copy()
            for c in members:
                v[c] = grid[c][i]
            mirror_pairs(v)
            ys.append(settled_read(pic, v, settle_s, repeats))
            session.keepalive()
        ys = np.asarray(ys)

        for c, p in ((int(c), int(p)) for c, p in members.items()):
            # ys[:, [p]] and not ys: `best_fringe` indexes its curves POSITIONALLY against
            # the pd list it is given, so handing it all four columns fits PD0's trace and
            # labels it PD p.
            f = best_fringe(
                grid[c], ys[:, [p]], [p], noise=noise[[p]], vmax=VOLTAGE_MAX_CH[c], **gates
            )
            f["port"], f["base"], f["parallel"] = int(ports[pi]), int(bi), len(members)
            f["label"] = LABEL_OF_DAC[c]
            f["p_amp"], f["p_shape"] = _evidence(grid[c], ys[:, p], f, noise[p])
            f["ok"] = accepts(f["p_amp"], f["p_shape"])
            f["noise_v"] = float(noise[p])  # the detector's read noise this fit was judged by
            results[c] = f
            if verbose:
                vpi, amp = f.get("vpi", float("nan")), 1e3 * f.get("amplitude", 0)
                phi0, span = f.get("phi0", float("nan")) / np.pi, f.get("span_pi", float("nan"))
                ev(
                    "heater",
                    "fit",
                    f"{f['label']:<14} PD{p} port{ports[pi]}  vpi {vpi:5.2f}  amp {amp:6.1f}mV"
                    f"  phi0 {phi0:.3f}p  span {span:.2f}  {'OK' if f['ok'] else 'REJECTED'}",
                    "ok" if f["ok"] else "warn",
                    pad=_pad(c),
                    dac=c,
                    pd=p,
                    port=int(ports[pi]),
                    vpi=vpi,
                    phi0_pi=phi0,
                    span=span,
                    amp_mv=amp,
                    ok=bool(f["ok"]),
                )
    return results, fps, [], []
