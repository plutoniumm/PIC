#!/usr/bin/env python
"""One heater state in, four photodiodes out. Everything else comes from the calibration.

The minimal experiment. Set VOLTS, run it, get numbers. Nothing here decides anything the
calibration already knows: the per-channel voltage ceilings, the detector offsets and gains,
the input ports and the block rails all come from `pic.config` and `pic_data/calib.json`, so
this file has no constants of its own to drift out of agreement with the rig. `./do calibrate`
is what puts them there.

    ./experiment.py                          # zeros, all four ports, normalised
    ./experiment.py --volts 3.5,0,2.1,...    # a heater state
    ./experiment.py --raw                    # photodiode volts as read
    ./experiment.py --rtdc                   # correct the reading for drift first
    ./experiment.py --ports 1,2 --repeats 64

Two readouts, and the difference matters:

  normalised (default)  each column divided by its own sum, giving |U|^2 -- the fraction of
                        the light entering that port which left through each output. A
                        unitary conserves power, so the column sums to 1 by definition of the
                        object rather than by assumption, and this is the only scale that is
                        correct AT THAT HEATER STATE. Launch power, fibre coupling and the
                        switch's per-position loss all divide out together.

  --raw                 photodiode volts, dark subtracted and gain corrected but not
                        normalised. Use it when the absolute level is the measurement --
                        checking headroom against the ADC, or watching a drift -- and never
                        for anything scale-free.

RTDC, realtime drift correction, is off by default and that is deliberate. It re-measures a
handful of states from the stored table, fits the 15-parameter output mixing that carries the
table onto today's chip, and expresses this reading in the table's frame. On a table minutes
old the motion is under the repeatability floor and the fit would be fitting noise, so the
gate declines and says so. On a genuinely stale one it is worth about -29%.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pic.config import N_HEATERS, VOLTAGE_MAX_CH
from pic.rig import Rig

# Edit these, or override them on the command line. They are the whole experiment.
VOLTS = [0.0] * N_HEATERS
PORTS = (0, 1, 2, 3)
RAW = False
RTDC = False
DBM = 10.0
REPEATS = 32
ANCHORS = 10          # states re-measured when RTDC is on; below 5 it cannot fit 15 parameters


def clamp(v) -> np.ndarray:
    """Volts, held to each channel's own ceiling.

    Not a formality. The ceilings are `I_max * R` per channel and the two resistance groups
    differ by 2x, so a single global clamp either cooks the 57 ohm heaters or wastes half the
    range on the 118 ohm ones. The firmware clamps too and `Rig.open` checks the two tables
    agree; this is the host end of that."""
    v = np.zeros(N_HEATERS) if v is None else np.asarray(v, float).ravel()
    if v.size < N_HEATERS:
        v = np.concatenate([v, np.zeros(N_HEATERS - v.size)])
    hi = np.asarray(VOLTAGE_MAX_CH, float)
    out = np.clip(v[:N_HEATERS], 0.0, hi)
    if np.any(v[:N_HEATERS] > hi + 1e-9):
        over = np.flatnonzero(v[:N_HEATERS] > hi + 1e-9)
        print(f"  clamped DAC {over.tolist()} to their ceilings "
              f"{np.round(hi[over], 2).tolist()} V", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--volts", help="comma-separated heater volts, per DAC channel")
    ap.add_argument("--ports", default=",".join(map(str, PORTS)))
    ap.add_argument("--raw", action="store_true", default=RAW,
                    help="photodiode volts instead of the normalised transfer")
    ap.add_argument("--rtdc", action="store_true", default=RTDC,
                    help="realtime drift correction: carry the reading into the stored "
                         "table's frame before reporting it")
    ap.add_argument("--anchors", type=int, default=ANCHORS)
    ap.add_argument("--dbm", type=float, default=DBM)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pic-port"), ap.add_argument("--tec"), ap.add_argument("--laser-port")
    a = ap.parse_args()

    v = clamp([float(x) for x in a.volts.split(",")] if a.volts else VOLTS)
    ports = [int(p) for p in a.ports.split(",") if p != ""]

    kw = dict(laser="mock", board="mock", switch="mock", tec="mock") if a.mock else dict(
        laser="hw", board="hw", switch="hw", tec=a.tec or "/dev/cu.usbmodem1101",
        pic_port=a.pic_port, laser_port=a.laser_port)
    rig = Rig(**kw)
    with rig:
        from pic.normalise import sweep
        calib = rig.calib
        rig.tec.wait_stable()
        with rig.session(duration_s=600, power_dbm=a.dbm):
            corr = None
            if a.rtdc:
                corr = _drift(rig, calib, a.anchors)
            rig.measure(v)                      # the heater write, and its thermal settle
            import time
            from pic.config import DEFAULT_SETTLE_S
            time.sleep(DEFAULT_SETTLE_S)
            P = sweep(rig, v, repeats=a.repeats)          # (pd, port), dark and gain applied

    T = P if a.raw else calib.to_transfer(P)
    if corr is not None and not a.raw:
        T = corr(T)

    print(f"\nheater state (nonzero): "
          f"{ {i: round(float(x), 3) for i, x in enumerate(v) if x > 0} or 'all zero'}")
    print(f"{a.repeats} repeats, +{a.dbm:g} dBm, chip {rig.tec.temperature():.3f} C\n")
    hdr = "raw photodiode volts" if a.raw else "|U|^2, each column divided by its own sum"
    print(f"  {hdr}   (rows = PD 0..3, cols = input port)")
    for j in range(T.shape[0]):
        print("   " + "  ".join(f"{T[j, k]:8.4f}" for k in ports))
    if not a.raw:
        print(f"\n  column sums {np.round(T[:, ports].sum(0), 4).tolist()}")
        print(f"  row sums    {np.round(T[:, ports].sum(1), 4).tolist()}   "
              f"(not imposed -- output gains survive the column normalisation)")
    return 0


def _drift(rig, calib, m):
    """Fit the output mixing that carries the stored table onto today's chip.

    Returns a function applying it, or None when the gate declines. The gate is not optional:
    the mixing has 15 free parameters and each re-measured state supplies 12 observations, so
    below m = 2 it is underdetermined outright, and when the measured motion sits under the
    repeatability floor the fit has nothing to fit but noise -- correcting there made a fresh
    table 15% worse."""
    import numpy as np
    from pic.matvec import load_transfers, REPEAT_FLOOR
    from pic.normalise import sweep
    from theory.design import Table, correction_floor, fit_mixing, select

    tab = load_transfers(calib=calib)
    if tab is None:
        print("  rtdc: no stored table to anchor against -- skipped")
        return None
    idx = select("maxmin", tab.block(((0, 1), (0, 1))).reshape(len(tab), -1), int(m))
    fresh = []
    for i in idx:
        rig.measure(tab.volts[i])
        import time
        from pic.config import DEFAULT_SETTLE_S
        time.sleep(DEFAULT_SETTLE_S)
        fresh.append(calib.to_transfer(sweep(rig, tab.volts[i], repeats=4)))
    fresh = np.stack(fresh)
    moved = float(np.abs(tab.T[idx] - fresh).mean())
    bar = REPEAT_FLOOR * correction_floor(1.0, int(m))
    if moved <= bar:
        print(f"  rtdc: table moved {moved:.4f}, inside the {bar:.4f} floor for m={m} "
              f"-- left alone")
        return None
    M = fit_mixing(tab.T[idx], fresh)
    print(f"  rtdc: {m} states re-measured, table moved {moved:.4f}, "
          f"15-parameter output mixing fitted")

    def apply(T):
        Y = np.asarray(M, float) @ np.asarray(T, float)
        s = Y.sum(0, keepdims=True)
        return np.divide(Y, s, out=np.zeros_like(Y), where=s > 0)

    return apply


if __name__ == "__main__":
    raise SystemExit(main())
