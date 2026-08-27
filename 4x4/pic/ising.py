"""Ising on the bench: programme the mesh, cycle the four input ports, score.

The maths is `theory.ising` and none of it needs hardware. What lives here is the
instrument: which channels a design may use, the four-shot probe that returns the
intensity transfer matrix, and the laser session around it.

The measurement is exactly the drift probe -- four switch positions at one held bias --
because it returns the same object, `T[j, k] = g_j |U[j, k]|^2 c_k`. That is the whole
optical cost of an n-spin problem: four shots, whatever n is, because once T is measured
every one of the 2^n configurations is a dot product against the same sixteen numbers.

    python -m pic ising --mock            # idealised chip: does the pipeline work
    python -m pic ising --sim             # the bench as delivered: will it survive
    python -m pic ising --n 3 --dbm 8     # on hardware, one input port at a time

Mirrors `6x6/pic/compute/ising.py`, but the route is different and has to be: that chip
launched spins as input phases through a splitting tree, this one has a 1x4 switch and no
coherent superposition of ports. See the `theory.ising` docstring.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from theory import ising as kernel
from theory.calib import VPI_NOMINAL
from theory.clements import NMODE

from .config import VOLTAGE_MAX_CH
from .layout import LABEL_OF_DAC, N_HEATERS, REACHABLE_DACS

DBM_CEIL = 15.0


def usable(calib) -> np.ndarray:
    """Channels a design is allowed to move: wired, in the mesh model, and characterized.

    An uncharacterized channel still carries the nominal Vpi of 1.5 V, which on this die is
    off by a factor of three and would let the optimiser plan a 4 pi swing it cannot make.
    `pic.characterize` leaves exactly that pair -- nominal Vpi, zero phi0 -- on any heater
    it failed to fit, so it is also the test for one."""
    fitted = ~(np.isclose(calib.vpi, VPI_NOMINAL) & (calib.phi0 == 0.0))
    m = np.zeros(N_HEATERS, bool)
    m[REACHABLE_DACS] = True
    return m & fitted & (np.asarray(VOLTAGE_MAX_CH, float) > 0)


def probe(rig, volts, repeats: int = 3) -> np.ndarray:
    """Cycle the four input ports at one held bias -> T[j, k], photodiode j with port k lit.

    Averaging repeats the whole port cycle rather than the read, because the switch is the
    less repeatable of the two."""
    acc = np.zeros((NMODE, NMODE))
    for _ in range(max(1, repeats)):
        for k in range(NMODE):
            rig.select_input(k)
            acc[:, k] += rig.outputs(volts)
    return acc / max(1, repeats)


def solve(rig, J, *, repeats: int = 3, settle: float = 0.0, steps: int = 400,
          restarts: int = 16, pd_limit: float = 4.5, seed: int = 0) -> dict:
    """One instance end to end: encode, hold, probe four ports, decode, score."""
    enc = kernel.encode(J, rig.twin, rig.calib, trainable=usable(rig.calib),
                        vmax=np.asarray(VOLTAGE_MAX_CH, float), steps=steps,
                        restarts=restarts, seed=seed)
    rig.measure(enc.volts)
    if settle > 0:
        time.sleep(settle)
    T = probe(rig, enc.volts, repeats)
    if T.max() > pd_limit:
        raise SystemExit(f"photodiode over limit ({T.max():.2f} V) -- aborting")

    res = kernel.decode(T, J)
    cfgs_true, E_true = kernel.brute_force(J)
    score = kernel._score(T, J, cfgs_true, E_true)
    return {"J": np.asarray(J).tolist(), "volts": enc.volts.tolist(),
            "design_err": enc.rel_err, "design_contrast": enc.contrast,
            "T": T.tolist(), "spins": res["spins"].tolist(),
            "gs_true": cfgs_true[0].tolist(), **score}


def report(rows, n: int) -> str:
    gs = float(np.mean([r["found_gs"] for r in rows]))
    return (f"n={n}  {len(rows)} instances | design err "
            f"{np.mean([r['design_err'] for r in rows]):.3f} | contrast "
            f"{np.mean([r['contrast'] for r in rows]):.4f} | GS found {gs:.0%} "
            f"(chance {np.mean([r['chance'] for r in rows]):.0%}) | excess "
            f"{np.mean([r['excess'] for r in rows]):.3f} | E-corr "
            f"{np.nanmean([r['pearson'] for r in rows]):.3f}")


def main(argv=None):
    from .rig import Rig

    ap = argparse.ArgumentParser(prog="python -m pic ising")
    ap.add_argument("--n", type=int, default=4,
                    help=f"spins; at most {kernel.NSPIN_MAX}, the number of optical modes")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dbm", type=float, default=8.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="laser on-time cap; 0 sizes it from the instance count")
    ap.add_argument("--repeats", type=int, default=3, help="port cycles averaged per probe")
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--pd-limit", type=float, default=4.5)
    ap.add_argument("--dynamic", action="store_true",
                    help="re-anchor the drift estimate between instances. The four columns "
                         "of a probe are four switch positions seconds apart, so a chip that "
                         "moves between them yields a matrix describing no single device -- "
                         "which shows up as a miss with a perfectly good design error")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--tec", default="hw")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.dbm > DBM_CEIL:
        raise SystemExit(f"--dbm {a.dbm} exceeds the {DBM_CEIL} dBm ceiling")
    if not 2 <= a.n <= kernel.NSPIN_MAX:
        raise SystemExit(f"--n must be 2..{kernel.NSPIN_MAX}: the mesh has {NMODE} modes "
                         "and there is nowhere to put a fifth spin")

    hw = not (a.mock or a.sim)
    # The mock chip *is* `Calibration.sample(seed=0)`, so handing the rig that law is what a
    # perfect characterization of it would have produced. Without it --mock measures the gap
    # between two unrelated chips and tells you nothing about the pipeline. --sim keeps the
    # loaded calibration on purpose: that mismatch is the real one.
    calib = None
    if a.mock:
        from theory.calib import Calibration
        calib = Calibration.sample(seed=0)
    rig = Rig(laser="mock" if not hw else "hw",
              board="sim" if a.sim else ("mock" if a.mock else "hw"),
              switch="mock" if not hw else "hw",
              tec="mock" if not hw else a.tec, calib=calib,
              laser_port=a.laser_port, pic_port=a.pic_port,
              dynamic=getattr(a, "dynamic", False)).open()

    live = usable(rig.calib)
    seen = kernel.movers(rig.twin, rig.calib.phases(np.zeros(N_HEATERS)), live)
    print(f"design may move {int(live.sum())} of {N_HEATERS} channels; "
          f"{int(seen.sum())} of those change |U|^2: "
          + ", ".join(LABEL_OF_DAC[d] for d in np.flatnonzero(seen)))
    span = np.asarray(VOLTAGE_MAX_CH, float)[live] ** 2 / rig.calib.vpi[live] ** 2
    print(f"phase span available: {span.min():.2f} to {span.max():.2f} pi "
          f"(a free phase needs 2.00)")
    if a.sim:
        print("note: pic_data/calib.json does not describe `pic.sim` -- different Vpi, "
              "different zero-bias phases. A --sim run therefore measures a stale "
              "calibration on top of the reachable set, and scores worse than either "
              "alone. Characterize the simulator first if you want the reach number.")

    rng = np.random.default_rng(a.seed)
    problems = [kernel.random_ising(rng, n=a.n) for _ in range(a.instances)]
    dur = a.duration or (40 + a.instances * (4 * a.repeats * 0.4 + a.settle + 1.0))
    rows = []
    try:
        with rig.session(duration_s=dur, power_dbm=a.dbm) as s:
            if hasattr(s, "emitted") and not s.emitted:
                print("WARNING: no emission detected -- every result below is noise")
            for i, J in enumerate(problems):
                r = solve(rig, J, repeats=a.repeats, settle=a.settle, steps=a.steps,
                          restarts=a.restarts, pd_limit=a.pd_limit, seed=a.seed + i)
                rows.append(r)
                print(f"  inst {i}: design err {r['design_err']:.3f}  "
                      f"contrast {r['contrast']:.4f}  "
                      f"GS {'hit ' if r['found_gs'] else 'miss'}  "
                      f"excess {r['excess']:.3f}", flush=True)
                if hasattr(s, "keepalive"):
                    s.keepalive()
    finally:
        rig.close()

    print(report(rows, a.n))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"n": a.n, "seed": a.seed, "dbm": a.dbm, "mock": a.mock, "sim": a.sim,
                   "rows": rows}, open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
