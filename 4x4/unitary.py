"""One 4x4 unitary multiply, end to end: given U and x, measure y = Ux on the chip.

This is the user-facing front door over `pic.matvec.run_block` -- everything real happens
there (planner, probe, readout, scale correction); this file only assembles the run and
prints it. Read the docstrings there for why each step is what it is.

`U` here is a REAL 4x4 matrix. That is not a simplification of the pipeline; it is what
the instrument can host. With a 1x4 switch only one input port is lit at a time, so a
reading is p_j |U_kj|^2 -- the phases of a complex U never enter the data, and what the
chip hosts is the |U|^2 block of a signed-real target under the shift+Sinkhorn
decomposition. A genuinely complex unitary cannot be multiplied on this bench without a
splitter or a local oscillator, neither of which it has (`theory.intensity_matvec.
dither_matvec` is the stub kept for that day). A complex U is loaded with its phase intact
and refused by `_hostable`, which is named after the READOUT and not after a dtype so that
there is one place to change when a phase reference arrives.

The 4x4 does not fit as one block on this die (see theory.matmat.TILE_K), so it goes on
as four 2x2 tiles -- one heater program each, held while every column of X goes through.

    python unitary.py --mock                           # random U and x, on the mock chip
    python unitary.py --target U.txt --vectors x.txt  # your files, on the bench
    python unitary.py --target U.txt --vectors x.txt --mock   # your files, rehearsal

Both files are whitespace-delimited text (`np.loadtxt`): U.txt is 4x4, x.txt is 4xN with
each COLUMN one input vector -- a single vector is four numbers in any layout. Omit
--vectors and four random x are drawn from --seed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pic import Rig
from pic.acquisition import estimate_sweep_job
from pic.matvec import (Transfers, bench_box, load_transfers, measure_transfer,
                        reanchor, rig_probe, run_block)

K = 2                # tile size: the 4x4 hosts as four 2x2 blocks, nothing bigger fits
READOUT = "intensity"  # what the photodiodes return. "field" is the day there is an LO.
TABLE_STATES = 100   # states in a table measured fresh on the mock chip


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default=None,
                   help="path to a real 4x4 matrix (whitespace text)")
    p.add_argument("--vectors", default=None,
                   help="path to 4xN inputs, one column per x (whitespace text)")
    p.add_argument("--mock", action="store_true",
                   help="run against the physical mock chip instead of the bench")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the random U/x when no --target/--vectors are given")
    p.add_argument("--dbm", type=float, default=8.0, help="laser output power")
    p.add_argument("--repeats", type=int, default=1,
                   help="photodiode reads averaged per port")
    p.add_argument("--reanchor", type=int, default=0, metavar="M",
                   help="re-measure M stored table states and transport the table onto "
                        "today's chip (~20 s at M=5; M<5 is underdetermined)")
    p.add_argument("--optical-input", action="store_true",
                   help="set the input weight with the laser per port, so the multiply "
                        "happens in light; costs a retune per port")
    p.add_argument("--tec", default="mock",
                   help="TEC: 'mock', 'none', or a serial port (hardware only)")
    p.add_argument("--laser-port", default=None, help="laser serial port")
    p.add_argument("--pic-port", default=None, help="PIC board serial port")
    return p.parse_args(argv)


def _load(path):
    """One array from whitespace text: complex if the file says so, real if it does not.

    Read as complex and narrowed back, so `1+2j` reaches `_hostable` with its phase intact
    instead of dying inside `np.loadtxt` or being silently cast to its real part. The
    narrowing is the point: `np.iscomplexobj` tests the dtype, so a real file loaded as
    complex would trip a gate meant for actual phases."""
    A = np.loadtxt(path, dtype=complex)
    return A.real.copy() if not np.any(A.imag) else A


def _hostable(U, X):
    """Refuse what the READOUT cannot deliver -- named after the readout, not the dtype.

    The mesh's own U is complex and always was; that was never the constraint. With one
    input port lit there is no cross term at any output, so a photodiode reads |U_kj|^2 p_j
    and the sum over j happens in software -- phase has nowhere to act on the data. A
    splitter would only make that sum optical (`dither_matvec`) and still square the
    modulus. A local oscillator is the one addition that turns a reading back into a field.

    Lifting this gate is necessary and not sufficient. Everything under `run_block` -- the
    shift, the Sinkhorn diagonals, the table pick, `measured_matrix` -- is real-valued too,
    so a coherent front end is work in `pic.matvec` and not a flag here."""
    if READOUT == "intensity" and (np.iscomplexobj(U) or np.iscomplexobj(X)):
        sys.exit("complex target: this bench reads intensity, so the phases of U never "
                 "reach the data and the answer would be (|U|^2)x, not Ux. Pass U.real or "
                 "np.abs(U), or add a phase reference at the output and set READOUT.")


def _targets(a):
    """The U and X this run will multiply, from files or from --seed."""
    rng = np.random.default_rng(a.seed)
    U = np.atleast_2d(_load(a.target)) if a.target else rng.normal(size=(4, 4))
    X = _load(a.vectors) if a.vectors else rng.normal(size=(4, 4))
    # `np.loadtxt` drops a one-vector file to 1-D, and `atleast_2d` would make it a ROW.
    # Each x is a COLUMN, so the single-vector case -- the whole point of this script --
    # has to be reshaped the other way.
    X = X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
    if U.shape != (4, 4):
        sys.exit(f"target must be 4x4, got {U.shape}")
    if X.shape[0] != 4:
        sys.exit(f"vectors must have 4 rows (one per input port), got {X.shape[0]}")
    _hostable(U, X)
    return U, X


def _table(rig, a, box) -> Transfers:
    """The transfer table the planner picks states from.

    On the bench: the stored table, optionally re-anchored. The table IS the instrument
    -- planning through the twin is not offered as a fallback (`_require_table` has the
    measurement that says why). Against the mock chip the stored table is data about a
    DIFFERENT chip, so it is measured fresh on the chip in hand, the way `twin_table`
    does it: random reachable states, one four-port sweep each. Same instrument for table
    and run, which is the one consistency any planner needs."""
    if not a.mock:
        tab = load_transfers(calib=rig.calib)
        if tab is None:
            sys.exit("no measured transfer table on file -- capture one first "
                     "(scratchpad/raw_capture.py, or `./do char` then a sweep)")
        if a.reanchor:
            tab = reanchor(rig, tab, m=a.reanchor, calib=rig.calib, dbm=a.dbm)
        return tab
    rng = np.random.default_rng(a.seed)
    tr = np.flatnonzero(box.trainable)
    volts = np.zeros((TABLE_STATES, len(box.vmax)))
    volts[:, tr] = np.asarray(box.vmax, float)[tr] * np.sqrt(
        rng.uniform(0, 1, (TABLE_STATES, tr.size)))
    probe = rig_probe(rig, rig.calib, power="laser" if a.optical_input else "digital",
                      dbm=a.dbm, repeats=a.repeats, settle_s=0.0)
    T = np.stack([measure_transfer(probe, v) for v in volts])
    return Transfers(volts, T, None)


def run_ux(a, rig=None, session=True):
    """The whole multiply, given parsed args. Returns the `run_block` result dict.

    `rig=None` builds and opens its own Rig (and closes it); pass an open one to embed
    this in a longer experiment. `session=False` assumes the caller already holds a laser
    session with the watchdog sized for this job."""
    U, X = _targets(a)
    mine = rig is None
    if mine:
        kind = "mock" if a.mock else "hw"
        rig = Rig(laser=kind, board=kind, switch=kind,
                  tec="mock" if a.mock else a.tec,
                  laser_port=a.laser_port, pic_port=a.pic_port).open()
    try:
        box = bench_box(rig.calib)
        # watchdog: the mock table capture is TABLE_STATES sweeps, then four tiles.
        # Over- rather than under-sized by design -- a watchdog that fires early kills
        # the run, one that fires late costs seconds of lit diode.
        states = (TABLE_STATES if a.mock else 0) + 4 + int(a.reanchor)
        dur = max(90.0, float(estimate_sweep_job(states, repeats=max(1, a.repeats),
                                                 batched=rig.batched)))

        def go():
            tab = _table(rig, a, box)
            return run_block(rig, U, box, k=K, X=X, transfers=tab,
                             power="laser" if a.optical_input else "digital",
                             dbm=a.dbm, repeats=a.repeats)

        if session:
            with rig.session(duration_s=dur, power_dbm=a.dbm):
                return go()
        return go()
    finally:
        if mine:
            rig.close()


def report(res, U, X) -> str:
    """The measured product beside the truth, per column, in the prose style of pic/."""
    Y, Yt = res["Y"], res["Y_true"]
    lines = [f"{'x':<24}{'U x':<24}{'measured':<24}{'err':>7}"]
    for j in range(Y.shape[1]):
        e = (float(np.linalg.norm(Y[:, j] - Yt[:, j]))
             / max(float(np.linalg.norm(Yt[:, j])), 1e-12))
        lines.append(f"{np.array2string(X[:, j], precision=2):<24}"
                     f"{np.array2string(Yt[:, j], precision=2):<24}"
                     f"{np.array2string(Y[:, j], precision=2):<24}{e:>7.3f}")
    lines.append(f"matrix error {res['matrix_err']:.4f} (MEASURED, off the photodiodes)  "
                 f"vector error {res['vec_err']:.4f}  sign accuracy {res['sign_acc']:.0%}")
    lines.append(f"cost: {res['cost']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    a = parse_args(argv)
    U, X = _targets(a)
    res = run_ux(a)
    print(report(res, U, X))
    if res["matrix_err"] > 0.05:
        print(f"\nthe mesh does not host this target: matrix error {res['matrix_err']:.4f} "
              "is MEASURED and in the MATRIX, not the readout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
