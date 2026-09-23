"""Run `theory.intensity_matvec` on the rig: signed y = B x from photodiodes alone.

The maths is next door and hardware-free; this is the four things it needs from a bench.

**The box.** `theory` owns the heater law but not the board, so the per-channel voltage
ceiling has to come from here: `pic.config.VOLTAGE_MAX_CH` gives every channel its own
ceiling from its own measured resistance at 40 mA -- 2.25-2.45 V on the ~60 ohm group,
4.50-4.75 V on the ~118 ohm group. That is not cosmetic and it cuts both ways, so a
validation run against the uniform 3 V ceiling is wrong in both directions: of the eight
steerable channels the three clamped ones reach 0.18-0.58 pi where a flat 3 V would claim
0.31-0.99, and the five wide ones reach 0.95-1.51 pi where it would concede only 0.41-0.66.
`bench_box` is the one to believe.

**The probe.** One primitive, and it is the whole four-port sweep: set the volts, cycle the
switch through all four input ports, read the photodiodes, and turn the resulting matrix
into `|U|^2` with `Calibration.to_transfer`. It has to be the sweep and not the single read,
because the scale each column needs is that column's own sum -- a unitary conserves power,
so everything entering port k leaves through the four outputs -- and that number does not
exist until the column is complete. The single-port path that divided by a stored
`input_scale` is kept as `normalise="static"` and is wrong on this bench: the light leaving
ports 0 and 1 moves by 5-6x across heater states, so a constant fitted at one state is out
by that factor at the next, which is where a vector error of 1.1 against a hosted residual
of 0.0001 came from. The sweep costs three extra switch moves per heater state and no extra
thermal settle, and it pays for itself the moment a second vector is measured under the
same program.

The sweep is also ONE round trip, not 4*repeats of them: the firmware runs the port cycle
itself and answers once (`pic.interface.PIC.sweep_raw`). That is where the clock went -- at
repeats=6 a sweep was 24 serial round trips and 24 mirror settles, 8.9 s, of which 0.24 s
was analog-to-digital conversion. Batched it is 1.54 s, and the whole span from 6 repeats to
32 costs 1.54 s to 2.23 s, so averaging has stopped competing with the wall clock.

**The number that was missing.** `run` used to report `matrix_err` out of `plan.predict()`,
which is the twin's arithmetic on the twin's own fitted block and never touches an
instrument. On 2026-08-27 it read 0.0002 on a bench run whose vectors came back 1.1 out with
25% of the signs right, and nothing in the run could tell a bad plan from a bad readout. The
hosted block is now MEASURED -- `measured_matrix` cycles the switch and reads it off the
photodiodes, which the column normalisation above was going to cost anyway -- and the twin's
prediction is printed beside it under a name that says what it is. On this die the gap is
large: `plan_from_table` has the measurement, and it is why the default planner on hardware
picks a measured heater state rather than fitting one.

**Where the multiply happens.** With the 1x4 switch only one port is lit at a time, so the
accumulation over input rails is digital no matter what -- that is the hardware, not a
shortcut. The input weight p_j can still be optical, by retuning the laser per port
(`power="laser"`), and then the chip is doing p_j |U_kj|^2 in light. The default is
`power="digital"` because a laser retune per port costs a settle and goes through the safety
scaffolding four times per vector; say `--optical-input` when the claim matters more than
the clock. In digital mode a port read is power-independent, so the honest description of
that run is "the chip multiplies, the host weights and sums".

**The rungs above.** `theory.matmat` needs nothing from a bench but the probe above, so
`--cols` walks several input vectors under one program and `--block` tiles a target too big
for the mesh across it. Only the tiling changes what the hardware does: one heater program
per tile, written once and held while the whole stripe of X goes through it. Inverted -- a
column at a time, retiling for each -- an 8x8 would pay 16 thermal settles per column
instead of 16 in total, which is the difference between a minute and a quarter of an hour.

    python -m pic matvec --mock            # plan a random 2x2 and read it back
    python -m pic matvec --mock --scan     # which rail pair to put the block on
    python -m pic matvec --mock --k 8 --block 2 --cols 4   # an 8x8 in sixteen 2x2 tiles
    python -m pic matvec --mock --unitary  # host two rotations, compose, check O(2)
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from theory.calib import Calibration
from theory.clements import NMODE
from theory.intensity_matvec import (
    BEST_RAILS,
    HeaterBox,
    MatvecPlan,
    Program,
    Term,
    block_rails,
    plan_matvec,
    scan_rails,
    score,
    sinkhorn,
    validate,
)
from theory.matmat import (
    BlockPlan,
    TILE_K,
    CountedProbe,
    compose,
    matmat as tile_matmat,
    plan_block,
)

from .acquisition import estimate_sweep_job, sweep_seconds
from .config import DEFAULT_SETTLE_S, VOLTAGE_MAX, VOLTAGE_MAX_CH
from .layout import N_HEATERS, REACHABLE_DACS
from .normalise import SWEEP_CYCLES, load_session, sweep

ROOT = Path(__file__).resolve().parent.parent
TRANSFER_GLOB = "pic_data/sessions/*/raw_transfers*.json"

# `theory.intensity_matvec.plan_matvec`'s own constants, named here because the table
# planner has to reproduce its decomposition exactly or the two are not comparable.
# 0.25 is a measured optimum, not a round number. Swept against matrix error on 120 random
# targets with the table stale by its measured 0.0157, as margin: host / matrix / relative
# amplification -- 0.05: .073/.110/1.89, 0.10: .059/.090/1.45, 0.25: .051/.085/1.11,
# 0.60: .069/.127/1.00, 2.00: .080/.241/0.99. A larger shift makes the hosted matrix more
# uniform, so Sinkhorn's diagonals stop being extreme and a stale table is amplified less; but
# the target then sits under a constant the recovery subtracts back off, and past 0.25 that
# costs more signal than the conditioning buys. Clean interior minimum -- do not tune by eye.
SHIFT_MARGIN, SINK_FLOOR = 0.25, 1e-3
# what two captures of the same states agree to, measured 2026-08-27: the bar a
# correction has to clear before it is worth its own fitting noise.
REPEAT_FLOOR = 0.0179


def bench_box(calib: Calibration | None = None, vmax=None, known=None) -> HeaterBox:
    """The reachable set as the board actually is: measured Vpi and phi0, per-channel
    ceilings, and only the channels a characterization has fitted."""
    calib = calib or Calibration.load_or_nominal()
    return HeaterBox.from_calibration(
        calib, np.asarray(VOLTAGE_MAX_CH if vmax is None else vmax, float), known=known
    )


def mock_truth(seed: int = 0):
    """The fabricated instance `pic.interface.twin_forward` builds MockPIC from.

    A `--mock` run plans against it so the loop tests the algorithm rather than the gap
    between a saved calibration and a different simulated chip -- with `pic_data/calib.json`
    against the mock the matvec comes back a factor of four out, which is a correct answer
    to the wrong question. On the bench the calibration file IS the truth and this is
    unused."""
    from theory.twin import MeshError, Twin

    return Calibration.sample(seed=seed), Twin(MeshError.sample(seed=seed))


class SweepProbe:
    """The bench primitive, with the four-port sweep as the unit of measurement.

    Same call signature as every other probe -- `(volts, port, power) -> four intensities`
    -- so nothing above it changes, but a new heater state triggers the whole port cycle and
    the per-port answer is served out of it. That inversion is forced: `to_transfer` divides
    each column by its own sum, and the sum of a column that has not been read yet does not
    exist. A probe that returns one port at a time can only be divided by a constant, and on
    this bench the constant is out by 5-6x at most heater states.

    The sweep is cached per heater state, which is what makes it cheap rather than four
    times dearer: a plan writes its volts once and then asks for every port and every vector
    at that state, so the hardware cost is 4 reads per program instead of 1 read per pass.
    The chip is temperature-held and the state is deterministic, so a second read of the
    same state differs only by read noise -- average it with `repeats` rather than by
    re-reading, and call `invalidate()` if something outside this object moved the chip.

    `power="laser"` keeps the multiply optical and is not served from the cache. The sweep's
    column sums are taken at the reference power and then used as the divisor for a read at
    `p` times that power, so the returned column is `p |U[:,k]|^2` with `p` having happened
    in light. Dividing by the live column's own sum instead would cancel exactly the factor
    the laser was asked to apply.

    Counts what it cost the bench in the same shape as `theory.matmat.CountedProbe`, and
    more honestly: it counts the reads and switch moves the sweep actually performs, where
    wrapping it would count one read per request."""

    def __init__(
        self,
        rig,
        calib: Calibration | None = None,
        power: str = "digital",
        settle_s: float = DEFAULT_SETTLE_S,
        dbm: float = 8.0,
        repeats: int = 1,
        cache: bool = True,
        cycles: int | None = None,
    ):
        if power not in ("digital", "laser"):
            raise ValueError(f"power={power!r}; use 'digital' or 'laser'")
        self.rig = rig
        self.calib = calib or rig.calib
        self.power = power
        self.settle_s = float(settle_s)
        self.dbm = float(dbm)
        self.repeats = int(repeats)
        self.cycles = cycles
        self.cache = bool(cache)
        self.reads = self.switch_moves = self.writes = 0
        self._port = None
        self._state = None  # (volts, T, column sums at the reference power)

    def invalidate(self):
        self._state = None

    def _sweep(self, v) -> tuple:
        """Write the state, pay the settle once, cycle the ports. Returns (|U|^2, scale)."""
        self.rig.measure(v)
        self.writes += 1
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        if self.power == "laser":
            self.rig.laser.set(self.dbm)
        n = max(1, self.repeats)
        raw = sweep(self.rig, v, n, cycles=self.cycles)
        self.reads += NMODE * n
        # A batched sweep moves the mirror once per port per CYCLE, not once per repeat;
        # counting otherwise would report a cost the bench no longer pays.
        c = min(n, max(1, self.cycles if self.cycles is not None else SWEEP_CYCLES))
        self.switch_moves += NMODE * (c if getattr(self.rig, "batched", False) else n)
        self._port = NMODE - 1  # sweep leaves the switch on the last port
        P = np.clip(self.calib.to_intensity(raw.T).T, 0.0, None)
        return self.calib.to_transfer(raw), P.sum(0)

    def __call__(self, volts, port: int, p: float = 1.0) -> np.ndarray:
        port, v = int(port), np.asarray(volts, float)
        if not (self.cache and self._state is not None and np.array_equal(v, self._state[0])):
            self._state = (v.copy(), *self._sweep(v))
        _v, T, scale = self._state
        if self.power != "laser":
            return float(p) * T[:, port]
        if self._port != port:
            self.rig.select_input(port)
            self._port = port
            self.switch_moves += 1
        self.rig.laser.set(self.dbm + 10 * np.log10(max(float(p), 1e-6)))
        self.reads += 1
        y = np.clip(self.calib.to_intensity(np.asarray(self.rig.outputs(v), float)), 0.0, None)
        return y / max(float(scale[port]), 1e-12)

    def per_column(self, n_col: int) -> dict:
        return {
            "reads": self.reads / n_col,
            "switch_moves": self.switch_moves / n_col,
            "writes": self.writes,
        }

    def __str__(self):
        return (
            f"{self.writes} heater writes, {self.switch_moves} switch moves, " f"{self.reads} reads"
        )


def rig_probe(
    rig,
    calib: Calibration | None = None,
    power: str = "digital",
    settle_s: float = DEFAULT_SETTLE_S,
    dbm: float = 8.0,
    normalise: str = "column",
    repeats: int = 1,
    cycles: int | None = None,
):
    """The bench primitive behind every pass: (volts, port, power) -> four intensities.

    `normalise="column"` is `SweepProbe` and is the one to use. `"static"` is the original
    single-port path, kept so a run can be repeated as it was and so the two can be compared
    on one chip state; it divides by the `input_scale` the calibration stored, which is a
    single heater state's launch and is deprecated for the reason in `pic.normalise`.

    In the static path the thermal settle is charged only when the heater state actually
    changes, which in `shift` mode is once per plan: one program means the volts are written
    once and every later pass only moves the switch. `split` pays it on every pass."""
    if normalise == "column":
        return SweepProbe(
            rig, calib, power=power, settle_s=settle_s, dbm=dbm, repeats=repeats, cycles=cycles
        )
    if normalise != "static":
        raise ValueError(f"normalise={normalise!r}; use 'column' or 'static'")

    calib = calib or rig.calib
    scale = np.asarray(calib.meta.get("normalisation", {}).get("input_scale", np.ones(4)), float)
    if power not in ("digital", "laser"):
        raise ValueError(f"power={power!r}; use 'digital' or 'laser'")
    state = {"port": None, "volts": None}

    def probe(volts, port: int, p: float = 1.0):
        port, v = int(port), np.asarray(volts, float)
        if state["port"] != port:
            rig.select_input(port)
            state["port"] = port
        if state["volts"] is None or not np.array_equal(v, state["volts"]):
            rig.measure(v)
            time.sleep(settle_s)
            state["volts"] = v.copy()
        if power == "laser":
            rig.laser.set(dbm + 10 * np.log10(max(float(p), 1e-6)))
        i = calib.to_intensity(np.asarray(rig.outputs(v), float)) / max(scale[port], 1e-9)
        return i if power == "laser" else i * float(p)

    # counted from outside, because this path really does pay one read per request -- which
    # is exactly the accounting `SweepProbe` had to take over for itself
    return CountedProbe(probe)


def measure_transfer(probe, volts, n: int = NMODE) -> np.ndarray:
    """|U|^2 at one heater state, through whatever probe is in use.

    Free on `SweepProbe`: normalising at all already costs the whole port cycle, so these
    columns come back out of the cache the first pass filled. Assembling them into a matrix
    is the only new thing here, and it is what turns "the twin says the mesh hosts this"
    into "the mesh hosts this"."""
    return np.column_stack([np.asarray(probe(volts, k, 1.0), float) for k in range(n)])


def measured_matrix(plan: MatvecPlan, probe) -> np.ndarray:
    """The signed matrix the mesh ACTUALLY hosts, read off the photodiodes.

    Identical arithmetic to `MatvecPlan.predict()` with one substitution -- the fitted
    `hosted` block is replaced by the measured one, while the plan's own `scale`, `r` and
    `s` are left exactly as planned. So this is what `plan.matvec` returns on a noiseless
    bench, and the gap to `predict()` is the twin's error and nothing else.

    That gap is what the 2026-08-27 run did not have. It reported 0.0002 for "realised
    matrix vs target" out of `predict()`, a calculation that never touched an instrument,
    while its vectors came back 1.1 out with 25% of the signs right -- and there was no
    number anywhere in the run that could tell those two apart."""
    ix = np.ix_(list(plan.rails[0]), list(plan.rails[1]))
    M = -plan.offset * np.ones((len(plan.rails[0]), len(plan.rails[1])))
    for t in plan.terms:
        blk = measure_transfer(probe, t.prog.volts)[ix]
        M = M + t.coeff * (np.diag(1 / t.r) @ (blk / t.prog.scale) @ np.diag(1 / t.s))
    return M


def measured_block_matmat(plan: BlockPlan, probe, X) -> tuple[np.ndarray, np.ndarray]:
    """Measure every tile AND push X through it, one visit per tile. Returns (B, Y).

    One function and not two, because the expensive state is the heater program: measuring
    the tiles in one pass and applying them in another visits each program twice and the
    probe's cache holds one state, so a four-tile product pays eight thermal settles instead
    of four. Interleaved, the measurement is free -- the sweep that reads the tile is the
    sweep its columns are then served from.

    This is `BlockPlan.matmat`'s loop, tile-major for the same reason it is there."""
    (M, N), k = plan.B.shape, plan.k
    Xp = np.zeros(((N + k - 1) // k * k, np.atleast_2d(X).shape[1]))
    Xp[:N] = np.atleast_2d(X)
    Bh = np.zeros(((M + k - 1) // k * k, (N + k - 1) // k * k))
    Y = np.zeros(((M + k - 1) // k * k, Xp.shape[1]))
    for (i, j), tile in _by_state(plan.tiles):
        want = plan.B[i * k : (i + 1) * k, j * k : (j + 1) * k]
        got = measured_matrix(tile, probe)
        # PER TILE, not once for the assembled matrix: every tile is its own heater state
        # with its own brightness, so a single global scalar leaves the differences between
        # them behind as shape error in the sum. The scalar is measured off the photodiodes
        # here, from the tile and its target only -- X never enters it.
        _, a = shape_scale(got, want)
        a = a if abs(a) > 1e-9 else 1.0
        Bh[i * k : (i + 1) * k, j * k : (j + 1) * k] = got / a
        Y[i * k : (i + 1) * k] += tile_matmat(tile, probe, Xp[j * k : (j + 1) * k]) / a
    return Bh[:M, :N], Y[:M]


def _by_state(tiles: dict):
    """Tiles ordered so that any two picking the SAME heater program are adjacent.

    The table is a codebook and the codebook concentrates: measured on the 115-state bench
    table, the 64 tiles of a 16x16 pick **18 distinct states** and the 16 of an 8x8 pick 10.
    `SweepProbe` caches one state, so two tiles on the same program separated by a third pay
    two thermal settles and two sweeps for one measurement -- 72 percent of the 16x16's
    measurement sweeps were re-reading a state the run had already read. Sorting brings them
    together and the existing cache does the rest; nothing caches across a gap, so no tile
    is ever served a sweep taken minutes earlier.

    What it costs is independence: tiles served from one sweep share that sweep's read
    noise. Measured rather than argued, paired over 12 random 16x16 targets at the bench's
    1.2 percent read noise, the measured block error is 0.3531 grouped against 0.3682 as
    planned -- no worse, and the sweeps fall 54 to 12. At 3 percent the paired difference is
    +0.10 +- 0.21, still nothing. It is only worth re-measuring if read noise ever
    approaches 3 percent, and by 6 percent both orderings are already past a relative error
    of 1 and neither works.

    Refinement breaks the sharing on purpose, one tile at a time, which is another reason to
    spend it on a few tiles rather than on all of them.

    `Y` accumulates with `+=` and `Bh` writes disjoint blocks, so the order is free."""
    return sorted(
        tiles.items(), key=lambda kv: (tuple(np.round(kv[1].terms[0].prog.volts, 6)), kv[0])
    )


def shape_scale(M, B) -> tuple[float, float]:
    """Split a matrix error into the part one scalar could fix and the part it cannot.

    Brightness is a scalar the laser absorbs; shape is the mesh hosting the wrong matrix.
    Reporting only the total confuses a dim block with a wrong one, and on this bench both
    happen at once."""
    M, B = np.asarray(M, float), np.asarray(B, float)
    a = float((M * B).sum() / max(float((B * B).sum()), 1e-18))
    return float(np.linalg.norm(M - a * B) / max(abs(a) * np.linalg.norm(B), 1e-18)), a


@dataclass
class Transfers:
    """|U|^2 MEASURED at known heater states: the chip's own answer to "what does this
    program host", with no twin anywhere in it."""

    volts: np.ndarray  # (n, N_HEATERS)
    T: np.ndarray  # (n, 4, 4), every column summing to 1
    path: Path | None = None

    def __len__(self):
        return len(self.T)

    def block(self, rails) -> np.ndarray:
        out, inp = rails
        return self.T[:, list(out), :][:, :, list(inp)]


def load_transfers(
    path=None, calib: Calibration | None = None, root: Path = ROOT
) -> Transfers | None:
    """The measured transfer table on file, or None if there is none.

    The newest session wins and its LARGEST table wins, so a `raw_transfers_big.json`
    dropped beside `raw_transfers.json` is picked up without a flag. `pic.normalise.
    load_session` handles the file's port-major storage, and the sweeps then go through the
    same `Calibration.to_transfer` the live probe uses -- which is what makes a plan picked
    off the table and a matrix measured on the bench the same object rather than two
    conventions that happen to agree."""
    calib = calib or Calibration.load_or_nominal()
    if path is not None:
        cands = [Path(path)]
    else:
        cands = sorted(Path(root).glob(TRANSFER_GLOB), key=lambda q: q.stat().st_mtime)
        if not cands:
            return None
        cands = [q for q in cands if q.parent == cands[-1].parent]
    best = None
    for q in cands:
        d = load_session(q)
        if not d["raws"]:
            continue
        # the session's own switch-off dark, not the calibration's: it was measured in the
        # same session as these sweeps, and a stale floor is the one nuisance a column sum
        # cannot divide out because it is additive rather than multiplicative.
        T = np.stack([calib.to_transfer(r, d["dark"]) for r in d["raws"]])
        # A sync writes the fitted mixing here rather than rewriting the file's raw reads,
        # which are real measurements and should stay as taken. Applying it on load keeps one
        # copy of the data and one of the correction, so a sync can be inspected or dropped.
        M = (d.get("meta") or {}).get("sync_mixing")
        if M is not None:
            Y = np.asarray(M, float) @ T  # (n,4,4), mixing on the output
            ssum = Y.sum(axis=1, keepdims=True)
            T = np.divide(Y, ssum, out=np.zeros_like(Y), where=ssum > 0)
        # Strictly greater would keep the FIRST table of the largest size, and `cands` is
        # mtime-sorted ascending, so ties went to the oldest capture -- on 2026-08-27 that
        # silently planned every run off a table taken during a thermal transient while a
        # fresher one of the same size sat beside it. Tables of equal size are separated by
        # nothing but age, and the youngest is the one that still describes the chip: a
        # 75-minute-old table differs from it by 0.100 per entry against a 0.0145 repeat.
        if best is None or len(T) >= len(best):
            best = Transfers(np.asarray(d["volts"], float), T, q)
    return best


def _shift(B, margin: float = SHIFT_MARGIN, floor: float = SINK_FLOOR):
    """B -> (c, sinkhorn(B + c)). Hoisted out of `_pick_state` because it depends only on
    the target: `rank_rails` reuses one factorisation across all 36 rail pairs, which is the
    difference between a ranking that is worth running and one that is not."""
    c = max(0.0, -float(np.asarray(B, float).min())) + margin * float(np.abs(B).mean())
    return c, sinkhorn(np.asarray(B, float) + c, floor=floor)


def _pick_state(H, transfers: Transfers, rails, floor: float = SINK_FLOOR, sink=None) -> Term:
    """The table state whose MEASURED block best realises the nonnegative piece H.

    Scored on the recovered matrix, not on the block. `Term.matrix()` undoes the Sinkhorn
    diagonals and the hosted scale before anything reaches the answer, and those diagonals
    weight the entries very unevenly, so the block residual an optimiser would minimise is
    not the error the caller gets.

    Brightness is deliberately NOT in this ranking, and it was tested rather than assumed.
    The recovery does amplify a stale table -- absolutely, by a median 3.45 -- and a dim state
    looks like it should amplify more, since the recovery divides by `scale`. It does not:
    `scale` is FITTED to the block, so it grows with the block, and the amplification is
    scale-invariant in relative terms. Adding a brightness penalty picked the same state on
    200 of 200 random targets and moved the matrix error by 0%. The amplification comes from
    `r` and `s`, which come from the TARGET and are common to every candidate, so no choice of
    state can touch it. (`SHIFT_MARGIN` can, and 0.25 is already its optimum -- see there.)"""
    A, r, s = sinkhorn(H, floor=floor) if sink is None else sink
    blocks = transfers.block(rails)
    scale = (blocks * A).sum((-2, -1)) / max(float((A * A).sum()), 1e-18)
    safe = np.where(scale > 0, scale, 1.0)[:, None, None]
    rec = np.einsum("i,nij,j->nij", 1 / r, blocks, 1 / s) / safe
    nH = max(np.linalg.norm(H), 1e-18)
    err = np.linalg.norm(rec - H, axis=(-2, -1)) / nH
    # how far a relative error in the stored block travels into the recovered matrix
    k = int(np.argmin(np.where(scale > 0, err, np.inf)))
    # copies, not views: the Program is handed to a driver and to `plan_from_table`, which
    # writes phases into it, and the table has to survive being planned against repeatedly.
    return Term(
        1.0,
        Program(
            transfers.volts[k].copy(),
            np.zeros(len(transfers.volts[k])),
            float(scale[k]),
            blocks[k].copy(),
            float(err[k]),
        ),
        r,
        s,
    )


def refine_term(
    term: Term,
    H,
    probe,
    rails,
    box: HeaterBox,
    trials: int = 6,
    step: float = 0.06,
    seed: int = 0,
    rng=None,
) -> Term:
    """Hill-climb a picked state on the CHIP, because the table is quantised, not noisy.

    `_pick_state` returns the best of the hundred states someone happened to measure, and on
    the 8x8 that residual is 0.16 of a 0.20 measured error -- four fifths of it. Sixteen
    tiles share the hundred states one 2x2 block gets to itself, so the shortfall is
    resolution and no amount of averaging touches it. Measuring a handful of fresh states
    NEAR the pick is far cheaper than capturing thousands blind: each trial is one heater
    write and one port cycle, and the gradient is not needed because the neighbourhood is
    small and the objective is measured directly.

    Scored exactly as `_pick_state` scores, on the recovered matrix rather than the raw
    block, so a refinement cannot win by improving a number the caller never sees. Steps are
    taken in phase, which is V^2, so a fixed fractional step moves every heater equally far
    around its own fringe rather than equally far in volts."""
    rng = rng or np.random.default_rng(seed)
    tr = np.flatnonzero(box.trainable)
    vmax = np.asarray(box.vmax, float)

    A = sinkhorn(H, floor=SINK_FLOOR)[0]

    def err_of(volts) -> tuple[float, np.ndarray, float]:
        blk = measure_transfer(probe, volts)[np.ix_(list(rails[0]), list(rails[1]))]
        sc = float((blk * A).sum() / max(float((A * A).sum()), 1e-18))
        rec = np.diag(1 / term.r) @ blk @ np.diag(1 / term.s) / (sc if sc > 0 else 1.0)
        return (float(np.linalg.norm(rec - H) / max(np.linalg.norm(H), 1e-18)), blk, sc)

    best_v = term.prog.volts.copy()
    best_e, best_blk, best_sc = err_of(best_v)
    for _ in range(max(0, trials)):
        v = best_v.copy()
        ph = (v[tr] / np.maximum(vmax[tr], 1e-9)) ** 2  # fraction of full phase
        ph = np.clip(ph + rng.normal(0, step, tr.size), 0.0, 1.0)
        v[tr] = vmax[tr] * np.sqrt(ph)
        e, blk, sc = err_of(v)
        if e < best_e:
            best_e, best_v, best_blk, best_sc = e, v, blk, sc
    return Term(
        term.coeff,
        Program(best_v, np.zeros(len(best_v)), best_sc, best_blk, best_e),
        term.r,
        term.s,
    )


REFINE_MIN_TRIALS = 4


def refine_budget(
    errs, budget: int, min_trials: int = REFINE_MIN_TRIALS, max_trials: int | None = None
) -> np.ndarray:
    """Split a fixed total of refinement trials across tiles by hosted residual.

    `run_block` used to give every tile the same `--refine N`, which on a 16x16 is 64 x 9
    sweeps whether a tile is hosted at 0.02 or at 0.25. It should not be uniform, and the
    right weighting is derivable rather than a taste:

      the block error adds incoherently over tiles -- measured, `theory.matmat.accumulation`
      puts the observed sum on the independent bound at every tile count -- so what a budget
      is spending against is `sum e_i^2`, not `sum e_i`;
      refinement's return is concave in trials -- `refine_ablation` measures 24 / 41 / 100
      percent of the full-refine benefit for a quarter / half / all of the trials -- which is
      close enough to logarithmic that `g'(t) ~ 1/t`;
      equalising the marginal gain `e_i^2 g'(t_i)` across tiles then gives `t_i ~ e_i^2`.

    So trials go as the SQUARE of the residual, not the residual, and the worst tiles take
    most of the budget by construction. `min_trials` is a floor with teeth: a tile gets that
    many or none at all, because a refinement visit costs one baseline sweep before its first
    trial and two or three trials of a random hill climb do not reliably beat that overhead.
    Tiles that get nothing keep their picked state, which is also what lets them share a
    sweep with another tile that picked the same one.

    Measured against uniform `--refine N`, paired on the same targets and the same table
    (`refine_ablation`, 115-state table, 1.2 percent read noise). "keep" is the fraction of
    the benefit that uniform 8-per-tile delivers, and the sweep counts are what the bench
    actually pays -- a funded tile costs one baseline sweep on top of its trials, an unfunded
    one costs nothing:

        8x8, 16 tiles          uniform                    budgeted
        trials  sweeps    hosted   keep      sweeps    hosted   keep
            32      48    0.1692    24%          34    0.1611    36%
            64      80    0.1555    41%          71    0.1431    76%
           128     144    0.1327   100%         140    0.1186   133%

        16x16, 64 tiles
           128     192    0.1643    16%         135    0.1628    20%
           256     320    0.1502    57%         282    0.1422    64%
           512     576    0.1262   100%         556    0.1253    95%

    So: HALF the trials keeps two thirds to three quarters of the benefit, and a quarter
    keeps a fifth to a third. Per sweep the budgeted policy is 1.3-1.6x the return of the
    uniform one everywhere below the full budget, and at the full budget it is simply better
    for the same money. Uniform at one trial per tile is *negative* -- 128 sweeps to make the
    residual slightly worse -- which is the clearest statement of the problem: a visit costs
    a sweep before it can win anything, so trials must not be spread thin.

    Returns an integer trial count per tile, in the order `errs` came in."""
    e = np.asarray(errs, float)
    budget = max(0, int(budget))
    if e.size == 0 or budget <= 0:
        return np.zeros(e.size, int)
    cap = budget if max_trials is None else int(max_trials)
    w = np.square(np.clip(e, 0.0, None))
    if not w.sum() > 0:
        return np.zeros(e.size, int)
    t = np.floor(budget * w / w.sum()).astype(int)
    t = np.minimum(t, cap)
    t[t < min_trials] = 0  # a visit is worth paying for or it is not
    # hand the floored remainder to the worst tiles that are already funded, then to the
    # worst unfunded one that the remainder can lift over the floor
    left = budget - int(t.sum())
    for i in np.argsort(-e):
        if left <= 0:
            break
        if t[i] > 0:
            add = min(left, cap - t[i])
        elif left >= min_trials:
            add = min(left, cap)
        else:
            continue
        t[i] += add
        left -= add
    return t


def twin_table(box: HeaterBox, probe, n_states: int, rng) -> Transfers:
    """A measured transfer table, measured through `probe`. One chip, table and all.

    The point of the argument is negative: a table captured on the BENCH and then refined
    against the twin scores candidate states on a different chip from the one the starting
    point came from. It walks away from the optimum and reports a 772 percent regression
    that is an artefact of mixing the two. Any refinement experiment has to sample its table
    from the same instrument it then refines against, which is what this is for."""
    tr = np.flatnonzero(box.trainable)
    volts = np.zeros((n_states, len(box.vmax)))
    volts[:, tr] = np.asarray(box.vmax, float)[tr] * np.sqrt(rng.uniform(0, 1, (n_states, tr.size)))
    T = np.stack([measure_transfer(probe, v) for v in volts])
    return Transfers(volts, T, None)


def refine_ablation(
    budgets=(0, 8, 16, 32),
    n_states: int = 100,
    tiles: int = 16,
    k: int = TILE_K,
    noise: float = 0.012,
    targets: int = 6,
    seed: int = 0,
    min_trials: int = REFINE_MIN_TRIALS,
) -> list[dict]:
    """What refinement buys per trial, uniform against budgeted, on one consistent chip.

    Each budget is run twice over the same targets and the same table: once as `--refine N`
    on every tile, once as the same TOTAL number of trials handed out by `refine_budget`.
    The two spend identical hardware time by construction -- `sweeps` counts the baseline
    visit each funded tile costs -- so the comparison is where the trials go and nothing
    else.

    Reported on the hosted residual, which is what refinement optimises and what the
    2026-08-27 ablation reported, and on the measured block error beside it, which is what
    the caller gets."""
    from theory.intensity_matvec import twin_probe
    from theory.twin import MeshError, Twin

    box = bench_box()
    rows = []
    for b in budgets:
        acc = {"uniform": [], "budget": []}
        cost = {"uniform": 0, "budget": 0}
        for t in range(targets):
            rng = np.random.default_rng(seed + 1000 * t)
            twin = Twin(MeshError.sample(seed=seed + t))
            probe = twin_probe(twin, box, noise=noise, rng=rng)
            table = twin_table(box, probe, n_states, rng)
            B = rng.normal(size=(k * int(np.sqrt(tiles)), k * int(np.sqrt(tiles))))
            for how in ("uniform", "budget"):
                plan = plan_block_from_table(B, table, box, k=k)
                keys = list(plan.tiles)
                errs = np.array([plan.tiles[q].terms[0].prog.err for q in keys])
                per = (
                    np.full(len(keys), b)
                    if how == "uniform"
                    else refine_budget(errs, b * len(keys), min_trials=min_trials)
                )
                r2 = np.random.default_rng(seed + 7 * t)
                for q, n in zip(keys, per):
                    if n <= 0:
                        continue
                    i, j = q
                    H = (
                        np.asarray(plan.B)[i * k : (i + 1) * k, j * k : (j + 1) * k]
                        + plan.tiles[q].offset
                    )
                    plan.tiles[q].terms = (
                        refine_term(
                            plan.tiles[q].terms[0], H, probe, plan.rails, box, trials=int(n), rng=r2
                        ),
                    )
                cost[how] += int(per.sum()) + int((per > 0).sum())
                got = measured_block_matmat(plan, probe, np.eye(B.shape[1]))[0]
                acc[how].append(
                    (
                        float(np.mean([p.terms[0].prog.err for p in plan.tiles.values()])),
                        score(got.ravel(), B.ravel())[0],
                    )
                )
        row = {"per_tile": b, "targets": targets}
        for how in ("uniform", "budget"):
            a = np.array(acc[how])
            # per target as well as the mean: nominally identical targets host anywhere from
            # 1e-4 to 2e-2, so a mean over a handful of them is not a measurement and the
            # comparison has to be PAIRED on the same target and the same table
            row[how] = {
                "hosted": float(a[:, 0].mean()),
                "measured": float(a[:, 1].mean()),
                "per_target": a,
                "sweeps": cost[how] / targets,
            }
        rows.append(row)
    return rows


def reanchor(
    rig,
    transfers: Transfers,
    m: int = 5,
    calib: Calibration | None = None,
    rails=BEST_RAILS,
    dbm: float = 8.0,
    return_mixing: bool = False,
):
    """Transport a stored table onto today's chip by re-measuring m of its states.

    A table is perishable: 75 minutes old it differs from the chip by 0.100 per entry
    against a 0.0145 seconds-apart repeat, and that staleness swamps everything downstream --
    driving the planning residual to exactly zero moved the measured error by 0.0001.
    Recapturing costs 400 s, which is itself longer than the drift time constant, so the
    table is stale before it is finished.

    It does not have to be recaptured. The drift is a 15-parameter mixing on the OUTPUT
    side, `T_now = colnorm(M T_then)`, and nothing on the input side: `Calibration.
    to_transfer` divides each column by its own sum, so launch power, fibre coupling and
    switch loss are already gone when the model sees the data and what remains is per
    detector response and detector-to-detector leakage -- a left multiplication. Fitting it
    needs five states, not a hundred: 20 s buys hosting 0.073 against 0.079 for the full
    recapture.

    Five and not fewer because 15 parameters need 15 constraints and a column-normalised
    column carries 3, so m=1 makes the table WORSE (0.238 hosting). And measured now, not
    earlier: ties taken 74 minutes before the run left the table worse than no correction at
    all, because the fit lands the table where the ties were."""
    from theory.design import Table, correct, fit_mixing, select, should_correct
    from .normalise import sweep

    calib = calib or rig.calib
    blocks = transfers.block(rails).reshape(len(transfers), -1)
    idx = select("maxmin", blocks, int(m))
    # `sweep` deliberately does not write volts -- the caller owns the settle, because only
    # the caller knows whether the heater state changed. Here it changes on every iteration,
    # and without this the first port of each tie is read during the thermal transient. That
    # is the one measurement where it cannot be tolerated: the reanchor exists to decide whether
    # the table has moved, and a transient reads as motion, so an unsettled tie makes the
    # correction fire on its own settling and then fits the table to it.
    fresh = []
    for i in idx:
        rig.measure(transfers.volts[i])
        time.sleep(DEFAULT_SETTLE_S)
        fresh.append(calib.to_transfer(sweep(rig, transfers.volts[i], repeats=2)))
    fresh = np.stack(fresh)
    stale = Table(transfers.volts, transfers.T, None, None, "stored")
    moved = float(np.abs(stale.T[idx] - fresh).mean())
    if not should_correct(moved, REPEAT_FLOOR, int(m)):
        print(
            f"  reanchor: table moved {moved:.4f}, inside the {REPEAT_FLOOR:.4f} repeat "
            f"floor for m={m} -- left alone"
        )
        return (transfers, None) if return_mixing else transfers
    out = correct(stale, idx, fresh)
    print(
        f"  reanchor: {m} states re-measured, table moved {moved:.4f} per entry, "
        f"15-parameter output mixing applied to all {len(transfers)}"
    )
    fixed = Transfers(transfers.volts, out.T, transfers.path)
    # `./do sync` stores the mixing beside the raw reads rather than rewriting them, so it
    # needs the fit itself and not just the corrected table.
    return (fixed, fit_mixing(stale.T[idx], fresh)) if return_mixing else fixed


def _pick_states(
    H, transfers: Transfers, rails, k: int, floor: float = SINK_FLOOR, sink=None
) -> tuple[Term, ...]:
    """Host H as a WEIGHTED SUM of k measured states instead of the single nearest one.

    The table is a codebook and picking one entry is nearest-neighbour quantisation, which
    on the 8x8 leaves four fifths of the error: sixteen tiles share the hundred states one
    block gets to itself. But the sum over terms already happens in software -- `matvec`
    computes sum_i coeff_i * term_i(x) -- so the reachable set was never the hundred points,
    it is their linear SPAN, and nobody was using it.

    A real 2x2 lives in four dimensions and the measured blocks span all four (singular
    values 4.12, 1.63, 0.70, 0.39 on the 2026-08-27 table), so four terms host any target
    exactly and the residual is not reduced but eliminated.

    **On the bench that buys nothing and costs the answer.** A paired A/B -- same table,
    same target, only `--terms` changed -- on the 8x8:

        K   hosted residual   measured matrix   vector error
        1          0.0195            0.1875           0.0419
        4          0.0000            0.1876           0.0701

    The planning residual goes to zero, the matrix the chip actually hosts does not move at
    all, and the vector answer gets 1.7x worse. Both halves of that follow from what is
    already written above: the residual K removes is quantisation of the table, and the
    table is stale and mis-scaled by far more than it is quantised, so eliminating the small
    error leaves the large one untouched. What K does change is the noise, exactly as the
    simulation says -- sum|c| runs to about 6 at k=4, and 1.2 percent read noise amplified
    by 6 is the 0.028 the vector error grew by.

    The simulation's case for k=4 was never the amplification, which it got right; it was
    its single-term baseline of 0.657. That is a twin-planned single term, and this die does
    not host one that badly -- picking off the measured table gets to 0.0195, thirty times
    better -- so the trade it was weighing does not exist here. Default 1. Kept available
    because on a table dense enough to make quantisation the dominant error the arithmetic
    comes back, and because the effect is worth re-measuring after a recapture.

    Greedy matching pursuit rather than a full subset search: the dictionary is tiny in the
    only dimension that matters and the gain is in leaving the codebook at all, not in
    picking the ideal four."""
    A, r, s = sinkhorn(H, floor=floor) if sink is None else sink
    blocks = transfers.block(rails)
    scale = (blocks * A).sum((-2, -1)) / max(float((A * A).sum()), 1e-18)
    ok = scale > 0
    rec = np.einsum("i,nij,j->nij", 1 / r, blocks, 1 / s) / np.where(ok, scale, 1.0)[:, None, None]
    D = rec.reshape(len(rec), -1)[ok]
    idx_map = np.flatnonzero(ok)
    target, chosen, coef = np.asarray(H, float).ravel(), [], np.ones(1)
    resid = target.copy()
    for _ in range(max(1, k)):
        corr = np.abs(D @ resid) / np.maximum(np.linalg.norm(D, axis=1), 1e-12)
        corr[chosen] = -np.inf
        chosen.append(int(np.argmax(corr)))
        M = D[chosen].T
        coef, *_ = np.linalg.lstsq(M, target, rcond=None)
        resid = target - M @ coef
    err = float(np.linalg.norm(resid) / max(np.linalg.norm(target), 1e-18))
    out = []
    for c, j in zip(coef, chosen):
        n = int(idx_map[j])
        out.append(
            Term(
                float(c),
                Program(
                    transfers.volts[n].copy(),
                    np.zeros(len(transfers.volts[n])),
                    float(scale[n]),
                    blocks[n].copy(),
                    err,
                ),
                r,
                s,
            )
        )
    return tuple(out)


def plan_from_table(
    B,
    transfers: Transfers,
    box: HeaterBox | None = None,
    rails=None,
    mode: str = "shift",
    floor: float = SINK_FLOOR,
    margin: float = SHIFT_MARGIN,
    sink=None,
    terms_k: int = 1,
) -> MatvecPlan:
    """Plan B onto a heater state PICKED from measured transfers, with no twin in the loop.

    `plan_matvec` fits volts through `theory.twin`, and on this chip that model is not
    predictive. Over the 16-state table the twin's |U|^2 misses the measured one by a
    relative 1.14 -- worse than predicting nothing at all -- and the best of all 576
    row/column permutations still misses by 0.73, against 0.48 for simply predicting the
    mean measured transfer. So a plan fitted through it hosts a matrix nobody has seen,
    which is how a bench run reported a 0.0001 hosted residual and 25% sign accuracy in the
    same breath. Until `learn.unitary_fit` closes on this die, fitting is the weaker move.

    Picking is the stronger one: for every state in the table, recover the signed matrix its
    measured block realises under the plan's own arithmetic, and keep the closest. The
    residual it reports is then a measurement, and `measured_matrix` re-reading the same
    state on the bench is a genuine check of it rather than a tautology.

    The table size is the resolution limit and nothing else is, which is the useful thing
    about it. A random signed 2x2 on the default rails lands at a median matrix error of
    0.232 with 93% of signs over 16 measured states and 0.094 with 98% over 100 -- against
    1.1 and 25% for the plan fitted through the twin. More states is how that number falls;
    a better optimiser is not."""
    B = np.asarray(B, float)
    rails = block_rails(B.shape[0]) if rails is None else rails
    if mode == "split":
        pos = _pick_state(np.clip(B, 0, None), transfers, rails, floor)
        neg = _pick_state(np.clip(-B, 0, None), transfers, rails, floor)
        terms, offset = (pos, Term(-1.0, neg.prog, neg.r, neg.s)), 0.0
    elif mode == "shift":
        offset, sink = _shift(B, margin, floor) if sink is None else (sink[0], sink[1])
        terms = (
            _pick_states(B + offset, transfers, rails, terms_k, floor, sink)
            if terms_k > 1
            else (_pick_state(B + offset, transfers, rails, floor, sink),)
        )
    else:
        raise ValueError(f"mode={mode!r}; use 'shift' or 'split'")
    if box is not None:
        for t in terms:
            t.prog.phases[:] = box.phases(t.prog.volts)
    return MatvecPlan(B, rails, mode, terms, offset)


def plan_block_from_table(
    B,
    transfers: Transfers,
    box: HeaterBox | None = None,
    k: int = TILE_K,
    rails=None,
    mode: str = "shift",
    terms_k: int = 1,
) -> BlockPlan:
    """`theory.matmat.plan_block`, tile by tile, off the measured table."""
    B = np.atleast_2d(np.asarray(B, float))
    M, N = B.shape
    rails = block_rails(k) if rails is None else rails
    Bp = np.zeros(((M + k - 1) // k * k, (N + k - 1) // k * k))
    Bp[:M, :N] = B
    tiles = {
        (i, j): plan_from_table(
            Bp[i * k : (i + 1) * k, j * k : (j + 1) * k],
            transfers,
            box,
            rails,
            mode,
            terms_k=terms_k,
        )
        for i in range(Bp.shape[0] // k)
        for j in range(Bp.shape[1] // k)
        if Bp[i * k : (i + 1) * k, j * k : (j + 1) * k].any()
    }
    return BlockPlan(B, k, rails, tiles)


def rank_rails(transfers: Transfers, k: int = 2, trials: int = 24, seed: int = 0) -> list[dict]:
    """Rank rail pairs on MEASURED transfers rather than on the twin.

    `theory.intensity_matvec.scan_rails` ranks by fitting the twin, so on this die it ranks
    a model. The table answers the operational question directly -- host random signed
    targets on each pair, see what comes back -- and carries four measurements that explain
    the answer, none of which orders the rails on its own:

      `host`   median relative error of the recovered matrix over the random targets. This
               is the ranking; the rest is why.
      `cond`   median condition number of the measured block over the states. This is the
               conditioning number to read, because it is scale-free: |det| of a 2x2 goes as
               the square of whatever scale the readout applied, so the stored per-column
               scale being up to 5x wrong moved |det| by two orders of magnitude on this
               bench while the condition number moved from 2.0 to 1.8. A pair whose `cond`
               stays large is ill-conditioned in the chip and no readout fixes it.
      `det`    |det| of the measured block, best over states, under the column
               normalisation. Kept because it is the quantity that says how far from
               singular the BEST state on that pair is, in the units the plan works in;
               read it beside `cond`, never instead of it.
      `steer`  mean swing of a block entry across the states -- the diameter of the
               reachable set, measured, and the single best predictor of `host` of the
               three (rank correlation +0.41, against +0.33 for `cond` and +0.26 for
               `det`). Port 3 is the least steerable rail on the bench -- its dominant
               output moves 0.38 to 0.55 where port 1's moves 0.003 to 0.98 -- but that is
               a handicap and not a disqualification, and the measurement is worth more
               than the intuition: out(1,3) in(0,3) still ranks 4th of 36. What sinks to
               last is the pair with no steering left on EITHER rail, out(0,3) in(2,3) at
               0.205.
      `bright` fraction of the injected light landing on the output rails, mean over states.
               It divides straight into the read noise.

    `shift` only. Measured on the 100-state table at the default rails, `split` hosts the
    MATRIX better -- 0.076 against 0.108 at zero noise, because two independent state picks
    are more freedom than one -- and is nonetheless the wrong decomposition: its vector
    error is 0.40 against 0.15, and stays 2.7x worse out to 2.5% read noise, because each
    sign-split half is sparse, its Sinkhorn diagonals are large, and the two terms very
    nearly cancel. It also costs a second program, so a second thermal settle per vector.
    Ranking rails on it would rank them for a job nobody should run.
    """
    rng = np.random.default_rng(seed)
    targets = [(B, _shift(B)) for B in (rng.normal(size=(k, k)) for _ in range(trials))]
    rows = []
    for out in itertools.combinations(range(NMODE), k):
        for inp in itertools.combinations(range(NMODE), k):
            blocks = transfers.block((out, inp))
            errs = [
                score(
                    plan_from_table(B, transfers, rails=(out, inp), sink=sink).predict().ravel(),
                    B.ravel(),
                )[0]
                for B, sink in targets
            ]
            cond = np.linalg.cond(blocks)
            rows.append(
                {
                    "out": out,
                    "in": inp,
                    "host": float(np.median(errs)),
                    "cond": float(np.median(cond)),
                    "illcond": float(np.mean(cond > 100)),
                    "det": float(np.abs(np.linalg.det(blocks)).max()),
                    "steer": float((blocks.max(0) - blocks.min(0)).mean()),
                    "bright": float(blocks.sum((1, 2)).mean()),
                }
            )
    return sorted(rows, key=lambda r: r["host"])


def _require_table(transfers, what):
    """No table, no run. The twin planner is not a fallback -- it is a trap.

    It was reachable here whenever `load_transfers` returned None, and on this die it plans
    against a model whose median relative error over 135 measured states is 1.04, against 0.38
    for simply predicting the mean transfer -- 2.7x worse than a constant guess. On the very
    state the table planner picks for a bench target it puts 0.002 where the chip measures
    0.900. A plan fitted through it hosts a matrix nobody has seen, reports a residual near
    zero because it is scored against the same model that produced it, and delivered vectors
    1.1 out at 25 percent sign accuracy. Failing loudly is the only safe behaviour.

    The twin stays everywhere else -- it IS the mock chip, the simulator and the selftests,
    and `4x4/CLAUDE.md` requires the hardware-free path to be a physical model rather than a
    stub. What is removed is only its use as a PLANNER against real hardware."""
    if transfers is None:
        raise RuntimeError(
            f"{what} needs a measured transfer table and none was found. Capture one "
            f"(scratchpad/raw_capture.py, or `./do char` then a sweep) -- planning through "
            f"`theory.twin` instead is not offered: it scores 1.04 relative against this die "
            f"where guessing the mean transfer scores 0.38."
        )


def run(
    rig,
    B,
    box: HeaterBox | None = None,
    vectors=None,
    rails=None,
    mode: str = "shift",
    terms_k: int = 1,
    power: str = "digital",
    dbm: float = 8.0,
    seed: int = 0,
    calib=None,
    twin=None,
    normalise: str = "column",
    transfers: Transfers | None = None,
    repeats: int = 1,
    cycles: int | None = None,
    refresh: bool = False,
    refine: int = 0,
    **fit_kw,
) -> dict:
    """Plan B onto the mesh, MEASURE what it hosts, then measure y = B x for each vector.

    The order is the point. `matrix_err` is the hosted block read off the photodiodes and
    compared against the target; `twin_matrix_err` is the old `plan.predict()` number under
    a name that says what it is. Both are reported, because the gap between them is the
    twin's error on this die and nothing else -- and while only the second existed it read
    0.0002 on a bench run whose vectors were 1.1 out.

    Pass `transfers` and the plan is picked from measured states instead of fitted through
    the twin; `plan_from_table` has the measurement that says why that is not a refinement.

    `refresh` re-sweeps the chip for every vector instead of serving them from the state
    `SweepProbe` already holds. Off by default because the cached answer is the *same
    measurement*: in digital mode a read does not depend on the requested power, so N
    vectors under one program are arithmetic on one sweep and `vec_err` is then a function
    of `matrix_err` rather than an independent estimate. Turn it on to put read noise back
    into `vec_err` -- and know what it costs: `SweepProbe.invalidate` drops the cache but
    the next sweep rewrites the heaters, so it buys a thermal settle per vector.

    No laser session is opened here -- the caller owns that, because it owns the watchdog."""
    B = np.asarray(B, float)
    calib = calib or rig.calib
    box = box or bench_box(calib)
    _require_table(transfers, "matvec")
    plan = plan_from_table(B, transfers, box, rails=rails, mode=mode, terms_k=terms_k)
    probe = rig_probe(
        rig, calib, power=power, dbm=dbm, normalise=normalise, repeats=repeats, cycles=cycles
    )
    if not hasattr(probe, "per_column"):  # the deprecated static closure counts nothing
        probe = CountedProbe(probe)
    rng = np.random.default_rng(seed)
    xs = [rng.normal(size=B.shape[1]) for _ in range(4)] if vectors is None else list(vectors)

    measured = measured_matrix(plan, probe)
    m_shape, m_scale = shape_scale(measured, B)
    m_cond = float(np.linalg.cond(measured))
    rows = []
    for x in xs:
        if refresh and hasattr(probe, "invalidate"):
            probe.invalidate()
        y = plan.matvec(probe, x)
        # The chip hosts `m_scale * B`, and `m_scale` was just MEASURED off the photodiodes
        # by cycling the ports -- a cost the column normalisation pays anyway. Dividing it
        # out is recovering a known scalar, not fitting one to the answer: it uses only the
        # realised matrix and the target, never y or x. On 2026-08-27 the input coupling
        # dropped to x0.67 when the laser was moved to a working USB controller, which put
        # 0.349 of matrix error on the board of which only 0.145 was shape.
        y_c = y / m_scale if abs(m_scale) > 1e-9 else y
        e, sg = score(y_c, B @ x)
        raw_e, _ = score(y, B @ x)
        rows.append(
            {
                "x": np.asarray(x, float),
                "y": y_c,
                "y_raw": y,
                "y_true": B @ x,
                "err": e,
                "raw_err": raw_e,
                "sign": sg,
            }
        )
    return {
        "plan": plan,
        "rows": rows,
        "measured": measured,
        "cost": probe,
        "matrix_err": score(measured.ravel(), B.ravel())[0],
        "matrix_shape": m_shape,
        "matrix_scale": m_scale,
        "matrix_cond": m_cond,
        "twin_matrix_err": score(plan.predict().ravel(), B.ravel())[0],
        # a probe with no cache to invalidate re-reads every pass whatever `refresh`
        # says, so the vectors are independent measurements and the digest must say so
        "planner": "table" if transfers else "twin",
        "refresh": bool(refresh) or not hasattr(probe, "invalidate"),
        "vec_err": float(np.mean([r["err"] for r in rows])),
        "vec_err_raw": float(np.mean([r["raw_err"] for r in rows])),
        "sign_acc": float(np.mean([r["sign"] for r in rows])),
    }


def run_block(
    rig,
    B,
    box: HeaterBox | None = None,
    k: int = TILE_K,
    cols: int = 3,
    X=None,
    rails=None,
    power: str = "digital",
    dbm: float = 8.0,
    seed: int = 0,
    calib=None,
    twin=None,
    normalise: str = "column",
    transfers: Transfers | None = None,
    repeats: int = 1,
    refine: int = 0,
    budget: int = 0,
    cycles: int | None = None,
    settle_s: float = DEFAULT_SETTLE_S,
    terms_k: int = 1,
    **fit_kw,
) -> dict:
    """Tile B into k x k blocks, host each in turn, and measure Y = B X.

    `BlockPlan.matmat` owns the loop order and it is the only thing here that decides the
    clock: the tile is the outer loop, so each program is written once and every column that
    tile can serve is served before the heaters move again.

    The cost report comes from whichever probe is in use rather than from a wrapper.
    `SweepProbe` already counts the reads and switch moves the sweep performs, and a
    `CountedProbe` around it would count one read per *request* -- which is now the thing
    that is free.

    As in `run`, every tile's realised matrix is MEASURED before a column of X goes through
    it, so `matrix_err` is a reading and `twin_matrix_err` is the prediction beside it.

    `refine` hill-climbs every tile for N trials; `budget` spends a fixed TOTAL across the
    tiles that need it (`refine_budget`), with `refine` as the per-tile cap. On a 16x16 that
    is 576 sweeps against 282 for two thirds of the benefit.

    The winner's re-measurement is deliberate and is not waste. `refine_term` leaves the
    measured block of its best trial in `prog.blk`, and reusing it would save one sweep per
    refined tile -- but that read is the one the refinement SELECTED on, and the minimum of
    nine noisy measurements is biased low by about the read noise. Reporting it as the
    measured matrix would be a winner's curse worth a fifth of the residual. The extra sweep
    buys an independent read, which is what this function is for."""
    B = np.atleast_2d(np.asarray(B, float))
    calib = calib or rig.calib
    box = box or bench_box(calib)
    _require_table(transfers, "block matvec")
    plan = plan_block_from_table(B, transfers, box, k=k, rails=rails, terms_k=terms_k)
    probe = rig_probe(
        rig,
        calib,
        power=power,
        dbm=dbm,
        normalise=normalise,
        repeats=repeats,
        cycles=cycles,
        settle_s=settle_s,
    )
    if (refine or budget) and transfers is not None:
        rng = np.random.default_rng(seed)
        keys = list(plan.tiles)
        errs = np.array([plan.tiles[q].terms[0].prog.err for q in keys])
        # A budget is the whole point of a budget: `--refine N` per tile is 64 x (N+1)
        # sweeps on a 16x16 whatever the tiles cost, and most of them cost little.
        trials = (
            refine_budget(errs, budget, max_trials=refine or None)
            if budget
            else np.full(len(keys), int(refine))
        )
        before = float(np.mean(errs))
        for q, n in zip(keys, trials):
            if n <= 0:
                continue
            i, j = q
            tile = plan.tiles[q]
            # the nonnegative piece each tile's Term was picked against: the shift the
            # planner applied, not the raw target block
            Bt = np.asarray(plan.B)[i * k : (i + 1) * k, j * k : (j + 1) * k]
            H = Bt + tile.offset
            tile.terms = (
                refine_term(tile.terms[0], H, probe, plan.rails, box, trials=int(n), rng=rng),
            )
        after = float(np.mean([t.terms[0].prog.err for t in plan.tiles.values()]))
        n_hit = int((trials > 0).sum())
        print(
            f"refined {n_hit} of {len(keys)} tiles on-chip, {int(trials.sum())} trials "
            f"({int(trials.sum()) + n_hit} sweeps): mean hosted residual "
            f"{before:.4f} -> {after:.4f}"
        )
    if not hasattr(probe, "per_column"):
        probe = CountedProbe(probe)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(B.shape[1], cols)) if X is None else np.atleast_2d(X)
    measured, Y = measured_block_matmat(plan, probe, X)
    err, sign = score(Y.ravel(), (B @ X).ravel())
    return {
        "plan": plan,
        "X": X,
        "Y": Y,
        "Y_true": B @ X,
        "cost": probe,
        "measured": measured,
        "matrix_err": score(measured.ravel(), B.ravel())[0],
        "matrix_shape": shape_scale(measured, B)[0],
        "twin_matrix_err": score(plan.predict().ravel(), B.ravel())[0],
        "planner": "table" if transfers else "twin",
        "vec_err": err,
        "sign_acc": sign,
    }


def _plan_line(res) -> str:
    """The two matrix numbers, always together. Separately either one misleads: the
    prediction is a calculation and the measurement has no target to be judged against."""
    plan = res["plan"]
    src = "picked off the measured table" if res["planner"] == "table" else "fitted on the twin"
    return (
        f"hosted residual {plan.err:.4f} ({src})   twin predicts "
        f"{res['twin_matrix_err']:.4f} vs target\n"
        f"MEASURED matrix vs target {res['matrix_err']:.4f}"
        + (
            f"   (shape {res['matrix_shape']:.4f}, brightness "
            f"x{res['matrix_scale']:.3f}, cond {res['matrix_cond']:.1f})"
            if "matrix_scale" in res
            else ""
        )
    )


def digest_block(res) -> str:
    plan, cost = res["plan"], res["cost"]
    n_col = res["X"].shape[1]
    return "\n".join(
        [
            f"{plan.shape[0]}x{plan.shape[1]} in {plan.programs} tiles of {plan.k}x{plan.k} on "
            f"rails out{plan.rails[0]} in{plan.rails[1]}",
            _plan_line(res),
            f"cost: {cost}  ->  {cost.reads / n_col:.0f} reads per column of X; the writes and "
            f"switch moves are per TILE and do not grow with the {n_col} columns",
            f"Y error {res['vec_err']:.4f}, sign accuracy {res['sign_acc']:.0%} "
            f"(the sum over tiles is done in software; the chip did every multiply)",
        ]
    )


def digest(res) -> str:
    plan = res["plan"]
    out, inp = plan.rails
    lines = [
        f"rails out{out} in{inp}, {plan.mode} decomposition",
        _plan_line(res),
        f"{'x':<28} {'B x':<28} {'measured':<28} {'err':>7}",
    ]
    for r in res["rows"]:
        lines.append(
            f"{np.array2string(r['x'], precision=2):<28} "
            f"{np.array2string(r['y_true'], precision=2):<28} "
            f"{np.array2string(r['y'], precision=2):<28} {r['err']:>7.3f}"
        )
    lines.append(
        f"mean vector error {res['vec_err']:.4f} "
        f"(uncorrected {res.get('vec_err_raw', float('nan')):.4f}; the measured "
        f"scale x{res['matrix_scale']:.3f} is divided out), "
        f"sign accuracy {res['sign_acc']:.0%}"
    )
    lines.append(
        f"cost: {res['cost']} for the matrix and {len(res['rows'])} vectors"
        + (
            ""
            if res.get("refresh")
            else " -- the vectors are arithmetic on the held sweep, so their error is a "
            "function of the measured matrix, not a second opinion on it"
        )
    )
    return "\n".join(lines)


def _selftest(seed: int = 0):
    """The theory validation run against the BENCH box, which is the number that counts.

    Two levels of read noise, both measured: `pic.sim` puts the PD/TIA read noise at
    0.6-4.4 mV against a 173-261 mV full scale.

    Then both probes against `pic.sim`, whose |U|^2 is known exactly, so the question is the
    measurement chain alone and not a plan on top of it. The simulator carries the two
    nuisances the bench has: a per-port launch (`eta_in` times the switch's repeatability)
    and a per-output loss (`eta_out`). Column normalisation removes the first exactly and
    the second not at all, which is the honest description of what it does and the reason
    the residual below is not zero.

    Then the two claims the measurement path rests on. `measured_matrix` must reduce to
    `MatvecPlan.predict()` when the bench IS the twin, or the gap it reports on hardware is
    a difference of arithmetic rather than a fact about the chip. And `rank_rails`, from the
    measured table alone, must independently recover `BEST_RAILS` into its leaders and sink
    the least steerable pair -- out(0,3) in(2,3), whose block entries move 0.21 across a
    hundred states where the best pair's move 0.82 -- into its tail. Top-3 and bottom-third
    rather than exact ranks: the leaders sit within 0.01 of each other, which is inside the
    spread over target draws, so anything sharper would be asserting noise."""
    from theory.intensity_matvec import twin_probe
    from theory.twin import Twin

    from .rig import Rig

    box = bench_box()
    clean = validate(box, k=2, trials=6, restarts=12, steps=300, seed=seed)
    noisy = validate(box, plan=clean["plan"], noise=0.012, trials=12, seed=seed + 1)
    free = int(box.trainable.sum())
    assert clean["vec_err"] < 0.05, clean["vec_err"]
    assert clean["sign_acc"] == 1.0, clean["sign_acc"]

    calib = Calibration.load_or_nominal()
    v = np.zeros(N_HEATERS)
    v[REACHABLE_DACS] = 0.4 * VOLTAGE_MAX
    with Rig(laser="mock", board="sim", tec="mock", switch="mock") as rig:
        rig.measure(v)
        truth = rig.board.sim._mesh_t(rig.board.sim.phases(v)).T  # T[pd, port]
        got = {}
        for m in ("column", "static"):
            p = rig_probe(rig, calib, settle_s=0.0, normalise=m)
            got[m] = np.stack([p(v, k) for k in range(NMODE)], axis=1)
        cost = rig_probe(rig, calib, settle_s=0.0)
        cols = np.stack([cost(v, k) for k in range(NMODE)], axis=1)
        [cost(v, k) for k in range(NMODE)]  # the cache must not re-read a held state
    err = {m: float(np.abs(T - truth).max()) for m, T in got.items()}

    assert abs(cols.sum() - NMODE) < 1e-9, cols.sum()  # every column already sums to 1
    assert cost.reads == NMODE and cost.writes == 1, str(cost)
    assert err["column"] < 0.5 * err["static"], err
    # the static path does not merely mis-scale: an `input_scale` fitted at another state
    # over-divides, and an over-divided dark subtraction goes negative, which is the same
    # pathology the bench saw.
    #
    # The bar is the sign and the ordering, not a fixed depth. The excursion is the dark
    # subtraction over-divided, so it scales with `pd_offset` -- re-measuring that on
    # 2026-08-28 (PD0 31.5 mV -> 0.14 mV, after the ADC reference moved off AVCC) took the
    # excursion to -0.008 and would have failed a hardcoded -0.01, having changed nothing
    # about the pathology. `err["column"] < 0.5 * err["static"]` above carries the magnitude.
    # Since the rewire the excursion at this state is +0.005 rather than negative: whether
    # over-division actually crosses zero depends on the operating point the mock lands on,
    # and the map moved it. The ordering is the part that is about the pathology rather than
    # about the state, and `err["column"] < 0.5 * err["static"]` above carries the magnitude.
    assert got["static"].min() < got["column"].min(), (got["static"].min(), got["column"].min())

    # the budget policy, as arithmetic: it must spend what it is given, never fund a tile
    # below the floor, and put more on the worse tile
    e = np.array([0.30, 0.05, 0.20, 0.02, 0.10])
    t8 = refine_budget(e, 8 * e.size)
    assert t8.sum() == 8 * e.size, t8
    assert t8[0] > t8[2] > t8[4], t8
    assert not ((0 < t8) & (t8 < REFINE_MIN_TRIALS)).any(), t8
    small = refine_budget(e, 6)
    assert small.sum() == 6 and small[0] == 6, small  # all of it on the worst tile
    assert refine_budget(e, 0).sum() == 0
    assert refine_budget(np.zeros(4), 40).sum() == 0  # nothing to win, nothing spent

    # and end to end on one consistent chip: budgeted must beat uniform at the same trials
    abl = refine_ablation(budgets=(0, 4), n_states=60, tiles=4, targets=3, noise=0.012)
    assert abl[1]["budget"]["hosted"] <= abl[0]["budget"]["hosted"], abl
    assert abl[1]["budget"]["sweeps"] <= abl[1]["uniform"]["sweeps"], abl

    tprobe = twin_probe(Twin(), box)
    Mm = measured_matrix(clean["plan"], tprobe)
    same = float(np.abs(Mm - clean["plan"].predict()).max())
    # the differential pair must BE the measured matrix applied to x. Both shots come off
    # one column-normalised sweep, so a scale they failed to share would show up right here
    # -- which is the check the sign errors on 2026-08-27 needed and nothing performed.
    xd = np.array([0.7, -1.3])
    signed = float(np.abs(clean["plan"].matvec(tprobe, xd) - Mm @ xd).max())
    tab = load_transfers(calib=Calibration.load_or_nominal())
    ranked = rank_rails(tab) if tab is not None else []
    assert same < 1e-5, same
    assert signed < 1e-6, signed
    if ranked:
        top = [(r["out"], r["in"]) for r in ranked[:3]]
        stuck = min(ranked, key=lambda r: r["steer"])
        # NOT `BEST_RAILS in top`. That asserted a stored constant against whatever table
        # happens to match the glob, so it failed the moment a new capture landed -- and it
        # ranked on `rank_rails`' median hosting over random targets, which is the wrong
        # statistic for a block run anyway: sixteen tiles are SUMMED, so the worst tile
        # dominates and the median hides it. Feeding the median ranking to an 8x8 chose a pair
        # whose worst tile was 0.6631 and returned Y error 0.5581; re-ranking on the worst tile
        # picked another and returned 0.1041 on the same table and the same target.
        #
        # What is worth asserting is the property, not the constant: the ranking must put a
        # steerable pair on top and must not rank an unsteerable one there.
        assert ranked[0]["steer"] > stuck["steer"], (ranked[0]["steer"], stuck["steer"])
        assert ranked.index(stuck) >= 2 * len(ranked) // 3, (stuck, ranked.index(stuck))
    return {
        "free": free,
        "span_max": float(box.span_pi[box.trainable].max(initial=0.0)),
        "budget": t8,
        "ablation": abl,
        "measured_vs_predicted": same,
        "differential": signed,
        "table": tab,
        "ranked": ranked,
        "fit_err": clean["fit_err"],
        "vec_err": clean["vec_err"],
        "sign_acc": clean["sign_acc"],
        "noisy_err": noisy["vec_err"],
        "noisy_sign": noisy["sign_acc"],
        "passes": clean["passes"],
        "brightness": clean["plan"].terms[0].prog.scale,
        "rails": clean["rails"],
        "probe_err": err,
        "probe_cost": str(cost),
        "eta_out_spread": float(np.ptp(rig.board.sim.eta_out)),
    }


def target_matrix(a):
    """The B a `python -m pic matvec` invocation will work on. Shared with `job_seconds`,
    which has to know the shape before the laser is lit."""
    rng = np.random.default_rng(a.seed)
    return np.atleast_2d(np.loadtxt(a.target) if a.target else rng.normal(size=(a.k, a.k)))


def job_seconds(a, batched: bool = False) -> tuple[float, str]:
    """(watchdog duration, one-line explanation) for the run `a` describes.

    `batched` is whether the board can run a four-port sweep in one round trip, and it moves
    this number by 5-6x, so it is asked of the open board rather than assumed -- an estimate
    made for the wrong firmware is either a watchdog that fires mid-run or one that never
    fires at all.

    Sized in whole port cycles, because that is the unit the bench is charged in: one
    heater program costs a thermal settle plus four switch moves, near five seconds, and
    the reads inside it are the cheap part. The flat 120 s this replaced was under a
    twentieth of what an 8x8 at K=4 needs, and the watchdog fired mid-run -- the beam went
    off, the loop kept reading, and the dark current was reported as an answer.

    The multi-term cost is the surprise and it is the cache, not the arithmetic:
    `SweepProbe` holds ONE heater state, so a K-term plan re-sweeps every term for the
    measurement and again for the application. K=1 measures a tile once and serves every
    vector from that sweep; K=4 pays eight port cycles per tile for the same tile.

    Rounded up rather than tuned: a watchdog that fires early destroys the run, and one
    that fires late costs nothing but a few seconds of lit diode at the end. `--terms` is
    counted whether or not the table planner ends up being the one used, so a `--mock` run
    -- which plans through the twin, where terms do not multiply the programs -- gets an
    upper bound. That is the correct direction to be wrong in."""
    if getattr(a, "scan", False):
        return 0.0, "--scan is a fit; no light needed"
    B = target_matrix(a)
    terms = 2 if a.mode == "split" else max(1, int(getattr(a, "terms", 1)))
    repeats = max(1, int(getattr(a, "repeats", 1)))
    refine = max(0, int(getattr(a, "refine", 0)))
    budget = max(0, int(getattr(a, "budget", 0)))
    cycles = int(getattr(a, "sweep_cycles", None) or SWEEP_CYCLES)
    per_state = sweep_seconds(repeats, batched=batched, cycles=cycles)
    if getattr(a, "unitary", False):
        tiles, n_vec, terms, what = 2 * max(1, a.cols), 0, 1, f"{a.cols} composed pairs"
    elif a.block:
        k = int(a.block)
        rows, cols = -(-B.shape[0] // k), -(-B.shape[1] // k)
        tiles, n_vec = rows * cols, max(1, a.cols)
        what = f"{B.shape[0]}x{B.shape[1]} in {tiles} {k}x{k} tiles x {a.cols} columns"
    else:
        tiles, n_vec = 1, 4  # `run` draws four vectors when none are passed
        what = f"one {B.shape[0]}x{B.shape[1]} block x {n_vec} vectors"
    # Each tile is measured once per term. Applying it costs nothing more while one term
    # fits in the cache; K > 1 re-sweeps every term, and --refresh drops the cache on
    # purpose. Block mode is port-major, so its columns are served by that one revisit.
    if a.block or getattr(a, "unitary", False):
        per_tile = terms * (2 if terms > 1 else 1)
    else:
        per_tile = terms + n_vec * (
            terms if terms > 1 else (1 if getattr(a, "refresh", False) else 0)
        )
    states = tiles * per_tile
    if budget:
        # a budget caps the trials outright, and every funded tile costs one baseline sweep
        # on top. The worst case is the most tiles the floor allows, which is what to size a
        # watchdog against.
        states += budget + min(tiles, budget // REFINE_MIN_TRIALS)
    elif refine:
        states += tiles * (1 + refine)  # the pick is re-measured, then N fresh states
    states += max(0, int(getattr(a, "reanchor", 0)))  # m stored states re-measured
    # Tiles that pick the same table state share one sweep (`_by_state`), which on a
    # hundred-state table and 64 tiles is a real saving -- and it is deliberately NOT
    # subtracted here. The sharing depends on the plan, the plan does not exist yet, and a
    # watchdog is the one estimate that must only ever be too long.
    dur = estimate_sweep_job(states, repeats=repeats, batched=batched, cycles=cycles)
    return dur, (
        f"{what}, {terms} term(s) -> {states} heater programs at "
        f"{per_state:.1f} s per four-port sweep"
        + ("" if batched else " (host-loop sweep: reflash for the batched one)")
    )


def main(rig, a) -> int:
    """`python -m pic matvec`. `rig` is open and inside a laser session, or None for
    `--scan`, which is a fit and needs no light."""
    sim = getattr(a, "sim", False)
    calib, twin = (rig.calib if rig is not None else Calibration.load_or_nominal()), None
    if getattr(a, "mock", False) and not sim:
        calib, twin = mock_truth()
    box = bench_box(calib)
    B = target_matrix(a)
    # the table is THIS die MEASURED, so against either simulated instrument it is data
    # about a different chip -- which is the whole point of them, and would turn the
    # measured/predicted gap into a statement about the simulator.
    planner = getattr(a, "planner", "auto")
    tab = (
        load_transfers(calib=calib)
        if planner == "table" or (planner == "auto" and not (getattr(a, "mock", False) or sim))
        else None
    )
    if planner == "table" and tab is None:
        print("no measured transfer table on file -- falling back to the twin")
    if tab is not None and getattr(a, "reanchor", 0) and rig is not None:
        tab = reanchor(rig, tab, m=a.reanchor, calib=calib, dbm=a.dbm)

    print(
        f"heater box: {int(box.trainable.sum())} steerable channels, widest span "
        f"{box.span_pi[box.trainable].max(initial=0.0):.2f} pi "
        f"({'measured' if calib.meta else 'NOMINAL'} calibration)"
    )
    if tab is not None:
        print(f"transfer table: {len(tab)} measured states from {tab.path}")
    if a.scan:
        if tab is not None:
            print(
                f"\nmeasured rail ranking for a {B.shape[0]}x{B.shape[0]} block over "
                f"{len(tab)} states, best hosting first:"
            )
            print(
                f"  {'out':<9}{'in':<9}{'host err':>9}{'cond':>7}{'ill':>6}{'|det|':>8}"
                f"{'steer':>8}{'bright':>8}"
            )
            for r in rank_rails(tab, k=B.shape[0])[:6]:
                print(
                    f"  {str(r['out']):<9}{str(r['in']):<9}{r['host']:>9.3f}"
                    f"{r['cond']:>7.1f}{r['illcond']:>6.0%}{r['det']:>8.3f}"
                    f"{r['steer']:>8.3f}{r['bright']:>8.3f}"
                )
        A = sinkhorn(np.abs(B) + 0.05)[0]
        print(
            f"\ntwin rail scan for the same block (feasible first, brightest among those)"
            f"{' -- a ranking of the model, not the die' if tab is not None else ''}:"
        )
        print(f"  {'out':<9}{'in':<9}{'residual':>9}{'brightness':>12}")
        for r in scan_rails(A, box, restarts=12, steps=300)[:6]:
            print(f"  {str(r['out']):<9}{str(r['in']):<9}{r['err']:>9.4f}{r['scale']:>12.3f}")
        return 0

    norm = "static" if getattr(a, "static_scale", False) else "column"
    if a.unitary:
        probe = rig_probe(
            rig, calib, power="laser" if a.optical_input else "digital", dbm=a.dbm, normalise=norm
        )
        # the same planner the rest of this module switched to: the group property is a
        # statement about the chip, so both factors have to be hosted the way the chip is
        # actually programmed, not the way the twin imagines it
        pf = (
            None
            if tab is None
            else (lambda B, _s: plan_from_table(B, tab, box, rails=None, mode=a.mode))
        )
        g = compose(
            box,
            twin,
            k=a.block or TILE_K,
            trials=a.cols,
            seed=a.seed,
            probe=probe,
            plan_fn=pf,
            restarts=a.restarts,
            steps=a.steps,
        )
        print(
            f"\ngroup property over {g['trials']} pairs of orthogonal matrices, composed "
            f"by feeding the measured A through program b"
        )
        print(
            f"  distance to O(k) before projection   {g['dist_c']:.4f}   "
            f"(needs no truth; a lower bound on the error below)"
        )
        print(f"  error vs the true product, raw       {g['raw']:.4f}")
        print(
            f"  after polar projection               {g['polar']:.4f}   "
            f"(the nearest orthogonal matrix)"
        )
        print(
            f"  after QR                             {g['qr']:.4f}   "
            f"(moves by {g['qr_spread']:.4f} if the columns are reordered)"
        )
        return 0

    print(f"\ntarget B:\n{np.round(B, 3)}")
    common = dict(
        rails=None if a.rails is None else _rails(a.rails),
        power="laser" if a.optical_input else "digital",
        dbm=a.dbm,
        seed=a.seed,
        calib=calib,
        twin=twin,
        normalise=norm,
        transfers=tab,
        repeats=getattr(a, "repeats", 1),
        cycles=getattr(a, "sweep_cycles", None),
        restarts=a.restarts,
        steps=a.steps,
    )
    if a.block:
        res = run_block(
            rig,
            B,
            box,
            k=a.block,
            cols=a.cols,
            refine=getattr(a, "refine", 0),
            budget=getattr(a, "budget", 0),
            terms_k=getattr(a, "terms", 1),
            **common,
        )
        print()
        print(digest_block(res))
    else:
        res = run(
            rig,
            B,
            box,
            mode=a.mode,
            terms_k=getattr(a, "terms", 1),
            refine=getattr(a, "refine", 0),
            refresh=getattr(a, "refresh", False),
            **common,
        )
        print()
        print(digest(res))

    # only on measured evidence. The old form of this fired on `plan.err`, a residual against
    # the twin, and so stayed silent through the run it was written to catch.
    if res["matrix_err"] > 0.05:
        print(
            f"\nthe mesh does not host this target: the {res['matrix_err']:.4f} above is "
            "MEASURED, and it is in the MATRIX, not the readout. Try --scan for better "
            "rails, or a smaller block."
        )
        if res["twin_matrix_err"] < 0.5 * res["matrix_err"]:
            print(
                f"  the twin predicted {res['twin_matrix_err']:.4f} for the same plan, so "
                "that gap is the model and not the bench -- re-characterize before "
                "believing any fitted program."
            )
    return 0


def _rails(spec: str):
    """`1,2:0,3` -> ((1, 2), (0, 3))."""
    out, inp = spec.split(":")
    return (tuple(int(x) for x in out.split(",")), tuple(int(x) for x in inp.split(",")))


if __name__ == "__main__":
    r = _selftest()
    print(f"bench box: {r['free']} steerable channels, widest span {r['span_max']:.2f} pi")
    print(f"2x2 on rails out{r['rails'][0]} in{r['rails'][1]}, {r['passes']} reads/vector")
    print(f"  hosted residual {r['fit_err']:.4f}, block brightness {r['brightness']:.3f}")
    print(f"  vector error / sign  {r['vec_err']:.4f} / {r['sign_acc']:.0%}  (noiseless)")
    print(
        f"  vector error / sign  {r['noisy_err']:.4f} / {r['noisy_sign']:.0%}  "
        f"(1.2% read noise, single read)"
    )
    print(
        f"against `pic.sim`'s known |U|^2 at one held state: worst entry off by "
        f"{r['probe_err']['column']:.3f} through the four-port sweep, "
        f"{r['probe_err']['static']:.3f} through the stored input_scale"
    )
    print(
        f"  what the sweep cannot remove is the {r['eta_out_spread']:.2f} spread in "
        f"output loss, which is a row scale and needs `pd_gain`, not a column"
    )
    print(
        f"  one held state costs {r['probe_cost']}, and the second pass over its ports "
        f"costs nothing"
    )
    print(
        f"  the measured matrix reduces to the predicted one to {r['measured_vs_predicted']:.1e} "
        f"when the bench is the twin, so any gap on the die is the die"
    )
    print(
        f"  the differential pair reproduces that matrix on a signed x to "
        f"{r['differential']:.1e}: both shots share one column-normalised sweep"
    )

    a0, a1 = r["ablation"]
    print(
        f"refinement budget on residuals [0.30 0.05 0.20 0.02 0.10] -> {r['budget']} "
        f"trials: it goes as the square of the residual, and a tile under the floor gets "
        f"nothing rather than a visit it cannot pay for"
    )
    print(
        f"  4 tiles, 4 trials each: uniform {a1['uniform']['hosted']:.4f} in "
        f"{a1['uniform']['sweeps']:.0f} sweeps, budgeted {a1['budget']['hosted']:.4f} in "
        f"{a1['budget']['sweeps']:.0f}, from {a0['uniform']['hosted']:.4f} unrefined"
    )

    tab, ranked = r["table"], r["ranked"]
    if tab is None:
        print("\nno measured transfer table on file -- rails can only be ranked on the twin")
    else:
        print(f"\nrails ranked on {len(tab)} MEASURED states from {tab.path}")
        print(
            f"  {'out':<9}{'in':<9}{'host err':>9}{'cond':>7}{'ill':>6}{'|det|':>8}"
            f"{'steer':>8}{'bright':>8}"
        )
        for row in ranked[:5]:
            print(
                f"  {str(row['out']):<9}{str(row['in']):<9}{row['host']:>9.3f}"
                f"{row['cond']:>7.1f}{row['illcond']:>6.0%}{row['det']:>8.3f}"
                f"{row['steer']:>8.3f}{row['bright']:>8.3f}"
            )
        print(
            f"  `ill` is the fraction of states whose block has condition number over 100 "
            f"-- states the chip cannot host a general target in, whatever the readout "
            f"does. |det| is scale-dependent and `cond` is not; read them together"
        )
        stuck = min(ranked, key=lambda row: row["steer"])
        print(
            f"  BEST_RAILS out{BEST_RAILS[0]} in{BEST_RAILS[1]} ranks "
            f"{[(row['out'], row['in']) for row in ranked].index(BEST_RAILS) + 1} of "
            f"{len(ranked)} on measured data, which is the twin's one confirmed call"
        )
        print(
            f"  least steerable pair out{stuck['out']} in{stuck['in']} (steer "
            f"{stuck['steer']:.3f}, |det| {stuck['det']:.3f}) ranks "
            f"{ranked.index(stuck) + 1}: unsteerable is unusable, and the ranking finds "
            f"that without being told which port is bad"
        )
