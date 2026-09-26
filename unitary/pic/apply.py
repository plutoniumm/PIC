"""The seam a front end sits on: a target U, some input vectors, and what the chip returned.

`unitary.py` is the terminal over `pic.matvec.run_block`. This is the same multiply with the
argparse Namespace, the printing and the `sys.exit` taken out, so a caller that is not a
terminal can have the numbers. Nothing is measured here -- `pic.matvec` does that -- and
nothing is constructed here either: the rig (bench or mock), the heater box and the transfer
table all arrive as arguments, because the one decision a front end must never make for
itself is which instrument it is talking to.

Two hosting modes, one call. `hosting="full"` asks for all of U, which goes on as four 2x2
tiles and four heater programs; `hosting="tile"` asks for one 2x2 sub-block of the same U on
one program. They are not a good and a degraded path, they are different questions: a random
signed 2x2 is hosted to ~3e-3 and read back at 0.006 relative on this die, and 3x3 and 4x4
are not hosted at all (`theory.intensity_matvec`, and it is heater span rather than
optimisation). The tile is what this chip does; the full 4x4 is what the tiling makes of it.

**What a number off this bench can mean.** One input port is lit at a time and the ports are
mutually incoherent, so a photodiode returns |U_kj|^2 p_j and never a field. Three
consequences ride in the API rather than in a comment:

  - Every DEVICE metric compares intensities, so it pins the target down only up to
    U -> D_out U D_in with D diagonal unitary (`GAUGE_ABS2`). That gauge is physical here:
    the two output-screen trimmers the v2 rewire dropped, H18 and H14, are exactly it, and
    they cost nothing to drop for exactly this reason.
  - The amplitude fidelity |tr(U^H V)|/N (`theory.program.fidelity`) needs a phase reference
    the bench does not have. It is reported against the plan's own predicted matrix, `source`
    names the planner that produced it, `measurable` is False, and it is never computed from
    a photodiode reading.
  - The per-x number is a DISTRIBUTION. `rel_err` and `sign_acc` come back one per column of
    X and are not averaged away; `spread` carries the quantiles beside the mean, because a
    mean over four vectors is the statistic that let a run report 0.0002 in the matrix while
    its vectors came back 1.1 out.

    python -m pic.apply        # both hostings on the mock chip
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.clements import NMODE
from theory.intensity_matvec import score
from theory.matmat import TILE_K
from theory.program import fidelity

from . import layout
from .config import VOLTAGE_MAX_CH, mirror_pairs, DAC_PAIRS, DEFAULT_SETTLE_S, PAIR_OF_DAC, pin_detectors
from .matvec import (
    Transfers,
    bench_box,
    load_transfers,
    measure_transfer,
    mock_truth,
    rig_probe,
    run_block,
)

READOUT = "intensity"  # what the photodiodes return. "field" is the day there is an LO.
HOSTING = ("full", "tile")
TABLE_STATES = 100  # states in a table measured fresh on a simulated chip

GAUGE_ABS2 = (
    "intensity readout: fixes the target only up to U -> D_out U D_in, the "
    "diagonal unitary gauges no photodiode can see"
)
NO_PHASE_REF = (
    "not measurable on this bench -- |tr(U^H V)|/N needs a phase reference, so V "
    "is the plan's predicted matrix and never a reading"
)


class ReadoutError(ValueError):
    """Refused by the READOUT. Named after the readout and not after a dtype, so that there
    is one place to change when a phase reference arrives."""


def _hostable(U, X):
    """The gate in `unitary.py._hostable`, raising where the CLI exits.

    With one input port lit there is no cross term at any output, so a photodiode reads
    |U_kj|^2 p_j and the sum over j happens in software: the phases of a complex U have
    nowhere to act on the data, and the answer would be (|U|^2)x. Lifting this is necessary
    and not sufficient -- the shift, the Sinkhorn diagonals and `measured_matrix` under
    `pic.matvec` are real-valued too. A library raises because a front end has to be able to
    put the refusal in front of a person and stay running."""
    if READOUT == "intensity" and (np.iscomplexobj(U) or np.iscomplexobj(X)):
        raise ReadoutError(
            "complex target: this bench reads intensity, so the phases of U never reach the "
            "data and the answer would be (|U|^2)x, not Ux. Pass U.real or np.abs(U), or add "
            "a phase reference at the output and set READOUT."
        )


@dataclass(frozen=True)
class Wiring:
    """Which heater each DAC channel reaches, read live from the modules that own it.

    Reported, never applied. Every volt vector in this path is DAC-indexed from the planner
    down to `Rig.measure`, so the map belongs at that one boundary -- `pic.layout`
    (`HEATER_OF_DAC` into the `theory.layout` table) for which heater, `pic.config.DAC_PAIRS`
    for which channels are shorted together -- and a front end that permuted volts on top
    would double-apply the day either changes. Read on every call rather than captured at
    import, so a UI showing this is showing the board as the library currently believes it,
    not as it was when the process started.

    `one_to_one` is the fact worth a banner: it went False with the v2 rewire, where sixteen
    channels came to drive thirteen heaters."""

    heater_of_dac: np.ndarray
    pads: tuple
    pairs: tuple
    one_to_one: bool

    @classmethod
    def current(cls) -> "Wiring":
        m = np.asarray(layout.HEATER_OF_DAC, int)
        pads = tuple(layout.HEATERS[int(d)].pad for d in m)
        return cls(m, pads, DAC_PAIRS, len(set(pads)) == len(pads))

    def __str__(self):
        return (
            f"{len(self.pads)} DAC channels on {len(set(self.pads))} heaters, "
            f"{len(self.pairs)} bonded pair(s) {self.pairs}"
        )


@dataclass(frozen=True)
class Metric:
    """One number with the two things that decide whether it may be shown as evidence:
    where it came from, and what it still does not pin down."""

    name: str
    value: float
    source: str  # "device" is off the photodiodes; anything else is a calculation
    measurable: bool  # False: no configuration of this bench produces this number
    caveat: str


@dataclass
class ApplyResult:
    """Everything a front end needs and nothing it has to recompute.

    `X`, `B` and both `Y` are the ones actually used: under `hosting="tile"` they are the
    sub-block and the rows of x it takes, and `rows`/`cols` say which entries of the caller's
    U and x those were."""

    hosting: str
    rows: tuple
    cols: tuple
    B: np.ndarray  # the target that was hosted
    X: np.ndarray  # (len(cols), n) inputs, one per column
    Y_device: np.ndarray  # (len(rows), n) measured on the chip
    Y_cpu: np.ndarray  # (len(rows), n) = B @ X, in software
    rel_err: np.ndarray  # (n,) per column, never averaged here
    sign_acc: np.ndarray  # (n,) per column
    matrix: Metric
    amp_fidelity: Metric
    measured: np.ndarray  # the hosted matrix read off the photodiodes
    orthogonal: bool  # is B orthogonal, i.e. is a fidelity defined at all
    wiring: Wiring
    cost: str
    raw: dict  # the `pic.matvec.run_block` result, for anything not above

    @property
    def programs(self) -> int:
        """Heater writes, and thermal settles with them: 4 for the full 4x4, 1 for a tile."""
        # the direct 4x4 writes the heaters once and has no plan to ask
        return self.raw["plan"].programs if "plan" in self.raw else 1

    @property
    def spread(self) -> dict:
        """The per-x distribution. The mean ships inside it, not instead of it."""
        e, s = self.rel_err, self.sign_acc
        lo, p25, med, p75, hi = (float(v) for v in np.percentile(e, [0, 25, 50, 75, 100]))
        return {
            "n": int(e.size),
            "mean": float(e.mean()),
            "min": lo,
            "p25": p25,
            "median": med,
            "p75": p75,
            "max": hi,
            "sign_mean": float(s.mean()),
            "sign_min": float(s.min()),
        }

    def digest(self) -> str:
        s = self.spread
        lines = [
            f"{self.hosting}: {self.B.shape[0]}x{self.B.shape[1]} of U on out{self.rows} "
            f"in{self.cols}, {self.programs} heater program(s) of {TILE_K}x{TILE_K}",
            f"{'x':<27}{'CPU  B x':<27}{'device':<27}{'rel err':>8}{'sign':>6}",
        ]
        for c in range(self.X.shape[1]):
            lines.append(
                f"{np.array2string(self.X[:, c], precision=2):<27}"
                f"{np.array2string(self.Y_cpu[:, c], precision=2):<27}"
                f"{np.array2string(self.Y_device[:, c], precision=2):<27}"
                f"{self.rel_err[c]:>8.3f}{self.sign_acc[c]:>6.0%}"
            )
        lines.append(
            f"rel err over {s['n']} x: min {s['min']:.3f} median {s['median']:.3f} "
            f"max {s['max']:.3f} (mean {s['mean']:.3f}); sign {s['sign_mean']:.0%} "
            f"mean, {s['sign_min']:.0%} worst"
        )
        for m in (self.matrix, self.amp_fidelity):
            lines.append(f"{m.name} {m.value:.4f} [{m.source}] -- {m.caveat}")
        lines.append(f"cost: {self.cost}; wiring: {self.wiring}")
        return "\n".join(lines)


def drive_units(box) -> list[tuple[tuple[int, ...], float]]:
    """What the board can be commanded independently: DRIVE UNITS, not DAC channels.

    Since the v2 rewire three heaters take a bonded pair of channels, shorted on the board,
    and `PIC._prep_dac` refuses a pair commanded to two voltages -- rightly, because two
    DAC81416 output stages tied together fight each other. So the free variable is the group,
    and anything that draws or perturbs volts has to draw one number per group. The grouping
    is read from `pic.config.DAC_PAIRS` on every call and is never restated here.

    A group is drivable if ANY of its channels is trainable, since they are one heater; its
    ceiling is the LOWEST of its channels', because the pair cannot straddle two clamps."""
    vmax, trainable = np.asarray(box.vmax, float), np.asarray(box.trainable, bool)
    seen, units = set(), []
    for d in range(len(vmax)):
        if d in seen:
            continue
        grp = PAIR_OF_DAC.get(d, (d,))
        seen.update(grp)
        if trainable[list(grp)].any():
            units.append((grp, float(vmax[list(grp)].min())))
    return units


def box_for(rig, *, mock: bool):
    """The reachable set of the chip in hand. Pair it with `table_for` on the same flag.

    On the bench the calibration file IS the truth. Against the mock it is a file about a
    DIFFERENT chip, and `pic.matvec.mock_truth` is the fabricated instance `MockPIC` was
    actually built from -- so the box has to come from there or the run measures the gap
    between two chips instead of the algorithm. It is not a small effect and it is not
    visible in the answer: on the stored calibration the mock's transfer moves 0.19 across a
    hundred drawn states where on its own it moves 0.91, and a 2x2 that hosts to 0.12 lands
    at 1.68. `unitary.py` builds its mock box from `rig.calib`, which is that mistake."""
    return bench_box(mock_truth()[0] if mock else rig.calib)


def table_for(
    rig,
    box,
    *,
    mock: bool,
    n_states: int = TABLE_STATES,
    seed: int = 0,
    power: str = "digital",
    dbm: float = 8.0,
    repeats: int = 1,
) -> Transfers:
    """The transfer table `apply_unitary` plans from, for whichever chip is in hand.

    On the bench the stored table IS the instrument (`pic.matvec._require_table` has the
    measurement that says why planning through the twin is not a fallback). Against a
    simulated chip the stored table is data about a DIFFERENT one, so it is measured fresh on
    the chip in hand -- random reachable states, one four-port sweep each -- and the table and
    the run then come off the same instrument, which is the only consistency a planner needs.

    `mock` is the caller's own flag rather than something sniffed off the rig: the seam
    refuses to guess what it is plugged into, and the front end already had to make this
    choice once to build the Rig at all.

    The states are drawn per DRIVE UNIT. `unitary.py._table` draws per channel, which since
    the v2 rewire is refused by the board on the first write -- one number per bonded pair is
    not a refinement of that draw, it is the only draw that can be sent."""
    if not mock:
        return load_transfers(calib=rig.calib)
    rng = np.random.default_rng(seed)
    units = drive_units(box)
    volts = np.zeros((n_states, len(box.vmax)))
    for grp, hi in units:
        # uniform in V^2 is uniform in phase, which spreads the states over the reachable
        # set instead of bunching them at the cold end
        volts[:, list(grp)] = (hi * np.sqrt(rng.uniform(0, 1, n_states)))[:, None]
    probe = rig_probe(rig, rig.calib, power=power, dbm=dbm, repeats=repeats, settle_s=0.0)
    return Transfers(volts, np.stack([measure_transfer(probe, v) for v in volts]), None)


def _direct_4x4(rig, U, X, *, repeats: int = 1, settle_s: float = DEFAULT_SETTLE_S):
    """Host the WHOLE 4x4 on the mesh: one heater program, four input ports, four reads.

    The mesh is a Clements rectangular interferometer, so a 4x4 unitary maps to its fifteen
    phases by formula (`theory.program.phases_for`) -- no search, no table, no tiling. Write
    the heaters once, then walk the switch: lighting input j and reading detector k gives
    |U_kj|^2 up to the per-detector gain and the per-port coupling, so four switch positions
    measure the whole transfer matrix.

    This is what `hosting="full"` should always have meant. It used to cut U into four 2x2
    tiles and sum them on the host, which is `pic.matvec`'s scheme for a target the mesh
    cannot hold -- a nonnegative block under Sinkhorn. A unitary is exactly the case the
    mesh CAN hold, so tiling it spends four programs and twelve reads to reproduce something
    one program and four reads measure directly, and leaves the 4x4 existing only in the
    host's addition rather than in the optics.

    What the switch still costs: one port is lit at a time, so a signed y = Ux cannot be read
    in one shot. Column j is measured with input power |x_j| and the signs are applied on the
    host. That sum is forced by the 1x4 switch, not by the decomposition."""
    from theory.program import phases_for

    calib = rig.calib
    vmax = np.asarray(VOLTAGE_MAX_CH, float)
    phases = phases_for(U)
    volts, ok = calib.volts(phases, vmax=vmax)
    reach = float(np.mean(ok))
    mirror_pairs(volts)

    T = np.zeros((NMODE, NMODE))
    for j in range(NMODE):
        rig.select_input(j)
        y = np.mean([rig.outputs(volts) for _ in range(max(1, repeats))], axis=0)
        T[:, j] = y
    dark = np.asarray(calib.pd_offset, float)[:NMODE]
    T = np.clip(T - dark[:, None], 0.0, None)
    seen = np.isfinite(T).all(1)  # an output with no detector on it reads NaN
    if not seen.all():
        # Sinkhorn needs every row. A row of |U|^2 still sums to 1, so each measured row is
        # scaled by its own sum: that removes its detector's gain but leaves the per-port
        # input coupling in, which the column margin would have taken out.
        T[seen] /= np.maximum(T[seen].sum(1, keepdims=True), 1e-12)
        return T, volts, reach, ok

    # |U|^2 of a unitary is doubly stochastic, so BOTH margins are known a priori and any
    # deviation is instrument, not mesh. Normalising columns alone leaves the per-detector
    # gain in the rows -- measured here as row sums 0.65 to 1.62 across four detectors that
    # differ by 2.5x in responsivity. Sinkhorn removes the output gain and the input coupling
    # together, which is the same argument `Calibration.to_transfer` makes for dividing each
    # column by its own sum, carried to the other margin as well.
    import torch
    from theory.drift import sinkhorn

    T = sinkhorn(torch.as_tensor(np.clip(T, 1e-12, None), dtype=torch.float64)).numpy()
    return T, volts, reach, ok


def apply_unitary(
    rig,
    U,
    X,
    *,
    transfers: Transfers | None,
    hosting: str = "full",
    tile: tuple = (0, 0),
    box=None,
    rails=None,
    power: str = "digital",
    dbm: float = 8.0,
    repeats: int = 1,
    settle_s: float = DEFAULT_SETTLE_S,
    seed: int = 0,
    **kw,
) -> ApplyResult:
    """Measure Y = B X on `rig`, where B is U or one 2x2 sub-block of it. The whole seam.

    `rig` is open and the caller owns the laser session, as everywhere under `pic.matvec`:
    the watchdog belongs to whoever knows how long the job is. `transfers` has no default
    for the same reason `mock` is a flag rather than a sniff -- a silent fall back to the
    newest file on disk is the seam choosing an instrument. `table_for` is the one call that
    produces it, and `pic.matvec._require_table` refuses the run when it is None.

    `**kw` reaches `pic.matvec.run_block`, so `mode="split"`, `terms_k`, `normalise` and
    `cycles` are available without being re-listed here; `mode` there is the nonnegative
    decomposition and has nothing to do with `hosting`. `refine` and `budget` also reach it
    and are a trap today: they hill-climb the volts one CHANNEL at a time, which splits a
    bonded pair and is refused by `PIC._prep_dac` on the first write.

    Both hostings are one `run_block` call and differ only in how much of U is asked for.
    That is deliberate: a tile is not a special case with its own code path, it is a smaller
    B, so the two answers are comparable and any difference between them is the chip."""
    U, X = np.asarray(U), np.asarray(X)
    X = X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
    _hostable(U, X)
    if U.shape != (NMODE, NMODE):
        raise ValueError(f"target must be {NMODE}x{NMODE}, got {U.shape}")
    if X.shape[0] != NMODE:
        raise ValueError(f"vectors need {NMODE} rows, one per input port, got {X.shape[0]}")
    if hosting not in HOSTING:
        raise ValueError(f"hosting={hosting!r}; use one of {HOSTING}")
    U = U.astype(float)

    # Both hostings take the WHOLE 4x4 U. They differ in how the chip is asked to hold it:
    #   full -- one heater program, the mesh IS U, four switch positions read it out
    #   tile -- U cut into four 2x2 blocks, one program each, the HOST sums the partials
    # `tile` used to mean a single 2x2 sub-block of U, which returned two outputs and quietly
    # ignored half of every input vector. That is a different experiment, not a comparison.
    rows = cols = tuple(range(NMODE))
    B, Xu = U[np.ix_(rows, cols)], X[list(cols)]

    if hosting == "full":
        # The mesh holds a 4x4 unitary outright; see `_direct_4x4`. Tiling it would be the
        # scheme for a target the mesh cannot hold, and would put the 4x4 in the host's
        # addition rather than in the optics.
        T, volts, reach, ok = _direct_4x4(rig, U, X, repeats=repeats, settle_s=settle_s)
        Y = T @ Xu  # NaN rows where no detector sits, scored on the measured rows only
        Ytrue = np.abs(U) ** 2 @ Xu
        m = np.isfinite(T).all(1)
        per = np.array([score(Y[m, c], Ytrue[m, c]) for c in range(Y.shape[1])])
        A = np.abs(U[m]) ** 2
        merr = float(np.linalg.norm(T[m] - A) / max(np.linalg.norm(A), 1e-12))
        return ApplyResult(
            hosting=hosting,
            rows=rows,
            cols=cols,
            B=B,
            X=Xu,
            Y_device=Y,
            Y_cpu=Ytrue,
            rel_err=per[:, 0],
            sign_acc=per[:, 1],
            matrix=Metric("measured |U|^2 vs target", merr, "device", True, GAUGE_ABS2),
            amp_fidelity=Metric(
                "phases reachable",
                reach,
                "calibration",
                False,
                "fraction of the 15 Clements phases this calibration can actually make; "
                "the rest are clipped to the nearest reachable voltage",
            ),
            measured=T,
            orthogonal=bool(np.allclose(U @ U.T, np.eye(NMODE), atol=1e-6)),
            wiring=Wiring.current(),
            cost=f"1 heater write, {NMODE} switch moves, {NMODE * max(1, repeats)} reads",
            raw={"T": T, "volts": volts, "reachable": reach, "ok": ok},
        )

    box = bench_box(rig.calib) if box is None else box
    res = run_block(
        rig,
        B,
        box=box,
        k=TILE_K,
        X=Xu,
        transfers=transfers,
        rails=rails,
        power=power,
        dbm=dbm,
        repeats=repeats,
        settle_s=settle_s,
        seed=seed,
        **kw,
    )

    Y = np.asarray(res["Y"], float)
    per = np.array([score(Y[:, c], res["Y_true"][:, c]) for c in range(Y.shape[1])])
    orth = bool(np.allclose(B @ B.T, np.eye(len(rows)), atol=1e-8))
    return ApplyResult(
        hosting=hosting,
        rows=rows,
        cols=cols,
        B=B,
        X=Xu,
        Y_device=Y,
        Y_cpu=np.asarray(res["Y_true"], float),
        rel_err=per[:, 0],
        sign_acc=per[:, 1],
        matrix=Metric(
            "hosted matrix vs target", float(res["matrix_err"]), "device", True, GAUGE_ABS2
        ),
        amp_fidelity=Metric(
            "amplitude fidelity |tr(U^H V)|/N",
            float(fidelity(B, res["plan"].predict())),
            res["planner"],
            False,
            (
                NO_PHASE_REF
                if orth
                else NO_PHASE_REF + "; and this B is not orthogonal, so the ratio is an overlap "
                "rather than a fidelity -- a 2x2 sub-block of a unitary almost never is one"
            ),
        ),
        measured=np.asarray(res["measured"], float),
        orthogonal=orth,
        wiring=Wiring.current(),
        cost=str(res["cost"]),
        raw=res,
    )


@pin_detectors("pd")  # the PD path, whatever the bench has selected
def _selftest(seed: int = 0, cols: int = 6):
    """Both hostings against the mock chip, which is the only instrument there is today.

    The board is flashed with a firmware that answers none of this library's protocol
    (`mrunal/board_firmware_v2.md`), so `--mock` is not a rehearsal here, it is the run. What
    is checked is the seam and not the physics underneath it, which `pic.matvec._selftest`
    owns: that a complex target is refused by name, that the per-x numbers survive as a
    distribution, that the device metric carries its gauge and the fidelity admits it is not
    a measurement, and that the CPU side is arithmetic on the target rather than anything the
    chip returned.

    Every state the run writes is checked against `DAC_PAIRS` here as well as in the driver.
    The driver's refusal is the damage gate and fires one write in; this is the seam's own
    claim that it never produced such a write, which is the difference between a path that
    is correct and one that is caught.

    The two hostings run on ONE table off one chip, so the difference between their matrix
    errors is the tiling and nothing else."""
    from theory.clements import random_unitary

    from .rig import Rig

    rng = np.random.default_rng(seed)
    U = np.real(random_unitary(rng))
    U, _ = np.linalg.qr(U)  # real orthogonal: a fidelity is defined
    X = rng.normal(size=(NMODE, cols))

    out = {}
    with Rig(laser="mock", board="mock", tec="mock", switch="mock", detectors="pd") as rig:
        box = box_for(rig, mock=True)
        tab = table_for(rig, box, mock=True, seed=seed)
        for grp in DAC_PAIRS:
            assert np.allclose(tab.volts[:, list(grp)], tab.volts[:, grp[0]][:, None]), grp
        for h in HOSTING:
            out[h] = apply_unitary(
                rig, U, X, hosting=h, box=box, transfers=tab, settle_s=0.0, seed=seed
            )
        try:
            apply_unitary(rig, U.astype(complex) * 1j, X, box=box, transfers=tab)
            raise AssertionError("a complex target was accepted")
        except ReadoutError as e:
            assert "intensity" in str(e), str(e)

    for h, r in out.items():
        assert r.Y_device.shape == r.Y_cpu.shape == (len(r.rows), cols), r.Y_device.shape
        assert r.rel_err.shape == r.sign_acc.shape == (cols,), r.rel_err.shape
        # the CPU column is the target applied in software, not a rescaled device reading;
        # the direct 4x4 reads intensities, so its target is |U|^2 x
        want = (np.abs(r.B) ** 2 if h == "full" else r.B) @ r.X
        assert np.allclose(r.Y_cpu, want), h
        assert r.matrix.source == "device" and r.matrix.caveat == GAUGE_ABS2, r.matrix
        assert not r.amp_fidelity.measurable and r.amp_fidelity.source != "device"
        assert np.array_equal(r.wiring.heater_of_dac, layout.HEATER_OF_DAC)
        assert r.wiring.pairs == DAC_PAIRS and not r.wiring.one_to_one
        assert r.spread["max"] >= r.spread["median"] >= r.spread["min"], r.spread
    # direct: one heater program; tiled: one per 2x2 block, both over the whole U
    assert out["full"].programs == 1, out["full"].programs
    assert out["tile"].programs == (NMODE // TILE_K) ** 2, out["tile"].programs
    assert np.array_equal(out["tile"].B, U) and np.array_equal(out["full"].B, U)
    spd = _selftest_spd(U, X[:, :2], box, seed)
    # NOT an assertion that the tile beats the full 4x4. It should, and on the stored
    # calibration it does -- but that ordering is a claim about a chip, and the only chip
    # available is a mock whose heater model still indexes the PRE-rewire map, so asserting
    # it here would gate the seam on a number the map makes meaningless.
    return {
        "U": U,
        "results": out,
        "wiring": Wiring.current(),
        "matrix_err": {h: r.matrix.value for h, r in out.items()},
        "spread": {h: r.spread for h, r in out.items()},
        "spd": spd,
    }


def _selftest_spd(U, X, box, seed):
    """SPD mode end to end on four mock SPDs, one per output, each on its own dark and max:
    fits, a transfer table, a DPNN and both hostings, all on SPD values. Then one SPD, as the
    bench has it today: every consumer that needs all four outputs refuses and names them."""
    import tempfile

    from learn import train_hw

    from . import interface
    from .characterize import characterize, random_bases
    from .interface import MissingOutputs
    from .acquisition import grid
    from .rig import Rig

    out = {}
    with pin_detectors("spd"), Rig(laser="mock", board="mock", tec="mock", switch="mock") as rig:
        assert rig.detectors == "spd" and rig.board.outputs == [0, 1, 2, 3], rig.board.outputs
        assert not np.any(rig.calib.pd_offset) and np.all(rig.calib.pd_gain == 1), rig.calib
        # one heater the mock shows plainly on PD3 from port 3, to keep this seconds long
        chans = [6]
        with rig.session(duration_s=120, power_dbm=5.0, calibrating=True) as s:
            res, _ = characterize(
                rig.board, s, channels=chans, bases=random_bases(1), switch=rig.switch,
                ports=[3], levels=grid(32), settle_s=0.0, repeats=1, verbose=False,
            )
            tab = table_for(rig, box, mock=True, n_states=24, seed=seed)
            for h in HOSTING:
                out[h] = apply_unitary(rig, U, X, hosting=h, box=box, transfers=tab, settle_s=0.0)
    f = res[6]
    out["vpi"], truth = f["vpi"], float(mock_truth()[0].vpi[6])
    assert f["ok"] and abs(f["vpi"] - truth) < 0.05 * truth, (f, truth)
    assert np.isfinite(tab.T).all(), tab.T
    for h in HOSTING:
        assert np.isfinite(out[h].Y_device).all() and np.isfinite(out[h].matrix.value), h
    with tempfile.TemporaryDirectory() as d, pin_detectors("spd"):
        argv = ["--mock", "--rounds", "1", "--n-per-round", "16", "--steps", "0", "--epochs",
                "20", "--settle", "0", "--repeats", "1", "--out", d + "/dpnn"]
        import contextlib
        import io
        import json

        quiet = lambda: contextlib.redirect_stdout(io.StringIO())
        with quiet():
            assert train_hw.main(argv) == 0

        out["dpnn_r2"] = json.load(open(d + "/dpnn/meta.json"))["r2_dpnn"]
        assert np.isfinite(out["dpnn_r2"]), out["dpnn_r2"]

        keep = interface.MOCK_SPDS
        interface.MOCK_SPDS = {k: v for k, v in keep.items() if v[0] == 1}
        try:
            with Rig(laser="mock", board="mock", tec="mock", switch="mock") as rig:
                r = apply_unitary(rig, U, X, hosting="full", box=box, transfers=tab, settle_s=0.0)
                for what, call in (
                    ("tiled hosting", lambda: apply_unitary(
                        rig, U, X, hosting="tile", box=box, transfers=tab, settle_s=0.0)),
                    ("heater characterization",
                     lambda: characterize(rig.board, None, channels=chans)),
                ):
                    try:
                        call()
                        raise AssertionError(f"{what} ran on one detector")
                    except MissingOutputs as e:
                        assert "PD1 only" in str(e) and "PD0, PD1, PD2, PD3" in str(e), str(e)
            with quiet():
                assert train_hw.main(argv) == 2  # the DPNN refuses too, before any round
        finally:
            interface.MOCK_SPDS = keep
    assert np.flatnonzero(np.isfinite(r.measured).all(1)).tolist() == [1], r.measured
    assert np.allclose(r.measured[1].sum(), 1.0), r.measured
    return out


if __name__ == "__main__":
    r = _selftest()
    for h, res in r["results"].items():
        print(res.digest(), "\n")
    sp = r["spd"]
    print(
        f"SPD mode, 4 mock SPDs: Vpi {sp['vpi']:.3f} V fitted, DPNN trained, "
        f"4x4 hosted to {sp['full'].matrix.value:.3f} full / {sp['tile'].matrix.value:.3f} tiled; "
        "1 SPD: tiling, characterization and DPNN refused by name\n"
    )
    print(
        "the target was orthogonal; the chip was the mock physical model, because "
        "the board answers no protocol this library speaks"
    )
