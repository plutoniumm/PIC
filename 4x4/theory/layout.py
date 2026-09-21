"""The mesh's phase inventory: what each of the 16 driven heaters does.

The delivered part (`mrunal/PIC1A_packaged_IO.pdf`) wires **UH1..UH18** to the 4x4 Unitary
block. MH1..MH3 belong to the separate single-MZI test block and are not part of this mesh.

Eighteen heater pads, sixteen DAC channels, and a 4x4 unitary has only 15 free phases:

    6 MZI x (theta, phi)   12
    output-screen phases    4   one per rail, of which only 3 are independent (a 4th
                                would be a global phase, invisible to any detector)
                          ---
                          15   the whole mesh

The packaged part has 18 pads, three more than the mesh needs, and every one of the three is
accounted for: the fourth output trimmer is the global phase above, and H6 and H9 sit in the
pass-through gap column 1 leaves free, each exactly degenerate with phases the mesh already
commands (REDUNDANT below). The DAC81416 that drives the part has 16 channels, two short of
18, and both unwired pads land on the output screen -- H16 and H17 -- so the true output
screen is driven at only 2 of its 4 rails, one short of the 3 a full unitary needs.
Identifying which pads those were took a labelled diagram, not a guess (see CONFIRMED
below): two of the four wearing "output trimmer" labels turned out to be mid-mesh phi
phases, not trimmers.

CONFIRMED: DAC-to-heater-to-role, from `ananya/4X4MZI_REPORT.pdf` p.2 Figure 2, a labelled
drawing of all 18 heaters against the mesh topology, cross-checked against that report's own
resistance table (its Table 2, 13 heaters by pin) and the one in `ananya/4X4MZI_Test2
(1).pdf` (the six internals, by MZI), the vendor pinout
(`mrunal/PIC1A_packaged_IO.pdf`) and the wiring notebook (`Arduino/pin_check/pin_check.ino`).
All four agree on DAC-to-pad; the role each pad plays is read off the figure. This replaces
the PROVISIONAL placeholder that stood here before (theta was already measured separately,
see THETA_DAC; the placeholder covered only phi/out_phase/aux, and predicted the bench data
at r = -0.04 held out -- worse than chance. The corrected map predicts it instead: DAC 12 and
13 fit clean fringes (r2 0.998, 0.999) because they are mid-mesh phi, not trimmers, exactly
as a trimmer cannot; DAC 8 and 6, the real output pads, fit weakly or not at all, exactly as
an invisible diagonal phase must; DAC 15 was never swept, correctly, being MZI 1's own input
phase.

SETTLED: DAC 7 (H9) and DAC 11 (H6) are placed, and both are redundant rather than
unidentified. Figure 2 draws each on a straight pass-through rail in the column-1 gap, which
exists because column 1's only MZI takes rails 1 and 2:

  DAC 11 (H6) is on rail 0, the very segment DAC 12 (H7) already phases -- nothing
  interrupts rail 0 between column 0 and column 2 -- so the two phases add:
  T(0,1,theta,phi) . diag(e^{i d},1,1,1) = T(0,1,theta,phi+d), exactly. DAC 11 is a second
  driver for MZI index 3's phi and can be nothing else.

  DAC 7 (H9) is on rail 3, the *lower* input of MZI index 4, whose phi (DAC 10) is on
  rail 2. No heater shares that segment and a lower-arm phase is not the Clements external
  phase, so it is degenerate with no single channel -- but it is degenerate with three: a
  phase psi there is exactly phi4, phi5 and alpha0 each shifted by -psi, plus a global
  phase. In intensity, where alpha is invisible, it is phi4 and phi5 shifted together, which
  is why DAC 7 fits a real fringe (r2 0.979, Vpi 3.61 V,
  `pic_data/sessions/2026-08-27/char_results_40mA.json`) where an output trimmer cannot.

Neither adds a reachable state. Differentiating the mesh in all 18 physical phases gives
rank 15 = dim PU(4); the 16 driven channels give 14; dropping DAC 7 and DAC 11 leaves 14.
`ACTIVE_IDX` is not a truncation of the 16 driven channels, it is all of them, and both
identities and both ranks are checked in `_selftest`.

The vendor pinout agrees independently, through its common-ground clusters, which group
heaters by where they sit on the die: H9 grounds to BP15 with H7 and H8, the column-1/2
cluster, and H6 to TP11, alone, between column 0's TP8/TP9 and DAC 10's TP13. Every
output-screen heater instead grounds to TP26, and neither of these does. That is the part
worth being sure of, because a spare heater on rail 3 would have made the mesh fully
specified. Resistance agrees too: H9 at 57.2 ohm falls in the 60-ohm group with H10 (56.4)
and H12 (62.3), the other two heaters of MZI index 4, and nowhere near the 114-119 ohm the
rest of the mesh runs at.

Neither is worth driving. DAC 11 has never been swept, its resistance is unconfirmed and it
is staged at 1.5 V (`pic.config.HEATER_OHMS`), and measuring it would buy a worse copy of
DAC 12, which is already fitted at 4.60 V and 1.36 pi of span. DAC 7 is the more interesting
of the two only as a range extender: phi4's own driver DAC 10 is a 56-ohm heater capped at
2.25 V with Vpi 5.35 V, which spans 0.18 pi, while DAC 7 spans 0.39 pi on the same phi4 with
phi5 riding along, and phi5 has 1.51 pi of range to give it back. Both stay at `aux`, out of
the model, until something needs that.

Edit `HEATERS` and nothing else; every consumer (twin, calibration, programming, surrogate
features) reads the roles from here. This is the chip, not the board; which DAC channel
reaches which heater is `pic.layout`.

Roles
  theta      internal (arm-difference) phase of MZI k, `index` = k in `clements.MESH`
  phi        external phase on MZI k's upper input arm
  out_phase  output-screen phase on rail `index`, of the 2 that are driven
  aux        wired, real, and exactly reproduced by phases the model already commands
             (REDUNDANT); held at 0 V and outside the model

Two rails of the output screen carry no driven trimmer: `REF_RAIL`, the gauge choice (any
rail's absolute output phase is invisible, so fixing one at zero costs nothing), and
`UNREACHABLE_RAIL`, which is not a gauge choice -- no heater reaches it, so whatever relative
phase that rail actually carries cannot be commanded and is not represented by this model at
all. Checked rather than assumed, because the two spare channels were the obvious candidates
to close it: handing rail 3's own pad (H16, which has no DAC) to the 16 driven channels
takes the mesh Jacobian from rank 14 to 15, so that direction really is outside their span,
and neither spare supplies it -- DAC 7 moves alpha0, DAC 11 moves no alpha at all. The mesh
therefore realises a full 4x4 unitary only up to an uncharacterised, fixed phase on
`UNREACHABLE_RAIL`; every consumer of `alpha` from `unpack` sees zero there, which is a
stand-in, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clements import COLUMN, MESH, NMODE, NMZI, T

# 16, not 18: the packaged part exposes 18 heater pads but the DAC81416 has 16 channels,
# so H16 and H17 have no driver at all. Index here is the DAC channel, which is what the
# firmware addresses and what every measurement is keyed by.
N_HEATERS = 16
# DAC channel -> the heater it drives, decoded from the wiring notebook
# (Arduino/pin_check/pin_check.ino). Three of these were never resistance-checked and are
# held dark; see pic.config.VOLTAGE_MAX_CH.
DAC_HEATER = ("H15", "H12", "H11", "H8", "H4", "H3", "H18", "H9",
              "H14", "H13", "H10", "H6", "H7", "H5", "H2", "H1")
REF_RAIL = 1         # output rail with no driver, used as the phase reference (gauge choice)
UNREACHABLE_RAIL = 3  # output rail with no driver, NOT the reference: phase unmeasured
AUX_GAP_COLUMN = 1   # column whose MZI leaves two rails free; both spare heaters sit there


@dataclass(frozen=True)
class Heater:
    h: int
    role: str
    index: int  # MZI number for theta/phi, rail number for out_phase

    @property
    def pad(self) -> str:
        """The heater this DAC channel actually drives, from the measured wiring map.

        This used to assume UH(n+1) ran in DAC order. It does not -- DAC0 drives H15 and
        DAC15 drives H1, very nearly reversed -- which is why the old role labels put
        strong modulators on output trimmers."""
        return DAC_HEATER[self.h]

    @property
    def label(self) -> str:
        if self.role == "aux":
            return f"{self.pad}:aux"
        kind = f"alpha{self.index}" if self.role == "out_phase" else f"{self.role}{self.index}"
        return f"{self.pad}:{kind}"

    @property
    def column(self) -> int:
        """Rectangular column this heater sits in; the output screen counts as one past the
        last mesh column. Heaters in different columns of one pass cannot confound each
        other, which is what lets a sweep drive a whole column at a time.

        `aux` is column 1, not the screen: both spares sit in the gap column 1 leaves on the
        rails its MZI does not take. That is what `characterize.transparent_base` needs to
        know, since it opens every theta upstream of this column and leaves the rest dark."""
        if self.role in ("theta", "phi"):
            return COLUMN[self.index]
        return AUX_GAP_COLUMN if self.role == "aux" else len(set(COLUMN))


# MEASURED, not assumed. `ananya/4X4MZI_Test2 (1).pdf` tables the six internal phase
# shifters by MZI with their pins and resistances; `ananya/4X4MZI_REPORT.pdf` and the
# wiring notebook table the same pins under the H1-H18 names. Matching them on (pins,
# resistance) identifies all six, and the report's optical paths
#     IN1->OUT1: MZI1->MZI3->MZI4        IN1->OUT2: MZI1->MZI3->MZI4->MZI6
#     IN4->OUT4: MZI2->MZI3->MZI5        IN4->OUT3: MZI2->MZI3->MZI5->MZI6
# place report MZI k at mesh index k-1, which is the only assignment consistent with all
# four (MZI6 cannot sit in column 0). Net result: DAC k drives theta of mesh index 5-k.
THETA_DAC = {5 - k: k for k in range(NMZI)}     # mesh index -> DAC channel

# CONFIRMED from ananya/4X4MZI_REPORT.pdf p.2 Figure 2 (see module docstring): mesh index
# -> DAC channel driving that MZI's external (upper-arm) phase. MZI 5 (index 4, DAC 10) was
# already correct in the old placeholder; the rest move.
PHI_DAC = {0: 15, 1: 14, 2: 13, 3: 12, 4: 10, 5: 9}

# CONFIRMED: the two output-screen pads with a driver, by rail. Rails 1 (REF_RAIL) and 3
# (UNREACHABLE_RAIL) have no driver; see the module docstring for what that costs.
ALPHA_DAC = {0: 8, 2: 6}

# SETTLED (see module docstring): the two channels left over once theta, phi and the two
# driven out_phase channels are assigned are not unidentified, they are exactly redundant.
# DAC channel -> (the rail it sits on in column 1's gap, {DAC channel of a modelled phase:
# how much of it a phase psi here is worth}). Written against PHI_DAC/ALPHA_DAC rather than
# as bare channel numbers so it follows the map if the map ever moves.
REDUNDANT = {
    11: (0, {PHI_DAC[3]: +1.0}),
    7: (3, {PHI_DAC[4]: -1.0, PHI_DAC[5]: -1.0, ALPHA_DAC[0]: -1.0}),
}
AUX_DACS = tuple(sorted(REDUNDANT))
AUX_RAIL = {d: r for d, (r, _) in REDUNDANT.items()}

HEATERS = tuple(sorted(
    [Heater(THETA_DAC[k], "theta", k) for k in range(NMZI)]
    + [Heater(PHI_DAC[k], "phi", k) for k in range(NMZI)]
    + [Heater(ALPHA_DAC[r], "out_phase", r) for r in ALPHA_DAC]
    + [Heater(d, "aux", -1) for d in AUX_DACS],
    key=lambda h: h.h,
))

assert len(HEATERS) == N_HEATERS
assert [h.h for h in HEATERS] == list(range(N_HEATERS))
assert set(THETA_DAC.values()) | set(PHI_DAC.values()) | set(ALPHA_DAC.values()) \
    | set(AUX_DACS) == set(range(N_HEATERS))

# Per-heater column and mesh index, in DAC order. `fit_staged` needs them to fit phases
# column by column: light crosses the columns in order, so column 0's phases are determined
# by data column 3 cannot touch, and fitting them together is a worse optimisation than
# fitting them in sequence.
COLUMN_OF_HEATER = np.array([h.column for h in HEATERS], int)
MESH_IDX = np.array([h.index for h in HEATERS], int)

def _by_index(role) -> np.ndarray:
    """Heater channels for `role`, ordered by the mesh element each one drives.

    Every consumer reads these positionally -- `twin.matrix` hands element k to `MESH[k]` --
    so ordering them by DAC number instead silently transposes the map. That is exactly what
    happened here: DAC k drives theta of mesh index 5-k, `HEATERS` records it, and sorting by
    `h.h` fed DAC k to MZI k, mirroring every theta in the mesh. On 100 measured four-port
    states the twin scored pearson -0.20 against the chip; ordering by `h.index` and refitting
    phi0 takes it to +0.94."""
    return np.array([h.h for h in sorted((x for x in HEATERS if x.role == role),
                                         key=lambda x: x.index)], int)


THETA_IDX = _by_index("theta")
PHI_IDX = _by_index("phi")
ALPHA_IDX = _by_index("out_phase")
ALPHA_RAIL = np.array(sorted(h.index for h in HEATERS if h.role == "out_phase"), int)
AUX_IDX = np.array([h.h for h in HEATERS if h.role == "aux"], int)
N_AUX = AUX_IDX.size
# the heaters the mesh model actually uses; aux are driven to 0 and left out
ACTIVE_IDX = np.array(sorted(set(range(N_HEATERS)) - set(AUX_IDX.tolist())), int)

assert THETA_IDX.size == PHI_IDX.size == NMZI
assert ALPHA_IDX.size == len(ALPHA_DAC)
assert REF_RAIL not in ALPHA_RAIL and UNREACHABLE_RAIL not in ALPHA_RAIL
assert AUX_IDX.size == len(AUX_DACS)


def pack(theta, phi, alpha=None) -> np.ndarray:
    """Mesh parameters -> the 16-long optical-phase vector, in DAC-channel order.

    `alpha` may be the full 4-rail screen, in any gauge (it is re-referenced onto `REF_RAIL`
    before use), or just the driven rails, in `ALPHA_RAIL` order. Either way the entry for
    `UNREACHABLE_RAIL` is dropped silently: no heater can realise it."""
    ph = np.zeros(N_HEATERS)
    ph[THETA_IDX] = np.asarray(theta, float).ravel()
    ph[PHI_IDX] = np.asarray(phi, float).ravel()
    if alpha is not None:
        a = np.asarray(alpha, float).ravel()
        if a.size == NMODE:
            # `a` may come from `clements.decompose`, whose gauge fixes a[0] = 0 -- a choice
            # tied to array position, not to which physical rail happens to have a driver.
            # Re-gauge onto REF_RAIL (a global additive shift; the whole diagonal is only
            # ever meaningful up to one) before dropping the two undriven rails, so the
            # rail we treat as "the reference" is the one with no driver instead of
            # whichever index decompose happened to zero.
            a = a - a[REF_RAIL]
            ph[ALPHA_IDX] = a[ALPHA_RAIL]
        else:
            ph[ALPHA_IDX] = a
    return ph


def unpack(ph):
    """The 16-long phase vector -> (theta[6], phi[6], alpha[4]).

    `alpha[REF_RAIL]` is the gauge zero; `alpha[UNREACHABLE_RAIL]` is not measured and comes
    back zero as a stand-in, not because the chip's phase there is actually zero."""
    ph = np.asarray(ph, float).ravel()
    if ph.size != N_HEATERS:
        raise ValueError(f"expected {N_HEATERS} phases, got {ph.size}")
    alpha = np.zeros(NMODE)
    alpha[ALPHA_RAIL] = ph[ALPHA_IDX]
    return ph[THETA_IDX], ph[PHI_IDX], alpha


def sweep_columns():
    """Groups of heaters that can be swept in the same pass without confounding each other.

    On the 6x6 this took an interference-graph colouring built from measured influence; at
    4x4 the mesh is shallow enough that the column index is the colouring."""
    groups = {}
    for h in HEATERS:
        if h.role != "aux":
            groups.setdefault(h.column, []).append(h.h)
    return [np.array(sorted(v), int) for _, v in sorted(groups.items())]


def describe() -> str:
    rows = [f"{'h':>3}  {'pad':<5} {'role':<10} {'index':>5}  {'col':>3}  label"]
    for h in HEATERS:
        rows.append(f"{h.h:>3}  {h.pad:<5} {h.role:<10} {h.index:>5}  {h.column:>3}  {h.label}")
    rows.append("")
    rows.append(f"mesh {MESH}  columns {COLUMN}")
    rows.append(f"{N_HEATERS} heaters, {ACTIVE_IDX.size} in the model; output rail {REF_RAIL} "
                f"is the phase reference, rail {UNREACHABLE_RAIL} has no driver at all")
    return "\n".join(rows)


def _physical(theta, phi, alpha, gap=()):
    """The mesh exactly as `ananya/4X4MZI_REPORT.pdf` p.2 Fig. 2 draws it.

    `gap` is a sequence of (rail, phase) for the pass-through heaters column 1 leaves room
    for -- H6 on rail 0, H9 on rail 3. Those two rails are untouched by column 1, so the
    insertion point anywhere in the gap is the same matrix; this puts them just before
    column 2. With both supplied this is all 18 physical phases of the part."""
    U = np.eye(NMODE, dtype=complex)
    for k, (m, n) in enumerate(MESH):
        if COLUMN[k] == 2 and COLUMN[k - 1] != 2:
            for rail, p in gap:
                d = np.ones(NMODE, complex)
                d[rail] = np.exp(1j * p)
                U = np.diag(d) @ U
        U = T(m, n, theta[k], phi[k]) @ U
    return np.diag(np.exp(1j * np.asarray(alpha, float))) @ U


def _selftest(n: int = 200, seed: int = 0):
    """Prove REDUNDANT and `UNREACHABLE_RAIL` rather than asserting them.

    Two claims, both from the module docstring. First, each spare channel is exactly the
    fixed combination of modelled phases `REDUNDANT` records -- checked as matrices, up to
    the global phase no detector sees. Second, the mesh Jacobian in all 18 physical phases
    has rank 15 = dim PU(4), the 16 driven channels reach 14 of it, the 14 in the model
    reach the same 14 (so the two spares cost nothing), and it is rail 3's own undriven pad
    that supplies the missing direction."""
    rng = np.random.default_rng(seed)
    phi_of_dac = {v: k for k, v in PHI_DAC.items()}
    alpha_of_dac = {v: r for r, v in ALPHA_DAC.items()}

    worst = 0.0
    for _ in range(n):
        th = rng.uniform(0, np.pi, NMZI)
        ph = rng.uniform(-np.pi, np.pi, NMZI)
        al = rng.uniform(-np.pi, np.pi, NMODE)
        psi = rng.uniform(-np.pi, np.pi)
        for rail, coeffs in REDUNDANT.values():
            ph2, al2 = ph.copy(), al.copy()
            for d, c in coeffs.items():
                if d in phi_of_dac:
                    ph2[phi_of_dac[d]] += c * psi
                else:
                    al2[alpha_of_dac[d]] += c * psi
            A = _physical(th, ph, al, gap=[(rail, psi)])
            B = _physical(th, ph2, al2)
            g = np.angle(np.vdot(B, A))   # the leftover global phase, which no detector sees
            worst = max(worst, float(np.abs(A - B * np.exp(1j * g)).max()))
    assert worst < 1e-10, f"REDUNDANT is wrong: residual {worst:.2e}"

    # 18 physical phases in one vector: theta, phi, the 4-rail screen, then the two spares.
    n_spare = len(REDUNDANT)
    rails = [r for r, _ in REDUNDANT.values()]

    def U_of(p):
        return _physical(p[:NMZI], p[NMZI:2 * NMZI], p[2 * NMZI:2 * NMZI + NMODE],
                         gap=list(zip(rails, p[-n_spare:])))

    p0, eps = rng.uniform(-1, 1, 2 * NMZI + NMODE + n_spare), 1e-6
    U0, cols = U_of(p0), []
    for i in range(p0.size):
        a, b = p0.copy(), p0.copy()
        a[i] += eps
        b[i] -= eps
        A = U0.conj().T @ ((U_of(a) - U_of(b)) / (2 * eps))
        A -= np.trace(A) / NMODE * np.eye(NMODE)   # quotient out the invisible global phase
        cols.append(np.concatenate([A.real.ravel(), A.imag.ravel()]))
    J = np.array(cols).T

    def rank(*drop):
        keep = [i for i in range(p0.size) if i not in drop]
        s = np.linalg.svd(J[:, keep], compute_uv=False)
        return int((s > 1e-7 * s.max()).sum())

    ref, unr = 2 * NMZI + REF_RAIL, 2 * NMZI + UNREACHABLE_RAIL
    spare = tuple(range(p0.size - n_spare, p0.size))
    ranks = dict(physical=rank(), driven=rank(ref, unr), model=rank(ref, unr, *spare),
                 model_plus_rail3=rank(ref, *spare))
    assert ranks["physical"] == 15, ranks           # the part could realise any 4x4 unitary
    assert ranks["driven"] == ranks["model"] == 14, ranks   # the two spares cost nothing
    assert ranks["model_plus_rail3"] == 15, ranks   # and rail 3's pad is what would close it
    return worst, ranks


if __name__ == "__main__":
    print(describe())
    w, r = _selftest()
    print(f"\nredundancy residual {w:.1e}; Jacobian rank {r['physical']} over 18 physical "
          f"phases, {r['driven']} over the 16 driven, {r['model']} over the {ACTIVE_IDX.size} "
          f"modelled, {r['model_plus_rail3']} once rail {UNREACHABLE_RAIL} has a driver")
