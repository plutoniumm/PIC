"""The mesh's phase inventory: what each of the 18 heaters does.

The delivered part (`mrunal/PIC1A_packaged_IO.pdf`) wires **UH1..UH18** to the 4x4 Unitary
block, and `mrunal/PIC_Design_Review.pptx` shows it as six MZIs across four input and four
output ports. MH1..MH3 belong to the separate single-MZI test block and are not part of this
mesh.

Eighteen heaters, but a 4x4 unitary has only 15 free phases:

    6 MZI x (theta, phi)  12
    output trimmers        3   on rails 1, 2, 3
    aux                    3   real heaters, role not yet assigned
                          --
                          18

Rail 0 gets no trimmer because the fourth output phase is a global phase, which no detector
can see and no heater needs to make. So three of the eighteen are redundant degrees of
freedom by construction -- a checkable claim, and the first thing a full characterization
should confirm.

PROVISIONAL: which UH number is which role. The count is from the packaging document; the
assignment is a reading of the standard rectangular layout and needs the GDS or a per-heater
sweep to confirm. Edit `HEATERS` and nothing else; every consumer (twin, calibration,
programming, surrogate features) reads the roles from here.

This is the chip, not the board. Which DAC channel reaches which heater is `pic.layout`.

Roles
  theta      internal (arm-difference) phase of MZI k, `index` = k in `clements.MESH`
  phi        external phase on MZI k's upper input arm
  out_phase  output trimmer on rail `index` (never rail 0)
  aux        wired and real, but not yet assigned a role; held at 0 V and outside the model
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clements import COLUMN, MESH, NMODE, NMZI

# 16, not 18: the packaged part exposes 18 heater pads but the DAC81416 has 16 channels,
# so H16 and H17 have no driver at all. Index here is the DAC channel, which is what the
# firmware addresses and what every measurement is keyed by.
N_HEATERS = 16
# DAC channel -> the heater it drives, decoded from the wiring notebook
# (Arduino/pin_check/pin_check.ino). Three of these were never resistance-checked and are
# held dark; see pic.config.VOLTAGE_MAX_CH.
DAC_HEATER = ("H15", "H12", "H11", "H8", "H4", "H3", "H18", "H9",
              "H14", "H13", "H10", "H6", "H7", "H5", "H2", "H1")
N_AUX = N_HEATERS - (2 * NMZI + NMODE - 1)  # 3 heaters the model does not use
REF_RAIL = 0         # the output rail with no trimmer: its phase is the global phase


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
        other, which is what lets a sweep drive a whole column at a time."""
        return COLUMN[self.index] if self.role in ("theta", "phi") else len(set(COLUMN))


# MEASURED, not assumed. `ananya/4X4MZI_Test2 (1).pdf` tables the six internal phase
# shifters by MZI with their pins and resistances; `ananya/4X4MZI_REPORT.pdf` and the
# wiring notebook table the same pins under the H1-H18 names. Matching them on (pins,
# resistance) identifies all six, and the report's optical paths
#     IN1->OUT1: MZI1->MZI3->MZI4        IN1->OUT2: MZI1->MZI3->MZI4->MZI6
#     IN4->OUT4: MZI2->MZI3->MZI5        IN4->OUT3: MZI2->MZI3->MZI5->MZI6
# place report MZI k at mesh index k-1, which is the only assignment consistent with all
# four (MZI6 cannot sit in column 0). Net result: DAC k drives theta of mesh index 5-k.
THETA_DAC = {5 - k: k for k in range(NMZI)}     # mesh index -> DAC channel

# MEASURED, partially. Scoring candidate assignments against 1600 measured transfer samples
# (2026-08-27, `pic_data/transfer_8dbm.csv`) says the placeholder below is not merely
# unverified but WRONG: it predicts the data at r = -0.04 held out, i.e. no better than
# chance, while the best assignment reaches r = +0.50. Two entries are solid -- they appear
# in every one of the top five candidates:
#
#     MZI2's phi  <- DAC12  (labelled alpha1 here; it produces a clean fringe at r2 0.998,
#                            which an output trimmer cannot, so it is not a trimmer)
#     MZI3's phi  <- DAC13  (labelled alpha2 here; same argument, r2 0.999)
#
# DAC7 and DAC10 are interchangeable between MZI1 and MZI5 to three decimal places, so the
# data does not determine them and neither does this file. MZI6's phi is unidentified.
# Wiring in a half-known table would replace one unjustified permutation with another, so
# the placeholder stands and the measurement is recorded here instead. Fix it properly by
# re-running the scoring with the loss term in the twin -- at r = 0.50 the model is still
# missing physics (16.8% configuration-dependent loss), and that limits how well any
# assignment can score.
#
# The external phases are NOT yet identified. Nothing in the reports assigns H7/H9/H10/H13
# and the rest to MZIs, and the bench has only ever driven the six internal shifters. These
# keep a provisional order so the model stays whole; treat any phi result as unverified.
_FREE = [c for c in range(N_HEATERS) if c not in THETA_DAC.values()]

HEATERS = tuple(sorted(
    [Heater(THETA_DAC[k], "theta", k) for k in range(NMZI)]
    + [Heater(_FREE[k], "phi", k) for k in range(NMZI)]
    + [Heater(_FREE[NMZI + i], "out_phase", r)
       for i, r in enumerate(x for x in range(NMODE) if x != REF_RAIL)]
    + [Heater(_FREE[NMZI + NMODE - 1 + i], "aux", -1) for i in range(N_AUX)],
    key=lambda h: h.h,
))

assert len(HEATERS) == N_HEATERS
assert [h.h for h in HEATERS] == list(range(N_HEATERS))

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
# the heaters the mesh model actually uses; aux are driven to 0 and left out
ACTIVE_IDX = np.array(sorted(set(range(N_HEATERS)) - set(AUX_IDX.tolist())), int)

assert THETA_IDX.size == PHI_IDX.size == NMZI and ALPHA_IDX.size == NMODE - 1
assert AUX_IDX.size == N_AUX


def pack(theta, phi, alpha=None) -> np.ndarray:
    """Mesh parameters -> the 18-long optical-phase vector, in heater order.

    `alpha` may be the full 4-rail screen (rail 0 must be the reference and is dropped) or
    just the 3 trimmer phases."""
    ph = np.zeros(N_HEATERS)
    ph[THETA_IDX] = np.asarray(theta, float).ravel()
    ph[PHI_IDX] = np.asarray(phi, float).ravel()
    if alpha is not None:
        a = np.asarray(alpha, float).ravel()
        ph[ALPHA_IDX] = a[ALPHA_RAIL] if a.size == NMODE else a
    return ph


def unpack(ph):
    """The 18-long phase vector -> (theta[6], phi[6], alpha[4]) with alpha[0] = 0."""
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
                f"carries no trimmer (global phase)")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
