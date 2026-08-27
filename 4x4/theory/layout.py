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

N_HEATERS = 18       # UH1..UH18 on the packaged part
N_AUX = N_HEATERS - (2 * NMZI + NMODE - 1)  # 3 heaters the model does not use
REF_RAIL = 0         # the output rail with no trimmer: its phase is the global phase


@dataclass(frozen=True)
class Heater:
    h: int
    role: str
    index: int  # MZI number for theta/phi, rail number for out_phase

    @property
    def pad(self) -> str:
        """The packaged heater name, UH1..UH18. One-indexed, and assumed to run in this
        order -- confirm against the GDS before trusting it on hardware."""
        return f"UH{self.h + 1}"

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


HEATERS = tuple(sorted(
    [Heater(2 * k + o, r, k) for k in range(NMZI) for o, r in ((0, "theta"), (1, "phi"))]
    + [Heater(2 * NMZI + i, "out_phase", r)
       for i, r in enumerate(x for x in range(NMODE) if x != REF_RAIL)]
    + [Heater(2 * NMZI + NMODE - 1 + i, "aux", -1) for i in range(N_AUX)],
    key=lambda h: h.h,
))

assert len(HEATERS) == N_HEATERS
assert [h.h for h in HEATERS] == list(range(N_HEATERS))

THETA_IDX = np.array([h.h for h in HEATERS if h.role == "theta"], int)
PHI_IDX = np.array([h.h for h in HEATERS if h.role == "phi"], int)
ALPHA_IDX = np.array([h.h for h in HEATERS if h.role == "out_phase"], int)
ALPHA_RAIL = np.array([h.index for h in HEATERS if h.role == "out_phase"], int)
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
