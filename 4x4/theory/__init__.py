"""Pure-math layer for the 4x4 chip: the mesh algebra, the differentiable twin, the
heater law, and the target-to-voltage path. No hardware, no serial ports.

    from theory import Twin, Calibration, phases_for, decompose

`clements` and `calib` are numpy only; `twin` and `program.refine` need torch.
"""

from .calib import Calibration
from .clements import MESH, NMODE, NMZI, decompose, random_unitary, reconstruct
from .layout import ACTIVE_IDX, HEATERS, N_HEATERS, pack, sweep_columns, unpack
from .program import fidelity, phases_for, refine, unitary_for, volts_for

__all__ = [
    "Calibration",
    "MESH",
    "NMODE",
    "NMZI",
    "decompose",
    "reconstruct",
    "random_unitary",
    "HEATERS",
    "N_HEATERS",
    "ACTIVE_IDX",
    "pack",
    "unpack",
    "sweep_columns",
    "fidelity",
    "phases_for",
    "unitary_for",
    "volts_for",
    "refine",
    "Twin",
    "MeshError",
]


def __getattr__(name):
    """torch-backed pieces load on demand, so `import theory` stays torch-free."""
    if name in ("Twin", "MeshError"):
        from . import twin as _t

        return getattr(_t, name)
    raise AttributeError(name)
