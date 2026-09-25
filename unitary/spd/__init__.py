"""Single-photon detector readout: the Vega TDC board and its VAUL vault stream."""

from .array import SPD, SPDs, discover
from .vega import BAUD, SPDError, Vault, VaultParser, Vega, autodetect, frame, resolve

__all__ = [
    "BAUD",
    "SPD",
    "SPDError",
    "SPDs",
    "Vault",
    "VaultParser",
    "Vega",
    "autodetect",
    "discover",
    "frame",
    "resolve",
]
