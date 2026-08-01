"""Which DAC channel drives which physical heater.

The chip's 6x6P structure has **120** heaters (geometric numbering H0..H119, descending
GDS x then y -- see `pic_data/heater_map.csv` and `src.pic.layout`). The daughter board
exposes only **64** DAC channels. Which 64 heaters those channels land on is *not*
recorded anywhere in the shipped documentation:

  * `NUS/PIC_update.pdf` says "each heater is connected to a DAC" and names heaters by
    number ("PD4 responds to heaters 48 and 53"), but those numbers are DAC-channel
    indices in NUS's own scheme, not the GDS geometric numbering;
  * the daughter-board wiring lives in `NUS/1010.190.PcbDoc` (Altium binary), untraced.

So this module keeps the mapping as **data, not code**: an editable CSV with a per-row
`verified` flag. Until a row is verified, callers (and the UI) must treat the heater
identity as provisional -- the *voltage* is real, the *position on the chip* is a guess.

    from pic.wiring import load_map
    wm = load_map()
    wm.heater_of(7)     # -> geometric heater index driven by DAC 7, or None
    wm.dac_of(7)        # -> DAC channel driving heater H7, or None
    wm.verified         # False while any row is unverified
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from .config import NUM_DAC

MAP_CSV = "pic_data/dac_heater_map.csv"
NUM_HEATERS = 120


@dataclass
class WiringMap:
    """DAC channel <-> geometric heater index, with provenance."""

    dac_to_h: dict[int, int] = field(default_factory=dict)
    verified_rows: set[int] = field(default_factory=set)
    source: str = "identity (provisional)"

    @property
    def verified(self) -> bool:
        """True only when every mapped channel has been confirmed on hardware."""
        return bool(self.dac_to_h) and len(self.verified_rows) == len(self.dac_to_h)

    @property
    def h_to_dac(self) -> dict[int, int]:
        return {h: d for d, h in self.dac_to_h.items()}

    def heater_of(self, dac: int) -> int | None:
        return self.dac_to_h.get(int(dac))

    def dac_of(self, heater: int) -> int | None:
        return self.h_to_dac.get(int(heater))

    def driven_heaters(self) -> set[int]:
        return set(self.dac_to_h.values())

    def as_json(self) -> dict:
        return {
            "dac_to_h": {str(d): h for d, h in sorted(self.dac_to_h.items())},
            "verified": self.verified,
            "verified_rows": sorted(self.verified_rows),
            "source": self.source,
            "n_driven": len(self.dac_to_h),
            "n_heaters": NUM_HEATERS,
        }


def identity_map() -> WiringMap:
    """Fallback: DAC c drives geometric heater Hc, for c in 0..63.

    This is a PLACEHOLDER, not a measurement. It happens to cover the encode stage and
    most of the V mesh while driving none of Sigma or U, which is physically suspicious
    for a board meant to program both meshes. Never present it as ground truth.
    """
    return WiringMap(dac_to_h={c: c for c in range(NUM_DAC)}, verified_rows=set(),
                     source="identity (provisional -- NOT measured)")


def load_map(path: str | None = None, root: str = ".") -> WiringMap:
    """Load the wiring CSV (`dac,H,verified`; `#` comments allowed). Falls back to the
    provisional identity map when the file is absent."""
    full = path or os.path.join(root, MAP_CSV)
    if not os.path.isfile(full):
        return identity_map()

    dac_to_h: dict[int, int] = {}
    verified: set[int] = set()
    with open(full) as f:
        rows = csv.DictReader(r for r in f if not r.lstrip().startswith("#"))
        for r in rows:
            h = (r.get("H") or "").strip()
            if not h or h.lower() in ("none", "null", "-"):
                continue  # channel deliberately unassigned
            d, hi = int(r["dac"]), int(h)
            if not 0 <= d < NUM_DAC:
                raise ValueError(f"{full}: dac {d} out of range 0..{NUM_DAC - 1}")
            if not 0 <= hi < NUM_HEATERS:
                raise ValueError(f"{full}: H {hi} out of range 0..{NUM_HEATERS - 1}")
            if d in dac_to_h:
                raise ValueError(f"{full}: DAC {d} listed twice")
            dac_to_h[d] = hi
            if str(r.get("verified", "0")).strip() in ("1", "true", "True", "yes"):
                verified.add(d)

    dupes = [h for h in set(dac_to_h.values())
             if list(dac_to_h.values()).count(h) > 1]
    if dupes:
        raise ValueError(f"{full}: heater(s) {sorted(dupes)} driven by >1 DAC")
    if not dac_to_h:
        return identity_map()
    return WiringMap(dac_to_h, verified, source=os.path.relpath(full, root))
