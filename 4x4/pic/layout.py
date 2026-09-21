"""Board wiring: which DAC channel reaches which heater.

The chip side of this -- what each heater does -- is `theory.layout`, and it is re-exported
here so callers have one import. What lives here is the board: 18 DAC channels, one per
heater, currently the identity map.

All 16 DAC channels are wired today, on the DAC81416 (`pic.config.NUM_DAC = 16`); this
superseded the original 6-channel bring-up board (`mrunal/Setup.ino`, one DAC chip on CS 10,
`NUM_DAC_CH = 6`), and `WIRED_DACS` in `pic.config` reflects the current board, not that
one. Only H16 and H17, of the 18 pads, have no driver at all.

Two loose ends, both about pads rather than channels.

UH15: the packaged-IO document lists it on TP26/TP24, while both bench reports list it on
BP24/TP26. The +ve pin disagrees; BP24 is not a pad the vendor table assigns to anything, so
the reports have almost certainly typed B for T.

DAC4: the bench reports name the heater UH7 but give its +ve pad as BP9, and in the vendor
table BP9 is **UH4** -- UH7 is BP15/BP12, a different pair entirely. The prose of
`Setup_Analysis.pdf` 1.5 lists the active set as "UH3, UH4, UH8, UH11, UH12, UH15", agreeing
with the pad. Two independent readings against one, so the pad wins and the entry below says
UH4. The ground column of those tables is unreliable in the same way -- it says BP1 for
heaters whose vendor ground is BP7, BP15 or BP20 -- which is harmless, because all five are
the bottom common-ground rail. The one case where that matters is UH3, whose vendor ground
is BP1 (bottom) but which the reports wire to TP26 (top): if those rails are not shorted in
the package, DAC5 drives nothing. DAC5 is the weakest-modulating channel in the measured
data, which is at least consistent with that.
"""

from __future__ import annotations

import numpy as np

from theory.layout import (  # noqa: F401  (re-exported: this is the one layout import)
    ACTIVE_IDX, ALPHA_IDX, AUX_IDX, HEATERS, N_HEATERS, PHI_IDX, THETA_IDX,
    Heater, describe, pack, sweep_columns, unpack,
)

from .config import WIRED_DACS

# The permutation that used to live here is gone, and deliberately. It existed because
# `theory.layout` indexed by heater pad on the guess that UH(n+1) ran in DAC order, so a
# shim was needed to get from a DAC channel to a heater. `theory.layout` is now indexed by
# DAC channel directly, from the measured wiring map, and the shim had become a second,
# contradictory opinion: it printed DAC0 as H2:alpha3 where theory.layout says H15:theta5.
# Two layouts is one too many. This module now re-exports the measured one.
HEATER_OF_DAC = np.arange(N_HEATERS)
DAC_OF_HEATER = np.arange(N_HEATERS)
VERIFIED = True   # the six internal phase shifters are measured, not assumed

ACTIVE_DACS = np.sort(DAC_OF_HEATER[ACTIVE_IDX])
AUX_DACS = np.sort(DAC_OF_HEATER[AUX_IDX])
WIRED = np.array(sorted(WIRED_DACS), int)
# what a sweep can actually drive today: wired channels that the mesh model uses
REACHABLE_DACS = np.array(sorted(set(WIRED.tolist()) & set(ACTIVE_DACS.tolist())), int)
LABEL_OF_DAC = {d: HEATERS[d].label for d in range(N_HEATERS)}


def to_dac(v_heater) -> np.ndarray:
    """A vector indexed by heater -> the same values indexed by DAC channel."""
    v = np.zeros(N_HEATERS)
    v[DAC_OF_HEATER] = np.asarray(v_heater, float).ravel()
    return v


def to_heater(v_dac) -> np.ndarray:
    """A vector indexed by DAC channel -> the same values indexed by heater."""
    return np.asarray(v_dac, float).ravel()[DAC_OF_HEATER]


def wiring_table() -> str:
    rows = [f"{'dac':>3}  {'heater':<6} {'role':<10} wired"]
    for d in range(N_HEATERS):
        h = HEATERS[int(HEATER_OF_DAC[d])]
        rows.append(f"{d:>3}  {h.pad:<6} {h.role:<10} {'yes' if d in WIRED_DACS else 'no'}")
    return "\n".join(rows)
