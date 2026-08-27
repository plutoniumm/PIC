"""Board wiring: which DAC channel reaches which heater.

The chip side of this -- what each heater does -- is `theory.layout`, and it is re-exported
here so callers have one import. What lives here is the board: 18 DAC channels, one per
heater, currently the identity map.

Only 6 of the 18 heaters are wired today: the bring-up board (`mrunal/Setup.ino`) drives
one DAC chip on CS 10 with `NUM_DAC_CH = 6`. That map is documented, not guessed --
`mrunal/Setup_Analysis.pdf` Table 1 and `mrunal/Drift_Report.pdf` Table 1 agree on it --
and it is the only part of this file that is not a hypothesis.

The remaining 12 channels are the identity map, which is a placeholder. On the 6x6 the
DAC-to-heater map was decoded from a schematic, never confirmed electrically, and two of its
rows contradicted the measured data for months; treat anything outside `WIRED` the same way
until a per-channel sweep has confirmed it, then set `VERIFIED = True`.

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

# DAC channel -> heater pad UHn, keyed on the +ve pad the bench reports give rather than on
# the heater name beside it; see the module docstring for the one place those disagree. The
# other 12 channels are not connected yet and hold the identity map as a placeholder.
WIRED_MAP = {0: "UH15", 1: "UH12", 2: "UH11", 3: "UH8", 4: "UH4", 5: "UH3"}

def _heater_of_dac():
    """A permutation, not a patched identity: the six documented channels take their
    heaters, and the rest fill in with whatever is left, in order. Overwriting entries of
    an identity map instead would leave five heaters on two channels each and five reachable
    from none, which is the kind of map that produces a plausible-looking dead channel."""
    fixed = {d: int(pad[2:]) - 1 for d, pad in WIRED_MAP.items()}
    rest = [h for h in range(N_HEATERS) if h not in set(fixed.values())]
    out, it = np.empty(N_HEATERS, int), iter(rest)
    for d in range(N_HEATERS):
        out[d] = fixed[d] if d in fixed else next(it)
    return out


HEATER_OF_DAC = _heater_of_dac()
DAC_OF_HEATER = np.argsort(HEATER_OF_DAC)
assert sorted(HEATER_OF_DAC.tolist()) == list(range(N_HEATERS))
VERIFIED = False  # true only for the six channels in WIRED_MAP

ACTIVE_DACS = np.sort(DAC_OF_HEATER[ACTIVE_IDX])
AUX_DACS = np.sort(DAC_OF_HEATER[AUX_IDX])
WIRED = np.array(sorted(WIRED_DACS), int)
# what a sweep can actually drive today: wired channels that the mesh model uses
REACHABLE_DACS = np.array(sorted(set(WIRED.tolist()) & set(ACTIVE_DACS.tolist())), int)
LABEL_OF_DAC = {d: HEATERS[int(HEATER_OF_DAC[d])].label for d in range(N_HEATERS)}


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
        rows.append(f"{d:>3}  {h.pad:<6} {h.role:<10} {'yes' if d in WIRED_MAP else 'no'}")
    return "\n".join(rows)
