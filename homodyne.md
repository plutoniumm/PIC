# Homodyne readout bring-up (PIC B)

Signed matrix-vector needs the **complex** output field, not just power. This is the
2-shot homodyne readout that recovers it: interfere each signal rail with a phase-coherent
LO (the splitting tree's two straight reference arms), step the LO phase, and solve for E.
The math lives in `theory/readout.py` (twin-only); the hardware counterpart is
`src/pic/homodyne.py` (`Homodyne.acquire`/`recover`) plus the three bring-up scripts below.

**This is scaffolding.** Everything here is `--mock`-verified only. The hardware runs
happen LATER, with a human present, at **+15 dBm max** (PD-damage ceiling). Ising needs
none of this — do Ising first (see `pic-b-theory-nextsteps`).

## The physics in one screen

Per signal rail `k`, monitor-tap fraction `t`, LO-combiner fraction `f`:

```
mon        = t |E|^2
homo(psi)  = (1-f)(1-t)|E|^2 + f|LO|^2 + 2 sqrt(f(1-f)(1-t)) |LO| Re(E e^{-i psi})
```

Two shots `psi ∈ {0, pi/2}` + the monitor + the LO-power tap → `E * e^{-i arg(LO)}` per
rail. The `arg(LO)` (one per arm, top/bottom) is the missing constant; without it the
estimate is a randomly-rotated field, **worse than not doing homodyne** (mock: rel err
1.0 vs ~0; real chip ~1.24–1.38).

| thing | value | where |
|---|---|---|
| LO source | splitting tree's two straight reference arms (rail 0 top, rail 7 bottom) | phase-coherent with signal |
| LO-power taps | raw PD **0** (top), PD **13** (bottom) | `PDMap.lo_tap_*` |
| LO-phase heaters | **H16** (top, DAC 75, Vpi 2.07) / **H23** (bottom, DAC 8, **uncharacterized**) | `LO_PHASE_NET` |
| signal→LO map | rails 0-2 → top LO, rails 3-5 → bottom LO | `LO_OF_RAIL` |
| open reference arms | zero H0/H1/H14/H15 (split MZIs balanced → bright LO) | `open_reference` |
| PD-role map | monitor `[2,4,6,7,9,11]`, homodyne `[1,3,5,8,10,12]` | `DEFAULT_PDMAP` (schematic, **unverified**) |

## Bring-up order

Run these in order. Each gates the next; do not skip Step 0.

| step | what | script | needs |
|---|---|---|---|
| **0** | program one known config, confirm the chip's PD pattern matches the twin's prediction | *(Ising track — see below)* | nothing |
| **1** | verify the PD-role map (which raw PD is monitor / homodyne / LO-tap) | `scripts/homodyne_pdmap.py` | Step 0 |
| **2** | characterize H23 (bottom LO phase) via a homodyne fringe | `scripts/char_h23_homodyne.py` | Step 1 |
| **3** | measure the two absolute LO-phase constants (**the crux**) | `scripts/homodyne_calibrate.py` | Steps 1–2 |
| **4** | recover fields on known inputs, target ~0.1–0.2 rel err | `Homodyne.recover` (in your run script) | Steps 1–3 |

### Step 0 — the trust gate (belongs to the Ising track)

The twin has **never** been checked against a real PIC-B read. Before any homodyne number
is trusted: program one known heater config (e.g. via `theory.program`), read the 14 PDs,
and confirm the pattern matches `twin.forward`'s prediction. This is the same gate the
Ising demo needs (`pic-b-theory-nextsteps` memory) and is the cheapest first hardware
demo — do it there. Homodyne Steps 1–4 assume it passed.

### Step 1 — verify the PD-role map

The monitor/homodyne/LO-tap assignment is schematic-derived (`layout.SIG_ROUTE` + PD0/PD13)
and unverified on board B. Two model-free discriminators fix it: homodyne PDs **oscillate**
under an LO-phase sweep (H16 / H23) while monitors and taps don't; LO taps respond to LO
**amplitude** (open/close a reference arm) but stay phase-flat. The remaining 6 are
monitors. Intra-group rail order (which top PD is rail 0 vs 1 vs 2) is not resolvable this
way — it stays schematic; confirm against the GDS PD-feeder trace by eye.

```
python scripts/homodyne_pdmap.py --dbm 15 --out runs/pdmap.json
```
→ `runs/pdmap.json` (a `PDMap`) + `_evidence.json` (raw per-PD Δ). Bottom arm is skipped
until H23 is characterized. Feed `--pdmap runs/pdmap.json` to Steps 2–3 if it differs from
the schematic default.

### Step 2 — characterize H23 (the bottom LO phase)

H23 is drivable but `Vpi=null`: its DC transmission is a monotonic ~37 mV ramp with no
turnover, so `fit_fringe` can't place it. But as the bottom reference-arm phase it shows a
clean **V² fringe on a bottom homodyne PD** (raw 8/10/12) — that's what we fit. Light all
rails, open both arms, sweep DAC 8, fit the strongest bottom-homodyne fringe.

```
python scripts/char_h23_homodyne.py --dbm 15 --levels 0:4.0:0.25 --write
```
→ raw sweep in `runs/char_h23_<ts>.csv`; `--write` merges Vpi/V0/Vnull into net 23 of
`pic_data/pic_b_config.json`, which makes the **bottom arm commandable** for Steps 3–4.
(Without `--write` it only prints the fit.)

### Step 3 — the two LO-phase constants (THE CRUX)

Sweep each LO phase heater over 2π on a bright known input, read that arm's homodyne PDs,
fit each fringe `homo(psi)=C+K cos(psi−delta)`; with the twin's predicted `arg(E)` for the
known input, `arg(LO)=arg(E)−delta`. This is model-limited by the twin's accuracy on
hardware — Step 0 must have passed, and use a calibration input that lights all rails well
(weak rails give noisy `delta`).

```
python scripts/char_h23_homodyne.py --dbm 15 --write        # first, if not done
python scripts/homodyne_calibrate.py --dbm 15 --out runs/lo_phase.json
```
→ `runs/lo_phase.json`: the 6-vector `lo_phase` that `recover(derotate=True)` consumes.
Both arms need commanding — top always works; bottom needs Step 2's `--write`.

### Step 4 — recover fields

In your hardware run script (build on `template.open_devices` + `template.laser_session`
for the watchdog/keepalive safety):

```python
import json, numpy as np
from src.pic.homodyne import Homodyne, PDMap
from theory.hw import Hardware
from theory.twin import Twin

pdmap = PDMap.load("runs/pdmap.json")
lo_phase = json.load(open("runs/lo_phase.json"))["lo_phase"]
hw, twin = Hardware(), Twin()
h = Homodyne(pic, hw, pdmap, t=twin.tap, f=twin.lo_frac,
             lo_phase=lo_phase, on_tick=ls.keepalive)   # ls from laser_session
E = h.recover(known_input_phases, derotate=True)         # complex field per rail
```

`acquire`/`recover` mirror `theory/readout.py` exactly, reading real PDs instead of the
twin. Target ~0.1–0.2 rel err against the twin's predicted field on known inputs.

## Files

| file | role |
|---|---|
| `src/pic/homodyne.py` | `Homodyne` (acquire/recover), `PDMap`, `open_reference`, `twin_pd_forward`/`build_mock` (the `--mock` stack), `dac_to_phase` |
| `scripts/homodyne_pdmap.py` | Step 1 — verify PD roles → `PDMap` JSON |
| `scripts/char_h23_homodyne.py` | Step 2 — H23 via homodyne fringe → config net 23 |
| `scripts/homodyne_calibrate.py` | Step 3 — the two LO-phase constants → `lo_phase` JSON |
| `theory/readout.py` | the twin-only reference implementation of the same math |

## Notes & caveats (mock-verified, no hardware yet)

- **PD-role map is a parameter, not a constant.** Everything takes `--pdmap`; the schematic
  default is the fallback. Verify it (Step 1) before trusting recover.
- **Phase↔volt** is the repo law `phi = pi (v/Vpi)^2 + phi0` via `theory.hw.Hardware`;
  `dac_of` scatters the 120-net vector onto the 128 firmware DACs. Uncommandable nets stay
  at 0 V (their built-in offset phase), which recover honestly cannot correct.
- **The bottom arm is blocked until Step 2.** `Homodyne.acquire(los=("bottom",))` raises a
  clear error if H23 is still uncharacterized — that is the current gap, not a bug.
- **Mock faithfulness.** `build_mock` injects a hidden per-arm LO phase (`MOCK_LO_BIAS`) the
  recover side doesn't know — a stand-in for the real chip's unmodeled reference-arm
  propagation. So the mock genuinely reproduces "uncalibrated recover is worse than
  nothing", and the calibration recovers that hidden constant. It is not proof of any
  hardware result.
- **Safety.** Hardware paths go through `template.laser_session` (watchdog hard-off +
  keepalive + emission check). +15 dBm ceiling. Never open the PIC serial port while
  `ui.py` holds it.
