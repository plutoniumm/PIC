# PIC — photonic matrix-multiplier control

Hardware-in-the-loop control/optimisation of a silicon-photonic **MZI-mesh PIC** that
optically computes a matrix–vector product **M = U·Σ·V** (SVD: input encoding → 6×6 mesh
`U` → absorbers `Σ` → 6×6 mesh `V` → photodiodes). Heaters set phase; **DAC voltages in
(64 ch, 0–4 V grid), photodiode voltages out (14 raw, 10 usable)**. End goal: program the
splitting-tree 6×6 for optical matmul. Treat the chip as a black box: volts in, PD volts
out. Chip/vendor background in `NUS/` and `CLAUDE.md`.


## Usage

Run everything from the **repo root** with the **`pic`** conda env (Python 3.14):
`conda activate pic` (or `/usr/local/Caskroom/miniconda/base/envs/pic/bin/python`).

### The rig — `from pic import Rig`

```python
from pic import Rig
with Rig(laser="hw", board="hw") as rig:        # each of laser/board: "hw" | "mock"
    with rig.session(duration_s=60, power_dbm=5):   # watchdog'd laser session
        pd = rig.measure(volts)                      # 128 DAC volts in -> 14 PD volts out
```

Laser control and the live PD dashboard are on the `./do` dispatcher (`./do laser state 1`,
`./do laser set 5`, `./do measure`); characterization is `./do heaters`. Only one process
may hold the Arduino serial port at a time.

> The live browser console (`ui.py` + `ui/`, an HTTP server on `:8787` that painted laser/
> heater/PD state onto the schematic) was **removed** in the paper cleanup — recoverable from
> git history. The scene graph it drew, `pic.layout`, stays: it still backs
> `scripts/pic_state_diagram.py`, and `theory/twin.py` matches its heater numbering.

### PIC replica — `src/pic_neurophox.py`

Runnable neurophox simulation of the whole chip: input → **U** (6×6 Clements) → **Σ** →
**V** (6×6 Clements) → PD intensities.

```
python src/pic_neurophox.py            # demo: build + self-check + sample readout
python src/pic_neurophox.py --diagram  # structure figure -> slides/images/pic_replica.png
```
```python
from src.pic_neurophox import PICReplica
pic = PICReplica(seed=0)
pic.encode_sigma(A)            # program Σ to a target matrix's singular values
pic.readout([1,0,0,0,0,0])     # 6 photodiode intensities  (pic.matrix = realized M)
```

A **structural** replica — mesh phases are free (random) until the heater→phase
calibration exists (`from_voltages` is the stub it will fill). Needs tensorflow (in the
`pic` env; ~6 s one-off load).

*Legacy:* `Monte Carlo for PIC.ipynb` (all-in-one MC optimiser over serial) was removed in the
cleanup — recoverable from git history. Firmware still flashes from
`Arduino/Fully_automated_PIC.ino` (Arduino, 115200 baud; 64 floats out → 14 mean PD volts in);
the live host optimizer scaffolds are `src/inverse.py` and `scripts/pic_gd.py`.


## Code layout

- **`pic/` — the unified rig library (`import pic`).** `from pic import Rig` is the primary
  entry: one object driving laser + board (+ an optional model), each `"hw"` or `"mock"`
  (`Rig(laser="mock", board="mock").open()`; then `.session(...)`, `.measure(v)`, `.predict(H)`,
  `.close()`). Under it: `interface` (`PIC` pyserial driver matching the `.ino` + `MockPIC`,
  no-hardware), `config`, `acquisition` (`sweep_channel`, `measure_averaged`, `collect_dataset`,
  `homodyne_sweep`/`fit_homodyne`), `layout`/`wiring`/`homodyne`, `session` (the dual-device
  experiment layer, was `template.py`), `devices` (`Laser`/`MockLaser`/`PDMv5`, was `laser/`),
  `model` (`MockModel`/`DpnnModel`/`TwinModel`), `compute.ising`/`compute.matvec`, `bringup`.
  CLI: `python -m pic <measure|ising|matvec|bringup>`. Legacy paths (`from src.pic import …`,
  `from laser.laser import Laser`, `from template import …`) still work as thin shims.
- **`theory/`** — pure-math simulation (differentiable digital twin + Ising/matvec kernels);
  imported by `pic.compute` and `pic.model`.
- **`src/`** — modelling: `data`, `inverse` (`MonteCarloInverse`, `GradientInverse`),
  `characterize` (per-heater φ²/φ⁰ fringe fits), `census` / `sweep_analysis` (the PIC-B fringe
  pipeline), `prune` (the DPNN pruning core), `pic_neurophox` (the replica above).
- **`scripts/`** — analysis / schematic / hardware scripts; index in **`scripts/README.md`**.


## Datasets

Both are **Section-A, power-only**, on the 0.5 V input grid; format: col 0 = 64 input
volts (string), cols 1–14 = 14 raw PD volts. **Not** the splitting-tree 6×6.

- **`pic_data/` — canonical clean set:** 7 multi-session runs (15 Jul / 22 Jul), 160k
  unique rows (135k excluding a laser-off dud). Rows are in acquisition order, so a
  within-run output trend *is* drift. This is what the drift proof runs on.
- **`100k_data_original.xlsx` — deleted:** verbatim copy of the four 22-July runs
  (byte-identical, recoverable from git); `src/data.py` auto-rebuilds it. "100k" findings
  = the 22-July campaign.

Every unique input was measured **exactly once** — no independent repeats, so √N averaging
can't be tested offline (needs fresh acquisition). ~38% of rows are exact duplicates.


## Prior work — `Anagha/` (PIC-A, untracked)

The Section-A predecessor work (Anagha Gayathri, Pranav): 128 full-range 0–5 V per-channel
sweeps, a 20×200 repeat-noise set, thermal-crosstalk studies, and a TEC-controlled forward
NN reaching **R² ≈ 0.86** (vs 0.70 without TEC). 353 MB, kept on disk and gitignored —
**inventory, paper mapping and open discrepancies in [`Anagha/README.md`](Anagha/README.md)**.
Read it before writing the characterisation section; it contradicts the findings below on
which PDs are dead, and it suggests TEC control is the missing lever behind the ceiling.


## Findings (compacted research log)

Single-shot, power-only characterisation of the Section-A path:

- **Learnability ceiling ≈ 0.55–0.78 R²** (held-out inputs→PD). Two model families *and*
  the physics model agree → it's a property of the **data**, not the fit; capacity doesn't
  move it.
- **Not readout noise.** The dark-PD floor (~5.8 mV) would permit R² ≈ 0.95; the gap is
  structure + drift + per-shot phase jitter, amplified on deep PDs (σ ∝ √signal).
- **4 dead PDs {0,2,7,11}** — std at the ~5 mV ADC floor; dropped by the 14→10 filter. They
  double as a **free common-mode drift reference** — feeding them back as inputs lifts mean
  R² by ~0.2 (PD8 revived 0.10→0.65).
- **The chip is non-stationary.** Same-day mV-level offset; **20–60 %/week** response
  collapse. **No time-invariant V→PD map exists** (cross-session R² ≤ 0) — characterisation
  is valid only *within* a thermal session. Likely thermal (heater self-heating + ambient/
  TEC baseline) → enable/verify TEC control.
- **Drift is two-tier and adaptable.** Same-day = offset; across-day = map change. A ~20-param
  affine adapter on ~100 fresh samples flips cross-day R² −0.44 → 0.48; a ridge learned-delta
  needs ~500+ samples for the rest.
- **The remaining lever is a measurement upgrade, not a bigger model:** homodyne (recover
  phase/sign, lift small-signal PDs off the ADC floor) and/or √N averaging.

**Section B / 128-channel era** (2026-07-23 the whole setup was disconnected, rewired for the
8-DAC 128-channel board, and reconnected; work is now on **PIC B**):

- **Everything measured before the rewire is SUPERSEDED for B** — the 2026-07-23 morning 64-ch
  census (`pic_data/census/probe_b_influence.csv`, swings up to 78 mV, 8 silent channels) was
  taken through the old PIC-A-era hookup and its DAC map. Kept for provenance only; do not use
  its channel→PD structure or PD classes for B.
- **No schematic DAC→heater map is trusted** (picpin decode contradicted on hardware). The B map
  is being derived empirically: `scripts/probe_channels.py` on the 128-ch firmware
  (`Arduino/pic128/`), one channel at a time, raw repeats + laser telemetry streamed to
  `pic_data/census/` (tracked in git).
- **Firmware facts learned the hard way:** these DACs need the per-write config sequence
  (`0x03 00 84` / `0x09 00 00` / `0x05 FF FF` after every value write — drive.ino's "proven"
  block); the one-time `0x03 00 FF` + `0x04 00 01` init from `Fully_automated_PIC.ino` does not
  reliably bring them up. `pic128.ino` = proven sequence + 0–2 V clamp + 10-sweep on-chip ADC
  averaging, CS pins {10,9,8,7,6,5,4,3}, accepts 64- or 128-value lines (pads with 0).
- **Post-rewire census + fringe sweep** (`census_b128_postcycle.csv`, `sweep_b128_fringes.csv`,
  fits in `fringe_fits_b128.csv`): **68/128 channels live; DAC chip 4 (CS pin 6, ch 64–79) is
  fully dead** (check its ribbon). **No PD is dead on B**; tiers (0–2 V sweep): strong =
  PD 3,5,6,7,8,9,10,12 (swing ≥25 mV from ≥2 ch; pd10 from 23 ch, pd8 up to 170 mV), weak/narrow
  = PD 0,1,2,4,11,13 (≤27 mV). **95 (ch,pd) fringes close within the 0–2 V clamp, Vπ median
  2.24 V** (range 0.9–4.9); the rest are monotonic partial fringes — full 2π/heater needs >2 V.
  Single-PD-dominant channels (σ/readout candidates): ch36→pd10, ch37→pd10, ch81→pd9, ch99→pd10;
  ch0/ch1 dim pd8+10+12 together ⇒ input-stage. Deep channels' 1.5 V fingerprints drift
  session-to-session — place them by fringes, not single-point deltas.

Per-PD classes (held-out R², 100k):

| class | PDs | R² | note |
|---|---|---|---|
| **good** | 1, 3, 5, 10, 12, 13 | 0.53–0.97 | PD12/13 ≈ solved |
| **noise / unlearnable** | 4, 6, 9 | 0.14–0.25 | irreducible σ∝√signal floor; needs homodyne/√N |
| **recovered** | 8 | 0.12→0.65 | small signal swamped by drift; rescued by dead-PD ref |
| **dead** | 0, 2, 7, 11 | ~0 | ADC floor; used as drift monitor |


## Hardware facts

- **Input (DAC):** 0–4 V, 0.5 V grid; operating regime mostly 0–2 V (~1.5 V ≈ Vπ). Firmware
  `sendDACvalue` clamps to **0–2 V** (note: the shipped 100k data shows *no* clamp active —
  reconcile before "fixing"). Phase ∝ heater power ∝ v².
- **Output (PD):** ~0–0.2 V. 14 read, 4 damaged {0,2,7,11}. Two phase shifters per MZI (full
  U/V reachable). Readback via 10-bit Arduino ADC (~4.9 mV/LSB) → floor ≈ 5 mV.
- **GDS (`NUS/AD_SiPhIS_180423.gds`, whole chip):** ~484 heaters, ~154 MZIs, 57 monitor PDs;
  topology reconstructable geometrically from placements + waveguide adjacency (`scripts/gds_trace.py`).

Theory: Clements mesh `1603.08788`; Cem data-driven OMM `2210.09171`.
