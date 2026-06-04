# PIC — Progress & Key Learnings

Living log of findings for the photonic matrix-multiplier work. Newest section last.
Goal: program the splitting-tree 6×6 to do optical matmul (A = U·E·V via SVD), with
heater voltages in and photodiode voltages out; homodyne (bypass readouts) for phase/sign.

---

## Hardware / device facts (confirmed)

- **Input (DAC):** 0–4 V, 0.5 V grid. Operating regime is mostly **0–2 V, centered ~1.5 V ≈ Vπ**
  (π phase shift). Thermal PS spec (briefing): π at **<25 mW**, insertion loss <0.2 dB,
  rise/fall **<30 µs** (device-level; chip-level thermal settling is the real bottleneck).
- **Output (PD):** ~**0–0.2 V** (max seen 0.30). Phase ∝ heater power ∝ v² (quadratic).
- **Photodiodes:** 14 read per sample; **4 are damaged → PD indices [0, 2, 7, 11]**. The
  14→10 filter drops exactly these. Confirmed dead in data (means ~0.002–0.011 V ≈ floor).
- **Phase shifters per MZI: two** (independent amplitude + phase → full U/V reachable, unlike
  Cem 2210.09171 whose chip had only one).
- **Readout precision:** outputs quantized to 1 mV; readback via 10-bit Arduino ADC
  (≈4.9 mV/LSB over 0–5 V). Signals ≤6% of full scale → small-signal PDs live in ~4–10 LSB.

## GDS inventory (`NUS/AD_SiPhIS_180423.gds`, whole chip, all 7 structures)

AMF PSOI C-band process. Authoritative component counts (parsed from `top` cell placements):
- **484** thermal phase shifters (heaters)
- **308** 2×2 MMI couplers → ~**154 MZIs**
- **224** 1×2 MMI splitters (splitting trees)
- **57** power-monitor photodiodes
- **6** input couplers, **1** grating coupler
- Units: 1 DB unit = 1 nm. Components on a regular grid (heaters Δy=70 µm; columns aligned in x).
- ⇒ **Topology can be reconstructed geometrically from the GDS** (by component position +
  waveguide adjacency). This is the right source — not raster diagrams.

## Wirebond / control PCB (`NUS/1010.190.PcbDoc`, Altium binary, parsed via `olefile`)

The PCB is a **passive fan-out board** (no active parts):
- **190 nets**, named numerically `1`–`190` (no semantic H#/DAC labels).
- **190 wirebond pads** (`Pad1`–`Pad190`, footprint "PAD"), one per net = chip-side bond pads.
- **2× 96-pin FPC connectors** (`FPC96Pin`) → 192 pins ≈ 190 nets, mate with the DAC board.
- 190 ≈ **~128 heater lines + ~57 PD lines + a few grounds** — consistent with the user's
  "128 inputs, 14 (of 57) outputs" framing.
- ⇒ Board just routes chip bond pads to the FPC connectors; the **Arduino + 4×16-ch DAC
  daughter board drive the FPC pins in fixed firmware order** (DAC i → chip i/16, ch i%16).
- ⇒ **To RUN the loop we already have everything** (chip + passive PCB + DAC board + firmware
  + dataset). The DAC-channel→heater map is only needed to attach *physical-mesh meaning* for
  the SAM physics model — it's reconstructable but multi-hop & geometric (no ready label table):
  net → chip-side pad XY (`Pads6`) → nearest GDS heater; and net → FPC pin → DAC chan (firmware).

---

## The dataset: `100k_data_original.xlsx` (the only dataset we have)

- 100,000 rows. **Col 0 = 64 input voltages (string); cols 1–14 = the 14 raw PD outputs.**
- This is the **OLD Section-A, power-only** path (no homodyne), NOT the splitting-tree 6×6.
- Row 2's 14 outputs, with [0,2,7,11] dropped, exactly equal the notebook `DESIRED_OUTPUT`.
- Input grid {0,0.5,…,4.0}; mass concentrated at 0.5–1.5 V; tail (2.5–4.0 V) sparse.
- **No exact-duplicate inputs** (no free noise-floor estimate); in 64-D the nearest pair
  differs by L1 > 3 (no near-twins either).

### ⭐ Learnability table (test R², the ceiling any memoryless model can hit)

Raw inputs (0–4 V). Two model classes agree → this is a real, model-independent ceiling.

| Model         | PD1 | PD3 | PD4 | PD5 | PD6 | PD8 | PD9 | PD10 | PD12 | PD13 | **mean** |
|---------------|-----|-----|-----|-----|-----|-----|-----|------|------|------|----------|
| HistGB(v)     |0.78 |0.62 |0.19 |0.62 |0.26 |0.34 |0.18 |0.58  |0.95  |0.98  |**0.55**  |
| MLP([v,v²])   |0.72 |0.63 |0.19 |0.64 |0.22 |0.17 |0.18 |0.59  |0.93  |0.96  |**0.52**  |

- **Hugely uneven:** PD12/PD13 ≈ solved (0.93–0.98); PD4/PD6/PD9 nearly unlearnable (~0.2).
  Pattern = deep-path / low-signal PDs are the hard ones (more interfering MZIs + worse SNR).
- Compare Cem 2210.09171: R²≈0.99 on a clean 3×3. The gap is the **measurement chain**
  (10-bit ADC, no thermal stabilization, random sampling), not proof the physics is wrong.

### Other dataset findings

- **No 2 V clamp was active when this data was collected.** Raw inputs (to 4 V) beat
  clamped-at-2 on **every** PD: mean R² **0.55 (raw) vs 0.33 (clamp)**. The current firmware's
  `voltage>2 → 2` clamp would NOT reproduce this data. ⚠️ contradicts CLAUDE.md gotcha — reconcile.
- **Noise floor (from damaged PDs):** per-channel std ≈ **5 mV**; signal std ≈ 44 mV. For the
  hard PDs the *unexplained* residual (~36 mV) ≫ 5 mV ⇒ ceiling is **not** white ADC noise;
  it's structured (deep-path interference complexity + hidden thermal/ambient state).
- **Temporal/thermal-memory probe inconclusive:** rows are shuffled (adjacent-row drive adds
  0 predictive power), so this dataset can't isolate noise vs drift.
- **Physics basis probe (Vπ anchor):** ridge on per-channel [cos(a·v²), sin(a·v²)] peaks at
  a≈0.3 (Vπ≈3.2 V, R²=0.42), NOT at the physical Vπ≈1.5 V (a=1.40 → R²=0.10). A *linear,
  per-channel* model can't recover Vπ because interference (products across MZIs) + per-heater
  φ⁰ offsets dominate. ⇒ Can't shortcut to the physics with global features; need the
  structured per-heater model + topology. Smooth bases all plateau ~0.42 linearly (vs 0.55
  black-box) → most "easy" variance is smooth; the rest needs true interference modeling.

---

## Physical map decode (PCB + data-driven)

**PCB (`Components6`, parsed via olefile + length-prefixed pipe-text):**
- 190 wirebond lands `Pad1`–`Pad190` (staggered double-row bond-pad ring; X 2513–4878 mil,
  Y 4484–6404 mil) + 2 FPC connectors `J1`,`J2` (96-pin each).
- `CHANNELOFFSET` ⇒ **PadN → board channel = N+1** (channels 2–191), sequential.
- `Pads6` (binary, net-per-pad linkage) parser still incomplete — not blocking; CHANNELOFFSET
  + positions already give the PCB-side map.

**Remaining hops for a *full geometric* DAC→MZI chain (some data missing):**
firmware DAC ch (0–63) → FPC pin → PCB net → PadN (bond land) → bond wire → chip bond pad →
**chip metal routing (in GDS)** → heater → MZI. The FPC↔DAC hop needs the **DAC daughter-board**
design (not in our files); the bond-pad↔heater hop needs chip metal-layer routing from the GDS.

**Data-driven functional map (η² main-effect, DAC ch → live PD, from 100k) — `/tmp/dac_pd_influence.npy` (64×10):**
- **~3 channels dominate:** ch1→PD1 (η²=0.68), ch14 & ch15→PD13/PD12. Then a steep cliff.
- **~20 of 64 channels have ≈0 influence** on any live PD (ch 0,3,10,32,33,39,43–47,51,55–62).
- Explains the low learnability (most inputs irrelevant) AND why the data "reliably predicts
  inputs for *some* targets" (those dominated by the few strong channels / predictable PDs).
- ⇒ Model each PD from its *few* influential channels (sparse), not dense 64→10. This sparsity
  is the lever for a structured model to beat the dense black-box ceiling.

## ⭐ Better-physics test — RESULT (the key decision)

Tested whether a physics-structured forward model beats the black box on the 100k set:

| approach | mean R² (10 live PDs) |
|---|---|
| dense black-box HistGB (all 64 ch) | 0.55 |
| **high-capacity** HistGB (800 trees) | **0.57** (barely moves) |
| high-capacity MLP (256³, low reg) | 0.29 (overfits → negative R² on hard PDs) |
| sparse MLP (top-10 ch/PD) | 0.48 |
| **sparse PHYSICS** (cos/sin(a·v²) + pairwise, top-10 ch) | **0.48** (ties sparse MLP) |

**Verdict: better physics does NOT help on this data. The ceiling (~0.57) is noise / hidden-
thermal-state / measurement-chain limited, not under-modeling.** Confirmed three ways:
1. Physics-basis model **ties** the sparse NN and **loses** to the dense black-box.
2. Sparse (few channels) loses to dense → deep PDs depend diffusely on *many* channels
   (deep-path interference + thermal crosstalk), so single-channel `v²` features are mis-specified.
3. **Capacity doesn't move the ceiling** (0.55→0.57 with 800 trees; the big MLP gets *worse* by
   fitting noise). No deterministic signal left to capture past ~0.57.
- **Vπ never emerges:** every phase-scale scan pins `a` at the low boundary (smooth ≈ polynomial),
  never the physical Vπ≈1.5 V. Crosstalk (φ_m depends on many v's, not v_m alone) + per-heater
  offsets + noise wash out the clean single-Vπ interference signature.
- ⇒ This validates the original worry: ambient temp / temporal interference / ADC-DAC noise
  dominate. **The ceiling is a hardware/protocol problem, not a modeling problem.** Compute/MLX
  cannot raise forward fidelity on this dataset (the 434 s high-capacity fit gained ~nothing).
- ⚠️ Caveat: this is the existing power-only, 10-bit-ADC, no-thermal-control, randomly-sampled
  set. Physics is **not** disproven — it just hasn't had a fair shot. **Same chip**, we simply
  have not yet collected **homodyne** data (complex-field linear ID) with better readout +
  thermal control + structured sweeps. That data collection is the next step (live, tomorrow).

## ⭐ Can we PROVE a hardware limit? — identifiability result (`scripts/prove_hardware_limit.py`)

The better-physics RESULT above is *empirical* ("capacity doesn't move the ceiling"). Asked to
**prove** a hardware ceiling. The rigorous, model-free statement: for ANY predictor g(V),
`MSE ≥ E[Var(Y|V)]`, so `R²_max = 1 − E[Var(Y|V)]/Var(Y)`. Estimating `E[Var(Y|V)]` from the 100k set:

- **(C) Dark-PD readout floor = 5.8 mV** (the 4 damaged PDs, pooled). If readout/ADC/DAC noise
  were the only limiter the ceiling would be **~0.95–0.99**. → **Readout noise is NOT the limiter.**
  Kills the simplest "it's just ADC noise" hypothesis.
- **(A) Exact-duplicate test: ZERO repeated input vectors** (random sampling on a 9⁶⁴ grid never
  repeats). So we cannot measure irreducible noise directly offline.
- **(B) Per-PD within-cell spread** (spread of Y among rows sharing the 2–3 dominant channels;
  well-populated cells, no overfit): PD12/13 = **1.4–1.6× floor** (≈ fully determined by 2 channels —
  *these are characterizable now*); PD4/6/9 = **6.6–8.5× floor** with peak few-channel R²≈0.05
  (their variance lives in *no small channel set*). Reproduces the learnability table exactly.
- **(D) Row order is ~iid** (consec-row channel match 0.195 vs 0.111 baseline; weak block
  structure, no clean trajectory) → no usable time axis → can't test drift offline either.

**Verdict — the honest answer to "can we prove it":**
1. **NOT from this dataset.** It has *zero input repeats* and *one shot per input*. That design makes
   **reducible** variance (crosstalk / high-order interference a richer model could capture) and
   **irreducible** variance (thermal/temporal hidden state) **mathematically unidentifiable** — you
   cannot separate "model too weak" from "hardware too noisy" without repeats. The 0.55→0.95 gap is
   confounded. (Note PD4/6/9 being invisible to *main-effect* η² yet high-variance is consistent with
   *interference-determined* = possibly **reducible** with a coherent model — so don't call them
   "noise" yet; homodyne may rescue them.)
2. **What IS proven offline:** 4 PDs dead (Tier 0); readout noise is not the bottleneck (Tier C);
   learnability is sharply per-PD.
3. **The decisive test is cheap (Tier 2, hardware, ~15 min):** *(i) Repeat test* — pick ~20
   representative V, measure each N≈50× back-to-back. Within-V variance = irreducible noise with **all
   channels fixed** (crosstalk confound gone). If ≈ dark floor → ceiling is reducible → characterization
   IS possible (better model wins). If ≈ the 38–50 mV within-cell spreads → **proven irreducible** →
   capped. *(ii) Drift/soak* — same V once/min for ~30–60 min (± heating neighbors); drift beyond the
   repeat noise → **thermal hidden state proven** → a static V→PD map is provably insufficient
   (needs temperature as an input / closed-loop). **Repeats break the confound.** Script this for the live session.

## Modeling verdict & plan

- A **naïve memoryless model — physics OR black-box NN — caps ~0.5 R²** on data like this and
  leaves the hard PDs unmodeled. Pure physics won't suffice; pure black-box also plateaus.
- But the ceiling is set by the **measurement chain**, not the physics: v² genuinely helps and
  clean PDs hit 0.98. A structured grey-box (known topology + per-MZI cos²·, φ=φ²v²+φ⁰) has a
  strong prior → identifiable from *few* structured measurements, and can plausibly beat the
  black-box ceiling that's data-starved on a high-frequency interference function.
- **Recommended hybrid:**
  1. **Grey-box physics forward model** (per-MZI transfer, two phase shifters, tiled over the
     GDS-derived topology).
  2. **+ learned residual / probabilistic head** for the genuinely stochastic part (drift,
     ambient, crosstalk) — the "fuzzier" model.
  3. **+ low-rank (LoRA) adapter** for session-to-session drift (validate rank by SVD of
     repeated-characterization offset variation).
- **Homodyne is also the precision/SNR lever**, not just phase: coherent gain (weak signal ×
  strong LO) lifts small-signal PDs off the ADC floor — exactly where this dataset collapses.
- **Characterization must control thermal state** (full settle, fixed approach direction,
  ordered sweeps) — random sampling like the 100k set destroys structured per-path ID.

## Open blockers

- **DAC→heater wirebond map**: data located (PcbDoc `Pads6`/`Nets6`, 190 numbered nets +
  pad XY). Numeric nets only → must be reconstructed geometrically (pad XY ↔ GDS heater; FPC
  pin ↔ DAC channel). Not needed to *run* the loop; needed to fit the literal SAM model.
- **Per-structure topology** (isolate the splitting-tree 6×6 from the 484-heater GDS) — the
  thing we actually need going forward; reconstructable from GDS placements + adjacency.
- **Firmware clamp reconciliation** (which firmware produced the no-clamp 100k data?).

## Code layout (scaffolded for live work — `tests/smoke_test.py` passes)

- **`lib/pic/`** — hardware interface (import `from lib.pic import PIC, MockPIC, acquisition`):
  - `config.py` — `PICConfig`, channel counts, damaged PDs, voltage/firmware constants.
  - `interface.py` — `PIC` (pyserial driver, matches the .ino: 64 floats out → 14 floats in,
    `measure()` does optional thermal-settle re-read) and `MockPIC` (forward-fn backed, no HW).
  - `acquisition.py` — `sweep_channel` (monotonic, dwell), `measure_averaged` (noise floor),
    `collect_dataset`, and **`homodyne_sweep` + `fit_homodyne`** (reference-arm homodyne that
    works with the *existing* firmware: step a reference heater, fit the fringe).
- **`src/`** — modelling (import `from src import ...`):
  - `mzi.py` — two-phase-shifter transfer (`mzi`, `mzi_closed`, `attenuator`), Clements
    `clements_layers` / `mesh_unitary` (decomposition target→angles is TODO).
  - `data.py` — load/cache the 100k xlsx; `live()` drops damaged PDs.
  - `influence.py` — `eta2_matrix`, `top_channels`, `dead_channels`.
  - `forward.py` — `BlackBoxForward` (working, ~0.55 ceiling), `GreyBoxForward` (scaffold).
  - `inverse.py` — `MonteCarloInverse` (refactor of the original heuristic, drives any
    `measure()`), `GradientInverse` (scaffold for MLX/autodiff inverse).
  - `characterize.py` — `fit_fringe` (per-heater φ2/φ0/Vpi from a sweep), `characterize_channels`.
- Run tests/scripts from the repo root so `lib`/`src` import.

## Environment

- Python work uses the **`pic`** conda env (Python 3.14): `conda activate pic` or
  `/usr/local/Caskroom/miniconda/base/envs/pic/bin/python`. Installed: numpy, pandas, scipy,
  scikit-learn, openpyxl, **pyserial**, olefile. Dataset cached `/tmp/pic_100k.npz`;
  influence matrix `/tmp/dac_pd_influence.npy`.
- Theory refs in `References/`: `1603.08788` (Clements mesh), `2210.09171` (Cem data-driven OMM).
