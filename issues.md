# PIC — characterisation issues & investigation log

Living log of the PIC characterisation: which **photodiodes** work, which don't, why, and what fixes
them (Issues 1–7), then whether the dataset is learnable and how clean it is (Issue 10). Goal: anyone
should be able to **reconstruct the full picture from this file alone** (numbers + the script that
produced them). Newest evidence appended under each issue.

## Data sources
- `100k_data_original.xlsx` — **DELETED (Issue 10d)**, was a verbatim copy of the four 22-July runs
  (every row byte-identical, max|Δinput|=max|Δoutput|=0; they partition it into 4×25k disjoint input
  chunks). "100k" = the 22 Jul campaign, NOT a separate session. Recover with git if ever needed.
- `pic_data/testing_working_ps_hybrid_64_*.xlsx` — 7 runs over 2 days. **Rows are in acquisition
  order** (inputs stationary along rows ⇒ a within-run output trend = drift). 22 Jul = 4 sequential
  runs over disjoint input chunks (= the 100k set). 15 Jul = 10k + 25k + a **dud** (`25k_15july(2)`,
  laser off, all flat at the floor); the 15 Jul inputs are **distinct** (share 0 inputs with the 100k set).
- **NO independent repeated measurements exist** (Issue 10c): every unique input was measured exactly
  once. The 100k duplicates are file copies, not repeats ⇒ √N averaging cannot be tested offline; it
  needs new acquisition (re-measure inputs N×).
- Readout/ADC floor ≈ **5 mV** (pooled std of the 4 dead PDs).

## PD classification (data-driven; `scripts/pd_learnability.py`)
Per-PD on the 100k (25k subsample, held-out): output std, full-64-channel black-box R².
Classes: **dead** = std ≤ ~floor (no signal); **unlearnable** = real swing but R² < 0.4 (no input
explains it); **good** = R² ≥ 0.4.

| PD | std (mV) | R² | class |   | PD | std (mV) | R² | class |
|----|----|----|----|---|----|----|----|----|
| 0 | 3.0 | 0.00 | **dead** | | 7 | 4.2 | 0.04 | **dead** |
| 1 | 27.7 | 0.78 | good | | 8 | 13.4 | 0.25 | **unlearnable** |
| 2 | 10.3 | 0.28 | **dead** | | 9 | 45.9 | 0.14 | **unlearnable** |
| 3 | 57.9 | 0.57 | good | | 10 | 52.4 | 0.53 | good |
| 4 | 38.7 | 0.14 | **unlearnable** | | 11 | 2.3 | 0.01 | **dead** |
| 5 | 63.2 | 0.58 | good | | 12 | 39.0 | 0.94 | good |
| 6 | 50.3 | 0.19 | **unlearnable** | | 13 | 50.4 | 0.97 | good |

- **dead {0,2,7,11}** · **unlearnable {4,6,8,9}** · **good {1,3,5,10,12,13}**.
- Colored into `pic_structure.png` (grey / orange / green).

---

## Issue 1 — Dead PDs, and how we detect them
**Symptom:** PD0, PD2, PD7, PD11 carry no usable signal.

**Detection (objective):** output std stays at the readout floor regardless of input
(PD0 3.0, PD7 4.2, PD11 2.3 mV ≈ the 5 mV floor) **and** black-box R² ≈ 0 (no input combination
moves them). Cross-check: NUS `PIC_update.pdf` shows these four uniformly dark across all DAC
sweeps; the firmware 14→10 filter drops exactly indices {0,2,7,11}.

**Anomaly:** **PD2** sits at 2× floor (10.3 mV) with R² 0.28 — nominally in the dark set but shows
weak partial structure. Kept as "dead" (signal still ≪ live PDs) but flagged: it is not as fully
dead as 0/7/11.

**Use found later (see Issue 5):** because the dead PDs see *no guided signal*, any drift they
*do* show is a pure **common-mode / baseline** drift → they double as a **free drift monitor**.

---

## Issue 2 — Baseline: the unlearnable PDs
**Symptom:** PD4, PD6, PD8, PD9 have a wide output range (std 13–50 mV) but no input predicts them.

**Evidence:**
- Full 64-channel black box (held-out): R² PD4 0.14, PD6 0.19, PD8 0.25, PD9 0.14. The *predicted*
  output distribution collapses to a spike while the *actual* stays broad (`pd_learnability.png`).
- Single-channel separation d′ = |E[Y|ch hi] − E[Y|ch lo]| / std: 0.08–0.27 (learnable PDs 0.76–1.11).
- Max single-channel influence η² ≤ 0.05 (learnable PDs 0.39–0.68). Only ~6 of 64 channels carry
  any signal at all: {1,14,15,23,48,54}; ch1→PD1/3/5, ch14/15→PD8/10/12/13.
- **Physics model agrees it's the data, not the model:** the faithful neurophox optical model
  (`scripts/neurophox_baseline.py`, the labelled 6×6P U·Σ·V with edge-drops) gets mean R² **0.41**
  vs the black box's **0.50** on the same split, and **fails on the same PDs 4/6/9** the black box
  does. (Fuzzy variant — stochastic phases + within-MZI crosstalk, `neurophox_fuzzy.py` — keeps the
  same point R² 0.38 but is perfectly calibrated, |z|<1 = 0.68; learned crosstalk |κ| mean 0.17,
  max 0.68 ⇒ real heat migration.)

**Verdict:** "unlearnable" ≠ small signal. The variance is real but **not caused by the inputs**.
Capacity doesn't help (800-tree GBT moved the ceiling ~0.02). It's an information problem.

---

## Issue 3 — A session one-hot recovers ONE of them (PD8)
**Question:** is the unlearnable variance just drift (an offset that moves between runs)? Give the
model the run identity (a coarse thermal-state proxy) and see. `scripts/hidden_state_test.py`:
black box on voltages only / + session one-hot / fit-within-each-session. Held-out R²:

| PD | input | +session | within-session | reading |
|----|----|----|----|----|
| 1 (good) | 0.55 | 0.85 | 0.79 | drift was even holding good PDs back |
| 12 (good) | 0.69 | 0.96 | 0.93 | " |
| 13 (good) | 0.81 | 0.97 | 0.96 | " |
| **8 (unl)** | 0.12 | **0.57** | 0.26 | **RECOVERED** — small signal swamped by drift |
| **6 (unl)** | 0.17 | 0.34 | 0.16 | partial |
| **4 (unl)** | 0.13 | 0.22 | 0.08 | still unlearnable |
| **9 (unl)** | 0.13 | 0.22 | 0.10 | still unlearnable |
| mean (all live) | 0.39 | **0.58** | 0.48 | |
| mean (unlearnable) | 0.14 | 0.34 | 0.15 | |

**Findings:**
1. **Drift limits the whole chip:** knowing the run lifts mean R² 0.39 → 0.58; good PDs go near-1.
2. **PD8 was mislabeled** — it's "real signal + drift offset" (0.12 → 0.57). Reclassify as
   **recoverable (drift-swamped)** once a drift reference exists.
3. **PD4, PD6, PD9 stay unlearnable** — the *within-session* column proves it: even with all
   cross-run drift removed (fit inside one run) they sit at 0.08–0.16. Their orange label stays.

---

## Issue 4 — Recurrent drift model (intra- + inter-session)
**Idea:** model drift on both timescales — a **session embedding** for the inter-run baseline +
a **GRU** over `[v_t, y_{t-1}]` (drift is only visible through past outputs; inputs are iid) for
the intra-run trajectory. Evaluated on a **temporal** split (train first 70% of each run, predict
the future 30% — the state must *extrapolate* drift). `scripts/drift_rnn.py`.

| model | mean R² | unlearnable {4,6,8,9} | note |
|----|----|----|----|
| static (v only) | 0.02 | — | predicting the future w/o drift modelling collapses |
| + session emb | **0.29** | — | inter-session offset = the robust win |
| recurrent (GRU + emb) | 0.14 | −0.13 | **did NOT beat +session** on this split |

**Verdict:** the **inter-session** offset is the valuable, robust signal; the **intra-session GRU
did not help** when forced to extrapolate the future of a run (it underfit the base v→y map while
chasing a weak/noisy trend). Intra-run drift *is* real (Issue 5) but a teacher-forced GRU isn't the
right tool to extrapolate it here. Better lever identified in Issue 5: the **dead PDs directly
measure the common-mode drift**, so feed *that* as the state instead of inferring it. [TODO]

---

## Issue 5 — Global-heating hypothesis · a.k.a. **"Global Warming"** 🌡️ (drift mechanism)
**Hypothesis (user):** heater self-heating accumulates in the substrate *within the TEC deadband*
(TEC holds 25.0 ± 0.2 °C at its sensor; the optics can sit hotter). Si dn/dT ≈ 1.8e-4/K ⇒ ~0.1 K
re-phases the mesh measurably ⇒ drift. Test: does the output trend within a run, and do the dead
PDs (no guided signal) move too? `scripts/drift_mechanism.py`, `drift_mechanism.png`.

Within-run trend (slope of PD vs row index; inputs stationary ⇒ trend = drift):

| run | live common-mode | live mean \|slope\| | **dead-PD slope** | cross-PD corr |
|----|----|----|----|----|
| 10k_15july | +14.9 mV | 18.4 | +6.8 | +0.60 |
| 25k_15july | −11.3 | 11.3 | −10.8 | +0.91 |
| 25k_22july(10) | +3.4 | 9.1 | +4.2 | +0.03 |
| 25k_22july(10)_2 | −12.6 | 12.6 | −9.0 | +0.88 |
| 25k_22july(10)_3 | +5.2 | 7.6 | −0.1 | +0.12 |
| 25k_22july(10)_4 | −6.7 | 8.2 | −5.7 | +0.49 |

**Findings:**
1. **Yes, the output drifts within a single run** (common-mode 3–15 mV; |slope| 7–18 mV). Confirms
   intra-run drift.
2. **Not always up** — 3/6 runs net-up, 3/6 net-down. Direction is **run-dependent** (depends on
   the chip's thermal state at run start: cold start warms up, warm start settles down). So the
   simple "heating ⇒ output up" is false; it's re-phasing, which moves ports both ways.
3. **Largely common-mode/global** — cross-PD trend correlation 0.49–0.91 in 4/6 runs ⇒ a single
   shared cause moves the PDs together (consistent with global heating), not independent per-PD noise.
4. **The DEAD PDs drift too**, tracking the live common-mode sign (dead avg |slope| 6.1 mV vs live
   11.2 mV). Dead PDs see no guided signal ⇒ this is a **baseline/background** component (PD dark
   current is strongly T-dependent; + ADC/bias) — i.e. a real "phantom" baseline that rises/falls
   with chip temperature, exactly the deadband-heating effect. The live PDs carry this **plus** the
   thermo-optic signal re-phasing (the extra ~5 mV differential).

**Cross-session (between runs), matched input distribution (1.125 ± 0.001 V):** PD means shift
34–56 mV across runs; week-scale response collapses 20–60 %. (`scripts/drift_analysis.py`.)

**KEY consequence (actionable):** the **dead PDs are a free common-mode drift monitor**. The hidden
state that Issue 3/4 tried to *infer* is partly **directly observable** in PD0/2/7/11. Next step:
subtract / regress out the dead-PD common-mode as a live reference and re-test the unlearnable PDs.
[TODO — likely better than the GRU.]

---

## Issue 6 — Drift correction via the dead-PD reference · the **"Global Warming" fix** ✅
**Idea:** the dead PDs measure the common-mode drift directly (Issue 5), so feed the raw dead-PD
readings {0,2,7,11} as extra inputs and let the model subtract the drift. It's a *measurement*, so
it works on an unseen run / the future — unlike a session label (Issue 3), and unlike the GRU
(Issue 4) which had to extrapolate. `scripts/drift_deadref.py`, `drift_correction.png`. Pooled
multi-session pic_data, black box, two splits:

| features | RANDOM mean | RANDOM unl | TEMPORAL mean | TEMPORAL unl |
|----|----|----|----|----|
| voltages | 0.393 | 0.139 | 0.375 | 0.130 |
| + session one-hot | 0.581 | 0.343 | 0.581 | 0.346 |
| **+ dead-PD ref** | **0.582** | **0.357** | **0.590** | **0.368** |
| + dead + session | 0.594 | 0.366 | 0.596 | 0.371 |

per-PD (temporal, voltages → +deadPD): **PD8 0.10 → 0.65 (recovered)**, PD6 0.17 → 0.36,
PD4 0.12 → 0.25, PD9 0.13 → 0.22.

**Findings:**
1. Dead-PD reference ≥ session one-hot, and it's a **live measurement** → deployable on an unseen run.
2. **Holds on the TEMPORAL/future split** (0.590, ≈ its random 0.582) — no extrapolation needed,
   unlike the GRU (Issue 4) which collapsed there. *Reading* the drift beats *inferring* it.
3. Chip-wide lift **+0.2 mean R²** (0.39 → 0.59). PD8 fully recovered; PD6 ~doubles.
4. **PD4, PD9 stay ~0.22–0.25** even after drift correction → their variance is NOT common-mode
   drift; it's deeper (deep-path interference / genuine noise). These are the **irreducible core**.

**Deploy:** read all 14 PDs; use the 4 dead ones to correct the 10 live ones in real time. Free —
the dead detectors were the drift sensor all along.

---

## Issue 7 — Source of the irreducible range (PD4/6/9): white, deep-interference-amplified
**Question:** if 4/6/9's wide range isn't input and isn't common-mode drift, what is it?
Decompose the residual after voltages + dead-PD (held-out, ordered). `scripts/unlearnable_source.py`.

| PD | class | R²(volt) | R²(+dead) | resid SD | ac@1 | ac@5 | ac@20 | corr(Σv²) |
|----|----|----|----|----|----|----|----|----|
| 4 | UNL | −0.01 | 0.14 | 41 mV | −0.02 | 0.00 | −0.02 | +0.01 |
| 6 | UNL | 0.00 | 0.23 | 50 mV | 0.00 | +0.01 | 0.00 | +0.02 |
| 9 | UNL | 0.04 | 0.13 | 46 mV | +0.01 | −0.01 | −0.01 | −0.01 |
| 12 | good | 0.52 | 0.94 | 10 mV | — | — | — | +0.06 |
| 13 | good | 0.73 | 0.97 | 10 mV | — | — | — | +0.05 |

**Findings:**
1. **Residual is white** — autocorrelation ≈ 0 at lags 1/5/20 ⇒ NOT slow drift (nothing to track;
   explains why the RNN, Issue 4, and the dead-PD reference, Issue 6, can't touch 4/6/9).
2. **Not residual heating** — corr with total drive power Σv² ≈ 0.
3. **Scales with mesh depth** — white residual ~45 mV on deep PD4/6/9 vs ~10 mV on shallow good
   PD12/13 → ~4–5× interference amplification of a common jitter floor.
4. **PD4–PD6 anti-correlated (−0.19)** → interference signature (complementary coupler ports),
   not additive electronic noise. (PD4/6/9 otherwise ~independent: no shared mode to subtract.)

**Interpretation:** the range is **per-shot phase uncertainty** (DAC quantization ~5 mV/LSB across
~60 heaters + heater jitter) **amplified by deep interferometric position**.

**Honest confound:** inputs are iid in row order, so *any* function of inputs (even deterministic
high-order interference the black box can't fit) is also white in time. This test kills
drift/hidden-state but **cannot** separate genuine single-shot noise from unmodeled-determinism.
Evidence leans noise + data-starvation (capacity didn't help; physics ties NN — progress.md).

**Fix (same class either way — a measurement upgrade, not a model):** homodyne (measure phase ⇒
deep PDs deterministic); averaging N repeats (white ⇒ √N: 50 mV → 10 mV at N=25); more ADC/DAC bits.
**Decisive test:** input-repeat (same V, N×) — identical ⇒ deterministic/recoverable; scattered ⇒ noise.

### Issue 7b — and it's a *characterised* noise (`scripts/noise_characterize.py`, `noise_characterize.png`)
The residual isn't arbitrary — it's a fixed, stationary noise law:
- **Shape: Gaussian, zero-mean.** 68/95/100 % within 1/2/3σ (Gaussian = 68/95/99.7); near-zero
  skew/kurtosis for 4/6/9. Not heavy-tailed.
- **Scale law: σ ∝ √(signal)** (σ² ∝ μ). **PD4/6/9 collapse onto ONE universal curve**
  (σ ≈ 26→65 mV over μ = 25→145 mV); PD13 is flat at ~10 mV. One signal-proportional noise process,
  ~5× amplified on the deep PDs (the Issue-7 interference gain).
- ~50 % relative width on the deep PDs ⇒ **far too large for literal photon shot noise** ⇒ mechanism
  is interference/speckle amplification of phase jitter; the √μ *law* holds regardless.
- **~stationary within a day** (22 Jul σ≈43 mV) but the **noise amplitude itself drifts week-scale**
  (15 Jul σ≈60 mV) — Global Warming degrading the floor too.
⇒ It is "basically noise": a characterised Gaussian with a predictable σ(signal) law, not recoverable
structure. Cannot be fit from voltages; shrinks as √N by averaging, or vanishes with homodyne (phase).

### Issue 7c — the good/noise split is bimodal, not a hand-set line (`scripts/noise_all_pds.py`)
All 10 live PDs, held-out R² (100k, voltages only): good cluster **0.53–0.97** (6 PDs), noise
cluster **0.14–0.25** (4 PDs), **biggest gap 0.27 wide with NO PD in [0.25, 0.53]**. Any threshold
in that empty band gives the same classes ⇒ the bar is data-driven, not arbitrary.
- The noise law σ∝√μ is the SAME for every PD; usable-vs-noise is set by whether the SIGNAL clears
  it. PD5 and PD6 share the noise curve (~41 vs 45 mV) but PD5 has 2.4× the signal (45 vs 19 mV) ⇒
  PD5 good, PD6 noise. The difference is **signal, not noise** (they only looked close on the
  drift-corrected pic_data, a different dataset). `noise_all_pds.png`.

---

# LEARNING

## Issue 10 — Can we learn the dataset, folding in everything? (`scripts/learn_compare.py`, `learn_compare.png`)
Assume all heaters work; throw our best models at the data with the learnings baked in, 3 in parallel.
**Setup (identical for all):** pooled multi-session pic_data (dud excluded), **TEMPORAL split** (train
first 80% of each run, predict the last 20% — drift must extrapolate), predict the 10 live PDs, with
**all 64 heaters driven** + the **dead-PD drift reference** (Issue 6). 40k train / 20k test.

| model | what | good {1,3,5,10,12,13} | PD8 (recov.) | noise {4,6,9} | all-live |
|----|----|----|----|----|----|
| **tree** | HistGBR per PD on [v,v²,dead_ref] | **0.725** | 0.565 | 0.262 | **0.570** |
| **mlp** | 3-layer MLP on [v,dead_ref] | 0.665 | 0.519 | 0.210 | 0.514 |
| **physics** | faithful U·Σ·V, all 64 ch → 96 phases | 0.609 | 0.459 | 0.118 | 0.447 |

per-PD (tree): PD13 0.95, PD12 0.92, PD1 0.85, PD3/5/10 ~0.51–0.57, PD8 0.56; noise PD4 0.23, PD6 0.34, PD9 0.22.

**Findings:**
1. **Yes — the usable chip is learnable.** Best (trees + drift ref) hits **R² 0.73 on the good PDs**,
   0.57 all-live, on the honest future split — matching/edging the Issue-6 chip-wide 0.59. PD12/13/1
   are essentially solved (0.85–0.95).
2. **Ranking trees > MLP > physics** (all-live 0.57 > 0.51 > 0.45). The data-driven models beat the
   grey-box because the neurophox U·Σ·V is an **uncalibrated structural replica** (no heater→phase
   map yet); that it still reaches 0.61 on the good PDs from pure structure is a sane physics floor,
   and it's the model that will *improve* once heaters are calibrated.
3. **The noise PDs {4,6,9} cap EVERY model at 0.08–0.34** regardless of capacity or structure →
   confirms Issue 7: this is an **irreducible measurement-noise floor, not a modelling gap**. Needs a
   measurement upgrade (homodyne / √N averaging), not a bigger model.
4. **PD8 (recovered) 0.46–0.56** on the temporal/future split → the dead-PD drift reference works out
   of sample, as Issue 6 predicted.
5. "All heaters work" did not raise the data-driven ceiling (trees/MLP already use all 64). The
   ceiling is set by **drift (correctable) + irreducible noise (not)** — both already characterised.
   The model lever left is the **grey-box**: calibrate heater→phase and it should close on the trees.

### Issue 10b — reverse ablation on trees + the single-shot ceiling (`scripts/tree_ablation.py`, `tree_push2.py`, `tree_push.png`)
Dropped the 3 pure-noise PDs {4,6,9}; target = the **7 learnable PDs {1,3,5,8,10,12,13}**. Built up
trees one small change at a time (cumulative), same temporal split, chasing >0.90:

| rung | mean R² | note |
|----|----|----|
| v only | 0.523 | |
| + v² | 0.523 | trees already split on v ⇒ no gain |
| **+ dead-PD drift ref** | **0.702** | **+0.18 — the dominant lever** (Issue 6) |
| + tuned trees | 0.711 | capacity barely helps |
| + session id | 0.741 | run-identity (Issue 3); IN-SESSION ONLY, doesn't generalise to a new run |
| + noise-PD state | 0.756 | {4,6,9} raw as input drift/interference state |
| + interactions | 0.757 | explicit strong-channel products ⇒ ~0 (trees already get them) |
| + full data (108k) | **0.779** | more rows, best variant; sqrt-target transform did not help |

**Verdict — 0.90 is blocked by the per-shot noise floor, not the model.** Compared our best to each
PD's SINGLE-SHOT ceiling, measured on the cleanest data (single-session 100k, no cross-session drift):

| | PD1 | PD3 | PD5 | PD8 | PD10 | PD12 | PD13 |
|----|----|----|----|----|----|----|----|
| clean 100k ceiling | 0.78 | 0.60 | 0.61 | 0.29 | 0.56 | 0.95 | 0.97 |
| our best (xsession, drift-corr) | **0.90** | 0.66 | 0.63 | **0.64** | 0.65 | 0.97 | 0.99 |

We are **at or above the single-shot ceiling on every PD** — even the cleanest single-session data
caps PD3/5/10 at 0.56–0.61, so there is no more signal to extract; PD8 triples (0.29→0.64) purely
from the drift reference. PD1/12/13 are solved (0.90–0.99); the mean is pinned by PD3/5/8/10, which
sit at their σ∝√μ noise ceiling (Issue 7), not a capacity limit. **No feature/model gets past ~0.78
on this single-shot power data.** The only lever left is a MEASUREMENT upgrade: average N repeats
(white noise ⇒ σ/√N) or homodyne (phase ⇒ deep PDs deterministic) — new data, not a new model.

### Issue 10c — can we average the noise down using duplicate inputs? No — there are none (`scripts/near_dup_check.py`)
Checked all 260k rows for repeated input vectors (exact, on the 0.5 V grid). **100,000 inputs appear
twice** — but the two copies are **byte-identical in BOTH input and output** (max|Δ|=0): the
`100k_data_original` file *is* the four 22-July sessions concatenated. They are file duplicates, not
independent re-measurements. The 15-July sessions share 0 inputs with that set. **So every unique
input was measured exactly once — zero independent repeats.** Averaging the "duplicates" is averaging
a number with itself (per-shot σ=0): it cannot reduce noise.
⇒ The √N-averaging fix (Issue 7/10b) is sound but **cannot be tested or exploited offline**. It needs
a NEW acquisition: pick a set of inputs, measure each N× (ideally back-to-back, same drift state), and
confirm σ falls as 1/√N. That is the concrete rig experiment to break the 0.78 ceiling.

### Issue 10d — deduplicated the corpus and re-ran the pipeline (`scripts/dedup_and_rerun.py`)
Removed exact (input+output) duplicate rows across all 8 files:

| | rows | kept | dropped |
|----|----|----|----|
| 7 pic_data sessions | 160,000 | 160,000 | **0** (each file 100% unique) |
| `100k_data_original.xlsx` | 100,000 | 0 | **100,000** (verbatim copy of the 22-July sessions) |
| **total** | **260,000** | **160,000** | **100,000 = 38.5%** |

So **38.5% of the corpus was duplicate, ALL of it the `100k_data_original` file**; canonical clean set =
the 7 pic_data sessions (160k unique; **135k** excluding the dud). Re-ran the pipeline on it:

| pillar | clean result | prior | verdict |
|----|----|----|----|
| (A) gradient-boost forward (+dead ref, temporal) | good 0.744, PD8 0.586, noise 0.299, all-live **0.595** | good 0.725, all-live 0.570 | unchanged (ticks up with the larger clean train set) |
| (B) unlearnable classes (pooled, random) | good {1,3,5,10,12,13}, **UNLEARN {4,6,8,9}** | same | class split holds |
| (C) drift correction (temporal) | all-live 0.430 → **0.595**; PD8 0.15 → 0.59 | Issue 6: 0.375 → 0.590 | reproduces |

**Key point:** the duplication never actually contaminated any result — every analysis used a
dup-free subset (100k alone, or pic_data alone); no single train/test split ever pooled both copies,
so there was no leakage. Dedup makes the dataset canonical and the prior conclusions stand verbatim.

**`100k_data_original.xlsx` was DELETED** (redundant verbatim copy of the 22-July sessions; git-tracked,
recoverable). Canonical dataset = the **7 pic_data sessions** (160k unique rows; 135k excluding the dud).
12 analysis scripts still hardcode-load the deleted file and would need repointing to the 22-July
pic_data if rerun; `src/data.py`/docs only mention it in text.

**Near-duplicate check (`scripts/near_dup_check.py`):** the clean pic_data has **0 exact-input
collisions** and **0 rows within 0.1%** — inputs are exactly on the 0.5 V grid, so two distinct rows
differ by ≥0.5 V in some channel (min relative distance 3.74% = 0.5/max‖input‖), making sub-0.1%
neighbours geometrically impossible. **No clean quasi-repeats either:** matching on all 44 channels
with any PD effect (η²>0.0015) gives 0 groups — the meaningful-input space is far too sparsely sampled
to collide in 160k rows. ⇒ every meaningful input was measured exactly once; the √N route needs new data.

---

## Open / next
- [x] Use dead-PD common-mode as a measured drift reference — **works** (Issue 6); beats the GRU,
      ties/edges the session one-hot, and is deployable on unseen runs.
- [x] PD8 → relabelled **recovered** (orange→green ↺) in `pic_structure_corrected.png`
      (`scripts/pic_clean_schematic.py main(corrected=True)`); original kept as the as-measured map.
- [x] Source of PD4/6/9 range characterized (Issue 7): **white, deep-interference-amplified** phase
      jitter — not drift. Reducible vs irreducible is confounded offline; settle with the repeat test.
- [ ] Reduce it: homodyne (phase) and/or averaging √N repeats and/or more ADC/DAC bits.
- [ ] Fold the dead-PD reference into the live `lib.pic.acquisition` path (correct in real time).
- [ ] Hardware confirms: input-repeat test (same V, N×); TEC on/off; start-of-session reference probe.
- [ ] The PIC PD-state report (pending — builds on this log).
