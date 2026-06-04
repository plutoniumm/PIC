# PIC — Results Summary

Scan-friendly digest of every key finding. Each block: a **table with the actual numbers**,
**Look** (what to read in the table), **Means** (the conclusion), **Fix** (what to do).
Full detail + methods in `progress.md`; reproduce with `scripts/*.py`.

**TL;DR:** 4 of 14 photodiodes are dead. Of the rest, ~2 are perfectly predictable and ~3 are
nearly unpredictable. The single biggest problem is **drift** — the same voltages give different
outputs at different times (a few mV/hour, **20–60 % over a week**), so **no fixed
voltage→output calibration exists**. The chip must be temperature-controlled and/or re-probed
every session.

---

## 1. Four photodiodes are dead

| PD          | 0\* | 2\* | 7\* | 11\* | 1 (live) | 13 (live) |
|-------------|----:|----:|----:|-----:|---------:|----------:|
| mean (mV)   | 5   | 15  | 7   | 4    | **101**  | **198**   |
| std (mV)    | 3   | 10  | 4   | 2    | ~44      | ~50       |

**Look:** PDs 0, 2, 7, 11 sit at ~2–15 mV and barely move (std ~5 mV); live PDs swing to 50–240 mV.
**Means:** those 4 are damaged — they carry no signal. Outputs are 14 → **10 usable**.
**Fix:** already handled — `filter_adc_output` drops indices [0, 2, 7, 11]. ✓ (no action needed)

---

## 2. Predictability is wildly uneven across PDs (within one session)

| PD        | 1 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 12 | 13 | **mean** |
|-----------|--:|--:|--:|--:|--:|--:|--:|---:|---:|---:|---------:|
| R² ceiling|.78|.62|**.19**|.62|**.26**|.34|**.18**|.58|**.95**|**.98**|**0.55**|

**Look:** PD12/PD13 = **0.95 / 0.98** (near-perfect); PD4/PD6/PD9 = **0.19 / 0.26 / 0.18** (≈ random).
**Means:** some outputs are fully determined by the inputs; others (deep in the mesh, low signal)
are not predictable from voltages alone. Two different model families agree → it's a real ceiling.
**Fix:** for matmul, read results off the **high-R² PDs**; the hard PDs need homodyne (coherent gain
lifts weak signals off the noise floor) or a full interference model — not more fitting.

---

## 3. ADC/readout noise is NOT what limits us

| quantity                                   | value     |
|--------------------------------------------|-----------|
| dark-PD readout noise floor                | **5.8 mV**|
| R² ceiling IF that were the only noise     | **~0.95** |
| R² ceiling actually achieved by any model  | **~0.55** |

**Look:** a 5.8 mV noise floor would allow R²≈0.95, but every model caps at 0.55.
**Means:** the gap is **not** ADC/DAC noise — it's structured (interference complexity + drift, §5).
**Fix:** a better ADC helps weak PDs but won't lift the ceiling; the real culprit is drift (§4).

---

## 4. ⭐ THE BIG ONE — the chip drifts; no fixed calibration exists

Same input distribution measured at different times. Mean shift = drift (error bar ~0.3 mV).

**Same day** (4 runs, 22 July), PD output mean in mV:

| PD  | run 1 | run 2 | run 3 | run 4 |
|-----|------:|------:|------:|------:|
| 1   | 101   | 90    | 86    | 88    |
| 6   | 68    | 60    | 54    | 58    |
| 4   | 48    | 43    | 38    | 41    |

**One week apart** (15 July → 22 July):

| PD  | 15 July | 22 July | change   |
|-----|--------:|--------:|---------:|
| 8   | 49      | 19      | **−60 %**|
| 6   | 107     | 60      | **−44 %**|
| 4   | 72      | 43      | **−41 %**|
| 1   | 124     | 92      | **−26 %**|
| 13  | 239     | 194     | **−19 %**|

**Model trained on one session, tested elsewhere** (R²):

| tested on…             | mean R²                          |
|------------------------|----------------------------------|
| its own held-out data  | up to **0.98**                   |
| later run, same day    | drops 0.05–0.30                  |
| the **other week**     | **NEGATIVE** (worse than guessing the mean) |

**Look:** identical inputs give outputs that slide **5–13 mV within a day** and **drop 20–60 % over a
week**; a model that scores 0.98 on its own data goes **negative** a week later.
**Means:** **a time-invariant voltage→output map does not exist.** This is the proven hardware issue —
the chip is non-stationary. Calibration is only valid *within a single thermal session*. (Within a
session it works fine — see PD12/13 at 0.98 — so the physics isn't wrong; the *operating point* moves.)
**Fix:** (a) **turn on the chip's TEC** (it has a thermistor + thermoelectric cooler) to hold
temperature; (b) take a **start-of-session reference measurement** and fit a small per-session drift
correction; (c) otherwise keep it **hardware-in-the-loop** — re-measure on the chip every iteration
instead of trusting a stored model.

---

## 5. More compute / better physics does NOT help (on this data)

| model                                  | mean R² |
|----------------------------------------|--------:|
| dense black-box                        | 0.55    |
| same, 800 trees (high capacity)        | 0.57    |
| big neural net (low regularization)    | 0.29 (overfits) |
| physics model (cos/sin of v², sparse)  | 0.48    |

**Look:** going from 0.55 → 0.57 with 8× the capacity; physics ties the small net at 0.48.
**Means:** no model wins — the limit is the data/hardware (drift + noise), not under-modeling.
*Caveat:* this is power-only data; we have **not** yet collected homodyne (phase) data, so physics
is not disproven — it just hasn't had a fair shot.
**Fix:** collect **homodyne data with the TEC on**, then re-test the physics model. Don't spend compute
trying to beat 0.55 on the existing data.

---

## 6. Data-quality flags (know before you use the files)

| flag                          | evidence                                              | so what                                  |
|-------------------------------|-------------------------------------------------------|------------------------------------------|
| **Dud session**               | `25k_15july(2)`: all 14 channels flat at **~14.5 mV** | laser off / fiber unplugged — **exclude**|
| **Inputs are sparse drivers** | ~**3** of 64 channels drive each PD; ~**20** do nothing| model each PD from its few real inputs    |
| **No 2 V firmware clamp**     | raw-to-4 V scores **0.55** vs clamp-at-2 V **0.33**    | the data was taken without the clamp the current firmware applies — reconcile before trusting either |

**Look:** the `(2)` file's live PDs (4.9 mV std) look identical to its dead PDs (4.8 mV) — no light.
**Means:** one of the seven files is unusable; most input channels are inert; firmware clamp mismatch.
**Fix:** drop the dud; build sparse per-PD models; confirm which firmware produced the dataset.

---

*Sources:* §1–3,5,6 from `100k_data_original.xlsx`; §4 from the 7 `pic_data/` sessions.
Scripts: `scripts/prove_hardware_limit.py` (§1–3), `scripts/drift_analysis.py` (§4,6).
