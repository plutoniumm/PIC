# PIC — Results Summary

Power-only (no-homodyne) characterization of the Section-A path. Inputs: 64 heater voltages on a
{0, 0.5, …, 4} V grid; outputs: 14 photodiode voltages, 10 usable. Three headline results: **4 PDs
are dead**, per-PD predictability spans **0.18–0.98 R²**, and **the chip is non-stationary
(20–60 % drift/week)** so no fixed voltage→output calibration holds. Methods in `progress.md`;
reproduce via `scripts/`.


## 1. Four dead photodiodes

*Metric:* per-PD mean and std of the output over the whole dataset, in mV, while inputs vary.
*Why:* a live PD must move with the inputs; one whose std ≈ the readout-noise floor (~5 mV) is dark.

|            | PD0\* | PD2\* | PD7\* | PD11\* | PD1 | PD13 |
|------------|------:|------:|------:|-------:|----:|-----:|
| std (mV)   | 3     | 10    | 4     | 2      | 44  | 50   |
| mean (mV)  | 5     | 15    | 7     | 4      | 101 | 198  |

PD0/2/7/11 don't respond to any input (std ≈ noise) → 14 outputs collapse to **10**. Already filtered.


## 2. Per-PD predictability ceiling

*Metric:* test-set R² of an inputs→PD regressor (gradient-boosted trees, 80/20 split). *Why:* R² =
fraction of output variance the inputs explain; an upper bound for any memoryless model. Two model
families agree, so it's a property of the data, not the fit.

| PD | 1   | 3   | 4   | 5   | 6   | 8   | 9   | 10  | 12  | 13  | mean |
|----|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|
| R² | .78 | .62 | .19 | .62 | .26 | .34 | .18 | .58 | .95 | .98 | .55  |

PD12/13 ≈ solved (.95/.98); PD4/6/9 ≈ unpredictable (.18–.26). The hard PDs sit deep in the mesh
(many interfering arms, low signal). → Read matmul results off the high-R² PDs; weak PDs need
homodyne gain, not more fitting.


## 3. The ceiling is not readout noise

*Metric:* the R² we *would* hit if the only error were ADC noise, = 1 − σ²/Var(signal), with σ
taken from the dead PDs (pure noise, no light). *Why:* separates a noise-limited chip from a
structure/drift-limited one.

| quantity                          | value     |
|-----------------------------------|----------:|
| readout-noise floor σ (dead PDs)  | 5.8 mV    |
| implied R² if noise were the limit| ~0.95     |
| R² actually achieved              | 0.55      |

Noise alone permits .95 but we get .55 → the gap is structure + drift, not the ADC. A faster
converter won't lift it.


## 4. ⭐ The chip drifts — no time-invariant calibration exists

*Metric:* per-PD output **mean** across sessions that sample the **same input distribution** at
different times (mV). *Why:* if inputs match, E[output] is constant *iff* the voltage→output map is
time-invariant; any mean shift is drift (sampling error on a 25k-row mean ≈ 0.3 mV, so mV shifts are real).

Same day, four consecutive runs:

| PD | run 1 | run 2 | run 3 | run 4 |
|----|------:|------:|------:|------:|
| 1  | 101   | 90    | 86    | 88    |
| 6  | 68    | 60    | 54    | 58    |
| 4  | 48    | 43    | 38    | 41    |

One week apart (15 → 22 July): PD8 −60 %, PD6 −44 %, PD4 −41 %, PD1 −26 %, PD13 −19 %.

*Metric:* R² of a model trained on one session and tested on others. *Why:* quantifies how fast the
calibration goes stale.

| tested on             | R²                          |
|-----------------------|-----------------------------|
| its own held-out data | up to 0.98                  |
| later run, same day   | drops 0.05–0.30             |
| the other week        | **negative** (worse than the mean) |

Same-day shift 5–13 mV; week-scale −20 to −60 %; a model scoring .98 on its own data goes negative a
week later → **no calibration survives past one thermal session**. (Within a session it works, so the
physics is fine — the *operating point* moves.) **Fix:** enable the chip's TEC; add a start-of-session
reference probe + per-session correction; otherwise stay hardware-in-the-loop (re-measure each iteration).


## 5. Capacity and physics don't move the ceiling

*Metric:* mean R² vs model class/size on the same data. *Why:* if adding capacity doesn't raise R²,
the limit is the data, not the model.

| model                              | mean R² |
|------------------------------------|--------:|
| dense black-box                    | 0.55    |
| same, 8× capacity (800 trees)      | 0.57    |
| large neural net, low regularization | 0.29 (overfits) |
| physics model (cos/sin of v², sparse) | 0.48 |

8× capacity buys +0.02; physics ties the small net → limit is hardware (drift + noise). Homodyne
(phase) data isn't collected yet, so physics isn't disproven — collect it with the TEC on, then re-test.


## 6. Data-quality flags

*Metric:* per-file/per-channel sanity checks. *Why:* one file is unusable and two settings affect how
the data must be modelled.

| flag                    | evidence                                                | consequence                          |
|-------------------------|---------------------------------------------------------|--------------------------------------|
| dud session             | `25k_15july(2)`: all 14 channels flat at ~14.5 mV       | laser off / fiber unplugged — exclude |
| inputs are sparse drivers | ~3 of 64 channels move each PD; ~20 do nothing        | model each PD from its few real inputs |
| no 2 V firmware clamp    | raw-to-4 V R² 0.55 vs clamp-at-2 V 0.33                 | dataset predates the firmware clamp — reconcile |


*§1–3, 5, 6 from `100k_data_original.xlsx`; §4 from the 7 `pic_data/` sessions.
Scripts: `prove_hardware_limit.py` (§1–3), `drift_analysis.py` (§4, 6).*
