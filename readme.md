# Device Comparison

1. [Heaters](#1-heaters)
2. [Photodiodes](#2-photodiodes)
3. [Phase reach](#3-phase-reach)
4. [Model quality](#4-model-quality)
5. [Cheapest upgrades, by measured leverage](#5-cheapest-upgrades-by-measured-leverage)

## 1. Heaters

`φ = π·V²/Vπ² + φ₀` on both. **Bold = better.**
`swing` = best-PD peak-to-peak over the sweep · `ASNR` = `swing/σ_slow` (σ_slow §2: 6x6 4.54, 4x4 3.25 mV)
· `score` = `rmse/amplitude` · `C` = `selectivity × N_pd`, `selectivity = max_pd(swing_pd)/Σ_pd(swing_pd)`,
1 = spread over every rail, N_pd = all on one · `bounded` = fitted period inside the swept range.

| | 6x6 PIC B | 4x4 PIC1A |
|---|---|---|
| pads on die | 240 → **120 nets** (mirror-shorted pairs) | 18 |
| DAC channels | 128 → 120 nets | 16 |
| drivable | **112 / 128 = 88 %** | 13 / 16 = 81 % |
| fringe fitted | **96 / 120 = 80 %** | 10 / 15 = 67 % |
| **yield** | **96 / 120 = 80 %** | 10 / 18 = 56 % |
| `Vπ` median | **2.86 V** (0.78–6.13) | 4.00 V (3.01–5.94) |
| `Vπ` CV | 38.1 % | **19.5 %** |
| span swept `(V_top/Vπ)²` | **1.96 π**, 91 % ≥1π, 41 % ≥2π | 0.56 π, **0 %** ≥1π |
| span at ceiling `(Vmax_ch/Vπ)²` | **3.06 π** @5 V, 98 % ≥1π, 90 % ≥2π | 0.96 π @40 mA, 30 % ≥1π, max 1.51 |
| `bounded` | 39 / 63 = 62 %; 13 / 96 op points extrapolated | **10 / 10**; every fringe a fragment |
| `Vπ` error | sign-swap inverted V0/Vnull on 34 / 96 | **±12 %** irreducible (24 % single trace) |
| `swing` median | 32.4 mV (p10 7.8, p90 83.1) | **139 mV** |
| **`ASNR` median** | 7.1, **p10 1.7** | **42.8** |
| `visibility` median | 0.286, 40 % >0.5 | **0.571**, 60 % >0.5 |
| `r²` median | 0.933, 54 % >0.9 | **0.998**, 100 % >0.95 |
| `score` | 0.148 | **0.010** |
| `C` | 1.93 | 1.78 |
| `R`, `P_π = Vπ²/R` | **never measured** | **13 / 16**: 56–62 Ω ×4, 114–119 Ω ×9 → **182 mW** median (119–301) |
| phase / DAC code @`Vπ` | 0.0096° | **0.0069°** |
| settling | τ 0.7–1.2 s, t99 3–5 s | **0.5 s**; TEC 300 s |
| stationarity | cross-session R² ≤ 0, −20–60 %/wk, no cooler | **25.032 ± 0.089 °C**, 0 % railed |

6x6 wins yield and reach; 4x4 wins every quality metric. `ASNR` p10 1.7 = the weakest tenth of 6x6
heaters move their best PD by under 2× the chip's own drift.

| 6x6 gate | n | lost to |
|---|---|---|
| driving | 112 / 128 | 11 dead (39, 51, 62, 98, 100, 111, 118, 120, 124–126); 4 shorted @1.25 V (112–115); 1 weak (99) |
| optically responsive | 95 | 44 / 112 move no PD at 0 V; `V0`-hold recovers most, Σ 1123 → 1964 mV |
| reliable fringe | 96 | 93 ok, 3 weak, 13 extrapolated |

| 4x4 gate | n | lost to |
|---|---|---|
| DAC driver | 16 / 18 | DAC81416 = 16 ch |
| `R` known | 13 | H18/H14/H6 never ohmmetered, staged 1.5 V |
| fitted | 10 / 15 | DAC 10, 14 at vis 0.01–0.02, SNR 4 — under-driven, not broken |
| mesh role known | 6 | φ/α assignment r = +0.50 vs 1600 transfers; shipped table −0.04 = chance |
| recovered by 40 mA ceilings | +2 | at 30 mA / 1.5 V the 60 Ω heaters bought 0.13 π; clamp is per channel in `VOLTAGE_MAX_CH` **and** `VMAX[]` |

| 6x6 by stage | `enc.split` | `enc.single` | `V` | `sigma` | `U` |
|---|---|---|---|---|---|
| drivable | 16 | 8 | 42 | 9 | 33 |
| characterized | 16 | 7 | 38 | 7 | 25 |

PIC A, same die: 60 / 120 nets, 63 / 128 fits, Vπ 2.68–4.47 median 2.90 sd 0.21 → **CV 7 %**,
rejected to 48.9 V. Stricter selection than PIC B's 38 %, but same process — most of that 38 % is
fit uncertainty, not silicon.

Per-channel 4x4 heater census (R, Vπ ± sd, P_π, swing): `4x4/README.md`.

## 2. Photodiodes

Absolute mV do not compare (TIA gain, ADC ref, launch power). All normalised. **Bold = quieter.**
`σ` = sd of repeated reads, state held · `FS` = per-PD 99.9th pct, same epoch · `%FS` = `100·σ/FS`
· `CV` = `100·σ/μ`, lit reads `μ > 0.1·FS` · `ENOB` = `log₂(FS/σ)` vs 10 nominal · `σ/swing` = inverse
`ASNR` · `σ/Σ` = σ over total power that read · LSB 6x6 4.888 mV (10-bit/5 V), 4x4 2.502 (10-bit/2.56 V)
· `q` = `LSB/√12/√AVG_N` = 6x6 0.446 (AVG_N 10), 4x4 0.181 mV (AVG_N 16).
**Slow** = held 15 s–55 min, the regime programs run in. **Fast** = back-to-back, ~0.1 s.

| slow | 6x6 @+15 dBm | 4x4 vendor | 4x4 current @+8 dBm |
|---|---|---|---|
| `σ/LSB` med | 0.93 | **0.66** | 3.38 |
| `%FS` med | 2.37 % | **0.76 %** | 1.36 % |
| `%FS` mean | 3.66 % | **0.69 %** | 1.43 % |
| `CV` med | 4.77 % | **2.79 %** | — |
| `ENOB` med | 5.4 | **7.0** | 6.7 |
| **`σ/swing`** | 18.7 % | **1.6 %** | 3.7 % |
| `σ/Σ` | **0.234 %** | 0.905 % | 1.17 % |
| σ med | 4.54 mV | 3.25 mV | 8.45 mV |
| n | 196 file×PD, 14 PDs | 32 traces × 900 reads @1 s | 4 (16 reads @1 s) |

| fast | 6x6 (AVG_N 10) | 4x4 (AVG_N 16) |
|---|---|---|
| `σ/LSB` med | 0.27 | **0.060** |
| `%FS` med | 0.53 % | **0.02 %** |
| `ENOB` med | 7.5 | **12.1** |
| σ med | 1.32 mV | **0.15 mV** |
| `σ/q` | 3.0× | **0.83×** |
| slow / fast | **3.4×** | 56× |

| uniformity | 6x6 (14 PDs) | 4x4 vendor (4) | 4x4 current (4) |
|---|---|---|---|
| σ span | 2.29–21.2 mV | **2.87–3.71** | 1.30–18.4 |
| max / min | 9.3× | **1.3×** | 14.2× |
| `%FS` span | 1.01–6.59 % | **0.37–0.88 %** | 0.22–2.79 % |
| worst | pd3 6.59 %FS, pd10 4.56, pd1 4.55, pd12 4.17 | PD1 0.88 % | PD0 2.79 %, PD3 2.34 % |

| range | 6x6 | 4x4 |
|---|---|---|
| PDs / dead | 14 / **0** (PIC A: 4, {0,2,7,11}) | 4 / **0** |
| ADC | 10-bit / 5.0 V, 10 sweeps | 10-bit / 2.56 V int, 16 sweeps |
| max @+5 / +10 / +15 dBm | — / — / 378 mV (per-PD 82–378) | **336** / **1128** / not run above +13 |
| ADC span used | 7.6 %, capped by +15 dBm PD damage | **44 %**, no clipped read |

- 4x4 quieter on every per-detector measure: 1.7× `CV`, 12× `σ/swing`, +1.6 bits. Per-read
  resolution is a wash (ENOB 5.4 vs 7.0) — the edge is 12× more signal per heater, not the PDs.
- `σ/Σ` is the only 6x6 win, an artefact of 14 rails vs 4.
- 6x6 fast σ = one **unaveraged** LSB/√12 (1.41 mV) despite 10 firmware sweeps; they take ~1.5 ms,
  too fast to average 1/f. 4x4 sits at its post-average floor.
- 4x4 PDs match to 1.3×; the current rig's 14.2× is PD0/PD3 degraded on **this bench** (PD0 dark
  pedestal 37.8 mV). 6x6's 9.3× is the chip — pd3/pd10/pd12 are its brightest and its noisiest.
- 4x4 slow column = **vendor bench** (5 V ref, 5-sample avg, 6 wired DACs). Current bench 1.8× worse
  normalised, all PD0/PD3, still beats 6x6 on `%FS` and `ENOB`.
- +8 dBm FS = measured +5 dBm table × 6.3/3.2 mW. +15, +5, vendor measured directly.
- 6x6 slow σ = noise **plus** drift; no cooler, no other way to hold a state.
- **Dead-PD set contested.** Vendor {0,2,7,11}; Pranav's responsive set includes 7; base-error data
  {0,5,9,11}; Anagha fits ADC13 at R² 0.984 (`Anagha/README.md` §1). On PIC B all 14 respond (pd0
  198 mV, pd2 82, pd7 360, pd11 180) — `pic/config.py:16` discards four working detectors.
- **6x6 dark anomaly.** No optical input, heaters still move PDs 0.5–0.7 V ≈ 3× the whole optical
  range. Vanishes when unplugged. Ribbon pickup, unexplained.
- 4x4 PD0 usable but **rank by SNR, never amplitude** — amplitude ranking gave eight false heater
  identifications in one afternoon.

## 3. Phase reach

| | 6x6 PIC B | 4x4 |
|---|---|---|
| Vπ | median **2.86 V** (0.78–6.13) | **3.0–5.9 V** on fitted channels |
| Ceiling | **5.0 V** firmware clamp (`pic128.ino:52`); the census swept to 4.0 V | 2.25–4.75 V, per channel from its own I·R at 40 mA |
| Span reached | swept: median **1.96 π**, 91 % reach π, 41 % reach 2π. At the 5 V clamp: median **3.06 π**, 90 % reach 2π | 0.39–1.51 π; **3 of 10 exceed π; none reaches 2π** |
| Vπ confidence | sign-swap bug inverted V0/Vnull on **34 of 96** | **±12 %** irreducible; 24 % one trace, 12 % eight jointly |
| per-config drift correction | **0 % recovered** | **75 %** of 12 h drift, mean abs ΔT 0.0227 → 0.0056 |

- 6x6 wins reach; neither knows its phase. The 4x4's ±12 % is structural — π needs ~4.5 V on a
  120 Ω heater, so the information is not in the measurement.
- Aliased fits are indistinguishable by residual: a true 11 π span fitted an **84 π** alias at
  r² = 1.0000 with a *better* residual. Gates `characterize.resolvable_span` and
  `src.census.robust_fit` exist for this — do not remove them.


## 4. Model quality

| | 6x6 PIC B | 4x4 |
|---|---|---|
| best HW R² | 0.576 mean, 0.625 good PDs | **0.976** (28k) / **0.890** (540 fresh, 8 ch) |
| params | 25,063 | 2,753 |
| samples | 2,072 | 28,000 / 540 |


## 5. Cheapest upgrades, by measured leverage

| Chip | Change | Effect |
|---|---|---|
| 6x6 | **die TEC** | R² 0.70 → 0.86 on the predecessor half — the only lever with direct evidence |
| 6x6 | re-align input fibre | +11 dB; stops operating at the PD-damage ceiling |
| 4x4 | ohmmeter DAC 6/8/11 | 3 staged → drivable; unblocks 2 φ heaters + the role fit |
| 4x4 | raise heater current limit | span ∝ V²; ×1.25 → k=3 from 0.138 to 0.020 |
| 4x4 | higher-res ADC or larger Rf | TIA 37 kHz vs 100 Hz sampled — 3 orders of gain-bandwidth to trade, cheaper than laser power |
| both | external splitter | 6x6 nothing; 4x4 the last 2 first-column external phases need 2 ports lit at once |
