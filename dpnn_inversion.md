# Can the DPNN surrogate characterize heaters by inversion? (offline study)

**Question.** Instead of sweeping each heater on the chip to measure its fringe
`P(V)=A+B·cos(φ₂V²+φ₀)` → `Vπ=√(π/φ₂)`, `V0`, sweep each heater *inside the trained DPNN*
(free inference) and fit the fringe to the model's predicted PD response. If the recovered
`Vπ/V0` match the hardware-swept values in `pic_data/pic_b_config.json`, characterization
could be done by training rather than sweeping.

**Verdict up front: no — not even as a rough seed.** The DPNN (R²≈0.58) learned a *smooth,
near-monotonic* surrogate of the aggregate heater→PD response. A single-heater sweep inside
it produces a **monotonic ramp, not a cosine fringe** (87% of channels are monotonic over
0–4 V), so there is essentially nothing to fit. Only **1 of 95** characterized channels
yields a physically real fringe from the model. This is model inference only — no hardware,
serial, or laser was touched.

---

## Method

- **Model load.** Set the PIC-B module globals before loading (else the checkpoint shape
  mismatches), exactly as `scripts/lora_retrain_hw.py` / `scripts/pic_gd.py` do:
  `T.FEAT = drivable channels (112)`, `T.DEAD = []`, `T.LIVE = range(14)`,
  `T.RESPONSIVE = B_RESPONSIVE`, then `T.load_ckpt("runs/dpnn_hw","relu",8)`.
  Loaded model: `din=116` (112 heaters² + dbm + bfm + temp + mA), `dout=14`,
  widths `[128,61,29]`, 25 k params, trained on 2072 hardware samples.
- **Feature row (`make_features`).** `F = [H[:,FEAT]², dbm, bfm, temp, mA]` — only the 112
  drivable-heater columns enter (squared; the linear V term was dropped in training because
  φ∝V²). No dead-PD refs on PIC B. Normalization is the frozen analytic scale from the
  checkpoint.
- **"Sweep heater h in the model."** Build a batch where DAC channel `h` ranges 0→4 V in
  0.1 V steps (41 points) and every other channel is held at a fixed BACKGROUND; predict the
  14-PD vector; take the PD the config says is `h`'s `fringe_pd` (apples-to-apples with the
  hardware fringe); fit with `src.census.robust_fit` (multi-start `fit_fringe`) and read
  `Vπ`, and `V0/Vnull` via `src.sweep_analysis.fringe_extrema`.
- **Operating point.** Held at the training regime, per the `pic_gd` recipe: `dbm =
  median(buffer) = 13.5`, telemetry `(bfm,temp,mA)=(26.9, 25.1, 166.7)` = mean of buffer rows
  nearest that power. (`NOMINAL_TEL` is PIC-A and wildly off this chip, so it is **not** used.)
- **Ground truth.** The 95 drivable channels in `pic_data/pic_b_config.json` that carry a
  reliable hardware-swept `Vπ/V0/fringe_pd`.

### Backgrounds tested

| background | how | why |
|---|---|---|
| **zero** | all other heaters at 0 V | matches how many hardware fringes were first measured, but **OOD** for the model |
| **v0** (light-routed) | every characterized heater at its config `V0` | closer to a lit, light-routed chip |
| **rand** (marginal) | average the predicted PD curve over 16 random dense 0–4 V vectors, then fit | matches the *training* distribution (77% of channels nonzero per row) |

---

## Results

### Vπ recovery vs config (all 95 channels)

| background | corr(Vπ_model, Vπ_cfg) | median \|ΔVπ\| (V) | median rel. err | within 10% | within 20% |
|---|---|---|---|---|---|
| zero | 0.05 | 7.0 | 5.8× | 3% | 4% |
| v0   | −0.11 | 5.2 | 2.5× | 1% | 4% |
| rand | 0.19 | 15.1 | 6.9× | 0% | 1% |

The median \|ΔVπ\| of 5–15 V and relative errors of **several hundred percent** are the
signature of degenerate fits: with a monotonic input curve, `robust_fit` returns a
near-zero `φ₂` (very long period), so `Vπ=√(π/φ₂)` blows up to 40–60 V. These numbers are
not "rough Vπ" — they are non-fits.

### Why: the model gives ramps, not fringes

| metric | how | value |
|---|---|---|
| monotonic single-channel curves | sign of `diff(P(V))` constant over 0–4 V, zero bg | **83/95 = 87%** |
| recovered a REAL fringe | physical Vπ (0.5–4.5 V) + turnover in [0,4 V] + ≥15 mV swing | **1/95 = 1%** |
| model moves config `fringe_pd` ≥15 mV | ptp on that PD | 21/95 = 22% |
| model routes heater to the SAME PD as hardware | argmax ptp over 14 PDs == `fringe_pd` | **5/95 = 5%** |

Example predicted curves on the config `fringe_pd` (mV, referenced to V=0, zero background) —
every one is a smooth ramp where hardware sees a full oscillation:

```
dac  0 enc.split pd10 cfgVπ=2.83 swing_hw=216 mV : 0.0 → −5.4 mV, monotonic
dac  1 enc.split pd12 cfgVπ=2.86 swing_hw=193 mV : 0.0 → −41 mV, monotonic
dac 15 V         pd8  cfgVπ=4.01 swing_hw= 25 mV : 0.0 → +8.7 mV, monotonic
dac 34 sigma     pd8  cfgVπ=2.75 swing_hw= 63 mV : 0.0 → +23 mV, monotonic
```

The model is **not dead** — across the random training buffer its 14-PD dynamic range is
75–90% of the real chip's, and a single-heater move does perturb *some* PD (median max-over-14
swing ≈22 mV). What it lacks is the per-heater **cosine periodicity** and the correct
heater→PD routing. It reproduces the locally near-linear part of `A+B·cos(φ₂V²+φ₀)` in the
operating region but never learned the turnover/oscillation, because it was fit to dense
random vectors and optimizes an aggregate MSE (R²≈0.58) that is dominated by the smooth trend.

### Background comparison

All three backgrounds fail. `v0` (light-routed) tightens the relative error slightly (median
2.5× vs 5.8× for zero) but still yields no fringes. The **random marginal is worst** for
shallow heaters — averaging over 16 dense backgrounds washes out even the monotonic slope,
inflating median \|ΔVπ\| to 15 V. The one weakly-positive correlation (rand, gated, corr 1.0)
is spurious: it survives on n=2. No background recovers Vπ.

### Shallow vs deep

Hypothesis was that shallow input-stage heaters (single optical path) recover better than
deep V/U-mesh heaters (many interfering paths). **Not supported** — both are near-flat in the
model:

| stage | n | median model swing on cfg PD (zero / v0 / rand) | hardware swing | within-20% (best bg) |
|---|---|---|---|---|
| shallow (enc.split, enc.single) | 23 | 5.6 / 8.0 / 6.1 mV | 47.9 mV | 4% |
| deep (V, U) | 63 | 6.4 / 6.7 / 3.6 mV | 21.5 mV | 6% |
| middle (sigma) | 7 | 8.9 / 5.0 / 7.3 mV | 51.8 mV | 0% |

If anything the model **under-represents the shallow fringes most** relative to their true
strength: shallow heaters have the strongest hardware fringes (48 mV) yet the model gives only
5–8 mV of monotonic response there. Section identity does not rescue the method.

### V0 recovery (the null-find seed use-case)

| background | median \|ΔV0\| (V) | within 0.5 V |
|---|---|---|
| zero | 6.0 | 6% |
| v0 | 4.0 | 14% |
| rand | 4.2 | 7% |

`V0` (voltage at fringe max, the quantity a re-anchor null-find would warm-start from) is no
better — best case 14% of channels land within 0.5 V, i.e. no better than guessing within the
0–4 V band.

---

## Relation to the other inversion sense (control-side)

`scripts/pic_gd.py` (PICGD) already inverts this **same** surrogate in the *design* direction:
target PD output → voltage vector, via analytic backprop `d(objective)/d(volts)` through the
differentiable model. That direction **works on this surrogate** for exactly the reason
parameter extraction fails here: gradient descent only needs the smooth, well-behaved
input→output gradient — which the monotonic surrogate supplies cleanly — and it optimizes the
*whole* voltage vector against the aggregate response, never asking any single heater to
reveal its cosine fringe. Parameter extraction needs the per-heater oscillation the model
never learned. So the surrogate is usable for closed-loop control/design but not for pulling
out physical fringe constants.

---

## Honest verdict

**DPNN inversion cannot replace hardware sweeps, and at R²≈0.58 it is not even a usable seed
or prior for the parabolic re-anchor.** The expectation of "rough Vπ, good enough to seed but
not to null" is *not* borne out: the model produces monotonic single-heater responses, so
1/95 channels give a real fringe, correlation with the ground-truth Vπ is ~0, and the recovered
values are non-physical (tens of volts). It cannot supply Vπ, V0, or even the correct
heater→PD routing (5% agreement).

For rapid re-characterization, the working levers remain the hardware ones: the parabolic/
null-find re-anchor warm-started from the existing config `V0/Vnull` (`./do heaters reanchor`)
and the ~20-param affine drift adapter — both of which consume *fresh single-channel hardware
sweeps*, which is precisely what this study hoped to eliminate and cannot.

A prerequisite for the idea to ever work would be a surrogate trained to represent per-heater
periodicity — e.g. including single-channel sweep rows (not only dense random vectors) in
training, or a physics-structured model (`src.pic_neurophox`) whose parameters *are* the
phases. The current black-box MLP, optimized for aggregate MSE, is structurally the wrong tool
for parameter extraction.

---

## Assumptions & caveats

- **`make_features`.** Features are `[H[:,FEAT]², dbm, bfm, temp, mA]` with `FEAT` = the 112
  `elec=="ok"` channels; the sweep sets one channel's column and holds the rest at the
  background. Sweep is 0→4 V × 0.1 V (finer than the 0.5 V training grid; the model is
  continuous so this is fine and only helps the fit).
- **Operating point** held fixed at training-regime telemetry (dbm 13.5, buffer-mean bfm/temp/
  mA), matching `pic_gd`. Using `NOMINAL_TEL` instead would push telemetry OOD; the buffer-mean
  choice keeps the fixed inputs in-distribution.
- **Which background "won":** none. `v0` (light-routed) is marginally least-bad on relative
  Vπ error and V0 hit-rate, but all three land at ~0–4% within-20%, driven by the shared root
  cause (monotonic single-channel response), so background choice is second-order.
- **No hardware claim.** Every number here is DPNN inference against the frozen
  `runs/dpnn_hw` checkpoint; "config"/"hardware-swept" values are the pre-existing entries in
  `pic_data/pic_b_config.json`, not re-measured.
```
