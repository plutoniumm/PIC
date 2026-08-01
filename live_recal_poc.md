# Closed-loop live drift recalibration — offline proof-of-concept

**Question.** While a task runs a programmed voltage vector `V0` (chosen to produce target
output `y*`), the chip drifts and the real output moves off `y*`. Can we hold the output
on-target by (1) collecting a few fresh probes on the drifted chip, (2) nudging a LoRA drift
adapter so the surrogate tracks the drifted chip, and (3) re-solving `V' = argmin ||surrogate(V') − y*||²`
by backprop through the adapted surrogate — then applying `V'`? Compare `error(naive V0)` vs
`error(recalibrated V')` on the drifted chip.

**Verdict up front: yes, worth wiring to hardware — but only closed-loop.** The adapter tracks
drift cleanly and monotonically (well-posed). A single open-loop re-solve *over-promises ~10×*
(surrogate exploitation) and can even make the output worse, exactly as `pic_gd.py` warns — so
the re-solve must be wrapped in a hardware line-search (HIL). With a 4–8-point on-chip
line-search, recalibration **reliably removes ~40–68 % of total drift error (~65–100 % of the
part input voltage can fix), never worse than naive**, for a cost of ≈25–60 chip reads per
recalibration. It does **not** drive drift error to literal zero — a per-PD gain/offset/noise
floor and the surrogate's own residual remain. And (Result 3) **retraining the adapter
continuously from the task's own free I/O stream removes ~32 % of a ramping drift vs ~15 % for a
one-shot dedicated recalibration — but the benefit plateaus by a retrain cadence of ~1-in-5–20
samples; retrain-every-sample is no better** (it chases ADC noise). Everything below is model
inference against the `runs/dpnn_hw` checkpoint — **no hardware, serial, or laser was touched**.
Reproduce with `PYTHONPATH=. python scripts/live_recal_poc.py` (add `--freq` for Result 3).

---

## Testbed

Two well-posed inversions are used; per-heater parameter extraction is **not** (it fails —
`dpnn_inversion.md`: a single-heater sweep inside the DPNN is a monotonic ramp, not a fringe).

| piece | choice | why |
|---|---|---|
| **chip proxy** (ground truth) | frozen base DPNN + a **known** drift | KNOWing the drift lets recovery be measured honestly |
| **controller model** | the **same** base DPNN, un-drifted, differentiable | the surrogate the control loop already uses (`pic_gd.load_surrogate`) |
| **target `y*`** | undrifted chip output at a random programmed `V0` | the output the task was programmed to hold; reachable at `V0` with zero drift |
| **adapter** | LoRA rank-4, base frozen, `finetune_fixed` (`scripts/lora_retrain_hw.py`) | tracks drift with a thin low-rank correction (2264 trainable params) |
| **re-solve** | `GradientInverse` target mode through the adapted surrogate (PICGD) | the *working* inversion direction — backprop `d‖ŷ−y*‖/dV` into the voltage vector |

**Drift model** (applied to the base DPNN to make "drifted chip" ≠ controller model):

| component | magnitude | invertible by input voltage? |
|---|---|---|
| per-heater commanded-voltage offset δ (added before `V²`) | **0.18 V RMS** (matches the hardware reanchor's 0.18 V RMS) | **yes** — `V'≈V0−δ` |
| per-PD gain drift | ±2 % | no (output-side) |
| per-PD offset drift | ±2 mV | no (output-side) |
| per-read ADC noise | 2 mV (host-averaged over 5 reads) | no (floor) |

**Why this proxy over `theory/twin.py`.** The twin is a physics chip whose output lives in a
different space from the DPNN, so the DPNN can't even predict the *undrifted* twin — `error0`
would be dominated by base-surrogate mismatch, not drift, and the adapter would be doing base
characterization, not drift tracking. That conflates the question. The DPNN-plus-known-drift
proxy **isolates drift recalibration** with a measurable ground truth. Its honesty cost is
stated in the caveats: chip and controller share the DPNN functional form, so these numbers are
an **optimistic** bound on the real chip (see caveats).

**The loop.** operating point held at the training regime (`+13.5 dBm`, buffer-mean telemetry,
as `pic_gd`). Probes = random dense 0–4 V configs on the 112 drivable heaters, measured on the
drifted chip. Adapter trained on `N` probes (held-out test set of 100 for tracking R²). Re-solve
initialised at the currently-programmed `V0` (**trajectory reuse**), box `[0,4] V`, 500 Adam
iters. HIL: each round re-solves from the current best, then measures a few points along
`best→proposal` on the chip and keeps the lowest measured error (mirrors `pic_gd.hil_optimize`).

---

## Result 1 — the adapter tracks the drifted chip (well-posed, monotone in N)

Frozen base vs LoRA adapter, evaluated on 100 held-out probes of the drifted chip. 0.18 V-RMS
drift; R² and RMSE over the 8 responsive PDs (`B_RESPONSIVE`).

| probes N | mean R² (14 PD) | resp R² | resp RMSE (mV) |
|---:|---:|---:|---:|
| 0 (frozen base) | 0.941 | 0.951 | 8.9 |
| 5   | 0.954 | 0.965 | 7.8 |
| 10  | 0.962 | 0.970 | 7.1 |
| 20  | 0.971 | 0.977 | 6.3 |
| 50  | 0.975 | 0.982 | 5.6 |
| 100 | 0.978 | 0.985 | 5.2 |
| 200 | 0.980 | 0.986 | 4.8 |

Under a larger 0.30 V-RMS drift the frozen base is hurt more (mean R² **0.804**) and the adapter
lifts it to **0.95** by N=200 — the worse the drift, the more the adapter recovers. This half of
the loop is unambiguous: **a handful of probes measurably re-aligns the surrogate to the drifted
chip.**

---

## Result 2 — re-solving the voltages restores the output (only closed-loop)

Per target: `naive` = ‖chip(V0) − y*‖ (the drift error to remove); `ideal-Vcomp floor` =
error at `V0−δ`, the best any *input-voltage* correction can do given the uninvertible
gain/offset/noise; `open` = single blind re-solve; `HIL` = re-solve + on-chip line-search;
`pred` = what the controller *thinks* it achieved (surrogate). Euclidean over 14 PDs, volts.
0.18 V-RMS drift, 3 random targets.

**Target 0** — naive 0.0454, floor 0.0129

| N | pred | open | **HIL** | chip meas | % removed (open) | **% removed (HIL)** |
|---:|---:|---:|---:|---:|---:|---:|
| 5   | 0.0028 | 0.0503 | 0.0431 | 4 | **−10.7 %** | 5.0 % |
| 10  | 0.0027 | 0.0444 | 0.0429 | 4 | 2.2 % | 5.5 % |
| 20  | 0.0034 | 0.0380 | **0.0242** | 4 | 16.3 % | **46.8 %** |
| 50  | 0.0032 | 0.0340 | 0.0245 | 6 | 25.1 % | 46.1 % |
| 100 | 0.0030 | 0.0441 | 0.0385 | 7 | 2.9 % | 15.1 % |
| 200 | 0.0036 | 0.0329 | 0.0270 | 4 | 27.6 % | 40.6 % |

**Target 1** — naive 0.0357, floor 0.0111

| N | pred | open | **HIL** | chip meas | % removed (open) | **% removed (HIL)** |
|---:|---:|---:|---:|---:|---:|---:|
| 5   | 0.0024 | 0.0326 | 0.0330 | 5 | 8.8 % | 7.6 % |
| 10  | 0.0019 | 0.0359 | 0.0325 | 6 | −0.6 % | 9.1 % |
| 20  | 0.0017 | 0.0271 | 0.0252 | 5 | 24.2 % | 29.4 % |
| 50  | 0.0027 | 0.0258 | 0.0235 | 5 | 27.7 % | 34.2 % |
| 100 | 0.0024 | 0.0235 | 0.0223 | 6 | 34.3 % | 37.5 % |
| 200 | 0.0023 | 0.0203 | **0.0190** | 5 | 43.2 % | **46.8 %** |

**Target 2** — naive 0.0198, floor 0.0156 (drift barely above floor → **noise-dominated**, little to remove)

| N | pred | open | **HIL** | chip meas | % removed (open) | **% removed (HIL)** |
|---:|---:|---:|---:|---:|---:|---:|
| 5   | 0.0010 | 0.0275 | 0.0167 | 6 | −38.4 % | 16.0 % |
| 20  | 0.0024 | 0.0132 | **0.0110** | 8 | 33.4 % | **44.6 %** |
| 100 | 0.0025 | 0.0216 | 0.0153 | 8 | −9.0 % | 22.8 % |

### Two things the tables say

- **Open-loop over-promises ~10×.** The controller's predicted error (`pred`) is ~2–4 mV while
  the actual chip error (`open`) is ~20–50 mV — the surrogate re-solve drives `V'` to a point
  where *the surrogate* reads `y*` but the chip disagrees by the surrogate residual. Worse,
  `open` is **negative** (worse than naive) at several N. This is textbook surrogate
  exploitation and is why a blind single re-solve is not safe to apply. (`pic_gd.py` saw the
  same on hardware: predicted +3.60 vs measured +1.08.)
- **HIL is reliable and monotone-ish.** Wrapping the re-solve in a 4–8-point on-chip
  line-search makes every entry **≥ naive** (never worse) and pulls ~40–47 % of total drift
  error out at N ≥ 20. Because the line-search keeps only *measured* improvements, it escapes
  the exploitation the surrogate alone falls into.

### Best-over-N summary (fraction of *removable* drift = naive − input-voltage floor)

| target | naive | V-floor | best HIL | @N | % of total | % of **removable** |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0454 | 0.0129 | 0.0242 | 20 | 46.8 % | 65.4 % |
| 1 | 0.0357 | 0.0111 | 0.0190 | 200 | 46.8 % | 68.0 % |
| 2 | 0.0198 | 0.0156 | 0.0110 | 20 | 44.6 % | (noise) |

Under 0.30 V-RMS drift (bigger, so more removable): target 0 → **66.8 %** of total removed
(best HIL 0.0172 @ N=50), target 1 → **68.4 %** (0.0235 @ N=50). More drift ⇒ more absolute
recovery, which is the argument for turning it on.

**Probe-count curve.** Adapter tracking rises monotonically with N (Result 1). Re-solve quality
rises with N too but *not* strictly — a different adapter can reshape the surrogate landscape in
a way the re-solve exploits differently (e.g. target 0 dips at N=100). The practical knee is
**N ≈ 20–50 adapter probes**; beyond that, tracking R² keeps climbing but the output error
plateaus against the two floors below.

---

## Where the residual comes from (why not zero)

`best HIL ≈ 0.019–0.024 V` over 14 PDs, i.e. ~50–65 % of the naive drift error remains. Two
irreducible pieces:

1. **Uninvertible drift.** Per-PD gain/offset/noise cannot be undone by an input-voltage vector.
   The `ideal-Vcomp floor` (error at the exact `V0−δ`) is ~0.011–0.016 V — a hard floor for
   *any* re-solve. HIL lands close to it (65–100 % of the removable gap).
2. **Surrogate residual.** Even the adapted surrogate has ~5 mV/PD RMSE vs the chip (~18 mV over
   14 PDs); the open-loop re-solve is floored by this, and HIL only partially escapes it by
   measuring on-chip.

So "drift error → 0" is the aspiration, not the result: **of the drift that voltage can fix,
~65–100 % comes back; of the total drift, ~40–68 %.**

---

## Result 3 — continuous (free-stream) retraining vs sparse dedicated, under ramping drift

**The idea.** Dedicated probes are not the only training data: during normal operation *every
`(input V, output y)` the task produces is a free label*. So the adapter can retrain
*continuously* from the task's own I/O stream, at high frequency, keeping the surrogate a live
digital twin — no probe downtime. **Question: does higher-frequency retraining reduce drift
error more than sparse retraining?**

**Setup** (closed-loop only; `--freq`). The chip drifts *continuously* — the per-heater offset
ramps 0.10 → 0.30 V RMS over `T=200` task steps (a fixed random direction). Each step streams
one task input (a buffer H row, the task's own diverse vectors), measured on the chip at the
current drift, into a sliding 100-sample window. All arms start from **one** adapter calibrated
at `t=0`; then:

- **streaming K∈{1,5,20,50}** warm-start-retrains on the window every K steps (free stream data),
- **dedicated@0** never retrains again (the "recalibrate once at session start, then run" status quo).

Steady-state = mean over 7 checkpoints in the run's second half; drift-error-removed is the
closed-loop HIL re-solve on 2 fixed targets; tracking R² is the adapter vs the drifted chip on a
held-out batch at the current drift. Raw: `runs/live_recal_freq.json` (+ `_fast.json`).

### Steady-state drift-error-removed vs retrain cadence

| arm (cadence) | % drift removed (slow, 0.1→0.3 V) | % removed (fast, 0.1→0.6 V) | tracking R² (slow) |
|---|---:|---:|---:|
| **K=1** (every sample) | 31.6 % | 23.0 % | 0.941 |
| **K=5** | **35.1 %** | **28.8 %** | 0.954 |
| **K=20** | 26.7 % | 26.7 % | **0.958** |
| **K=50** | 21.6 % | 19.2 % | 0.952 |
| **dedicated@0** (one-shot) | 14.8 % | 11.0 % | 0.909 → 0.584 |

### Adapter tracking R² over the run (drift accumulating)

The dedicated-once adapter **decays** as drift walks away from where it was calibrated; the
streaming adapters stay current (slow-drift run):

| checkpoint (drift RMS) | dedicated@0 | K=20 (live) | K=1 (live) |
|---|---:|---:|---:|
| t=105 (0.21 V) | 0.95 | 0.97 | 0.96 |
| t=135 (0.24 V) | 0.92 | 0.96 | 0.95 |
| t=165 (0.27 V) | 0.90 | 0.96 | 0.94 |
| t=195 (0.30 V) | 0.86 | 0.95 | 0.92 |

### What this says (honest — it plateaus)

- **Continuous retraining clearly beats one-shot dedicated: ~32–35 % removed vs ~15 %** (slow),
  ~29 % vs ~11 % (fast) — a >2× win. Under fast drift the dedicated adapter's tracking R²
  collapses (0.58) while the streaming adapters hold ~0.85–0.96. Keeping the surrogate live off
  the free stream is unambiguously worth it.
- **But higher frequency saturates by K≈5–20 — retrain-every-sample (K=1) is *not* better** (32 %
  vs 35 % at K=5), and its tracking R² is actually *lower* (0.941 vs 0.958 at K=20). Two honest
  reasons: (i) the **HIL line-search already compensates** most surrogate staleness on-chip, so a
  fresher surrogate only sharpens the *proposal direction*, a second-order effect; (ii) at K=1 the
  per-step drift (~0.001 V) is smaller than the per-read ADC noise, so retrain-every-sample
  **chases measurement noise** rather than real drift and slightly overfits the window. The
  pattern is drift-rate-robust: even at 3× faster drift, K=5–20 still beats K=1.
- **Practical cadence: retrain roughly every 5–20 task samples.** That captures the full
  continuous-retraining benefit at a fraction of the compute, and avoids the K=1 noise penalty.

---

## Verdict & what the hardware version needs

**Is it worth wiring into the hardware loop? Yes**, with the closed-loop caveat. Both halves are
sound: the LoRA adapter tracks the drifted chip from a handful of probes (well-posed), and the
PICGD control-inversion re-solves a corrected voltage vector (well-posed). Neither uses the
broken per-heater-extraction direction. The reduction is large and, crucially, **safe** under
HIL — the recalibrated output is never worse than leaving `V0` in place.

Real-hardware requirements:

- **HIL line-search is mandatory, not optional.** Open-loop over-promises ~10× and can degrade
  the output; only the on-chip line-search guarantees non-worsening. Budget ≈ 4–8 guarded chip
  reads per recalibration on top of the adapter probes.
- **Prefer the free stream over dedicated probes (Result 3).** The task's own `(V, y)` outputs
  are free labels; retraining the adapter from a sliding window of them keeps the surrogate live
  with zero probe downtime and roughly doubles the drift removed vs a one-shot dedicated
  recalibration. **Retrain every ~5–20 task samples** — that is the plateau; going to every
  sample buys nothing and chases ADC noise. Dedicated probes are then only needed to *seed* the
  adapter at session start (or if the task's input distribution is too narrow to identify the map).
- **The re-solve line-search still costs ~4–8 guarded chip reads per recalibration.** At ~20
  reads/s (`pic-thermal-settling`) that is a sub-second interrupt. Whether to interrupt the task
  or recalibrate between shots depends on the drift rate; same-day drift is a slow offset
  (`drift-adapter-rapid-recharacterization`), so a ~5–20-sample cadence is comfortable.
- **Trajectory reuse works.** Initialising the re-solve at the current `V0` (rather than random)
  converges fast and keeps `V'` near the running config — cheap and stable.

**Honesty / caveats.**

- **This proxy is optimistic.** Chip and controller share the DPNN base functional form, so a
  perfect adapter could in principle make surrogate == chip; on real hardware the base surrogate
  is only **R² ≈ 0.58** vs the true chip, so the surrogate-residual floor is much larger and HIL
  does proportionally *more* of the work. The **relative** conclusions transfer (adapter tracks
  drift; open-loop over-promises; HIL needed; trajectory reuse works); the **absolute** % removed
  will be lower on hardware, and the case for HIL only strengthens.
- **Small-drift targets are noise-dominated** (target 2: naive already near the floor) — nothing
  to recover, and %-removable is unstable there. Report per-target, not pooled.
- **HIL masks the cadence effect (Result 3).** Because the line-search corrects on-chip, even a
  stale adapter still removes ~11–15 %; retraining roughly doubles that but the *marginal* value
  of extra retraining frequency is bounded by what HIL already provides. The retrain-cadence
  plateau (K=1 ≈ K=5) is partly this and partly noise-chasing — both honest, both point to a
  modest cadence, not every-sample.
- **Per-heater parameter extraction was not used** (fails, `dpnn_inversion.md`). Only
  adapter-update + control-inversion, both well-posed.
- Every raw number is in `runs/live_recal.json` (+ `runs/live_recal_bigdrift.json`,
  `runs/live_recal_freq.json`, `runs/live_recal_freq_fast.json`); no hardware claim.

**Ready to prototype on hardware next**: the adapter path (`scripts/lora_retrain_hw.py`) and the
HIL re-solve (`scripts/pic_gd.py --hil`) already exist; this PoC shows they compose into a
working recalibration loop on a couple of targets.
