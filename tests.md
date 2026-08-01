# tests.md — archived results of removed analysis/modelling experiments

Archive of the old-dataset (legacy 100k / pooled `pic_data`) software experiments and the
modelling helpers they used. All were software-only (no hardware) and have been deleted to
keep the tree focused on the live PIC-B pipeline. The code lives in git history — recover any
file with `git log --diff-filter=D -- <path>` then `git show <commit>^:<path>`. Numbers below
are the results those scripts printed; they are preserved so nobody re-runs a dead experiment.

Context (from `Readme.md` Findings): the Section-A learnability ceiling is **R² ≈ 0.55–0.78**,
agreed by tree, MLP and physics models → a property of the *data*, not the fit. Dark-PD floor
~5.8 mV would permit R² ≈ 0.95, so readout noise is not the limiter. Chip is non-stationary
(same-day mV offset; 20–60 %/week collapse; cross-session V→PD R² ≤ 0). Per-PD classes: good
{1,3,5,10,12,13} 0.53–0.97; noise/unlearnable {4,6,9} 0.14–0.25 (σ∝√signal); recovered {8}
0.12→0.65 via dead-PD ref; dead {0,2,7,11} ~0 (used as drift monitor).

## Forward / surrogate modelling

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `forward_search.py grid` | Parallel config grid (MLP/HistGB/ridge × features × scalers) for the forward map, 7 PDs | Best test R² clusters at the **0.78** band; no config beats it; cross-day raw R² negative, affine recovery only | Capacity/knobs don't move the ceiling |
| `forward_search.py replicate` | Reproduce Anagha's exact net (64/192/128 GELU, MinMax[-1,1], batch32, 200ep) | Recovers her **≈0.78** avg R² on 7 PDs (single 25k run and pooled 100k) | Baseline confirmed reproducible |
| `forward_search.py regimes` | HistGB single-session vs single-day (train 3 runs → run 4) reliable-PD R² | Single-session > single-day > cross-day; drift, not fit, sets the gap | Accuracy regime = f(session distance) |
| `cos_model.py` | Cos-activation nets (additive Σ A·cos(w v²+…) + cos-MLP) vs polynomial, random & temporal splits | Interference structure representable but no better than tree/MLP; same 0.55–0.78 band | Sum-of-cos structure ≠ higher ceiling |
| `nn_variants.py` | Model-family bake-off on 100k: quad-OLS, MLP, conv-local, MLP+time, drift-GRU | No static family beats the ceiling; temporal split drops all; only GRU (sees yₜ₋₁) helps | Static V→PD map is saturated; the lever is state |
| `neurophox.py` (script) | Faithful differentiable U·Σ·V optical model (`FaithfulPIC`) + stochastic `FuzzyPIC` on 100k | Physics model ties black box (sharp mean R² ≈ **0.407** on 20k subset); fuzzy model gives calibrated per-PD σ, sorts noise PD4/6/9 to high-σ end | Physics-faithful ≈ black box → ceiling is in the data |
| `src/forward.py` | `BlackBoxForward` = per-output HistGB on [v, v²] (the surrogate every experiment used); `GreyBoxForward` physics stub | Workhorse surrogate; grey-box never implemented (needs topology + homodyne char) | Superseded by `src.census` / live pipeline |
| `src/mzi.py` | MZI 2×2 transfer matrices, Clements mesh composition, φ = φ2·v²+φ0 | Forward composition + closed-form check only; Clements-nulling inverse was TODO | Modelling helper, unused by live path |
| `src/influence.py` | η² (ANOVA main-effect) DAC→PD influence map | Coarse routing map; under-counts purely-interferometric channels | Superseded by empirical census |

## Drift & noise studies

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `drift.py marginal` | Same-day 22july run-to-run + week-scale 15→22july mean shifts; cross-session generalization gap | Optical excess drift above electronic; **20–60 %/week** collapse; within−cross R² gap is irreducible to any static model | Chip has a session hidden state |
| `drift.py mechanism` | Within-run PD slope vs row index; cross-PD correlation of the trend | Drift direction is **run-dependent** (not "always up"); high cross-PD corr ⇒ common-mode; dead PDs drift too | Drift is largely common-mode |
| `drift.py deadref` | Dead PDs {0,2,7,11} as a live common-mode reference, random vs temporal split | Feeding dead PDs as inputs lifts mean R²; **PD8 recovered**; PD4/9 stay (not common-mode) | Free drift reference works |
| `drift.py decompose` | Same-day vs across-day drift + rapid re-char adapter ladder | Same-day = additive **offset** (map stable); across-day = **map drift** (affine can't fully close); ~20-param affine on ~100 samples flips cross-day **−0.44→0.48**, ridge learned-delta needs 500+ | Two-tier drift; cheap adapter for same-day |
| `drift.py rnn` | Static vs +session-onehot vs GRU (h from [vₜ, yₜ₋₁]) on temporal 70/30 | GRU tracks drift better than static, but unlearnable PD4/6/8/9 stay low | Recurrence helps drift, not the noise floor |
| `dedup.py nearcheck` | 0.5 V grid exactness, exact-duplicate & quasi-repeat groups | Inputs exactly quantised; ~**38 %** exact-duplicate rows; no independent repeats on the driving channels | Offline √N averaging untestable |
| `dedup.py rerun` | Re-run forward/drift pipeline after dedup | Class split unchanged; prior pre-dedup tree (good **0.725**, PD8 0.565, noise 0.262, all-live **0.570**) holds; drift lift 0.375→0.590 survives | Conclusions robust to duplicates |
| `noise.py characterize` | Residual shape + scale law σ²(μ)=a+bμ+cμ² per PD | Residual ~Gaussian; noise PDs dominated by **multiplicative/phase** term (σ∝√signal), not additive read noise | Deep-PD noise is phase jitter |
| `noise.py allpds` | Per-PD SNR = signal/noise, R² spread | Bimodal usable-vs-noise split with a **wide empty R² gap** (not an arbitrary line) | Good/noise split is real |
| `unlearnable_source.py` | Decompose PD4/6/9 residual after V+deadPD: white-vs-structured, drive-power dep, shared mode | Residual mostly white (low autocorr), weak Σv² dependence, no strong shared cross-PD mode | Irreducible per-shot noise, not missing feature |

## Learnability & hardware-ceiling

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `pd_learnability.py` | Actual-vs-predicted output spread of the 64-ch black box over all 14 PDs | Reproduces classes: damaged {0,2,7,11}=floor, learnable R²>0.4, weak 0.2–0.4, UNLEARNABLE <0.2 | Predicted-spread collapse ⇒ unlearnable |
| `power_budget.py` | Square-law field chain: does PD4/6/9 signal clear the ADC floor? | Measured dark floor ≈ **5.8 mV**; signal clears it, but the limiter is noiseSD (σ∝√signal), **not** the 5 mV floor | Floor is not the ceiling |
| `prove_hardware_limit.py` | Model-free variance bounds (exact-duplicate, CV-lookup, within-cell, dark floor) | Dark floor would allow R² ≈ 0.95; CV-lookup peak reproduces split (PD12/13 ≈ **0.95**, PD4/6/9 ≈ **0.05**); within-cell spread conflates crosstalk + hidden state | Dataset **cannot** prove a hardware ceiling; needs hardware repeat protocol |
| `hidden_state_test.py` | Input-only vs +session-onehot vs within-session per-PD R² | Session hidden state does **not** rescue PD4/6/9 (within-session fit doesn't lift them) | Those PDs are noise, not session state |
| `tree_ablation.py` | Cumulative reverse ablation pushing 7 learnable PDs toward >0.90 | v→+v²→+deadPD→+tuned→+sessionID→+noise-PD→+interactions; 0.90 only reached by leaking session/noise-PD state; near per-PD single-shot ceiling otherwise | 0.90 needs oracle state, not honest inputs |
| `retrain_curve.py` | Held-out R² vs #training samples, tree (HistGB) vs MLP, 7 PDs | Both climb toward the **0.78** full-data NN benchmark; reports N to reach 0.65/0.70/0.75; wrote `slides/images/retrain_curve.png` | Sample-efficiency curve for retrain cost |
| `learn_compare.py` | Tree vs MLP vs physics on pooled `pic_data`, temporal split, +dead-PD ref | All three agree per-PD-class (good/PD8/noise/all-live) | Ceiling is a data property, model-agnostic |

## Prune / w2b architecture sweeps (dynamically-pruned MLP study)

The Week-2 Part-II study: is there a model between the tree (cheap retrain, lower ceiling) and
the dense MLP (higher ceiling, full retrain)? Answer: a dynamically-pruned MLP — dense-MLP
accuracy at tree-scale parameter count, LoRA-adaptable after drift. Pooled held-out R² band
across all variants ≈ **0.545–0.641**; tree ≈ 62,837 decision paths, dense 256-256-128 ≈ 134k
params. Depends on `src/prune.py` (a keeper).

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `prune_nn.py` | Pruned MLP vs tree vs dense MLP, pooled temporal split | Pruned reaches ≈ dense accuracy at ≈ tree-scale params; wrote `prune_nn.png` | Middle ground exists |
| `prune_retrain.py` | Cross-session retrain cost: train 2 same-day sessions, adapt to a different day | K=0 cold collapse (cross R² ≤ 0) reproduced; pruned warm-start fine-tune adapts with fewer fresh samples than from-scratch tree/dense | Warm-start pruned = cheap adapt |
| `w2b_pooled.py` | Ceteris-paribus tree/dense/pruned, same [v,v²,deadPD]+ELU; tree "params"=2·internal+leaves | Pruned matches dense at many× fewer params; the shipped pooled numbers for the slide figures | Only the model class differs |
| `w2b_dense_sweep.py` | Prune the dense net's own 256-256-128; knob sweep (std_k, epochs, prune-every, warmup, wd, min-neurons) | k has an interior optimum on the dense shape (unlike the over-wide net) | Gentle prune has a sweet spot |
| `w2b_arch.py` | Does the starting architecture matter, or only that pruning happens? (dense/wide3/deep-narrow + LoRA ranks) | Starting arch **barely matters**; pruning lands in the same R² band regardless | Prune, don't design |
| `w2b_stdk.py` | Aggressiveness knob k (threshold exp(μ−k·σ); larger k prunes less) | Larger k keeps more neurons and **gains nothing** | Prune hard; it's free |
| `w2b_retrain.py` | Cross-session retrain ceteris-paribus + LoRA arm (full fine-tune vs rank-r LoRA) | Small LoRA ranks win when fresh data is scarce; full fine-tune needs more | Low-rank adapt suffices post-drift |
| `w2b_figs.py` | Slide-figure generator (reads `/tmp/w2b_*.json`), no training | Wrote `slides/images/week2b/*.png` | Plot-only |

## Legacy Section-A (Monte Carlo, Anagha) & per-heater char

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `Monte Carlo for PIC.ipynb` | The original all-in-one HW-in-loop optimizer: 64 DAC volts → 10 filtered PDs, minimise Euclidean distance to `DESIRED_OUTPUT`, perturb best inputs (1/√iter decay), seed via k-nearest on 100k | The shipped Section-A optimizer; writes best_input/output/summary | Superseded by `src.inverse` + `template.py` |
| `anagha/…gen18…optuna.py` | Optuna hyperparam search, single PD (ADC10), [v,v²] MinMax[-1,1] | Search that produced the **64/192/128 GELU, lr 4.85e-4, batch32** net | Origin of the 0.78 baseline arch |
| `anagha/…gen20…best.py` | The fixed best net on 7 PDs (ADC13,12,10,8,5,3,1) + ENOB | Average R² ≈ **0.78**, per-PD R²/MAE + ENOB per PD | The reference Section-A forward result |
| `anagha/pic_fw_inverse.ipynb` | Forward net on 10 PDs (incl. noise PD4/6/9) then gradient inverse target-PD → DACs, N=5 targets | Autodiff inverse through the forward surrogate; reports per-target / per-PD MAE | Model-based inverse (offline analogue of MC) |

## Laser slide figures (not an old-dataset experiment; deleted with this batch)

| Experiment | What it tested | Key result | Verdict |
|---|---|---|---|
| `laser_slides_figs.py` | Week-2 laser slide figures + printed calibration fits from `runs/laser_*cal*.csv` | Setpoint→current linear **I ≈ 2.5·s − 7 mA**, power **P ≈ 0.344·s − 1.44 mW**, ceiling **+15.2 dBm** (same calibration baked into `laser/laser.py`); wrote `slides/images/week2/*.png` | Plot-only; cal constants live on in `laser.py` |
