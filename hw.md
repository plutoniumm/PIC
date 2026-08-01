# hw.md — archived drift-correction hardware investigation (removed, disproven)

This file is the permanent record of ~15 hardware and offline experiments that tried to make
**per-config / per-heater drift correction** work on this specific PIC-B chip. They all failed
for the same reason, so the code was deleted to stop anyone re-attempting a disproven path.

The scripts are recoverable from git history:

```
git log --diff-filter=D --name-only            # find the deletion commit
git show <commit>^:scripts/twin_drift_correct_hw.py   # recover one file
```

Deleted files: `scripts/{collect_clean_dpnn,collect_seq_hw,thermal_test,diff_drift_hw,
dpnn_drift_hw,drift_correct_hw,twin_drift_correct_hw,lora_retrain_hw,live_recal_poc}.py`,
`theory/drift_infer.py`, and the prose write-ups `dpnn_inversion.md` + `live_recal_poc.md`
(folded in below).

## Conclusion

Per-config drift correction is a **hardware observability / controllability limit on this
chip, not a modelling failure.** Every route was tried — twin backprop, DPNN input-δ
inference, model-free differential Jacobian, thermal-state observers, LoRA/full re-training,
control-inversion re-solve — and none beat the drifted baseline on held-out configs; several
made it *worse*. The clincher is the **fair single-session control-inversion** test
(`twin_drift_correct_hw.py`): a design where the target output is genuinely reachable, with an
injected drift **4.5× above the repeat-noise floor**, still recovered **0%** — the on-chip line
search rejected **all 24** gradient proposals. The correction machinery is sound (it recovers
injected drift exactly in simulation, and a twin *does* work on healthy photonic chips); it is
this chip's static model-error, hidden thermal state, and underdetermined intensity readout
that make the drift neither predictable nor invertible. Ising still works because its
summed-Gram readout averages over per-config variation.

## Results

| Experiment | Approach | Result | Why it failed |
|---|---|---|---|
| `twin_drift_correct_hw.py` | **FAIR single-session control-inversion.** Park at V0→y*, inject known drift δ, re-solve V with DPNN as gradient guide + hardware line-search (never-worse guard). One laser session so y* is genuinely reachable. | drift 0.15 V → **0% recovered**; drift 0.40 V at **4.5× noise → 0% recovered, all 24 line-search proposals rejected**. | The definitive negative. Even with a reachable target and drift well above noise, no surrogate-proposed voltage move measured better than the drifted start — the DPNN gradient does not point where the chip actually improves. |
| `drift_correct_hw.py` | **Twin backprop drift-inference.** Fit a phase-offset field dφ (on characterised heaters) so `theory/twin.py` reproduces the 6 monitor PDs; one joint fit over K probes. | train rel-resid 0.82→0.54, but **held-out 0.79→0.85 (worse)**. | Twin has **r≈0.10** vs the real monitors on random configs — the fit absorbs static twin↔chip model-error, not drift, and over-corrects on held-out. |
| `dpnn_drift_hw.py` | **DPNN input-δ inference.** Backprop a per-heater voltage offset δ through the frozen hardware DPNN so `DPNN((H+δ)²)` matches fresh probes; restore a target by commanding H−δ. | selftest recovers injected δ; hardware **fresh DPNN r²≈0.556**, held-out drift correction **worse than uncorrected**. | Same disease as the twin, milder: DPNN r²0.54–0.67 residual is still dominated by static model-error, so δ fits the bias not the drift. |
| `diff_drift_hw.py` | **Model-free differential Jacobian.** Compare the SAME 13-h-old configs re-measured now: `Δy = f₁(V)−f₀(V) ≈ J(V)·δ`; fit δ using only the DPNN's local gradient (static bias cancels). | 13-h drift **SNR 1.9**, inferred **|δ|≈0.67 V**; held-out restoration **0.076→0.19 (worse)**. Recovers injected δ exactly only in ideal sim. | Even removing the absolute model-error isn't enough: the Jacobian is wrong enough and the offset-drift model too simple, so V−δ misses. Real drift is not a clean input-referred offset. |
| `thermal_test.py` | **Thermal-state observer.** Ridge-regress the DPNN residual on EWMA-of-applied-power features (substrate thermal state at several time constants + per-region groups). | residual variance explained **−0.90 (time-split) / −0.46 (random-split)** → **no signal**. | The hidden thermal state (8× ordering/history dependence, below) is **not** an EWMA of power history — the observer features carry zero predictive information about the residual. |
| `collect_seq_hw.py` | **Fast sequential logged collection** (no reset/shuffle) to feed the thermal observer with natural accumulated history + timestamps. | data collector only (~0.5 s/config, ~300 configs/3 min). | Fed `thermal_test.py`, which found no thermal signal. |
| `collect_clean_dpnn.py` | **Controlled-history clean collection + retrain.** Fixed zero-reset before every config to make the config→output map history-independent, then retrain. | reset only **halved ordering noise 0.034→0.016**; old DPNN on clean **r²≈0.14**, fresh-trained clean **≈0**, LoRA-transfer **≈0.19**. | Killing the history dependence did not lift R² above the dirty-protocol 0.54–0.67 ceiling — the ceiling is static model error, not protocol. Reset removes only half the ordering noise; the rest is irreducible per-config variation. |
| `lora_retrain_hw.py` | **Cross-session rapid re-training.** Frozen base DPNN vs LoRA adapter vs full fine-tune, as a function of fresh-sample budget on the drifted chip. | Neither LoRA nor full fine-tune on tens–hundreds of fresh samples broke the responsive-R² ceiling. | Retraining cannot fix a chip whose config→output map has an unmodellable hidden state; more samples buy diminishing R², not correction. |
| `live_recal_poc.py` + `live_recal_poc.md` | **Offline closed-loop recal PoC** (simulated drift on a DPNN proxy that *shares the model's functional form*). LoRA adapter tracks drift + PICGD control-inversion re-solve + HIL line-search. | In sim: adapter tracks (R² 0.94→0.98); HIL removes **~40–68% of total drift** (~65–100% of removable); continuous retrain **~32% vs ~15%** one-shot. Open-loop over-promises **~10×**. | **Worked only in simulation**, explicitly flagged optimistic because proxy chip and controller share the DPNN. On real hardware the base surrogate is only **R²≈0.58**, so the surrogate-residual floor dominates — and the hardware version (`twin_drift_correct_hw.py`) recovered 0%. |
| `theory/drift_infer.py` | **Backprop drift-inference (theory).** `infer_drift`: joint fit of a phase-offset field to monitor residuals through the differentiable twin; noise averages 1/√(K·NPD). | selftest **17° injected → ~8° residual** for observable (encode-stage) drift; **global mesh phase 100% unobservable** from intensity monitors. | Correct backprop, but intensity-only monitors cannot see the global-phase gauge of the mesh — a hard **gauge floor** on how much mesh-stage drift is even recoverable. |
| `dpnn_inversion.md` | **DPNN heater characterisation by inversion** (offline). Sweep one heater *inside* the DPNN, fit the cosine fringe → Vπ/V0, to replace hardware sweeps. | **1/95 channels** yield a real fringe; **87% monotonic ramps**; corr(Vπ_model, Vπ_cfg)≈0; recovered Vπ non-physical (tens of V). | The MLP (R²≈0.58, aggregate MSE on dense random vectors) never learned per-heater **cosine periodicity** — it smoothed it away. Control-inversion (PICGD, design direction) works on the same model because it only needs the smooth gradient, not the fringe. |

## Root cause

- **Twin r≈0.10** vs the real monitor PDs on random configs — the physics twin does not predict this chip, so twin-based inference fits model-error not drift.
- **DPNN ceilings at r²0.54–0.67** (fresh, hardware-trained on 2072 samples); the residual is dominated by static model error, so any drift-δ backprop through it fits the bias.
- **8× thermal ordering/history dependence** is a hidden state that is **not** an EWMA of applied power (`thermal_test`: residual explained −0.90/−0.46) and is only half-removed by a fixed reset (0.034→0.016).
- **Gauge-unobservable mesh phase**: intensity-only monitors cannot see the global-phase gauge of the U·Σ·V mesh (`drift_infer`: 100% unobservable), so mesh-stage drift is not fully recoverable in principle.
- **Underdetermined readout**: 14 PDs (≈8 responsive) for ~93 characterised heaters, plus dead in-mesh heaters — the drift field is not identifiable from the monitors available.

## What still works

**Ising** (`scripts/ising_hw.py`): its summed-Gram / spin-energy readout averages over per-config variation, so the very per-config drift that defeats direct correction cancels out in the aggregate — first optical compute on PIC-B holds up despite all of the above.
