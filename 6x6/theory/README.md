# theory/ — what PIC B can actually compute

Simulation sandbox answering one question: given a characterized PIC B, what real problem
can it solve? **No hardware, serial or laser imports anywhere in here.** The twin is built
to match `src.pic.layout.build_scene()` element for element, so heater indices agree with
`pic_data/pic_b_config.json` (`net` == geometric heater id H0..H119).

    python -m theory.study            # all sections
    python -m theory.study gram size  # named sections

## Bottom line

**Two things work, both on the heaters already characterized** — neither blocks on finishing
characterization.

> The counts below say "82" because that is what `pic_b_config.json` held when they were
> measured. The mask is read live (`Hardware().known.sum()`) and characterization is ongoing,
> so it only goes up — re-run rather than trusting the literal number.

| | end-to-end, at the ADC floor | readout needed |
|---|---|---|
| **6-spin Ising** | 88 % ground state, E-corr 0.982 | monitor PDs only |
| **5×5 signed matvec** | 0.101 vec err, 98.0 % sign accuracy | homodyne + LO cal + per-channel D |

Ising is the cheaper demo: no homodyne, no LO phase constant, no diagonal calibration.
Signed matvec is the stronger claim but needs all three working.

## Modules

| file | what it is |
|---|---|
| `twin.py` | differentiable chip: tree → encode → V → Σ → U → monitor + homodyne PDs. Torch, autograd, batched over restarts (`matrix()` takes `(R,120)` and returns `(R,6,6)`) |
| `hw.py` | real constraints from `pic_b_config.json`: drivable / characterized masks, (Vpi, phi0), ADC floor |
| `readout.py` | homodyne: 2 LO-phase shots → complex output field |
| `program.py` | inverse design for a target **transfer** matrix (`fit_matrix`, `fit_submatrix`, `inner_rails`, `encode_phases`) |
| `gram.py` | inverse design for a target **Gram** matrix — the right objective for Ising |
| `matvec.py` | signed matrix-vector benchmark |
| `ising.py` | 6-spin Ising: power route and homodyne route |
| `study.py` | 7 sections, caches to `results/*.json` |

## Why Ising beats matmul on this chip — the structural reason

The power route measures `|Ms|² = sᵀ Re(MᴴM) s`, so **only the Gram matrix must match, not
M**. Every left-unitary is invisible: the missing output phase shifters, the whole U output
basis, global phase. That is roughly half the constraints of a transfer-matrix match, and it
is why an arbitrary 6-spin instance programmes *exactly* while a 6×6 matvec cannot.

Practical consequence: shift `J + cI = AᵀA`, launch spins as 0/π input phases at full
amplitude, and **total monitor-PD power is the Ising energy plus a constant**. No reference
beam. `NUS/Solution to the problems.docx` proposes the reference beam precisely because
intensity-only detection loses the sign — the power route sidesteps that entirely.

## Results

| result | number |
|---|---|
| arbitrary 6-spin SK Ising, Gram match | **exact** — hosted J vs target J to 9.7e-9 |
| …ground state, noiseless | 100 % |
| …ground state, at the ADC floor | **88 %**, E-corr 0.982, excess 0.002 (`throughput=0.05`) |
| Ising energy from monitor PDs alone | exact: E-corr 1.0000, residual 3e-7 |
| signed matvec *programming*, 2×2 … 5×5 | exact (0.000; 0.011 at 5×5 on the characterized 82) |
| signed matvec *programming*, 6×6 | 0.040 all heaters / 0.115 characterized-82 |
| homodyne field recovery, noiseless | 3e-7 |
| homodyne without the two-LO phase constant | **1.65 — fails completely** |
| homodyne at the ADC floor | ~0.12 |
| chip singular values (median) | `[1, .76, .48, .25, .068, .0075]` vs gaussian `[1, .76, .56, .38, .22, .057]` → effective rank ~4-5 of 6 |

End-to-end signed matvec (programme → encode → homodyne → compare to Mx), characterized-82:

| size | clean | at the ADC floor |
|---|---|---|
| 4×4 | 0.000 / 100 % sign | 0.141 / 95.6 % |
| **5×5** | 0.005 / 99.5 % | **0.101 / 98.0 %** |
| 6×6 | 0.132 / 93.8 % | 0.242 / 83.8 % |

**5×5 is the operating point.** 6×6 more than doubles the error and puts ~1 output component
in 6 on the wrong sign. 4×4 is *worse* than 5×5 under noise — fewer lit rails means less total
signal against a fixed ADC floor, so the extra dimension pays for itself right up until the
edge rails' drop losses take over at 6.

Amplitude, not fidelity, is the bench limit. Programming stays exact under noise; what
degrades is config-to-config power differences approaching the 4.9 mV ADC floor. Two fixes,
both free:

- **Pick the loudest restart, not the most accurate one.** The Gram match solves exactly with
  slack, so many restarts tie at zero error and differ only in amplitude. Selecting on
  amplitude among the shape-matching restarts alone took ground state 62 % → 88 %.
- **Spend the leftover slack:** `fit_gram(..., throughput=w)` adds `-w log|scale|`.

| w | gram err | amplitude | E-corr | ground state | excess |
|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.275 | 0.923 | 88 % | 0.017 |
| **0.05** | 0.0168 | **0.550** | **0.982** | 88 % | **0.002** |
| 0.20 | 0.0596 | 0.588 | 0.982 | 75 % | 0.003 |

## Traps — each of these cost a debugging cycle

- **The two reference arms serve different rail groups** (rails 0-2 top, 3-5 bottom, per
  `layout.SIG_ROUTE`). Their relative phase is a fixed chip constant. Uncalibrated, homodyne
  returns garbage (1.65 rel error — worse than not doing it).
- **Hold both reference arms at full transmission** (`readout.open_reference`) or the LO can
  sit near a null and the homodyne term vanishes.
- **`D` is per-output-channel, not global.** The mesh has no output phase shifters after U, so
  it realises `D·M`. Calibrate `D` once per programming from a few known input vectors, then
  take the real part. Absorbing it into one global scale *cannot* work and looks exactly like
  the chip failing — it produced a bogus 0.70 error at 4×4 where the true answer is 0.000.
- **`gram_scale` can be negative** (the chip hosts −J — harmless, the readout's linear
  calibration absorbs the sign). But `log` of it is a dead gradient, so the objective and the
  restart selection must both use `|scale|`. The signed version actively degrades results.
- **Optimiser, not chip.** An early "6×6 is unreachable, 0.57 error" was pure optimiser
  failure — too few restarts and a strict metric that scored the unavoidable `D`. Batched
  restarts + scoring modulo `D` took it to 0.040. Suspect the optimiser before the physics;
  the self-fit test (target drawn from the chip's own image) separates the two.
- **Put smaller problems on the inner rails** (`program.inner_rails`). Odd-column edge drops
  bleed rails 0 and 5.

## Where this left off

`results/` currently holds only `size.json` — most numbers above came from ad-hoc runs during
the session. **Re-run `python -m theory.study` to repopulate all seven sections.** Roughly an
hour; every section prints as it goes.

Open items, in the order I would take them:

1. **Settle the edge-drop model on hardware.** `twin.drop` treats an edge drop as a 2×2 MZI
   fed on one port with the cross port taken and the bar port dumped, keeping both heaters
   live. `layout.py` draws a single input coupler, which would instead make the second heater
   optically dead. `Twin(drop_live=False)` switches interpretations. This changes how many
   U/V heaters actually do anything and every number above depends on it.
2. **Measure the two-LO relative phase** once homodyne is up. Everything matvec-shaped is
   gated on it.
3. **Run the Ising demo on hardware.** It needs the least — monitor PDs, 82 heaters, no
   homodyne — so it is the fastest path to a real result.
4. **A real annealing loop.** Everything here brute-forces all 64 configs, which is only
   honest at n=6. The homodyne route (`ising.chip_energy_homodyne`) gives the per-spin field,
   i.e. the gradient, which is what a search would actually use.
5. **Drift.** A programmed matrix or J is valid within a thermal session, not across days
   (20-60 %/week). Nothing here models that; the twin is stationary.

## Standing caveat

All of this is simulation against a twin built from `layout.py` geometry. It has never been
checked against a hardware measurement. Treat every number as "what the structure implies",
not "what the chip does", until item 1 and item 3 above are done.
