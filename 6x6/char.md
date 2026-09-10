# PIC B characterization — heater-characterization method

How PIC B's heaters are mapped and characterized, and how the result stays current under
drift. A re-characterization pass is **in progress**; this documents the method, not a
frozen snapshot. The single loadable result is `pic_data/pic_b_config.json` (source of
truth, rebuild with `scripts/build_config.py`); the per-channel CSV is
`pic_data/dac_heater_map_b128.csv`. **Live counts live in the config's `summary` block —
read them from there, not from this doc.** Everything here is **PROVISIONAL, verified=0** —
schematic + electrical + optical, not cross-checked against a second board.


## 1. The chip & board

PIC B is a silicon-photonic 6×6 MZI mesh computing **M = U·Σ·V** (input encoding → V →
Σ → U → photodiodes). It is the **mirror twin of Section A**: the chip has 240 physical
heaters shorted in Section-A/Section-B pairs into **120 electrically-independent nets**
(geometric numbering `net` 0–119 = `heater_gds`), same structure, different DAC map.

The 128-channel board is **8× DAC81416 chips × 16 = 128 channels** across two 96-way
headers (A = chips 0–3 / DAC 0–63, B = chips 4–7 / DAC 64–127). Firmware
`Arduino/pic128/pic128.ino`, 14 photodiodes (A0–A13), ADC averages **10 sweeps/reply**.

| fact | value | why it matters |
|---|---|---|
| nets | 120 (240 heaters, mirror-shorted) | join maps on `net`/`heater_gds`, never a raw heater id |
| firmware | `Arduino/pic128/pic128.ino` | proven per-write config seq, 0–4 V clamp, 10-sweep avg |
| DAC per-write config | `0x03 00 84` / `0x09 00 00` / `0x05 FF FF` after **every** value | DAC81416 won't stay up on the old one-time init |
| CS pins | `{10,9,8,7,5,4,3,2}` | corrected 07-30 — see §2 |
| ADC | 14 PDs, avg 10 sweeps | firmware-side averaging, no host √N yet |


## 2. The CS-pin bug (why "chip 4" looked dead)

Firmware shipped with chip-select array `{10,9,8,7,6,5,4,3}`. The board actually wires
chips 4–7 to `{5,4,3,2}`. So firmware "chip 4" asserted **D6, where no DAC lives** →
channels 64–79 were never driven and read fully dead in the 07-23 census — which also
**mis-attributed the whole B half**, since every B-side CS was shifted by one pin.

Fix: corrected the array to `{10,9,8,7,5,4,3,2}`, recompiled + reflashed → chip 4 came
alive and the B-half channels landed on their real pins.

**Lesson:** a whole-chip clean boundary of "dead" channels points at CS / addressing, not
at the DAC or the ribbon. A dead *ribbon* frays; a dead *CS pin* dies on a 16-channel line.


## 3. Electrical map-check — which DACs drive (laser OFF)

`scripts/probe_hold.py` energizes one 16-channel block with an alternating pattern (even
ch → 1 V, odd ch → 2 V) and **holds it** — the script keeps the serial port open and
re-asserts the vector every ~2 s (opening the port resets the Mega), so each pin can be
probed with a multimeter. Went block by block, all 128 channels.

This measures **DAC drive presence at the pin**, nothing optical. To separate "DAC not
driving" (electrical) from "heater optically silent" you *need* a multimeter — the PD
census alone can't tell them apart. The fault sets (`ELEC_DEAD` / `ELEC_WEAK` /
`ELEC_SHORT`) are hard-coded in `scripts/map_b128.py` and surface in the config `summary`.

| status | channels | reading | meaning |
|---|---|---|---|
| ok | 112 of 128 | pattern follows the 1 V / 2 V set | drives (incl. 4 ok spares) |
| dead | 39, 51, 62, 98, 100, 111, 118, 120, 124, 125, 126 | ~0 V regardless of set | per-channel fault |
| short | 112, 113, 114, 115 | all sit ~1.25 V regardless of set | tied together / shunted cluster |
| weak | 99 | ~1.5 V into a 2 V set | partial drive |

Config `summary.drivable = 112` counts every `elec=="ok"` channel; **4 are unmapped
spares**, so **108 nets** are electrically reachable. All faults are isolated per-channel —
**no chip-level failures remain** after the CS fix.


## 4. Coupling loss — run at +15 dBm and average, don't push power

Input-fiber coupling is **~13× (~11 dB) below the 07-23 reference** (PD8 read 191 mV @
+5 dBm before; now needs +15 dBm for ~140 mV of the same shape). The laser is fine —
current scales, emission verified via the `bfm` monitor rise — and the fingerprint
**shape** is intact, so the loss is the input fiber alignment drifting during the rewire,
and it is **not recoverable**. All optical runs therefore use **+15 dBm**.

**⚠ Never exceed +15 dBm — it can damage the photodiodes** (`laser.max_dbm = 15` in the
config). +15 dBm is the ceiling, not a target to raise. Because we can't buy SNR with more
power, we buy it by **averaging**: the firmware means 10 sweeps/reply and the census keeps
**every raw repeat** (§5) so slow noise averages out in analysis, rather than pushing power
into the PD-damage regime.

Zero-heater PD fingerprint @ +15 dBm (config `laser.zero_heater_fingerprint_mV_at_15dBm`),
for a quick coupling sanity check:

| PD | 8 | 12 | 10 | 5 | 1 | 3 |
|---|---|---|---|---|---|---|
| mV | 206 | 210 | 168 | 165 | 147 | 146 |


## 5. Census sweeps — one channel at a time

`scripts/probe_channels.py` is the empirical DAC→heater ground truth. Per channel it reads
a **fresh all-zero baseline** (or the light-routed baseline of §9), drives the channel, and
re-reads — so each channel's response is measured against its own baseline, immune to slow
drift between channels. It streams **every raw repeat** (`--repeats`, no averaging) plus
laser telemetry (`bfm`, temp, mA) to `pic_data/census/` and flushes per channel, so a crash
loses at most one channel — **raw-data retention is a hard requirement**.

Two modes, set by `--levels`:

| mode | flag | what it produces |
|---|---|---|
| influence probe | default (single `--drive`, 1.5 V) | Δ per PD at one drive level — the footprint / DAC→PD map |
| fringe sweep | `--levels 0,0.5,1.0,…,4.0` | the full P(V) curve `census_fringes` fits (§6) |

Run at **+15 dBm** (`--dbm 15`; the script *default* is +5, kept for the old higher-coupling
reference). The characterization sweep is the full **0–4 V at 0.5 V steps**; the firmware
clamp was raised **2 V → 4 V** for it. NUS characterized the chip at 0–5 V; the census
energizes only **one heater at a time**, so 4 V is thermally safe — but **many-heater
operation must watch total thermal load** (the clamp is not a per-net safety margin once you
drive several nets at once). Why 0–4 V and not 0–2 V is §8.


## 6. Fringe fit → operating points

`scripts/census_fringes.py` reads a census CSV and, per channel, fits its most-responsive
PD's transmission to **P(V) = A + B·cos(φ₂V² + φ₀)** (via `src.characterize.fit_fringe`;
visibility = |B|/A). `src.sweep_analysis.fringe_extrema` then reads off the operating points:

| point | definition | use |
|---|---|---|
| **V0** | `v_at_max` — max pass | "transparent" / brightest — route light through (§9) |
| **Vnull** | `v_at_min` — 0 light out, φ = π | the dark point; the drift anchor (§10) |
| **Vpi** | √(π/φ₂) | half-fringe spacing; comes out even if the null is past the clamp |

Operate a heater by **normalizing between V0 and Vnull**. `fringe_extrema` also flags
`max_reachable` / `min_reachable` — whether V0 / Vnull actually fall inside the swept band
or are model-extrapolated past it.

**Multi-start fit (critical).** The V² law has many local minima over a wide sweep, so a
single φ₂ seed fails — e.g. ch0 has a clean null at 2.5 V that a bad seed rejects.
`robust_fit` runs `fit_fringe` from **several Vπ seeds** (`0.8, 1.2, 1.6, 2.0, 2.5, 3.0,
3.5`) and keeps the **lowest-rmse physical** fit. In the 2 V pass this took reliable fits
from 7 → 44. `python scripts/census_fringes.py <census.csv> [--vmax V] [--write]`.


## 7. Reliability filter

A fit is only trusted (`reliable = True`, the flag `map_b128` reads) if it is physical,
well-fit, and turns over in range. All three bounds must hold:

| bound | condition | why |
|---|---|---|
| visibility | `0.10 ≤ |B|/A ≤ 1.05` | real contrast, not noise; one floor on both devices (the 4x4's weakest real heater reaches 0.103); not >100% (a fit artifact) |
| fit error | `rmse ≤ 0.2·swing + 1 mV` | the cosine tracks the data, not noise |
| Vπ range | `0.5 ≤ √(π/φ₂) ≤ vmax + 1` | fringe turns over near the swept band; rejects Vπ extrapolated to nonsense |

The digest additionally gates on **swing ≥ 6 mV** (2σ of the 3.1 mV slow read noise, the
same 2σ margin as the 4x4 — a channel has to move a PD before there is anything to fit).
The `Vπ ≤ vmax + 1` term is the one that couples the filter to the sweep range — see §8.


## 8. The 0–4 V extension (`--vmax`)

`census_fringes` defaults to **`vmax = 2.0`**, and the first characterization pass swept and
fit at 2 V. But 0–2 V spans only **~0.8π** of phase (Vπ median ~2.24 V), so ~2/3 of heaters
are still monotonic within 2 V — their null sits in the **2–4 V band**. Such a heater fits a
large Vπ that **trips the reliability cap** (`Vπ ≤ vmax + 1 = 3.0 V`) and is rejected as
"monotonic / null past clamp", even though the fringe is real.

The fix is to see the null: sweep the full **0–4 V** (§5) and re-fit with **`--vmax 4.0`**,
which raises the reliability cap to `Vπ ≤ 5.0 V` and accepts those fringes. This is the
mechanism **extending coverage from the initial 57 toward all 112 drivable heaters** —
re-sweeping the not-yet-reliable channels at 0–4 V and re-fitting at `--vmax 4.0` recovers
the ones whose null lives beyond 2 V.


## 9. PD resurrection via light-routing (`--base-map`)

At an all-zero baseline the heaters sit at random **built-in offset phases**, so light
scatters and several PDs (4, 6, 13) read ~0 — their heaters look dim only because **no
light reaches them**, not because they're broken. Setting the **already-characterized**
heaters to their **V0** (transparent / max-pass) routes light through the mesh so it reaches
the dim, deep-mesh heaters:

| effect | before | after |
|---|---|---|
| total 14-PD baseline sum | 1123 mV | 1964 mV (**~1.75× brighter**) |
| PD4 / PD6 / PD13 | ~0 | 40–121 mV (lit up) |
| characterized heaters | 40 | **57** (+15 recovered) |

`scripts/probe_channels.py --base-map <map.csv>` reads `V0` for every characterized channel,
holds those at V0, and sweeps each dim channel from 0 against that **bright** baseline (the
swept channel itself always starts from 0; the others hold V0). **Verify the baseline
actually brightens** (check the baseline PD sum it prints) before a long run — hot heaters
also add thermal crosstalk, so a brighter-but-noisier baseline can cost more than it buys.
`reanchor.py` takes the same `--base-map` for re-anchoring the dim ones (§10).


## 10. Drift & rapid re-anchoring (`./do heaters`)

Drift on this chip is **offset-only same-day**: φ₀ moves, the fringe **shape** (φ₂ / Vπ)
holds. **Across days it becomes a map change** and the channel must be re-characterized from
a fresh sweep. Because same-day drift is offset-only, you **don't re-sweep** to track it —
you warm-start from the stored null and re-find it locally.

`scripts/reanchor.py` (the `./do heaters` backend) does exactly this. For each characterized
channel it warm-starts at the stored `Vnull` and runs a small **parabolic minimization** of
its dominant PD (`fringe_pd`) around that point — three probes `(v−δ, v, v+δ)`; if they
bracket a convex minimum it steps to the parabola vertex (δ then shrinks), otherwise it
steps downhill (gradient fallback); `maxiter = 5`, so **a few measurements per heater**, not
a full sweep. The null is a *minimum* of PD output, so a local parabola finds it directly.

`./do heaters` runs `reanchor.py --update`, which:

| step | detail |
|---|---|
| re-null | warm-start parabolic re-find of each stored `Vnull` |
| trust filter | only drift **≤ 0.5 V** is written back; larger is flagged `<-- big, suspect` |
| V0 follow | recompute `V0` from the new null holding the shape: `V0² = Vnull² ± Vpi²`, root nearest the old V0 |
| write | new `Vnull` (+ `V0`) back into `pic_data/pic_b_config.json` in place |
| report | drift-since-characterization stats + speedup vs a full 0–4 V re-sweep |

Baseline is **all-zero by default** (matching how most nulls were first measured); add
`--base-map pic_data/dac_heater_map_b128.csv` to light-route the dim ones, `--limit N` for a
quick subset, `--dbm` (default 15).

**Parallel characterization** (`./do heaters schedule` / `parallel`). Characterize heaters
with **disjoint PD footprints** in lockstep — one sweep recovers several nulls at once, cost
~independent of heater count. `scripts/parallel_groups.py` (`./do heaters schedule`) makes the
READ-PD a decision variable and does resource-constrained list-coloring over the
heater×PD swing/r²/Vπ matrix (`fringe_fits_b128.csv`) → confound-free rounds in
`pic_data/parallel_schedule.json` (~4× speedup; a naive same-PD colouring caps at ~2× because
one bright PD dominates ~26 heaters). `scripts/parallel_char.py` (`./do heaters parallel
--channels …`) is the validated lockstep runner: it drives the given heaters together, recovers
each from its own dominant PD, and cross-checks every recovered V0/Vnull against a solo sweep
(no cross-path confounding). **Still to come:** a full runner that consumes
`parallel_schedule.json` round-by-round and merges the recovered nulls into the config.


## 11. The single config

Everything lands in **`pic_data/pic_b_config.json`** — the loadable source of truth, and the
file you correct by hand when something is wrong. `scripts/build_config.py` folds the
per-channel map (`dac_heater_map_b128.csv`, from `map_b128.py`) together with device /
firmware / laser / PD metadata and the fresh-census footprint; **rebuild after any
re-characterization**. Operating points are set to `null` on non-drivable (`dead`/`short`)
channels — they aren't independently actionable there.

Top-level: `meta`, `devices`, `firmware`, `laser`, `photodiodes`, `summary`, `channels`.
Per-channel schema:

| field | meaning |
|---|---|
| `dac` / `chip` / `header` / `pin` | address: DAC index, its chip 0–7, header A/B, header pin |
| `net` | mirror-shorted net 0–119 (= `heater_gds`); the join key |
| `stage` | mesh stage: `enc.split` / `enc.single` / `V` / `sigma` / `U` (or `spare`) |
| `vendor` | `[A, B]` geometric heater ids of the shorted pair |
| `elec` | `ok` / `dead` / `weak` / `short` (§3 map-check) |
| `swing_mV` / `pds` | fresh 07-30 footprint: max |Δ| and the PDs it moves, strongest first |
| `V0` / `Vnull` / `Vpi` | operating points (§6) — `null` if not drivable / no reliable fringe |
| `fringe_pd` | the PD the fringe was fit on |
| `fringe_src` | **provenance**: which census run (timestamp = thermal frame) the point came from |

`summary` is the **live count block** — `nets`, `drivable`, `characterized`, and the
`dead`/`short`/`weak` lists. Read counts from here.

**Provenance & thermal frame.** The rig heats over a session, so later data sits in a warmer
frame — later is **not** better. `map_b128.load_operating_points` lets the **earliest**
reliable fit win per channel (cooler, less-drifted); a later run only fills channels the
earlier one missed. `fringe_src` records the frame per point, so mixed-frame points can be
re-anchored (§10) before use.


## 12. Artifacts

| artifact | what |
|---|---|
| `pic_data/pic_b_config.json` | consolidated single config — source of truth (§11) |
| `pic_data/dac_heater_map_b128.csv` | per-channel DAC↔heater map (join on `net`) |
| `pic_data/census/` | raw census streams + fringe fits (retention requirement) |
| `src/census.py` | **shared fringe library**: `robust_fit`, `load_census`, `analyze`/`digest`, `apply_fringe`, `stamp`, `CENSUS_HEADER` — the scripts below are thin CLIs over it |
| `scripts/probe_hold.py` | electrical map-check — hold a block, probe pins with a multimeter (§3) |
| `scripts/probe_channels.py` | one-channel census: influence probe / fringe sweep, `--base-map` light-routing (§5, §9) |
| `scripts/census_fringes.py` | multi-start fringe fit → V0/Vnull/Vpi, `--vmax` sweep-range cap (§6–§8) |
| `scripts/merge_fringes_to_config.py` | fold reliable `fringes_*.csv` into the config (earliest-wins) |
| `scripts/map_b128.py` | fuse picpin structure + census + elec status → the CSV map |
| `scripts/build_config.py` | fold map + metadata → `pic_b_config.json` (§11) |
| `scripts/rechar_outsidein.py` | outside-in transparent-frontier re-characterization (`--plan`/`--mock`) |
| `scripts/reanchor.py` | rapid drift re-anchor, warm-start parabolic re-null; `./do heaters` backend (§10) |
| `scripts/parallel_groups.py` | confound-free parallel-sweep scheduler → `parallel_schedule.json`; `./do heaters schedule` (§10) |
| `scripts/parallel_char.py` | validated lockstep parallel-sweep runner; `./do heaters parallel` (§10) |


## Current state

Counts are **as of the latest pass** and live in `pic_b_config.json` › `summary` — read
them there. Electrical faults below are the 07-30 map-check (stable this session).

| metric | source of truth |
|---|---|
| nets | `summary.nets` (120) |
| drivable | `summary.drivable` (112 ok channels; 108 nets after 4 spares) |
| characterized (V0/Vpi) | `summary.characterized` — **57** after the initial 0–2 V + light-routing pass; the in-progress 0–4 V / `--vmax 4.0` pass extends it toward 112 |
| dead | 39, 51, 62, 98, 100, 111, 118, 120, 124, 125, 126 |
| short (tied cluster) | 112, 113, 114, 115 |
| weak | 99 |
| dead PDs | none (0 on PIC B) |
| strong PDs | 3, 5, 6, 7, 8, 9, 10, 12 |
| weak PDs | 0, 1, 2, 4, 11, 13 |
| laser | +15 dBm, hard ceiling +15 dBm (PD-damage risk) |
