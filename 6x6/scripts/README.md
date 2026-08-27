# scripts/

Schematic, geometry, and hardware/characterization scripts for the PIC project. Run every script **from the repo root** (not from inside `scripts/`) so that `src` imports resolve, e.g. `python scripts/<name>.py`. Use the project conda env **`pic`** (interpreter `/usr/local/Caskroom/miniconda/base/envs/pic/bin/python`, or `conda activate pic`).

The schematic/geometry scripts are hardware-free (they read the shipped datasets/GDS); the **Hardware** and **Characterization** sections below drive the live Arduino/PIC (and laser). Scripts with a `--which` flag default to `all` (run every sub-analysis); many hardware scripts take `--mock` for a no-hardware dry run.

### Modelling & analysis (removed — see [`../tests.md`](../tests.md))

The old-dataset (100k / pooled `pic_data`) modelling and analysis scripts — forward-model search, drift & noise studies, learnability/hardware-ceiling probes, cosine/NN-variant bake-offs, prune & w2b architecture sweeps, and the Anagha legacy code — were removed in the cleanup. Their results are archived in [`../tests.md`](../tests.md); recover any script from git history (`git log --diff-filter=D -- scripts/<name>.py`).

### Chip geometry & schematic figures

| Script | What it does | How to run |
|---|---|---|
| [gds_trace.py](gds_trace.py) | Wire-by-wire trace of one 6×6P structure from the GDS into a netlist + geometry render. Reads `NUS/AD_SiPhIS_180423.gds`; writes `pic_data/netlist_6x6P.json`, `schematic_ref/gds_6x6P_geom.png`. Needs `gdstk`. | `python scripts/gds_trace.py` |
| [pic_clean_schematic.py](pic_clean_schematic.py) | Clean fully-labelled 6×6P schematic (all 120 heaters), plain and Section-B-state variants (B census PD classes, dead paths greyed, undriven heaters hollow). Reads the netlist JSON + `pic_data/dac_heater_map.csv`; writes `pic_data/heater_map.csv`, `pic_structure.png`, `pic_structure_corrected.png` (in the CWD — run from root). | `python scripts/pic_clean_schematic.py` |
| [pic_state_diagram.py](pic_state_diagram.py) | PIC B heater-state schematic: reuses `src.pic.layout.build_scene()` geometry and colours every heater by its current state (characterized / drivable-uncharacterized / dead / shorted / weak), joining config `net` → geometric heater id; marks the dead + shorted-together DAC 112–115 cluster and annotates the summary counts. Reads `pic_data/pic_b_config.json`; writes `pic_structure_state.png` (repo root). Re-run after re-characterization. | `python scripts/pic_state_diagram.py` |
| [picpin_map.py](picpin_map.py) | Decodes `NUS/picpin.drawio` into the board pin map: both 96-way connectors, the 120 mirror-shorted heater pairs, and DAC channel → heater. Self-checks against the 29 DAC→pin edges drawn in the file. Writes `pic_data/dac_heater_map_picpin.csv`, `pic_data/connector_pin_map.csv`. Schematic-derived, **not** hardware-verified — and contradicted by the Section-B census; superseded by the empirical census below. | `python scripts/picpin_map.py [--write]` |
| [probe_channels.py](probe_channels.py) | One-channel-at-a-time influence census (fresh all-zero baseline per channel, laser lit) — the empirical DAC→heater ground truth. Streams every RAW repeat + laser telemetry to a tracked CSV in `pic_data/census/` as measured (crash loses ≤ 1 channel). | `python scripts/probe_channels.py [--n 128] [--channels 0-15,63]` |

### Hardware (Arduino/PIC in the loop)

**Bring-up and the optical-compute runners moved into the `pic` library.** `hw_tests` + `settling` now live in `pic.bringup` (`python -m pic bringup <step|warming|dark|pd13|snr>`), and the Ising / signed-matvec runners in `pic.compute` (`python -m pic ising|matvec`, or `from pic.compute import ising, matvec`). The same-named scripts here — `settling.py`, `hw_tests.py`, `ising_hw.py`, `ising_ab.py`, `intensity_matvec_hw.py` — are now **thin shims** onto those, kept so the old invocations still work. `probe_channels.py` and `pd_learn_test.py` below are unchanged CLIs.

| Script | What it does | How to run |
|---|---|---|
| [settling.py](settling.py) | Thermal step-response settling-time fit and serial loop-speed budget. Talks to the live PIC over serial; `step --selftest` runs on a synthetic trace with no hardware. Optional CSV via `--csv`. Shim onto `pic.bringup`. | `python -m pic bringup step [PORT] [--channel N\|--all]` · `... bench [PORT]` · `... step --selftest` (or `python scripts/settling.py ...`) |
| [hw_tests.py](hw_tests.py) | Bring-up suite: **warming** (thermal drift, TEC-on vs off; dead-PD slope = pure baseline), **dark** (zero-input noise floor σ), **pd13** (find + trim the PD13 bypass to the middle-PD level), **snr** (drive level for SNR=1). Auto-detects the board; writes CSVs + PNGs to `runs/hwtests/`. `--mock` dry-runs the whole flow. Shim onto `pic.bringup`. | `python -m pic bringup <warming\|dark\|pd13\|snr\|compare> [...]` · add `--mock` for no hardware (or `python scripts/hw_tests.py ...`) |
| [pd_learn_test.py](pd_learn_test.py) | Live PD learnability probe: randomize the config's drivable heaters over the {0..4 V} grid, read all 14 PDs, split each PD's variance into signal (across configs) vs noise (within repeats) → per-PD frac_learnable/SNR/swing verdict. Picks the DPNN's output PDs. Laser at +15 dBm. Streams raw reads to `pic_data/census/`, writes `pic_data/pd_learnable.json`. `--mock` runs the whole flow with no hardware. | `PYTHONPATH=. python scripts/pd_learn_test.py --n 150 --repeats 3` · `--mock --n 40` for no hardware |

### Characterization pipeline (PIC B) — see [`char.md`](../char.md)

Probe → census → fringe fit → config, its drift re-anchor, and the parallel-sweep scheduler/runner. The shared fringe primitives live in [`src/census.py`](../src/census.py); these scripts are thin CLIs over it. Driven from **`./do heaters`** (`reanchor` default · `schedule` · `parallel`).

| Script | What it does | How to run |
|---|---|---|
| [probe_hold.py](probe_hold.py) | Electrical map-check: drive one 16-ch block to a fixed pattern and HOLD it (re-asserts every ~2 s) so pins can be probed with a multimeter (laser off). | `python scripts/probe_hold.py --start 0 --count 16` |
| [probe_channels.py](probe_channels.py) | (also above) one-channel census: influence probe (`--drive`) or fringe sweep (`--levels`), `--base-map` light-routing. | `python scripts/probe_channels.py --levels 0,0.5,...,4 --dbm 15` |
| [census_fringes.py](census_fringes.py) | Fit a census CSV → per-channel V0/Vnull/Vpi with a reliability flag (`src.census.analyze`/`digest`); `--vmax` sets the sweep-range cap, `--write` emits `fringes_*.csv`. | `python scripts/census_fringes.py <census.csv> --vmax 4.0 --write` |
| [merge_fringes_to_config.py](merge_fringes_to_config.py) | Fold reliable fringes into `pic_b_config.json` (earliest-wins; `--force` overwrites) via `src.census.apply_fringe`. | `python scripts/merge_fringes_to_config.py fringes_*.csv` |
| [map_b128.py](map_b128.py) / [build_config.py](build_config.py) | Fuse picpin structure + census + elec status → `dac_heater_map_b128.csv`, then fold map + metadata → `pic_b_config.json`. | `python scripts/map_b128.py --write` · `python scripts/build_config.py` |
| [rechar_outsidein.py](rechar_outsidein.py) | Outside-in orchestrator: hold characterized heaters at V0 (transparent frontier), sweep dark deep heaters input-first, fit + merge as it goes. `--plan`/`--mock` need no hardware. | `python scripts/rechar_outsidein.py --plan` · `--mock` · `--dbm 15` |
| [reanchor.py](reanchor.py) | Rapid drift re-anchor (`./do heaters` backend): warm-start parabolic re-null of each stored Vnull, `--update` writes back. | `./do heaters` · `python scripts/reanchor.py --limit 12` |
| [parallel_groups.py](parallel_groups.py) | Confound-free list-coloring scheduler → `pic_data/parallel_schedule.json` (no hardware). | `./do heaters schedule` · `python scripts/parallel_groups.py --only-knobs` |
| [parallel_char.py](parallel_char.py) | Validated lockstep parallel sweep: drive several heaters together, recover each from its own PD, cross-check vs solo. | `./do heaters parallel --channels 85,109,0,1` |
| [probe_refine.py](probe_refine.py) | Stage-2 fine refinement of interior fringe extrema (re-sweeps a narrow bracket around each coarse null/peak). | `python scripts/probe_refine.py --coarse <census.csv>` |

### Paper figures & analysis (`paper/`)

Hardware-free; each reads committed results and writes a PDF into `paper/images/`. The
"in paper" column tracks the current `paper/main.tex`; the others are kept because their
underlying analysis is still cited, only the figure was cut for length.

| Script | What it does | In paper |
|---|---|---|
| [fig_calib.py](fig_calib.py) | One heater's calibration start to finish: raw sweep points, the fitted cosine-in-squared-voltage, and the derived $V_0$/$V_{\rm null}$/$V_\pi$. | yes (`calib.png`) |
| [fig_census.py](fig_census.py) | Every reliably characterised channel of both sections against its sweep limit, PIC-A full-range vs PIC-B under the 4 V clamp. | yes (`census.png`) |
| [fig_chip.py](fig_chip.py) | Paper-proportioned PIC-B chip state map (per-heater state colours), a 7-inch redraw of `pic_state_diagram.py`. | yes (`chip_state.png`) |
| [fig_learnability.py](fig_learnability.py) | Per-PD held-out $R^2$ of the hardware-trained DPNN, split by signal tier. Also the check behind the Section-B block of the per-PD class table. | no (numbers only) |
| [ising_ab_analysis.py](ising_ab_analysis.py) | Pairs the baseline/corrected Ising runs in `runs/ising/` into `theory/results/ising_ab.json` (per-instance paired deltas, GS rates). | table only |
