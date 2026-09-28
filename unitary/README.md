# unitary — PIC1A 4x4: calibrate and run a unitary matvec from the browser

Self-contained copy of the 4x4 runtime and its UI. Carry the directory; nothing outside it
is imported. Works on macOS, Linux and Windows: instruments are found by USB serial number
(`pic/config.py` `USB_SERIAL`), not by device path.

**Portable:** copy this directory to any machine and run the installer. It needs only
Python 3.11+ and internet access for the packages. A `.venv` carried over from another machine
is detected (its interpreter does not run here) and rebuilt; don't bother copying it.

| | Install (once) | Start the UI |
|---|---|---|
| Windows | double-click `install.bat` | double-click `run.bat` |
| macOS / Linux | `./install.sh` | `python3 bootstrap.py run` (or `make run`) |

Or directly, on any OS (Windows has no `make`):

```bash
python bootstrap.py          # .venv with numpy, scipy, pyserial, psutil, torch (uv if installed)
python bootstrap.py run      # UI on http://127.0.0.1:8744, armed; turns the laser off on start
python bootstrap.py run --lan  # also reachable from the network; prints http://<lan-ip>:8744
python bootstrap.py test     # every self-test, no hardware
```

On macOS and Linux `make`, `make run` and `make test` do the same (`make run ARGS=--lan`).
`--lan` exposes laser and heater control to anyone on the network: use it on a trusted LAN only.
Everything runs in Python's UTF-8 mode (`-X utf8`), because Windows' default cp1252 cannot carry the ±, ° and φ in our logs.

CLI, from inside this directory: `.venv/bin/python -X utf8 -m pic {selftest,status,measure,tec,
char,fastchar,capture,program,matvec,...} [--mock]` (`.venv\Scripts\python.exe` on Windows). In a
terminal, events print as readable lines; piped (as the UI runs them) they are `@ev` JSON.

Instruments are named by chip id (USB serial number), shown on the Tools tab; Windows' FTDI
driver appends a letter (`AU05XLI8A`), which is matched too. COM port names do not matter.

| Path | What |
|---|---|
| `ui.py`, `static/` | browser UI: Run, Settings, Tools, Logs |
| `events.py`, `pic/log.py` | every component's state changes as structured events; the Logs tab reads them |
| `bootstrap.py` | cross-platform setup and launch (the Makefile wraps it) |
| `pic/` | rig: serial driver, laser, TEC, switch, acquisition, characterization, `Rig` |
| `spd/` | Single-photon detectors (Vega, FTDI) by chip id: `count`, `assign`, `dark`, `max`; VAUL parser, physical mock. See [Detector modes](#detector-modes-pd-and-spd) |
| `theory/` | Clements, calibration law, twin, signed intensity matvec, drift, stats |
| `Arduino/` | board firmware (`pic4x4`) and TEC firmware (`tec_pid`) |
| `laser/` | AeroDiode PDMv5 notes and the vendor Windows installer; the driver is `pic/devices/laser.py` |
| `pic_data/` | PD mode's calibration, fits, transfer tables; `pic_data/spd/` SPD mode's; `spd_map.json`, `spd_max.json`, `detector_mode.json` |

Instruments are named by chip id (USB serial number), shown on the Tools tab; any `--*-port`
flag takes an id or a role name. A replaced board, TEC Arduino or laser adapter has a new id: update
`USB_SERIAL`, or pass `--pic-port` / `--laser-port`.

Split from `../4x4/` on 2026-09-23. Left behind there: Ising, the learned surrogates, the
standalone scripts. `pic/sim.py` and `pic/model.py` came along because the self-tests use them.

## Detector modes: PD and SPD

Settings → Detectors picks what reads the four mesh outputs (`pic_data/detector_mode.json`).
Heaters and switch are the board's in both. One seam: `pic/interface.py:SPDBoard`, chosen in
`pic/rig.py:make_board`. An SPD read is `(rate − dark)/(max − dark)` per detector, NaN on an
output with no SPD.

| | PD | SPD |
|---|---|---|
| Readout | board ADC, volts | one Vega SPD per output, 0.2 s of 20 ms frames, all read in parallel |
| Readout law | `pd_gain`/`pd_offset` in calib.json | identity; per-SPD dark/max in `pic_data/spd_max.json` |
| Calibration, fits | `pic_data/calib.json`, `char_results.json` | `pic_data/spd/…` (first load starts from PD's heater law) |
| Transfer tables | `pic_data/sessions/` | `pic_data/spd/sessions/` |
| Physics fit (`learn.train_hw`) | `runs/hw`, `runs/table` | `runs/spd/hw`, `runs/spd/table` |
| Error log | `pic_data/errors.jsonl` | `pic_data/spd/errors.jsonl` |

`pic.config.data_path` does the routing; flipping back to PD finds PD's files untouched.
Everything that needs all four outputs (char, fastchar, recal, physics-fit training, capture/sync,
tiled runs, normalise sweep) refuses with fewer SPDs and names the missing outputs; the full
4x4 run then returns the covered rows only. An SPD on USB that is not in `pic_data/spd_map.json`
is refused by id. `python -m pic selftest` runs the whole SPD pipeline on four mock SPDs.

### Bring-up, four SPDs, in order

1. Plug in all four. `python -m spd count` lists each id and its slot (`unmapped` at first).
2. `python -m spd assign <id> PD<n>` for each (one SPD per slot; `none` unmaps).
3. Laser OFF: `python -m spd dark` stores every connected SPD's dark.
4. Laser ON at the lowest setting. SPDs are destroyed by high power: never raise it without
   watching `python -m spd count`. The UI's −10 dBm gives −8.0 dBm at the fibre.
5. Per output: set its brightest heater/port state, then `python -m spd max <id>` (max, then
   dark with the switch parked). All heaters at max is not the brightest state.
6. Settings → Detectors → SPD. Tools → SPDs card must pass (every slot mapped, on USB, dark
   2–200 /s, dark+max stored).
7. Characterize (fastchar), then table capture, then runs.

### Bench facts, 2026-09-26

| What | Measured |
|---|---|
| SPD | `DQ00QQ2C` on PD1, the only one fitted |
| Dark | 16–18 /s |
| Rates at −5 dBm setting, heaters 0 V | P0 72, P1 503, P2 912, P3 49 /s |
| All heaters at max | not brightest: P2 fell to 242 /s |
| Frames | 20 ms, each a per-frame count, not cumulative |
| Laser set → meter (dBm) | −10→−8.0, −5→−6.2, 0→−0.6, 5→4.7, 10→9.75, 12→11.8; driver floor 12.9 mA below 0 dBm |
| TEC thermistor | 5 V – 10k – A0 – NTC – GND; A0 ≈ 2.5 V healthy. `tec_pid.ino` `r` → one `DIAG` line (flashed on the bench board) |

### Not yet run on hardware

- Everything SPD-mode beyond one SPD's counts per input port: `assign`, per-id `max`, `dark`,
  the SPDs card, SPD characterization/table/runs, more than one SPD at once, dropout handling.
- Fit gates are in volts (`characterize.MIN_AMPLITUDE_V` 11 mV, fastchar's ADC-step noise floor)
  and apply unchanged to SPD's 0..1 values; mock SPD fits match PD fits, the bench is unchecked.
- The TEC hardware-target input (Settings). The Tools TEC pins/SPI rows: checked once by a direct
  `DIAG` read, not through the UI.
