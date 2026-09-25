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
python bootstrap.py test     # every self-test, no hardware
```

On macOS and Linux `make`, `make run` and `make test` do the same. Everything runs in Python's
UTF-8 mode (`-X utf8`), because Windows' default cp1252 cannot carry the ±, ° and φ in our logs.

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
| `spd/` | Single-photon detectors by chip id: `PHOTON_COUNT` frames added over an integration time, 0..1 against a stored max (`python -m spd count`, `python -m spd max`), VAUL parser, physical mock |
| `theory/` | Clements, calibration law, twin, signed intensity matvec, drift, stats |
| `Arduino/` | board firmware (`pic4x4`) and TEC firmware (`tec_pid`) |
| `laser/` | AeroDiode PDMv5 notes and the vendor Windows installer; the driver is `pic/devices/laser.py` |
| `pic_data/` | calibration, fringe fits, the 2026-09-22b transfer table |

Instruments are named by chip id (USB serial number), shown on the Tools tab; any `--*-port`
flag takes an id or a role name. A replaced board, TEC Arduino or laser adapter has a new id: update
`USB_SERIAL`, or pass `--pic-port` / `--laser-port`.

Split from `../4x4/` on 2026-09-23. Left behind there: Ising, the learned surrogates, the
standalone scripts. `pic/sim.py` and `pic/model.py` came along because the self-tests use them.
