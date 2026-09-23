# unitary — PIC1A 4x4: calibrate and run a unitary matvec from the browser

Self-contained copy of the 4x4 runtime and its UI. Carry the directory; nothing outside it
is imported. Works on macOS, Linux and Windows: instruments are found by USB serial number
(`pic/config.py` `USB_SERIAL`), not by device path.

```bash
make            # .venv with numpy, scipy, pyserial, psutil, torch (uv if installed)
make run        # UI on http://127.0.0.1:8744, armed; turns the laser off on start
make test       # every self-test, no hardware
```

CLI, from inside this directory: `.venv/bin/python -m pic {selftest,status,measure,tec,char,fastchar,program,matvec,...} [--mock]`.

| Path | What |
|---|---|
| `ui.py`, `static/` | browser UI: Matvec, Calibration, Diagnostics |
| `pic/` | rig: serial driver, laser, TEC, switch, acquisition, characterization, `Rig` |
| `theory/` | Clements, calibration law, twin, signed intensity matvec, drift, stats |
| `Arduino/` | board firmware (`pic4x4`) and TEC firmware (`tec_pid`) |
| `laser/` | AeroDiode PDMv5 notes and the vendor Windows installer; the driver is `pic/devices/laser.py` |
| `pic_data/` | calibration, fringe fits, the 2026-09-22b transfer table |

A replaced board, TEC Arduino or laser adapter has a new USB serial number: update
`USB_SERIAL`, or pass `--pic-port` / `--laser-port`.

Split from `../4x4/` on 2026-09-23. Left behind there: Ising, the learned surrogates, the
standalone scripts. `pic/sim.py` and `pic/model.py` came along because the self-tests use them.
