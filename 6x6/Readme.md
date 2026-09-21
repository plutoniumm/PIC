# PIC 6x6

Python control library for the 6x6 MZI-mesh chip (PIC B). Two 6x6 meshes sit on a splitting tree. Odd columns dump an edge output, so each mesh is a non-unitary contraction and matrices are reached by optimization through a surrogate instead of a formula. The board is 128 DAC channels of thermo-optic heaters in, 14 photodiodes out, lit by an AeroDiode PDMv5 laser.

This directory is an archive. The chip is characterized, optical Ising computing is demonstrated, and the code is kept as a method reference for `../4x4`. Methods, measurements and results are in the report. This file covers installation and use.

Every command and import path below is relative to `6x6/`.

## Install

There is no package to install, the directory is the import root. Use the `pic` conda env (Python 3.14) with numpy, scipy, pyserial, torch and matplotlib.

```bash
conda activate pic
./do          # prints the command list
```

`./do` selects an interpreter that has the imports a command needs. Set `PYTHON=/path/to/python` to force one. The 4x4 and 6x6 directories both define packages named `pic` and `theory`, so never put both on one `sys.path`.

## Quick start without hardware

```bash
python -m pic measure --mock            # live photodiode stream from the simulator
python -m pic bringup step --selftest   # settling check
```

## Python API

`Rig` is the single entry point. It holds the laser, the board and an optional model, each either `"hw"` or `"mock"`. It is a context manager, and every lit measurement runs inside `session`.

```python
import numpy as np
from pic import Rig

with Rig(laser="mock", board="mock") as rig:
    with rig.session(duration_s=30, power_dbm=5):
        y = rig.measure(np.zeros(128))   # 128 DAC volts in, 14 photodiode volts out
```

| Call | Effect |
|---|---|
| `Rig(laser, board, model)` | `model` takes `"mock"`, `"twin"` or `"dpnn"`, or `None` |
| `session(duration_s, power_dbm)` | guarded laser session with a watchdog timer and an emission check |
| `measure(v)` | set the DAC volts, return the 14 raw photodiode volts |
| `predict(H)` | forward prediction of the attached model |

Other modules in `pic`.

| Module | Contents |
|---|---|
| `pic.interface` | `PIC` serial driver that matches the firmware, and `MockPIC` |
| `pic.acquisition` | sweeps, settled reads, settling-time tools, reference-arm homodyne |
| `pic.layout`, `pic.wiring` | the chip as a scene graph, and the DAC channel to heater map |
| `pic.model` | the swappable predictors behind `Rig.predict` |
| `pic.compute.ising`, `pic.compute.matvec` | hardware runners for the Ising machine and the signed intensity matrix-vector product |
| `pic.bringup` | bring-up checks and thermal settling characterization |

`theory/` is pure math. It needs no hardware and is imported by `pic.compute` and `pic.model`. It holds the differentiable twin, the Ising, matrix-vector and homodyne kernels, and the drift-correction window.

## Command line

| Command | Effect |
|---|---|
| `./do laser state 1`, `state 0` | laser on, off |
| `./do laser set 10`, `set 5 -t 30` | ramp to +10 dBm, ramp and hold 30 s then off |
| `./do laser status` | full device readout |
| `./do measure [out.csv]` | live photodiode dashboard, or the raw stream as CSV |
| `./do heaters` | re-anchor the null of every characterized heater (`--limit N` for a subset) |
| `./do heaters schedule` | build the parallel-sweep schedule, no hardware |
| `./do heaters parallel --channels 85,109` | lockstep parallel sweep |
| `python -m pic ising`, `matvec`, `bringup` | computation runners and bring-up tests, `-h` lists their flags |

The heater characterization pipeline (probe, census, fringe fit, config, drift re-anchor, parallel scheduler) is documented in `char.md`.

## Running on hardware

- Flash `Arduino/pic128/pic128.ino` from the Arduino IDE before the first run. Firmware protocol is 115200 baud, DAC volts in, mean photodiode volts out.
- `laser_status == 1` does not prove the diode is emitting. Each session baselines the optical power monitor off against on and reports whether light was emitted.
- The board is a numeric CH340 serial device and the laser is an FTDI device, and the board's port glob also matches the laser. When both are plugged in, pass explicit ports with `laser_port=` and `pic_port=`, or set `$PIC_PORT`.
- Only one process may hold a serial port. Stop any running reader before starting another.
- The firmware clamps every DAC voltage to 0 to 5 V (`setDAC` in `pic128.ino`), so a voltage above 5 V is silently lowered.

## Layout

| Path | Contents |
|---|---|
| `pic/` | the rig. Serial drivers, laser, acquisition, session layer, the `Rig` facade, computation runners |
| `theory/` | pure-math models and algorithms |
| `src/` | characterization and modelling support that `pic` imports |
| `scripts/` | analysis, schematic and hardware scripts, indexed in `scripts/README.md` |
| `Arduino/` | board firmware |
| `pic_data/` | datasets, the DAC to heater maps and the PIC B configuration |
| `paper/` | the manuscript source |

## Tests

There is no pytest suite. `python -m pic bringup step --selftest` and `python -m pic measure --mock` run on the mock rig, which is a model of the chip and reproduces the instrument's refusal paths.
