# PIC

Python control library for the 4x4 unitary Clements mesh on a Quanfluence PIC1A die. The chip has 18 thermo-optic heaters (16 driven by the DAC board), four inputs behind a 1x4 optical switch, four photodiode outputs, a die TEC, and an AeroDiode PDMv5 laser. The library programs a unitary onto the mesh, reads it back, and computes matrix-vector and matrix-matrix products optically from photodiode intensities.

Methods, measurements and results are in the report. This file covers installation and use. The 6x6 chip is documented in `6x6/Readme.md`.

```bash
cd 4x4
```

Every command and import path below is relative to `4x4/`.

## Install

There is no package to install, the directory is the import root. Use the `pic` conda env (Python 3.14) with numpy, scipy, pyserial, torch and matplotlib.

```bash
conda activate pic
./do          # prints the command list
```

`./do` selects an interpreter that has the imports a command needs. Set `PYTHON=/path/to/python` to force one. The 4x4 and 6x6 directories both define packages named `pic` and `theory`, so never put both on one `sys.path`.

## Quick start without hardware

Every hardware command takes `--mock`. The mock is a physical model of the chip, and it reproduces the instrument's refusal paths, so a mock run exercises the same logic as a real one.

```bash
./do selftest                  # every model self-test plus a mock rig pass
./do status --mock             # laser, TEC and calibration state
./do char --mock               # sweep the heaters and fit their fringes
./do program --mock --random   # decompose a random unitary onto the chip, read it back
```

## Programming a unitary

A target unitary maps to heater phases by a closed-form Clements decomposition, and to voltages through the heater law `phi = pi (V/Vpi)^2 + phi0`.

```python
import numpy as np
from pic import Rig
from theory.clements import random_unitary

U = random_unitary(np.random.default_rng(0))

with Rig(laser="mock", board="mock", tec="mock", switch="mock") as rig:
    with rig.session(duration_s=30, power_dbm=8):
        volts, reachable, y = rig.realize(U)   # program U, read the four outputs
```

`reachable` is False for a heater whose target phase has no image inside its voltage ceiling. The decomposition and the calibration also run without a rig.

```python
from theory.clements import decompose, reconstruct
from theory.calib import Calibration
from theory.program import phases_for, unitary_for, volts_for, refine

volts, reachable = volts_for(U, Calibration.load_or_nominal())
```

- `refine(U, twin)` polishes the phases by gradient descent through the twin. It is a local step warm-started from the exact decomposition.
- A diagonal phase screen on the outputs is invisible in intensity, so a programmed unitary is defined up to it.
- The result is as accurate as `pic_data/calib.json`. Characterize with `./do char --write` before programming a real chip.
- `Rig(..., dynamic=True)` pre-distorts each target by the drift inferred from a four-port probe.

## Matrix-vector product

`y = B x` for a real signed `B` is computed from photodiode intensities alone, with no homodyne. A plan holds the mesh programs that encode `B`. Each program is applied to the positive and negative parts of `x`, and the difference is the signed result. A 2x2 block is what the die encodes.

```python
from pic import matvec
from theory.twin import Twin
from theory.intensity_matvec import plan_matvec, twin_probe

box, twin = matvec.bench_box(), Twin()      # reachable heater set from the calibration
probe = twin_probe(twin, box)               # simulated instrument, no hardware
plan = plan_matvec(B, box, twin)            # B is 2x2
y = plan.matvec(probe, x)
plan.predict()                              # the matrix the plan encodes, compare to B
```

On the rig, planning uses a measured transfer table of the die and the probe is the rig itself.

```python
tab = matvec.load_transfers()                          # pic_data/sessions/*/raw_transfers*.json
with Rig(laser="hw", board="hw", tec="none", switch="hw") as rig:
    with rig.session(duration_s=120, power_dbm=8):
        res = matvec.run(rig, B, vectors=V, transfers=tab)   # V holds one vector per row
```

`res` holds the plan, the measured matrix and its error against `B`, one row per vector, and the cost in heater writes, switch moves and reads. `./do matvec` runs the same path. The hardware run needs the table, and under `--mock` the table describes a different chip, so run it on the real device.

## Matrix-matrix product

`Y = B X` under one program serves every column of `X` from that program's photodiode reads. A matrix larger than 2x2 is cut into 2x2 tiles, each tile is encoded in turn, and the partial products are summed in software.

```python
from theory.intensity_matvec import plan_matvec
from theory.matmat import matmat, plan_block

Y = matmat(plan, probe, X)                  # one 2x2 program, all columns of X
bp = plan_block(B4, box, twin, k=2)         # any size, tiled into 2x2 blocks
Y = bp.matmat(probe, X4)                    # tile-major, each program written once
bp.programs                                 # heater writes for the whole product
```

On the rig:

```python
res = matvec.run_block(rig, B4, k=2, X=X4, transfers=tab)   # res["Y"], res["Y_true"], res["cost"]
```

From the command line, `./do matvec --block 2 --cols 4` runs an 8x8 target against four columns.

## Unitary operands

A product of orthogonal matrices is orthogonal. `compose` encodes two random orthogonal matrices A and B, reads A back, feeds that measured matrix through B, and reports the distance of the result from the orthogonal group. That distance needs no ground truth. When the operands are known to be orthogonal, the polar projection (the nearest orthogonal matrix) removes the error that lies off the group.

```python
from theory.matmat import compose, polar, qr_orth, manifold_dist

g = compose(box, twin, k=2, trials=4)
g["dist_c"], g["raw"], g["polar"], g["qr"]   # distance, then error raw, polar-projected, QR-projected

C = polar(C_measured)                        # project a measured product
manifold_dist(C_measured)                    # distance to the nearest orthogonal matrix
```

On the rig, pass `probe=matvec.rig_probe(rig, calib)`. From the command line, `./do matvec --unitary --cols 4` composes four pairs. Do not project a target that is not orthogonal, the error grows.

## Ising

```python
from pic import ising
from theory.ising import random_ising

res = ising.solve(rig, random_ising(np.random.default_rng(0), n=3))   # inside a session
```

`solve` encodes the couplings, holds the state, reads the four-port transfer matrix, decodes the spins and scores them against the brute-force ground state. `./do ising` runs it from the command line.

## Rig API

`Rig` holds the laser, board, TEC, switch and an optional model, each either `"hw"` or `"mock"`. It is a context manager, and every lit measurement runs inside `session`.

| Call | Effect |
|---|---|
| `Rig(laser, board, tec, switch, model)` | `board` also takes `"sim"`, the device as delivered. `tec` takes `"none"` or a serial port, `switch` takes `"hw"`, `"none"` or a serial port, `model` takes `"mock"`, `"twin"` or `"dpnn"` |
| `session(duration_s, power_dbm)` | guarded laser session with a watchdog timer, an emission check and a TEC settle gate |
| `select_input(port)` | route the laser to input port 0 to 3 |
| `measure(v)` | set 16 DAC volts, return the raw photodiode volts |
| `outputs(v)` | as `measure`, reduced to the four mesh outputs |
| `sweep_ports()` | all four input ports in one round trip, returns the transfer matrix `T[pd, port]` or `None` |
| `program(U)` | target unitary to `(volts, reachable mask)` |
| `realize(U)` | `program` followed by `outputs` |
| `predict(v)` | forward prediction of the attached model |
| `status()` | laser, TEC, switch, drift and calibration state |

## Command line

| Command | Effect |
|---|---|
| `./do selftest` | model self-tests plus a mock rig pass |
| `./do e2e` | staged bring-up check that names the broken link of the chain |
| `./do laser state 1`, `state 0`, `set 8`, `set 5 -t 30`, `status` | laser on, off, ramp to +8 dBm, ramp and hold 30 s then off, readout |
| `./do status` | laser, TEC and calibration state |
| `./do measure [--ports 0,1,2,3]` | live photodiode dashboard, optionally cycling the input switch |
| `./do tec [setpoint]` | live chip temperature, retarget when a setpoint is given |
| `./do char [--write]` | sweep every heater, fit fringes, write `pic_data/calib.json` |
| `./do calibrate`, `recal`, `sync` | agree every rig constant, re-anchor phi0 fast, carry the transfer table to the current chip state |
| `./do program --random`, `--target FILE` | decompose a unitary onto the chip and read it back |
| `./do matvec` | matrix-vector product. `--block K --cols N` tiles and serves N columns, `--unitary` composes orthogonal pairs, `--planner table` plans from the measured table |
| `./do ising` | Ising runner |
| `./do dataset --combos N` | N heater states times 4 ports into `pic_data/transfer.csv` |
| `./do train --rounds N` | train both surrogates from hardware |
| `./do figs` | redraw the rig diagrams into `figs/`, no hardware |

Append `-h` to any subcommand for its flags.

## Running on hardware

- Flash `Arduino/pic4x4/pic4x4.ino` from the Arduino IDE before the first run.
- `Rig.close()` turns the laser off. Pass `keep_laser=True` to leave it lit.
- `laser_status == 1` does not prove the diode is emitting. Each session baselines the optical power monitor off against on and reports whether light was emitted.
- Only one process may hold a serial port. When more than one device is plugged in, pass explicit ports with `laser_port=` and `pic_port=` (or `--pic-port`), or set `$PIC4_PORT`.
- Heater voltage ceilings are per channel. On open, `Rig` compares the firmware's ceiling table with `pic.config.VOLTAGE_MAX_CH` and refuses on any disagreement.
- `./do char --write` overwrites `pic_data/calib.json`. Programming accuracy depends on this file, so never write it from a mock run.

## Layout

| Path | Contents |
|---|---|
| `theory/` | pure math with no hardware. Clements decomposition, heater law, twin, programming, drift, Ising, matvec and matmat |
| `pic/` | the rig. Serial drivers, laser, TEC, switch, acquisition, characterization, the `Rig` facade and the computation runners |
| `learn/` | surrogates, the 52-parameter physics fit and the pruned network |
| `Arduino/` | board firmware |
| `pic_data/` | calibration files and measured tables |

`theory` imports neither `pic` nor `learn`. `pic` imports `theory`. `learn` imports both.

## Tests

There is no pytest suite. Each module carries a `_selftest` run as `python -m <module>`, for example `python -m theory.clements`, `theory.twin`, `theory.calib`, `theory.program`, `learn.unitary_fit` and `learn.dpnn`. `./do selftest` runs them together with a mock rig pass.
