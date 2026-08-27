# laser/ — AeroDiode PDMv5 laser control

Safe on/off + current-ramp control for the lab's **AeroDiode PDMv5** current driver,
straight from Mac/Linux Python. **No Windows, no Wine, no vendor GUI, no vendor library** —
just pyserial talking the PDMv5 serial protocol. Confirmed on hardware (first light
2026-07-07: OPM −50 → −25 dBm at 20 mA).

## Install & run

```
pip install -r requirements.txt          # just pyserial
```

Shortest path — the repo-root `./do` dispatcher (`state` is an alias for `hw`):

```
./do laser state 1     # power ON:  check key+interlocks, TEC off, enable at floor
./do laser set 10      # ramp to +10 dBm output (uses the optical calibration)
./do laser state 0     # ramp down to 0, then disable output + TEC off
./do laser status      # full device readout
```

`set` takes **output power in dBm** (ceiling +15.2 dBm ≈ 32.8 mW, above which the diode
current-clamps). Use `set --raw N` to write a raw setpoint instead (for re-calibration).

`./do` auto-picks the first python that can `import serial` (override with
`PYTHON=/path/to/python ./do ...`). Or call the module directly:

```
python -m laser hw 1      # from the repo root (laser/ is a package)
```

The port is auto-detected (the FTDI adapter, e.g. `/dev/cu.usbserial-AU05XLI8`);
override with `--port /dev/cu.usbserial-XXX` on the `python -m laser` form.

```python
from laser.laser import Laser
with Laser() as L:      # autodetects the port; serial close on exit does NOT change laser state
    L.hw_on()           # safe power-on (raises LaserError if key off / interlock open / not connected)
    L.set(10)           # smooth ramp to +10 dBm output
    L.hw_off()          # ramp to 0, verify, disable, TEC off
```

## Calibration (2026-07-08, external power meter)

The device "setpoint" is a **raw drive register, not mA and not mW**. Two fixed maps,
both measured on hardware and baked into `laser.py`:

- **setpoint → current:** `I[mA] ≈ 2.5·setpoint − 7` (linear, R²≈1). Current **clamps at
  250 mA** (the diode avg rating, = `CEIL_MA`) around setpoint ~100 — the optical knee.
- **setpoint → output:** `P[mW] ≈ 0.344·setpoint − 1.44`; threshold ~setpoint 4, slope
  efficiency ~0.14 mW/mA, **ceiling 32.8 mW (+15.2 dBm)**. `set <dBm>` inverts this.

So driving above setpoint ~100 (+15.2 dBm) adds only heat, not light. Raw sweep +
external L-I data and the plot are in `runs/laser_cal_*.csv` / `laser_extcal_*.csv` /
`laser_calibration_final.png`. Re-run `set --raw N` sweeps against a meter to refresh
`CAL_A`/`CAL_B`/`PMAX_MW` if the diode or coupling changes.

## Safety (first priority — enforced in `laser.py`)

- **`hw 1` refuses to enable** unless the driver responds on serial (connected) **and** the
  **key switch is ON** **and** both interlocks are closed. It tells you exactly what to fix
  (e.g. "turn the KEY switch to ON").
- **Output is never disabled while current flows.** Every off-path ramps the setpoint to 0
  and *verifies* the measured current is at baseline before dropping the enables; if it isn't,
  it refuses to disable and leaves the output on for you to investigate.
- **`set` won't enable a laser that is off** — run `hw 1` first (it will not silently emit).
- **Ceilings hard-clamped** to `CEIL_MA` (25 mA) so the hardware can't exceed the working cap
  even on a bug; `set` clamps to `MAX_MA` (20 mA). Raise these constants deliberately — the
  diode is rated ~250 mA average.
- **Always run `hw 0` before turning the key off or unplugging.**

The module refuses to turn on into a **0 mA setpoint** (pulsed-diode driver quirk), so `hw 1`
enables at a small non-zero **FLOOR** current (5 mA); every change after that ramps smoothly.

## Files

| file | role |
|---|---|
| `laser.py` | `Laser` policy class (on/off/ramp + safety) and the CLI |
| `pdmv5.py` | `PDMv5` serial driver: framing, register map, read/write/measure |
| `__main__.py` | entry point for `python -m laser` (delegates to `laser.main`) |
| `requirements.txt` | pyserial |
| `ControlSoftware_AeroDIODE_Setup_2.10.0.exe` | vendor Windows GUI installer — reference, and the proper tool for **TEC calibration** (see Known issues) |

## Protocol notes (in `pdmv5.py`)

Frame `[LEN][ADDR][CMD][DATA][CHK]`, `CHK=(XOR(prior)−1)&0xFF`, register id 2 bytes big-endian,
values F32 big-endian / U08 / U32; 125000 baud. Commands: read-addr `0x01`, version `0x02`,
write `0x10`, read `0x11`, apply `0x12`, measure `0x14`. Map reverse-engineered from the
vendor `PDMv5.dll` and confirmed against captured GUI traffic. Key registers: cw_current `0x3f`,
cw_enable `0x3c`, master_enable `0x20`, tec `0x14`, operating_mode `0x41` (0=ACC), cw_source
`0x3d` (0=INT). Measures: key `0x00`, interlocks `0x01`/`0x02`, diode current `0x1f`, diode
voltage `0x1e`, monitor optical power `0x2a`, alarms `0x46`.

## Known issues

- **The enable needs a status-read handshake after a key cycle.** Proven on hardware (A/B):
  after a key power-cycle the driver refuses to latch the master until the host has issued
  `measure()` calls (reads its alarms/status) — a safety handshake, once per power-on. A clean
  enable loop that only polls `laser_status` hangs for >60 s; the same loop that also calls
  `measure("alarms")`/`measure("diode_cw_current")` each poll latches in a few seconds. So
  `_latch_master` re-does the full enable (`cw_current`→CW-enable→master) each attempt **and
  measures the alarms+current each poll**. Do NOT strip those "useless" reads out — they're the
  gate. (Warm on/off cycling works without them because the handshake is already satisfied for
  that power session.) Ceilings/TEC are clamped after the latch; enable at a non-zero floor.
- **TEC is uncalibrated → we run it OFF (`USE_TEC=False`).** Both the slope (`TECSlope=0x1a`)
  and the calibration table (`size=0x18`) are blank on this unit, so enabling the TEC drives a
  thrashing open-loop control effort with no calibration behind it. With the TEC off the diode
  sits at the lab's controlled ~25 °C (measured stable), the benign `0x10` alarm doesn't block
  emission, and `state 1` enables reliably. The temperature *readout* and the factory drive
  limits (TEC 3 A / 4.3 V, temp window 15–55 °C) are intact. To properly calibrate later, use
  the vendor GUI (the installer) or write the per-unit table+slope via the reversed register
  map; then flip `USE_TEC=True`. Open-loop is fine at low/moderate power; at full 250 mA the
  diode self-heats without the TEC, which shifts output and isn't ideal for sustained high power.
- **Current monitor reads low / noisy** (~8–15 mA at floor) — uncalibrated scale/offset. Fine
  for control; characterize (setpoint vs OPM) if you need absolute current.
