# laser/ — AeroDiode PDMv5 laser control

Safe on/off + current-ramp control for the lab's **AeroDiode PDMv5** current driver,
straight from Mac/Linux Python. **No Windows, no Wine, no vendor GUI, no vendor library** —
just pyserial talking the PDMv5 serial protocol. Confirmed on hardware (first light
2026-07-07: OPM −50 → −25 dBm at 20 mA).

## Install & run

```
pip install -r requirements.txt          # just pyserial

python laser.py hw 1      # power ON:  check key+interlocks, TEC on, enable at floor
python laser.py set 20    # smoothly ramp to 20 mA (up or down)
python laser.py hw 0      # ramp down to 0, then disable output + TEC off
python laser.py status    # full device readout
```

Run from the repo root as `python -m laser hw 1`, or from inside `laser/` as
`python laser.py hw 1`. The port is auto-detected (the FTDI adapter, e.g.
`/dev/cu.usbserial-AU05XLI8`); override with `--port /dev/cu.usbserial-XXX`.

```python
from laser import Laser
with Laser() as L:      # autodetects the port; serial close on exit does NOT change laser state
    L.hw_on()           # safe power-on (raises LaserError if key off / interlock open / not connected)
    L.set(20)           # smooth ramp to 20 mA
    L.hw_off()          # ramp to 0, verify, disable, TEC off
```

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
| `requirements.txt` | pyserial |
| `ControlSoftware_AeroDIODE_Setup_2.10.0.exe` | vendor Windows GUI installer — reference only, **not needed to run this**; safe to delete |

## Protocol notes (in `pdmv5.py`)

Frame `[LEN][ADDR][CMD][DATA][CHK]`, `CHK=(XOR(prior)−1)&0xFF`, register id 2 bytes big-endian,
values F32 big-endian / U08 / U32; 125000 baud. Commands: read-addr `0x01`, version `0x02`,
write `0x10`, read `0x11`, apply `0x12`, measure `0x14`. Map reverse-engineered from the
vendor `PDMv5.dll` and confirmed against captured GUI traffic. Key registers: cw_current `0x3f`,
cw_enable `0x3c`, master_enable `0x20`, tec `0x14`, operating_mode `0x41` (0=ACC), cw_source
`0x3d` (0=INT). Measures: key `0x00`, interlocks `0x01`/`0x02`, diode current `0x1f`, diode
voltage `0x1e`, monitor optical power `0x2a`, alarms `0x46`.

## Known issues

- **Current monitor reads low** (~8 mA at 20 mA setpoint) — uncalibrated scale/offset. Fine
  for control; characterize (setpoint vs OPM) if you need absolute current.
- **TEC calibration unset** (`TECSlope=0`) → diode-temp readout wanders and the TEC doesn't
  regulate cleanly. Fix before any temperature-sensitive work (e.g. TEC-drift test).
