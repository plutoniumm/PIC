# 2026-08-27 — first full characterization of the PIC1A 4x4

Chip at ~25-27 C (uncontrolled, see TEC below), laser at +5 dBm, 15 levels, 4 input ports.

## What was measured

`calib.json` is the operative calibration; `char_results.json` carries the per-heater fits
including each channel's 4-port x 4-detector modulation fingerprint, which is what locates a
heater in the mesh. Both are the run logged in `char_run.log`.

### Heaters — 4 of 6 internal phase shifters characterized

| DAC | pad | MZI | R | ceiling | Vpi | span | r2 |
|---|---|---|---|---|---|---|---|
| 5 | H3 | 1 | 113.5 | 3.0 V | 4.09 | 0.54 pi | 1.000 |
| 4 | H4 | 2 | 57.8 | 1.5 V | — | 0.13 pi | rejected |
| 3 | H8 | 3 | 114.1 | 3.0 V | 4.84 | 0.38 pi | 1.000 |
| 2 | H11 | 4 | 118.8 | 3.0 V | 5.00 | 0.36 pi | 1.000 |
| 1 | H12 | 5 | 62.3 | 1.5 V | — | 0.13 pi | rejected |
| 0 | H15 | 6 | 114.1 | 3.0 V | 4.15 | 0.52 pi | 1.000 |

The two rejections are physics, not measurement failure: the 60R heaters are capped at 1.5 V
by the 30 mA rating, which buys 0.13 pi -- about 23 degrees of phase, too little to move the
outputs measurably. **MZI2 and MZI5 are not tunable within rating.** The `1.5` entries in
`calib.json`'s `vpi` are the nominal placeholder, NOT measurements; check `n_ok` before use.

### Photodiodes

| | dark | full scale | contrast | read noise |
|---|---|---|---|---|
| PD0 | 37.82 mV | 227.88 mV | 27.9 dB | ~26 mV |
| PD1 | 1.35 mV | 157.62 mV | 17.3 dB | 1.8 mV |
| PD2 | 0.12 mV | 159.46 mV | 25.2 dB | 4.9 mV |
| PD3 | 0.00 mV | 153.35 mV | 27.0 dB | 0.0 mV |

PD0 is ~14x noisier than its neighbours and carries a real 37.8 mV DC pedestal. It is usable
but must be ranked by SNR, never by amplitude -- ranking by amplitude let PD0's noise win
every channel and produced eight false identifications earlier the same day.

Input coupling at full scale: port 0 430.4, port 1 245.3, port 2 293.6, port 3 172.3 mV.
Port 3 is 4.0 dB below port 0, independently reproducing the archive's bad-launch claim.

### Laser

`laser_cal.csv` is the raw sweep. `P[mW] = 0.4223 * setpoint - 1.890`, valid setpoint 8-70,
rms residual 0.067 mW. Slope efficiency 163.6 uW/mA, threshold 11.7 mA, 2.578 mA per count.
Supersedes the 2026-07-08 fit, which was 23% low in slope with an unchanged threshold -- the
signature of a measurement-scale change, not a diode that aged. Absolute scale is therefore
only as good as the meter coupling was; the shape is solid. A reproducible kink sits at
setpoint 82 and the current clamps at 250 mA around setpoint 95, so the usable ceiling is set
at the end of the clean line (27.67 mW / 14.4 dBm) rather than at maximum light.

### TEC

`tec_coupling_test.log`. Heaters used as a calibrated heat source on the die:

    baseline, heaters OFF    28.68 -> 27.25 C  (-1.43)
    heaters ALL AT CEILING   27.24 -> 29.28 C  (+2.04)   856 mW
    heaters OFF again        29.28 -> 27.03 C  (-2.25)

Reversible, so both questions are answered at once: the thermistor IS coupled to the die
(heater power moves it by degrees) and the cooler DOES actuate (it recovers the load).
Spare cooling authority is roughly equal to the full heater load, so at a reachable setpoint
the die should hold steady across any heater configuration.

**RESOLVED: a single unseated SPI pin.** For most of the session the LT8722 received nothing.
The PID ran, telemetry streamed plausible temperatures, and every register write returned
without error -- but the part was never configured out of its default disabled state, drew no
current from the supply, and did not respond to either polarity. Four register "fixes" were
made against a part that could not hear them, and the original sketch failed identically,
which is what finally exonerated the code.

The diagnostic that settles it in seconds, now permanent in the sketch: read SPIS_STATUS,
SPIS_COMMAND and SPIS_DAC_ILIMN back and print them. Three *different* values means the bus
is alive. Three *identical* values -- we saw 0x87A50000 on every address, with a valid CRC --
is impossible for three distinct registers and means the part is not driving MISO at all.

With the pin seated, at setpoint 25.00 over 90 s:

    mean 25.032 C   sd 0.089 C   error +0.032 C
    drive -1.06 .. +1.91         railed 0% of samples

Authority in both directions, never saturated, inside the +-0.05 C band. The one code change
worth keeping is ILIMN 0x1B5 -> 0x4B; the clamp values in the original are correct as written
(the digest computes OV_CLAMP 0x04 = +6.3 V, so it never restricted anything).

**The cooler works** -- it pulled the die 28.68 -> 27.25 C in 45 s
with the heaters off. Earlier tests read as "not actuating" only because they ran at
equilibrium, where it had already reached its floor of ~26.3 C and had nothing left to give.
It cannot reach the 25.0 C setpoint against ambient plus load, so the PID sits railed.

Two caveats in `Arduino/tec_pid/tec_pid.ino`: `calibrationOffset = 1.3` is added to every
reported temperature, so the loop servos 1.3 C colder than it displays; and `readTemperature`
returns the *previous* value when the ADC rails, so a disconnected thermistor freezes the
reading at a plausible number instead of failing loudly.

Heater dissipation is 856 mW with all 13 drivable channels at their ceilings, and the die
reached 28.7 C under sweep load -- so heater power moves the die by degrees, and the
thermistor sees it.

## Firmware fixed this session

- `ILIMN` 0x1B5 -> 0x4B. The two LT8722 current-limit registers have *different* transfer
  functions (Eq. 9 vs Eq. 10) and cannot share a code; 0x1B5 meant -5.80 A on the negative
  side, past the part's own -4.5 A rating.
- Per-channel `VMAX[]` in `pic4x4.ino`, and the same array enforced in the host driver and in
  sweep generation. A flat 3.0 V clamp puts 52 mA through a 58 ohm heater.
- ADC averaging 5 -> 16.
- `P0` added: the Sercalo's open channel, giving a true optical dark for the normalisation.

## The mapping, which was the standing blocker

`ananya/4X4MZI_Test2 (1).pdf` tables the six internal phase shifters by MZI with pins and
resistances; `ananya/4X4MZI_REPORT.pdf` and the wiring notebook table the same pins under the
H1-H18 names. Matching on (pins, resistance) identifies all six, and the report's optical
paths place report-MZI k at mesh index k-1. Net: **DAC k drives theta of mesh index 5-k.**

Confirmed on hardware by the fingerprints, on two structural predictions that could have
failed:
- MZI1 (DAC5, column 0, rails 0-1) responds at ports 0 and 1 and is silent at 2 and 3 --
  a first-column MZI on rails 0-1 is unreachable from the other half of the mesh.
- MZI6 (DAC0, column 3, rails 1-2) modulates only PD1 and PD2; PD3 reads 0.1-0.2 mV.

## Transfer dataset and what it says about the model

`transfer_8dbm.csv` -- 400 combinations x 4 ports = 1600 rows, the same DAC state measured
at every input, which no dataset in the archive does. That is what makes `T = |U|^2`
assemblable.

**How unitary the chip is.** Doubly-stochastic error: rows (detectors) 0.099, columns
(input ports) 0.250. The column figure is mostly the 3.6 dB launch imbalance, measured
independently three ways today. Total power per port varies **16.8% across combinations** --
a lossless unitary conserves power whatever the heaters do, so the mesh has
configuration-dependent loss, and `theory/twin.py` (exactly unitary by construction) cannot
represent that at any parameter values.

**The joint fit does not converge**: `learn.unitary_fit.fit_staged` plateaus at validation
R^2 = 0.26, worse than the R^2 = 0.42 that CLAUDE.md records for an *unseeded* fit. Three
causes, and Vpi is not the main one:
 1. the twin is lossless and the chip is not (above);
 2. the phi/alpha role table is wrong -- it predicts held-out data at r = -0.04 where the
    best assignment reaches +0.50 (see `theory/layout.py`);
 3. Vpi is uncertain to +-12% and no fitter can fix that (below).

**Why Vpi will not pin down.** Span is (Vmax/Vpi)^2 and every heater covers 0.08-0.54 pi, so
each fringe is a *fragment* of a cosine, within which amplitude, offset and period trade off
against each other. Measured on synthetic data with a known answer: fitting one trace gives
24% mean Vpi error, fitting all eight of a heater's (port, base) traces with shared Vpi and
phi0 gives 12% (`theory.calib.fit_shared`). Better, not solved. Reaching pi needs ~4.5 V on
the 120R heaters against a 3 V ceiling, so the information is not in the measurement.
The route with real headroom is the joint fit, once the model above is right.
