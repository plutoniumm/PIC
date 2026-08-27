"""Hardware constants for the 4x4 chip: the firmware protocol, the DAC/ADC widths, and
the operating limits.

Everything here that is not yet confirmed against the delivered board is marked
PROVISIONAL. The 6x6 taught this the hard way -- three different voltage ranges lived in
three layers of that stack and disagreed silently -- so the host range, the firmware
clamp and the DAC reference are all stated here and the firmware sketch reads the same
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.calib import VOLTAGE_MAX, VPI_NOMINAL  # noqa: F401  (one source for both)
from theory.layout import N_HEATERS

BAUD_RATE = 115200
RESET_WAIT_S = 2.0  # the board resets when the serial port is opened
READY_BANNER = "pic4x4 ready"

NUM_DAC = 16          # DAC81416 channels; must match NUM_DAC in pic4x4.ino
NUM_ADC_RAW = 4       # A0..A3, one per mesh output through a PD-TIA (mrunal/PD_TIA_circuit.pdf)
OUT_PDS = (0, 1, 2, 3)  # the four mesh outputs, in rail order
MON_PDS = ()          # this block carries no tap monitors
NUM_OUT = len(OUT_PDS)

# The DAC81416 has 16 channels on one chip at CS 10, and the packaged part exposes 18 heater
# pads -- so two heaters have no driver at all. Of the 16, three (H18, H14, H6) were never
# resistance-checked and are held dark, leaving 13 drivable. Map and resistances are from
# Arduino/pin_check/pin_check.ino, which is the notebook decode.
DAC_HEATER = ("H15", "H12", "H11", "H8", "H4", "H3", "H18", "H9",
              "H14", "H13", "H10", "H6", "H7", "H5", "H2", "H1")
HEATER_OHMS = (114.1, 62.3, 118.8, 114.1, 57.8, 113.5, None, 57.2,
               None, 114.9, 56.4, None, 116.1, 117.1, 116.8, 114.1)
# Raised from the documents' 30 mA on a deliberate call: nothing in ananya/ or mrunal/ gives
# an absolute maximum, a derating curve or a source for the 30 -- it is asserted twice and
# never justified. Span goes as V^2, so 40 mA buys (40/30)^2 = 1.78x more phase on every
# channel -- enough to put several heaters past pi, which is the threshold that decides
# whether an MZI can reach every splitting ratio. The risk is lifetime, not immediate
# failure; nothing here is a datasheet limit. Put this back to 30 to revert everything.
HEATER_MAX_MA = 40.0

# Per-channel ceiling, not a global one. The heaters come in two resistance groups and a
# single 3 V clamp puts 52 mA through the 60R group -- 1.7x its rating. `None` resistance
# means no confirmed value, so that channel stays at 0 V.
# Per channel from its own measured resistance, not a two-group approximation: V = I*R, so
# a 118.8 ohm heater may take more than a 113.5 ohm one at the same current. Rounded down to
# 0.05 V so a rounding error cannot push a channel over the limit.
VOLTAGE_MAX_CH = tuple(0.0 if r is None
                       else float(int(1e-3 * HEATER_MAX_MA * r / 0.05) * 0.05)
                       for r in HEATER_OHMS)
WIRED_DACS = tuple(i for i, v in enumerate(VOLTAGE_MAX_CH) if v > 0)   # 13 channels

# One command scale for every channel. Callers work in "drive volts" 0..DRIVE_MAX_V and the
# scale opens each channel up to its own real ceiling, so nothing above has to carry a
# per-channel limit around. The 3 V channels get x2, the 1.5 V channels x1, the dark ones 0.
#
# Quantisation survives it: the DAC is 16 bits over 5 V, so an LSB is 76 uV and doubling it
# still leaves 39,300 steps across a 3 V channel. Phase goes as V^2, so the coarsest phase
# step sits at full drive and is pi*(2*76uV)*2*Vmax/Vpi^2 -- under 1e-4 rad. Not a limit.
DRIVE_MAX_V = 1.5
DRIVE_SCALE = tuple(v / DRIVE_MAX_V for v in VOLTAGE_MAX_CH)


def drive_to_volts(drive):
    """Uniform 0..DRIVE_MAX_V command -> real per-channel volts.

    This equalises the *command range*, not the phase range: span goes as (Vmax/Vpi)^2, so
    a 60R heater at 1.5 V still covers ~0.11 pi against a 120R heater's ~0.43 pi. Uniform
    drive makes the model and the surrogates rectangular; it does not buy reach."""
    d = np.clip(np.asarray(drive, float), 0.0, DRIVE_MAX_V)
    return d * np.asarray(DRIVE_SCALE, float)


def volts_to_drive(volts):
    """Real per-channel volts -> the uniform command, for code that has to cross back.

    A dark channel has no inverse and maps to 0, which is the only voltage it can be at
    anyway. Volts above a channel's ceiling clamp, exactly as the driver clamps them."""
    s = np.asarray(DRIVE_SCALE, float)
    live = s > 0
    return np.clip(np.asarray(volts, float) / np.where(live, s, 1.0), 0.0, DRIVE_MAX_V) * live

VOLTAGE_MIN = 0.0
FIRMWARE_VMAX = 3.0  # clamp in pic4x4.ino; must match, or host volts vanish at the DAC
DAC_REF_V = 5.0
DAC_BITS = 16
ADC_REF_V = 5.0
ADC_BITS = 10

ADC_AVG_N = 16       # full ADC sweeps averaged per firmware reply; must match pic4x4.ino AVG_N
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_SETTLE_S = 0.5  # thermo-optic settle before a read (mrunal/Setup.ino HEATER_DELAY_MS)

# 1x4 optical input switch (Sercalo) in front of U_IN1..4, on its own serial line. It selects
# one input port at a time, which is what makes a per-port characterization automatable.
SWITCH_BAUD = 9600
SWITCH_SETTLE_S = 1.0  # mrunal/Setup.ino SWITCH_DELAY_MS

# Chip TEC. Unlike the 6x6 rig this one has a calibrated cooler, so the substrate sits at
# a fixed temperature instead of wandering -- which is what made drift the dominant error
# there. Hold the chip here and treat a reading outside the band as a stale measurement.
# 25 C, and nothing moves it. This is asserted on every TEC port open -- the Arduino resets
# on connect and forgets whatever the last process set -- so it is the only place the chip's
# operating point actually lives.
#
# 20 C was measured to hold tighter (sd 0.030 at drive 0.68 V) and was the operating point
# for part of 2026-08-27. It stopped being reachable: after several hours the loop railed at
# drive +2.00 V and sat flat at 26.5 C, 6.5 C above target, having slewed 25 -> 20 in 25
# seconds earlier the same evening. The TEC pumps heat into a sink whose own temperature had
# risen, so the achievable delta is measured from the sink and not from room air. 25 C was
# comfortable throughout (drive +0.52 V, sd 0.020) and is what the bench can hold all day.
#
# Changing it costs more than it looks: every thermal excursion re-anchors the transfer
# table, and the chip was cycled 25->30->25->30->25->20 over one session chasing this.
TEC_SETPOINT_C = 25.0
TEC_TOLERANCE_C = 0.05
# A reset lands the chip near ambient, and pulling ~7 C back down runs at ~0.5 C/min. The
# old 60 s gate was sized for holding a setpoint, not for reaching one from cold, and it
# failed a capture that was merely still on its way.
TEC_SETTLE_S = 600.0


@dataclass
class PICConfig:
    port: str | None = None
    baud: int = BAUD_RATE
    num_dac: int = NUM_DAC
    num_adc_raw: int = NUM_ADC_RAW
    out_pds: tuple = OUT_PDS
    voltage_min: float = VOLTAGE_MIN
    voltage_max: float = VOLTAGE_MAX
    timeout_s: float = DEFAULT_TIMEOUT_S
    settle_s: float = DEFAULT_SETTLE_S
    reset_wait_s: float = RESET_WAIT_S


def out_mask(num_adc_raw: int = NUM_ADC_RAW, out_pds=OUT_PDS) -> np.ndarray:
    """Boolean mask selecting the mesh-output photodiodes out of a raw ADC reply."""
    m = np.zeros(num_adc_raw, dtype=bool)
    m[list(out_pds)] = True
    return m


def dac_code(voltage: float, ref_v: float = DAC_REF_V, bits: int = DAC_BITS) -> int:
    """Host volts -> DAC code, the same conversion the firmware does."""
    return int(np.clip(voltage, 0.0, ref_v) * (2**bits - 1) / ref_v)
