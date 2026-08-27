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

NUM_DAC = N_HEATERS   # 18; the firmware sends exactly this many, no padding
NUM_ADC_RAW = 4       # A0..A3, one per mesh output through a PD-TIA (mrunal/PD_TIA_circuit.pdf)
OUT_PDS = (0, 1, 2, 3)  # the four mesh outputs, in rail order
MON_PDS = ()          # this block carries no tap monitors
NUM_OUT = len(OUT_PDS)

# The bring-up board (mrunal/Setup.ino) drives only 6 DAC channels of the 18 heaters, one
# chip on CS 10. Everything here is written for the full 18; see pic.layout.WIRED_DACS for
# what is actually reachable today.
WIRED_DACS = tuple(range(6))

VOLTAGE_MIN = 0.0
FIRMWARE_VMAX = 3.0  # clamp in pic4x4.ino; must match, or host volts vanish at the DAC
DAC_REF_V = 5.0
DAC_BITS = 16
ADC_REF_V = 5.0
ADC_BITS = 10

ADC_AVG_N = 5        # full ADC sweeps averaged per firmware reply
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_SETTLE_S = 0.5  # thermo-optic settle before a read (mrunal/Setup.ino HEATER_DELAY_MS)

# 1x4 optical input switch (Sercalo) in front of U_IN1..4, on its own serial line. It selects
# one input port at a time, which is what makes a per-port characterization automatable.
SWITCH_BAUD = 9600
SWITCH_SETTLE_S = 1.0  # mrunal/Setup.ino SWITCH_DELAY_MS

# Chip TEC. Unlike the 6x6 rig this one has a calibrated cooler, so the substrate sits at
# a fixed temperature instead of wandering -- which is what made drift the dominant error
# there. Hold the chip here and treat a reading outside the band as a stale measurement.
TEC_SETPOINT_C = 25.0
TEC_TOLERANCE_C = 0.05
TEC_SETTLE_S = 60.0


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
