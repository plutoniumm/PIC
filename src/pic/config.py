"""Hardware configuration mirroring the Arduino firmware and the physical chip."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DEFAULT_PORT = "COM4"
BAUD_RATE = 115200
RESET_WAIT_S = 2.0  # Arduino auto-resets when the serial port is opened
READY_BANNER = "DACs initialized and ready"

NUM_DAC = 64  # 4 DAC chips x 16 channels
NUM_ADC_RAW = 14  # photodiodes A0..A13
DAMAGED_PDS = (0, 2, 7, 11)  # 4 damaged photodiodes -> dropped (14 -> 10)
LIVE_PDS = tuple(i for i in range(NUM_ADC_RAW) if i not in DAMAGED_PDS)
NUM_ADC_LIVE = len(LIVE_PDS)  # 10

VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 4.0  # host operating range (dataset grid step 0.5 V)
FIRMWARE_VMAX = 2.0  # clamp in the current .ino (configurable, not a chip law)
DAC_REF_V = 5.0
DAC_BITS = 16
VPI_NOMINAL = 1.5  # ~Vpi (pi phase shift), per the team

ADC_AVG_MS = 75  # firmware ADC averaging window (~50 sweeps of the 14 pins)
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_SETTLE_S = 0.0  # extra host-side thermal-settle dwell before a read


@dataclass
class PICConfig:
    port: str = DEFAULT_PORT
    baud: int = BAUD_RATE
    num_dac: int = NUM_DAC
    num_adc_raw: int = NUM_ADC_RAW
    damaged_pds: tuple = DAMAGED_PDS
    voltage_min: float = VOLTAGE_MIN
    voltage_max: float = VOLTAGE_MAX
    timeout_s: float = DEFAULT_TIMEOUT_S
    settle_s: float = DEFAULT_SETTLE_S
    reset_wait_s: float = RESET_WAIT_S

    @property
    def live_pds(self) -> tuple:
        return tuple(i for i in range(self.num_adc_raw) if i not in self.damaged_pds)


def live_mask(num_adc_raw: int = NUM_ADC_RAW, damaged=DAMAGED_PDS) -> np.ndarray:
    """Boolean mask selecting the live (undamaged) photodiode channels."""
    m = np.ones(num_adc_raw, dtype=bool)
    m[list(damaged)] = False
    return m


def dac_code(voltage: float, ref_v: float = DAC_REF_V, bits: int = DAC_BITS) -> int:
    """Firmware voltage -> DAC code conversion (for reference / simulation)."""
    return int(np.clip(voltage, 0.0, ref_v) * (2**bits - 1) / ref_v)
