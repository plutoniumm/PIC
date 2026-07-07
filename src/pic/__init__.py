"""PIC hardware interface: serial driver, config, and acquisition routines.

Everything the host needs to talk to the board is re-exported here, so callers
import from one place:

    from src.pic import PIC, MockPIC, find_port, NUM_DAC, NUM_ADC_RAW
"""

from .config import (
    PICConfig, live_mask, dac_code,
    NUM_DAC, NUM_ADC_RAW, NUM_ADC_LIVE, DAMAGED_PDS, LIVE_PDS,
    VOLTAGE_MIN, VOLTAGE_MAX,
)
from .interface import PIC, MockPIC, PICError, find_port
from . import acquisition, config

__all__ = [
    "PIC", "MockPIC", "PICError", "find_port",
    "PICConfig", "live_mask", "dac_code", "acquisition", "config",
    "NUM_DAC", "NUM_ADC_RAW", "NUM_ADC_LIVE", "DAMAGED_PDS", "LIVE_PDS",
    "VOLTAGE_MIN", "VOLTAGE_MAX",
]
