"""Back-compat shim -- the PIC hardware interface moved to the top-level `pic` package.

    from src.pic import PIC        ==  from pic import PIC
    from src.pic.acquisition ...   ==  from pic.acquisition ...

New code should import from `pic`. The submodules (`src.pic.config`, `.interface`,
`.acquisition`, `.layout`, `.wiring`, `.homodyne`) are aliased to their `pic.*` originals.
"""
from pic import (  # noqa: F401
    PICConfig, live_mask, dac_code, PIC, MockPIC, PICError, find_port, mock_fringe_forward,
    NUM_DAC, NUM_ADC_RAW, NUM_ADC_LIVE, DAMAGED_PDS, LIVE_PDS, VOLTAGE_MIN, VOLTAGE_MAX,
    acquisition, config,
)

__all__ = [
    "PIC", "MockPIC", "PICError", "find_port", "mock_fringe_forward",
    "PICConfig", "live_mask", "dac_code", "acquisition", "config",
    "NUM_DAC", "NUM_ADC_RAW", "NUM_ADC_LIVE", "DAMAGED_PDS", "LIVE_PDS",
    "VOLTAGE_MIN", "VOLTAGE_MAX",
]
