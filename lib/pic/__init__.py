"""PIC hardware interface: serial driver + acquisition routines.

    from lib.pic import PIC, MockPIC, PICConfig, acquisition
"""
from .config import PICConfig, live_mask, dac_code
from .interface import PIC, MockPIC, PICError
from . import acquisition

__all__ = [
    "PICConfig", "PIC", "MockPIC", "PICError",
    "live_mask", "dac_code", "acquisition",
]
