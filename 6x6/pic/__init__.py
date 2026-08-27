"""pic -- the unified PIC hardware-in-the-loop rig library.

One import for the whole rig: the laser and the Arduino/board (each with a hardware
and a mock implementation), the acquisition primitives, the session/experiment
orchestration, and a `Rig` facade that swaps any component hardware<->mock.

    from pic import Rig
    rig = Rig(laser="mock", board="mock").open()      # each arg: "hw" | "mock"
    with rig.session(duration_s=30, power_dbm=13):
        y = rig.measure(v)                            # 14 raw PD volts
    rig.close()

Legacy import paths still work as thin shims: `from src.pic import PIC`,
`from laser.laser import Laser`, `from template import open_devices, laser_session`.
"""
# board / Arduino
from .config import (
    PICConfig, live_mask, dac_code,
    NUM_DAC, NUM_ADC_RAW, NUM_ADC_LIVE, DAMAGED_PDS, LIVE_PDS,
    VOLTAGE_MIN, VOLTAGE_MAX, FIRMWARE_VMAX, VPI_NOMINAL, ADC_AVG_MS,
)
from .interface import PIC, MockPIC, PICError, find_port, mock_fringe_forward
from . import acquisition, config, layout, wiring, homodyne
# laser
from .devices import Laser, LaserError, MockLaser, PDMv5, PDMv5Error
from . import devices
# orchestration + facade
from .session import open_devices, laser_session, run_experiment, run_step_response
from .rig import Rig
# model / predictor (third swappable component). model.py's heavy deps (torch, scripts.pic_gd,
# theory.*) are all lazy, so this import stays torch-free.
from .model import make_model, MockModel, DpnnModel, TwinModel, Predictor

__all__ = [
    "Rig", "open_devices", "laser_session", "run_experiment", "run_step_response",
    "make_model", "MockModel", "DpnnModel", "TwinModel", "Predictor",
    "Laser", "LaserError", "MockLaser", "PDMv5", "PDMv5Error",
    "PIC", "MockPIC", "PICError", "find_port", "mock_fringe_forward",
    "PICConfig", "live_mask", "dac_code",
    "NUM_DAC", "NUM_ADC_RAW", "NUM_ADC_LIVE", "DAMAGED_PDS", "LIVE_PDS",
    "VOLTAGE_MIN", "VOLTAGE_MAX", "FIRMWARE_VMAX", "VPI_NOMINAL", "ADC_AVG_MS",
    "acquisition", "config", "layout", "wiring", "homodyne", "devices",
]
