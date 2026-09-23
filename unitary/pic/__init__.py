"""pic -- the 4x4 hardware-in-the-loop rig library.

One import for the whole rig: the laser, the board, the chip TEC, the acquisition
primitives and a `Rig` facade that swaps any component between hardware and mock.

    from pic import Rig
    rig = Rig(laser="mock", board="mock", tec="mock", model="mock").open()
    with rig.session(duration_s=30, power_dbm=13):
        v, ok, y = rig.realize(U)      # program a 4x4 unitary, read the outputs
    rig.close()

Same API as the 6x6 package next door, so anything written against that rig ports across.
What differs is the chip: 18 heaters instead of 128, a mesh that is unitary rather than a
contraction, and a TEC that holds the substrate still.
"""

from . import acquisition, characterize, config, devices, layout, sim
from .config import (
    ADC_AVG_N,
    FIRMWARE_VMAX,
    MON_PDS,
    NUM_ADC_RAW,
    NUM_DAC,
    NUM_OUT,
    OUT_PDS,
    TEC_SETPOINT_C,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    VPI_NOMINAL,
    WIRED_DACS,
    PICConfig,
    out_mask,
)
from .devices import (
    Laser,
    LaserError,
    MockLaser,
    MockSwitch,
    MockTEC,
    NoSwitch,
    NoTEC,
    OpticalSwitch,
    SwitchError,
    TECError,
    make_switch,
    make_tec,
)
from .interface import PIC, MockPIC, PICError, find_port, twin_forward
from .layout import ACTIVE_DACS, HEATERS, N_HEATERS, REACHABLE_DACS, pack, unpack
from .model import DpnnModel, MockModel, Predictor, TwinModel, make_model
from .rig import Rig
from .sim import BenchPIC, BenchSim
from .session import laser_session, open_devices

__all__ = [
    "Rig",
    "open_devices",
    "laser_session",
    "PIC",
    "MockPIC",
    "BenchPIC",
    "BenchSim",
    "PICError",
    "find_port",
    "twin_forward",
    "Laser",
    "LaserError",
    "MockLaser",
    "make_tec",
    "MockTEC",
    "NoTEC",
    "TECError",
    "OpticalSwitch",
    "MockSwitch",
    "NoSwitch",
    "SwitchError",
    "make_switch",
    "make_model",
    "MockModel",
    "TwinModel",
    "DpnnModel",
    "Predictor",
    "PICConfig",
    "out_mask",
    "pack",
    "unpack",
    "HEATERS",
    "N_HEATERS",
    "ACTIVE_DACS",
    "REACHABLE_DACS",
    "WIRED_DACS",
    "NUM_DAC",
    "NUM_ADC_RAW",
    "NUM_OUT",
    "OUT_PDS",
    "MON_PDS",
    "VOLTAGE_MIN",
    "VOLTAGE_MAX",
    "FIRMWARE_VMAX",
    "VPI_NOMINAL",
    "ADC_AVG_N",
    "TEC_SETPOINT_C",
    "acquisition",
    "characterize",
    "config",
    "devices",
    "layout",
    "sim",
]
