"""Device layer: the AeroDiode PDMv5 laser, the chip TEC, and a mock for each.

Every physical instrument in this rig pairs a hardware class with a hardware-free stand-in
that reproduces its refusal paths, so a mock run exercises the same safety logic as the
real one.
"""

from .laser import Laser, LaserError, main
from .mock import MockLaser
from .pdmv5 import PDMv5, PDMv5Error
from .switch import MockSwitch, NoSwitch, OpticalSwitch, SwitchError, make_switch
from .tec import TEC, MockTEC, NoTEC, SerialTEC, TECError, make_tec

__all__ = [
    "Laser", "LaserError", "MockLaser", "PDMv5", "PDMv5Error", "main",
    "TEC", "SerialTEC", "MockTEC", "NoTEC", "TECError", "make_tec",
    "OpticalSwitch", "MockSwitch", "NoSwitch", "SwitchError", "make_switch",
]
