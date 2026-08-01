"""Laser device layer: the real `Laser`, its drop-in `MockLaser`, and the PDMv5 wire
protocol. Every physical component in `pic` pairs a hardware class with a mock so the
rig can run with either."""
from .laser import Laser, LaserError, main
from .mock import MockLaser
from .pdmv5 import PDMv5, PDMv5Error
from . import pdmv5

__all__ = ["Laser", "LaserError", "MockLaser", "PDMv5", "PDMv5Error", "pdmv5", "main"]
