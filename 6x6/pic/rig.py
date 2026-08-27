"""Rig -- one object that runs the whole PIC hardware-in-the-loop, with every component
swappable between hardware and its mock.

    rig = Rig(laser="mock", board="mock").open()      # "hw" | "mock" | a device instance
    with rig.session(duration_s=30, power_dbm=13):
        y = rig.measure(v)                            # 14 raw PD volts
    rig.close()

`model=` accepts a predictor (see `pic.model`) so `rig.predict(H)` runs the DPNN / twin /
mock surrogate through the same interface -- the third swappable component.
"""
from __future__ import annotations

from .devices.laser import Laser
from .devices.mock import MockLaser
from .interface import PIC, MockPIC, mock_fringe_forward
from .session import laser_session, _resolve_pic_port


def make_laser(spec, port=None):
    """'hw'|'mock' -> a Laser/MockLaser; a device instance passes through."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockLaser()
    if spec == "hw":
        return Laser(port=port)
    raise ValueError(f"laser={spec!r}; use 'hw', 'mock', or a Laser instance")


def make_board(spec, port=None, laser_port=None):
    """'hw'|'mock' -> a PIC/MockPIC; a device instance passes through."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockPIC(mock_fringe_forward(seed=0), noise=3e-4)
    if spec == "hw":
        return PIC(port=_resolve_pic_port(port, laser_port))
    raise ValueError(f"board={spec!r}; use 'hw', 'mock', or a PIC instance")


class Rig:
    """Coordinated laser + board (+ optional model), each independently hw or mock."""

    def __init__(self, laser="hw", board="hw", model=None, *, laser_port=None, pic_port=None):
        self.laser = make_laser(laser, laser_port)
        self._board_spec = board
        self._pic_port = pic_port
        self.board = None            # opened lazily so the pic port can resolve off the laser
        if isinstance(model, str):   # "mock"|"dpnn"|"twin"; lazy so `import pic` stays torch-free
            from .model import make_model
            model = make_model(model)
        self.model = model           # swappable predictor; see pic.model

    def open(self):
        self.laser.open()
        laser_port = getattr(getattr(self.laser, "dev", None), "port", None)
        self.board = make_board(self._board_spec, self._pic_port, laser_port)
        self.board.open()
        return self

    def session(self, *, duration_s, power_dbm, **kw):
        """Guarded laser session (background watchdog + emission verify). Context manager."""
        return laser_session(self.laser, duration_s=duration_s, power_dbm=power_dbm, **kw)

    def measure(self, v):
        """Set 64/128 DAC volts, return the 14 raw PD volts."""
        return self.board.measure_raw(v)

    def predict(self, H):
        """Run the attached model's surrogate forward (needs model=; see pic.model)."""
        if self.model is None:
            raise RuntimeError("Rig has no model -- pass model= (see pic.model)")
        return self.model.predict(H)

    def close(self):
        try:
            if self.board is not None:
                self.board.set_zero()
                self.board.close()
        finally:
            try:
                self.laser.off()
            finally:
                self.laser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
