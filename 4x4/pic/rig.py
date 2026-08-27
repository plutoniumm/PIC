"""Rig: one object that runs the whole 4x4 hardware-in-the-loop.

Four swappable components -- laser, board, TEC, model -- each independently hardware or
mock, so every path in this package can be exercised with nothing plugged in.

    rig = Rig(laser="mock", board="mock", tec="mock", model="mock").open()
    with rig.session(duration_s=30, power_dbm=13):
        y = rig.measure(v)          # 18 DAC volts in, 8 raw photodiode volts back
        f = rig.realize(U)          # program a target unitary and measure what came out
    rig.close()
"""

from __future__ import annotations

import numpy as np

from .config import VOLTAGE_MAX, out_mask
from .devices.laser import Laser
from .devices.mock import MockLaser
from .devices.switch import make_switch
from theory.layout import N_HEATERS
from .devices.tec import make_tec
from .interface import PIC, MockPIC
from .session import _resolve_pic_port, laser_session


def make_laser(spec, port=None):
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockLaser()
    if spec == "hw":
        return Laser(port=port)
    raise ValueError(f"laser={spec!r}; use 'hw', 'mock', or a Laser instance")


def make_board(spec, port=None, laser_port=None, switch=None):
    """``'hw'|'mock'|'sim'`` -> a board; a PIC instance passes through unchanged.

    'mock' is the idealised chip this package was designed against -- all 18 heaters live,
    Vpi near 1.5 V, a clean readout. 'sim' is the bench that exists: six wired channels, a
    3 V clamp, the measured loss table and the measured noise (`pic.sim`). Use 'mock' to ask
    whether an algorithm is right and 'sim' to ask whether it will survive the bench."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockPIC(switch=switch)
    if spec == "sim":
        from .sim import BenchPIC
        return BenchPIC(switch=switch)
    if spec == "hw":
        return PIC(port=_resolve_pic_port(port, laser_port))
    raise ValueError(f"board={spec!r}; use 'hw', 'mock', 'sim', or a PIC instance")


class Rig:
    def __init__(self, laser="hw", board="hw", tec="mock", switch="mock", model=None, *,
                 keep_laser: bool = False,
                 calib=None, laser_port=None, pic_port=None, dynamic=False):
        self.laser = make_laser(laser, laser_port)
        self.tec = make_tec(tec)
        self.switch = make_switch(switch)
        self.keep_laser = bool(keep_laser)
        self._board_spec = board
        self._pic_port = pic_port
        self.board = None  # opened lazily so the board port can exclude the laser's
        if isinstance(model, str):  # lazy so `import pic` stays torch-free
            from .model import make_model
            model = make_model(model)
        self.model = model
        self._calib = calib
        self._mask = out_mask()
        self._twin = None
        from .drift import DriftTracker
        self.drift = DriftTracker(self, enabled=dynamic)

    @property
    def calib(self):
        """The heater law in force. Falls back to the nominal one, which is enough to
        drive the chip but not to program a target accurately."""
        if self._calib is None:
            from theory.calib import Calibration
            self._calib = Calibration.load_or_nominal()
        return self._calib

    @property
    def twin(self):
        """The differentiable mesh model, used by the drift fit. Lazy: importing it pulls
        in torch, and `import pic` stays torch-free."""
        if self._twin is None:
            from theory.twin import Twin
            self._twin = Twin()
        return self._twin

    def open(self):
        self.laser.open()
        self._laser_was_on = bool(getattr(self.laser, "is_on", lambda: False)())
        if self._laser_was_on and not self.keep_laser:
            print("  note: laser was already on; it will be turned off on exit "
                  "(pass --keep-laser to leave it lit)")
        self.tec.open()
        self.switch.open()
        laser_port = getattr(getattr(self.laser, "dev", None), "port", None)
        self.board = make_board(self._board_spec, self._pic_port, laser_port, self.switch)
        self.board.open()
        if hasattr(self.switch, "attach"):   # board-routed switch: it needs the open board
            self.switch.attach(self.board)
        return self

    def select_input(self, port: int) -> int:
        """Route the laser to one of the four input ports. See `pic.devices.switch` for why
        a characterization has to visit more than one."""
        return self.switch.select(port)

    def session(self, *, duration_s, power_dbm, **kw):
        """Guarded laser session: background watchdog, emission verify, TEC gate."""
        return laser_session(self.laser, duration_s=duration_s, power_dbm=power_dbm,
                             tec=self.tec,
                             read_pds=lambda: self.outputs(np.zeros(N_HEATERS)), **kw)

    def measure(self, v) -> np.ndarray:
        """Set the 18 DAC volts, return the raw photodiode volts."""
        return self.board.measure_raw(v)

    def outputs(self, v) -> np.ndarray:
        """Same, reduced to the four mesh outputs."""
        return self.measure(v)[self._mask]

    def program(self, U, vmax: float = VOLTAGE_MAX):
        """Target unitary -> (18 DAC volts, reachable mask). Closed form, no search.

        With `dynamic=True` the target phases are pre-distorted by the drift the tracker
        has inferred, so what lands on the chip is the target as the chip is *now* rather
        than as it was at calibration."""
        from theory.program import phases_for
        return self.calib.volts(self.drift.correct(phases_for(U)), vmax=vmax)

    def realize(self, U, vmax: float = VOLTAGE_MAX):
        """Program `U` onto the chip and read what the outputs did. Returns
        (volts, reachable mask, output photodiode volts)."""
        v, ok = self.program(U, vmax=vmax)
        return v, ok, self.outputs(v)

    def predict(self, v) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Rig has no model -- pass model= (see pic.model)")
        return self.model.predict(v)

    def status(self) -> dict:
        return {"laser_on": bool(getattr(self.laser, "is_on", lambda: False)()),
                "tec": self.tec.status(),
                "switch": self.switch.status(),
                "drift": self.drift.status(),
                "calib": self.calib.meta or "nominal (uncharacterized)"}

    def close(self):
        try:
            if self.board is not None:
                self.board.set_zero()
                self.board.close()
        finally:
            try:
                # Off by default, always. Leaving a lit diode behind with no watchdog on it
                # is the one failure mode that damages hardware unattended, and convenience
                # is not worth it -- `keep_laser` makes the exception explicit and opt-in.
                if not (self.keep_laser and getattr(self, "_laser_was_on", False)):
                    self.laser.off()
            finally:
                self.laser.close()
                self.tec.close()
                self.switch.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
