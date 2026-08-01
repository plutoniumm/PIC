"""Hardware-free stand-in for the AeroDiode PDMv5 laser.

Mirrors the slice of `laser.laser.Laser` / `laser.pdmv5.PDMv5` that callers actually use,
so the UI and experiment harnesses can run with no laser plugged in. It reproduces the
behaviours that bite on real hardware:

  * won't enable into a 0 setpoint (`FLOOR_SP`);
  * refuses `set()` while the laser is off;
  * `hw_off()` ramps to 0 and verifies the current before dropping the enables;
  * `laser_status == 1` does NOT imply emission -- below `LASE_THRESHOLD_SP` the mock
    reports ARMED/READY with a flat monitor photodiode, exactly like the real unit
    (see CLAUDE.md, task #9). Callers that baseline `bfm_optical_power` off-vs-on will
    correctly conclude "no light" in that regime.

Key + interlocks default to satisfied; flip them to exercise the refusal paths:

    laser = MockLaser(key=0)          # hw_on() raises LaserError
    laser = MockLaser(lase_sp=1e9)    # never emits -- armed but dark
"""

from __future__ import annotations

import math
import time

from .laser import Laser, LaserError


class _MockDev:
    """Stands in for `PDMv5`, the register-level transport."""

    port = "mock-laser"
    addr = 1

    def __init__(self, parent: "MockLaser"):
        self.p = parent
        self._reg: dict[str, float] = {
            "operating_mode": 0.0,
            "cw_current_source": 0.0,
            "tec_status": 0.0,
            "temperature": 25.0,
            "cw_laser_status": 0.0,
            "laser_status": 0.0,
            "cw_current": 0.0,
            "factory_max_current": 250.0,
            "max_current": 250.0,
            "max_avg_current": 250.0,
            "cw_max_current": 250.0,
        }

    def open(self):
        return self

    def close(self):
        pass

    def version(self):
        return "mock-1.0"

    def read_address(self) -> int:
        return self.addr

    def read_setting(self, name: str):
        if name not in self._reg:
            raise KeyError(f"unknown setting {name!r}")
        return self._reg[name]

    def write_setting(self, name: str, value, *, verify: bool = True, apply: bool = True):
        # the real driver refuses to latch the master on while the setpoint is 0
        if name == "laser_status" and value and self._reg["cw_current"] <= 0:
            return
        self._reg[name] = float(value)

    def measure(self, name: str):
        p = self.p
        sp = self._reg["cw_current"]
        on = self._reg["laser_status"] == 1
        if name == "key":
            return p.key
        if name in ("bnc_interlock", "ext_interlock"):
            return p.interlock
        if name == "driver_enable":
            return 1 if on else 0
        if name == "diode_cw_current":
            return max(0.0, 2.52 * sp - 6.8) if on else p.DARK_MA
        if name == "bfm_optical_power":
            # flat until the diode is actually lasing -- the whole point of task #9
            if on and sp >= p.lase_sp:
                return p.BFM_DARK + 0.04 * (sp - p.lase_sp)
            return p.BFM_DARK
        if name == "diode_temperature":
            return 24.8
        if name == "temperature_consign":
            return 25.0
        if name == "alarms":
            return 0
        raise KeyError(f"unknown measure {name!r}")

    def status(self) -> dict:
        s = {"address": self.addr}
        for n in ("key", "bnc_interlock", "ext_interlock", "driver_enable",
                  "diode_cw_current", "diode_temperature", "temperature_consign"):
            s[n] = self.measure(n)
        for n in ("operating_mode", "cw_current_source", "tec_status", "temperature",
                  "cw_laser_status", "laser_status", "cw_current"):
            s[n] = self.read_setting(n)
        return s


class MockLaser(Laser):
    """`Laser` with the serial transport swapped for `_MockDev`.

    Subclasses the real class so the safety policy under test is the *real* policy --
    only the device underneath is fake. Ramps are compressed so the UI stays responsive.
    """

    DARK_MA = 1.6
    BFM_DARK = 0.20
    LASE_THRESHOLD_SP = 8.0  # below this the diode is armed but not lasing

    def __init__(self, port: str | None = None, *, key: int = 1, interlock: int = 1,
                 lase_sp: float | None = None, rate_sp_s: float = 400.0,
                 dt_s: float = 0.01, **kw):
        self.key = int(key)
        self.interlock = int(interlock)
        self.lase_sp = self.LASE_THRESHOLD_SP if lase_sp is None else float(lase_sp)
        self.dev = _MockDev(self)
        self.sp_max = float(kw.get("sp_max", self.SP_MAX))
        self.rate = abs(float(rate_sp_s)) or self.RATE_SP_S
        self.dt = max(1e-3, float(dt_s))

    def open(self):
        return self

    def close(self):
        pass  # serial only; does not change laser state -- same as the real class

    def emitting(self) -> bool:
        """Ground truth the real hardware won't give you directly."""
        return self.is_on() and self.dev.read_setting("cw_current") >= self.lase_sp
