"""Chip thermoelectric cooler: hold the substrate at a fixed temperature.

This is the piece the 6x6 rig did not have, and its absence was the dominant error there.
Without it the substrate wanders, every heater's phase offset wanders with it, and the
calibration has a shelf life measured in hours; a correction only pays inside a narrow
window of drift magnitudes. With the chip clamped, phi0 is a constant rather than a
slowly moving target, so calibration becomes something you do once instead of a running
cost. Every acquisition path in this package therefore takes a temperature reading with
its data, and refuses to trust a sample taken outside the band.

The controller is an **LT8722** H-bridge driver run by a PID loop on its own Arduino
(`mrunal/tec_pid_1.ino`): a 10k NTC (beta 3450, Steinhart, EMA-filtered) on a 10k divider
reads the die, and the loop drives the LT8722 DAC over SPI at +/-2 V into a +/-1 A limit.
Kp = 5, no I or D term, one update per second.

From the host it is a serial line, not an SPI device: write a bare temperature in degrees to
retarget it, and it streams `temperature,setpoint,dac_volts` at 5 Hz. `SerialTEC` speaks
exactly that. The sketch ignores targets outside 5..80 C.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import TEC_SETPOINT_C, TEC_SETTLE_S, TEC_TOLERANCE_C

# The controller sketch itself ignores anything outside 5..80 C; this is tighter, because
# below the dew point water condenses on the die and well above room the TEC just saturates.
T_MIN_C, T_MAX_C = 15.0, 45.0
DRIVE_MAX_V = 2.0   # LT8722 DAC swing the PID loop is limited to
CURRENT_LIMIT_A = 1.0


class TECError(RuntimeError):
    pass


class TEC:
    """Interface every rig component talks to. Subclasses supply `_read` and `_write`."""

    def __init__(self, setpoint_c: float = TEC_SETPOINT_C, tolerance_c: float = TEC_TOLERANCE_C):
        self.tolerance = float(tolerance_c)
        self._setpoint = None
        self.target = float(setpoint_c)

    @property
    def target(self) -> float:
        return self._setpoint

    @target.setter
    def target(self, c: float):
        c = float(c)
        if not T_MIN_C <= c <= T_MAX_C:
            raise TECError(f"setpoint {c:.1f} C outside the safe band "
                           f"{T_MIN_C:.0f}..{T_MAX_C:.0f} C")
        self._setpoint = c
        if self.is_open:
            self._write(c)

    @property
    def is_open(self) -> bool:
        return False

    def open(self):
        return self

    def close(self):
        pass

    def temperature(self) -> float:
        raise NotImplementedError

    @property
    def error_c(self) -> float:
        return abs(self.temperature() - self._setpoint)

    @property
    def stable(self) -> bool:
        return self.error_c <= self.tolerance

    def wait_stable(self, timeout_s: float = TEC_SETTLE_S, poll_s: float = 1.0) -> float:
        """Block until the chip is inside the band. Returns the seconds it took; raises
        if it never gets there, because a measurement taken while the substrate is still
        moving is worse than no measurement."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.stable:
                return time.time() - t0
            time.sleep(poll_s)
        raise TECError(f"chip did not reach {self._setpoint:.2f} +/- {self.tolerance:.2f} C "
                       f"within {timeout_s:.0f} s (now {self.temperature():.2f} C)")

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def status(self) -> dict:
        return {"setpoint_c": self._setpoint, "temperature_c": self.temperature(),
                "tolerance_c": self.tolerance, "stable": self.stable}


class SerialTEC(TEC):
    """The PID controller of `mrunal/tec_pid_1.ino` over its serial line.

    It streams a CSV line every 200 ms whether or not anyone asks, so a read is "take the
    freshest line", not "send a query". Retargeting is a bare float on its own line."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 2.0, **kw):
        self.port, self.baud, self.timeout_s, self.ser = port, baud, timeout_s, None
        self._last = None
        super().__init__(**kw)

    @property
    def is_open(self) -> bool:
        return self.ser is not None

    def open(self):
        import serial

        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        time.sleep(2.0)  # the board resets when the port is opened
        self.ser.reset_input_buffer()
        self._write(self._setpoint)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _write(self, c: float):
        self.ser.write(f"{c:.2f}\n".encode())

    def _read_line(self):
        """The most recent complete telemetry line, or None. Returns (temp, setpoint, dac)."""
        deadline = time.time() + self.timeout_s
        latest = None
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if not raw:
                break
            parts = raw.split(",")
            if len(parts) == 3:
                try:
                    latest = tuple(float(x) for x in parts)
                except ValueError:
                    continue
            if self.ser.in_waiting == 0:
                break
        return latest

    def temperature(self) -> float:
        row = self._read_line()
        if row is None:
            if self._last is None:
                raise TECError(f"no telemetry from the TEC controller on {self.port}")
            return self._last[0]
        self._last = row
        return row[0]

    def status(self) -> dict:
        s = super().status()
        if self._last is not None:
            s["controller_setpoint_c"] = self._last[1]
            s["drive_v"] = self._last[2]
        return s


class MockTEC(TEC):
    """First-order thermal model. Settles in real time so `wait_stable` is exercised."""

    def __init__(self, ambient_c: float = 23.0, tau_s: float = 8.0, noise_c: float = 0.005,
                 **kw):
        self.ambient, self.tau, self.noise = float(ambient_c), float(tau_s), float(noise_c)
        self._rng = np.random.default_rng(0)
        self._t = self.ambient
        self._last = time.time()
        self._open = False
        super().__init__(**kw)

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self):
        self._open = True
        self._last = time.time()
        return self

    def close(self):
        self._open = False

    def _write(self, c: float):
        pass

    def temperature(self) -> float:
        now = time.time()
        dt, self._last = now - self._last, now
        drive = self._setpoint if self._open else self.ambient
        self._t += (drive - self._t) * (1 - np.exp(-dt / self.tau))
        return float(self._t + self._rng.normal(0, self.noise))


class NoTEC(TEC):
    """A rig with no cooler. Reads nan and never claims stability, so anything that gates
    on temperature fails loudly instead of recording an unlabelled drifting session."""

    def __init__(self, **kw):
        super().__init__(**kw)

    @property
    def is_open(self) -> bool:
        return True

    def temperature(self) -> float:
        return float("nan")

    @property
    def stable(self) -> bool:
        return False

    def wait_stable(self, timeout_s: float = 0.0, poll_s: float = 1.0) -> float:
        raise TECError("no TEC on this rig; temperature is uncontrolled and unmeasured")


def make_tec(spec, **kw):
    """``'mock'|'none'|<port>`` -> a TEC; a TEC instance passes through unchanged."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockTEC(**kw)
    if spec in ("none", "off"):
        return NoTEC(**kw)
    return SerialTEC(spec, **kw)
