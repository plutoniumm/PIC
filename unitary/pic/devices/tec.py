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
reads the die, and the loop drives the LT8722 DAC over SPI at +5/-1.26 V (DAC_MAX, UV clamp) into +2 A / -1 A limits.
Kp = 5, Ki = 0.1, no D term, one update per second.

From the host it is a serial line, not an SPI device: write a bare temperature in degrees to
retarget it, and it streams `temperature,setpoint,dac_volts` at 5 Hz. `SerialTEC` speaks
exactly that. The sketch ignores targets outside 5..80 C.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import TEC_SETPOINT_C, TEC_SETTLE_S, TEC_TOLERANCE_C

# The controller sketch itself ignores anything outside 5..80 C; this is tighter, because
# below the dew point water condenses on the die and well above room the TEC just saturates.
T_MIN_C, T_MAX_C = 15.0, 45.0
DRIVE_MAX_V = 5.0  # DAC_MAX in Arduino/tec_pid/tec_pid.ino
CURRENT_LIMIT_A = 2.0  # ILIMP in Arduino/tec_pid/tec_pid.ino


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
            raise TECError(
                f"setpoint {c:.1f} C outside the safe band " f"{T_MIN_C:.0f}..{T_MAX_C:.0f} C"
            )
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
        raise TECError(
            f"chip did not reach {self._setpoint:.2f} +/- {self.tolerance:.2f} C "
            f"within {timeout_s:.0f} s (now {self.temperature():.2f} C)"
        )

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def status(self) -> dict:
        return {
            "setpoint_c": self._setpoint,
            "temperature_c": self.temperature(),
            "tolerance_c": self.tolerance,
            "stable": self.stable,
        }


class SerialTEC(TEC):
    """The PID controller of `mrunal/tec_pid_1.ino` over its serial line.

    It streams a CSV line every 200 ms whether or not anyone asks, so a read is "take the
    freshest line", not "send a query". Retargeting is a bare float on its own line."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 2.0, **kw):
        self.port, self.baud, self.timeout_s, self.ser = port, baud, timeout_s, None
        self.status_line = None
        self._last = None
        super().__init__(**kw)

    @property
    def is_open(self) -> bool:
        return self.ser is not None

    def open(self):
        import serial

        # Hold DTR low across the open. The Arduino auto-resets on DTR, and a reset here
        # dumps the integrator and restarts the ramp from room temperature -- a minute of
        # settling lost every time any command touches the rig. The board is already
        # regulating; connecting to watch it must not disturb it.
        self.ser = serial.Serial(baudrate=self.baud, timeout=self.timeout_s)
        self.ser.port = self.port
        self.ser.dtr = False
        self.ser.open()
        # Holding DTR low does not reliably suppress the reset on every macOS/FTDI pairing,
        # and the sketch prints "Enter target temperature:" and then WAITS for one. A
        # setpoint written while the bootloader is still running is swallowed, and the
        # controller never starts streaming -- which reads exactly like a dead TEC. So wait
        # out a possible reset, then write, then confirm telemetry actually arrives.
        time.sleep(2.5)
        self.ser.reset_input_buffer()
        self._write(self._setpoint)
        for _ in range(3):
            if self._read_line() is not None:
                return self
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
                continue  # a read timeout, not the end of the stream
            if raw.startswith(("STATUS 0x", "FAULT 0x")):
                self.status_line = raw  # the LT8722's own word, printed every 5 s
                continue
            parts = raw.split(",")
            if len(parts) == 3:
                try:
                    latest = tuple(float(x) for x in parts)
                except ValueError:
                    continue
                if self.ser.in_waiting == 0:
                    break  # caught up, and we have a real sample
            # Anything else is the sketch's boot banner ("Starting LT8722...", "Enter target
            # temperature:"). Opening the port resets the Arduino, so those two lines arrive
            # before any telemetry does -- giving up on them meant every freshly-opened
            # connection reported the controller as silent.
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

    def __init__(self, ambient_c: float = 23.0, tau_s: float = 8.0, noise_c: float = 0.005, **kw):
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

    def _write(self, c: float):
        pass  # `is_open` is True, so the `target` setter writes through even with no device

    def temperature(self) -> float:
        return float("nan")

    @property
    def stable(self) -> bool:
        return False

    def wait_stable(self, timeout_s: float = 0.0, poll_s: float = 1.0) -> float:
        raise TECError("no TEC on this rig; temperature is uncontrolled and unmeasured")


class FeedTEC(TEC):
    """A TEC read through someone else's connection: `read()` returns (temp, setpoint, drive)
    from a port another component already holds. The port is opened once, by its owner, and
    never again -- two readers on one serial line split its telemetry between them, and a
    reopen can reset the controller."""

    def __init__(self, read, write=None, **kw):
        self._read_fn, self._write_fn, self._last = read, write, None
        super().__init__(**kw)

    @property
    def is_open(self) -> bool:
        return True

    def _write(self, c: float):
        if self._write_fn is not None:
            self._write_fn(c)
        elif self._last is not None and abs(c - self._last[1]) > 1e-6:
            raise TECError(f"cannot retarget to {c:.2f} C through a shared TEC connection")

    def temperature(self) -> float:
        self._last = self._read_fn()
        self._setpoint = self._last[1]  # the controller's own setpoint, whoever set it
        return self._last[0]

    def status(self) -> dict:
        s = super().status()
        if self._last is not None:
            s["controller_setpoint_c"], s["drive_v"] = self._last[1], self._last[2]
        return s


UI_TEC_URL = os.environ.get("PIC_UI_TEC", "http://127.0.0.1:8744/api/tec")


def remote_tec(url: str = UI_TEC_URL, **kw) -> FeedTEC:
    """The TEC as the UI server reads it, over HTTP, so a job never opens the port itself."""
    import json
    import urllib.request

    def read():
        d = json.loads(urllib.request.urlopen(url, timeout=5).read())
        if "c" not in d:
            raise TECError(f"UI server has no TEC reading: {d.get('error') or d}")
        return d["c"], d["set"], d["drive"]

    return FeedTEC(read, **kw)


def auto_tec(find_port):
    """The UI server's TEC feed when a server is up, else the port itself."""
    t = remote_tec()
    try:
        t.temperature()
        return t
    except Exception:
        return SerialTEC(find_port())


def make_tec(spec, **kw):
    """``'mock'|'none'|<port>`` -> a TEC; a TEC instance passes through unchanged."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockTEC(**kw)
    if spec in ("none", "off"):
        return NoTEC(**kw)
    return SerialTEC(spec, **kw)
