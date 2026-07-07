"""Serial interface to the PIC via the Arduino firmware (one round trip per call:
64 DAC voltages out, 14 mean photodiode voltages back)."""

from __future__ import annotations
import glob
import os
import time
import numpy as np
from .config import PICConfig, live_mask


class PICError(RuntimeError):
    pass


# Serial-device name patterns the Arduino shows up as, across platforms.
_PORT_PATTERNS = (
    "/dev/cu.usbmodem*",
    "/dev/tty.usbmodem*",
    "/dev/cu.usbserial*",
    "/dev/tty.usbserial*",
    "/dev/cu.wchusbserial*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
)


def find_port(explicit: str | None = None) -> tuple[str | None, list[str]]:
    """Locate the Arduino serial device. Precedence: explicit arg, ``$PIC_PORT``,
    then a glob of the usual device names. Returns ``(chosen_or_None, candidates)``."""
    if explicit:
        return explicit, [explicit]
    env = os.environ.get("PIC_PORT")
    if env:
        return env, [env]
    cands = sorted({p for pat in _PORT_PATTERNS for p in glob.glob(pat)})
    return (cands[0] if cands else None), cands


class PIC:
    """Synchronous driver: one set-and-read round trip per call. Usable as a context manager."""

    def __init__(self, config: PICConfig | None = None, **kw):
        self.cfg = config or PICConfig(**kw)
        self.ser = None
        self._mask = live_mask(self.cfg.num_adc_raw, self.cfg.damaged_pds)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self):
        import serial  # lazy import so MockPIC needs no pyserial

        self.ser = serial.Serial(
            self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_s
        )
        time.sleep(self.cfg.reset_wait_s)  # board resets on port open
        self.ser.reset_input_buffer()
        return self

    def close(self):
        if self.ser is not None and getattr(self.ser, "is_open", False):
            try:
                self.set_zero()
            finally:
                self.ser.close()
                self.ser = None

    def measure_raw(self, voltages) -> np.ndarray:
        """Apply 64 voltages, return the 14 raw photodiode voltages."""
        v = np.asarray(voltages, float).ravel()
        if v.size != self.cfg.num_dac:
            raise PICError(f"expected {self.cfg.num_dac} voltages, got {v.size}")
        v = np.clip(v, self.cfg.voltage_min, self.cfg.voltage_max)
        line = ",".join(f"{x:.1f}" for x in v) + "\n"
        self.ser.reset_input_buffer()
        self.ser.write(line.encode())
        return self._read_floats(self.cfg.num_adc_raw)

    def measure(self, voltages, settle_s: float | None = None) -> np.ndarray:
        """Return the 10 live photodiode voltages, optionally after a thermal settle."""
        settle_s = self.cfg.settle_s if settle_s is None else settle_s
        if settle_s and settle_s > 0:
            self.measure_raw(voltages)
            time.sleep(settle_s)
        return self.measure_raw(voltages)[self._mask]

    def set_zero(self):
        """Drive all DACs to 0 V (safe idle state)."""
        self.measure_raw(np.zeros(self.cfg.num_dac))

    def _read_floats(self, n: int) -> np.ndarray:
        deadline = time.time() + self.cfg.timeout_s
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) == n:
                try:
                    return np.array([float(p) for p in parts])
                except ValueError:
                    continue
            # non-matching line (startup banner / partial) -> keep reading
        raise PICError(f"timeout waiting for {n} values from {self.cfg.port}")

    @property
    def live_pds(self) -> tuple:
        return self.cfg.live_pds


class MockPIC(PIC):
    """Hardware-free stand-in backed by a forward function ``f(v64) -> raw14``."""

    def __init__(
        self, forward, noise: float = 0.0, config: PICConfig | None = None, **kw
    ):
        super().__init__(config, **kw)
        self.forward = forward
        self.noise = noise
        self._rng = np.random.default_rng(0)

    def open(self):
        self.ser = "mock"
        return self

    def close(self):
        self.ser = None

    def measure_raw(self, voltages) -> np.ndarray:
        v = np.asarray(voltages, float).ravel()
        if v.size != self.cfg.num_dac:
            raise PICError(f"expected {self.cfg.num_dac} voltages, got {v.size}")
        v = np.clip(v, self.cfg.voltage_min, self.cfg.voltage_max)
        y = np.asarray(self.forward(v), float).ravel()
        if self.noise:
            y = y + self._rng.normal(0.0, self.noise, y.shape)
        return y
