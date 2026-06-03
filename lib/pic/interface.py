"""Serial interface to the PIC via the Arduino firmware.

Protocol (matches `Fully_automated_PIC.ino`), one synchronous round trip per call:
    host -> "v0,v1,...,v63\n"   (64 DAC voltages, "%.1f")
    dev  -> "a0,a1,...,a13\n"   (14 mean photodiode voltages, 3 decimals)
The firmware sets all DACs then averages every ADC for 31 ms before replying.
"""
from __future__ import annotations
import time
import numpy as np
from .config import PICConfig, live_mask


class PICError(RuntimeError):
    pass


class PIC:
    """Synchronous driver: one set-and-read round trip per call.

    Use as a context manager:
        with PIC(port="/dev/tty.usbmodemXXXX") as pic:
            y = pic.measure(voltages)      # 10 live photodiode voltages
    """

    def __init__(self, config: PICConfig | None = None, **kw):
        self.cfg = config or PICConfig(**kw)
        self.ser = None
        self._mask = live_mask(self.cfg.num_adc_raw, self.cfg.damaged_pds)

    # --- lifecycle ---
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self):
        import serial  # pyserial; imported lazily so the mock needs no hardware deps
        self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_s)
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

    # --- core round trip ---
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
        """Return the 10 live photodiode voltages, optionally after a thermal settle.

        With ``settle_s > 0`` the setpoint is applied, we dwell to let the chip
        equilibrate, then re-apply and read -- so the result reflects the *settled*
        state (the firmware itself only averages 31 ms after setting).
        """
        settle_s = self.cfg.settle_s if settle_s is None else settle_s
        if settle_s and settle_s > 0:
            self.measure_raw(voltages)   # apply setpoint
            time.sleep(settle_s)         # equilibrate
        return self.measure_raw(voltages)[self._mask]

    def set_zero(self):
        """Drive all DACs to 0 V (safe idle state)."""
        self.measure_raw(np.zeros(self.cfg.num_dac))

    # --- helpers ---
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
                    continue  # malformed line; keep reading
            # otherwise: startup banner / partial line -> ignore and keep reading
        raise PICError(f"timeout waiting for {n} values from {self.cfg.port}")

    @property
    def live_pds(self) -> tuple:
        return self.cfg.live_pds


class MockPIC(PIC):
    """Hardware-free stand-in backed by a forward function ``f(v64) -> raw14``.

    Lets acquisition / inverse logic be developed and tested without the chip.
    """

    def __init__(self, forward, noise: float = 0.0, config: PICConfig | None = None, **kw):
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
