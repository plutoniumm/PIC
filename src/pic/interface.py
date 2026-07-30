"""Serial interface to the PIC via the Arduino firmware (one round trip per call:
64 DAC voltages out, 14 mean photodiode voltages back)."""

from __future__ import annotations
import glob
import os
import time
import numpy as np
from .config import PICConfig, live_mask, NUM_DAC, NUM_ADC_RAW, DAMAGED_PDS, VPI_NOMINAL


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

    def _prep_dac(self, voltages) -> np.ndarray:
        """Validate + clip the 64-vector before it can reach the DACs -- the format
        early-exit. Refuses wrong length or non-finite values: np.clip does NOT fix NaN,
        so it would otherwise reach the firmware as the literal 'nan' and drive garbage.
        """
        try:
            v = np.asarray(voltages, float).ravel()
        except (ValueError, TypeError) as e:
            raise PICError(f"DAC voltages not numeric ({e})")
        if v.size != self.cfg.num_dac:
            raise PICError(f"expected {self.cfg.num_dac} voltages, got {v.size}")
        if not np.all(np.isfinite(v)):
            bad = np.where(~np.isfinite(v))[0][:8].tolist()
            raise PICError(
                f"non-finite DAC voltage(s) at index {bad}; refusing to send"
            )
        return np.clip(v, self.cfg.voltage_min, self.cfg.voltage_max)

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        """Apply the DAC voltages, return the raw photodiode voltages. Re-sends and retries
        on a transient serial timeout (the CH340 occasionally drops a reply on long runs) so
        one glitch doesn't kill a whole census/training run."""
        v = self._prep_dac(voltages)
        line = ",".join(f"{x:.1f}" for x in v) + "\n"
        for attempt in range(retries + 1):
            self.ser.reset_input_buffer()
            self.ser.write(line.encode())
            try:
                return self._read_floats(self.cfg.num_adc_raw)
            except PICError:
                if attempt == retries:
                    raise
                time.sleep(0.15)

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
        v = self._prep_dac(voltages)
        y = np.asarray(self.forward(v), float).ravel()
        if self.noise:
            y = y + self._rng.normal(0.0, self.noise, y.shape)
        return y


def mock_fringe_forward(
    seed: int = 0,
    dead_pds=DAMAGED_PDS,
    dead_channels=(0,),
    vpi: float = VPI_NOMINAL,
    num_dac: int = NUM_DAC,
    num_adc_raw: int = NUM_ADC_RAW,
):
    """A physically faithful forward for :class:`MockPIC`: each (PD, channel) pair
    interferes with phase ``phi2*v^2 + phi0`` (thermo-optic, phase ~ dissipated power),
    so a one-at-a-time sweep traces a real ``cos(a*v^2+..)`` fringe. Dead PDs stay dark
    and dead channels contribute nothing, so dead-element detection has something to find.
    """
    rng = np.random.default_rng(seed)
    phi2 = (np.pi / vpi**2) * rng.uniform(0.6, 1.6, (num_adc_raw, num_dac))
    phi0 = rng.uniform(0, 2 * np.pi, (num_adc_raw, num_dac))
    amp = rng.uniform(0.008, 0.05, (num_adc_raw, num_dac))
    base = 0.015 + 0.01 * rng.random(num_adc_raw)
    live_pd = np.ones(num_adc_raw, bool)
    live_pd[list(dead_pds)] = False
    amp[:, list(dead_channels)] = 0.0

    def forward(v):
        v = np.asarray(v, float)
        contrib = amp * np.cos(phi2 * v[None, :] ** 2 + phi0)
        y = (base + contrib.sum(axis=1)) * live_pd
        return np.clip(y, 0.0, None)

    return forward
