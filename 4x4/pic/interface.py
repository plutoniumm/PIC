"""Serial interface to the 4x4 board: 18 DAC volts out, 8 mean photodiode volts back,
one round trip per call.

Same line protocol as the 6x6 firmware, so the driver is deliberately the same shape --
what changed is the widths and the mock, which here is a real physical model of the chip
rather than a bag of random fringes.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np

from .config import NUM_ADC_RAW, NUM_DAC, OUT_PDS, VOLTAGE_MAX_CH, PICConfig, out_mask


class PICError(RuntimeError):
    pass


_PORT_PATTERNS = (
    "/dev/cu.usbmodem*", "/dev/tty.usbmodem*",
    "/dev/cu.usbserial*", "/dev/tty.usbserial*",
    "/dev/cu.wchusbserial*", "/dev/ttyACM*", "/dev/ttyUSB*",
)


def find_port(explicit: str | None = None) -> tuple[str | None, list[str]]:
    """Locate the board. Precedence: explicit argument, ``$PIC4_PORT``, then a glob.

    The glob also matches the laser's FTDI, so pass a port explicitly whenever both
    instruments are plugged in -- opening the laser as if it were the board is the one
    mistake in this rig that is annoying to diagnose."""
    if explicit:
        return explicit, [explicit]
    env = os.environ.get("PIC4_PORT")
    if env:
        return env, [env]
    cands = sorted({p for pat in _PORT_PATTERNS for p in glob.glob(pat)})
    return (cands[0] if cands else None), cands


class PIC:
    """Synchronous driver: one set-and-read round trip per call. A context manager."""

    def __init__(self, config: PICConfig | None = None, **kw):
        self.cfg = config or PICConfig(**kw)
        self.ser = None
        self._mask = out_mask(self.cfg.num_adc_raw, self.cfg.out_pds)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self):
        import serial  # lazy so MockPIC needs no pyserial

        port, cands = find_port(self.cfg.port)
        if port is None:
            raise PICError(f"no serial port found; candidates were {cands}")
        self.cfg.port = port
        self.ser = serial.Serial(port, self.cfg.baud, timeout=self.cfg.timeout_s)
        time.sleep(self.cfg.reset_wait_s)  # the board resets on port open
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
        """Validate and clip before anything reaches the DACs. np.clip does not fix NaN,
        so a non-finite value would otherwise arrive at the firmware as the literal 'nan'
        and drive garbage into a heater."""
        try:
            v = np.asarray(voltages, float).ravel()
        except (ValueError, TypeError) as e:
            raise PICError(f"DAC voltages not numeric ({e})") from None
        if v.size != self.cfg.num_dac:
            raise PICError(f"expected {self.cfg.num_dac} voltages, got {v.size}")
        if not np.all(np.isfinite(v)):
            bad = np.where(~np.isfinite(v))[0][:8].tolist()
            raise PICError(f"non-finite DAC voltage(s) at index {bad}; refusing to send")
        # Per-channel, not the scalar ceiling. The firmware clamps too, and that is what
        # protects the hardware -- but if the host clips at 3 V and the firmware clips a
        # 60R channel to 1.5 V, every fit on that channel is against volts that never
        # landed, and the Vpi it produces is wrong in a way nothing downstream can see.
        return np.clip(v, self.cfg.voltage_min, np.asarray(VOLTAGE_MAX_CH, float))

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        """Apply the DAC voltages, return all `num_adc_raw` photodiode volts. Retries a
        transient serial timeout so one dropped reply does not kill a long run."""
        v = self._prep_dac(voltages)
        line = ",".join(f"{x:.3f}" for x in v) + "\n"
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
        """Return the four mesh-output photodiode volts, after a thermal settle."""
        settle_s = self.cfg.settle_s if settle_s is None else settle_s
        if settle_s and settle_s > 0:
            self.measure_raw(voltages)
            time.sleep(settle_s)
        return self.measure_raw(voltages)[self._mask]

    def select_port(self, port: int) -> int:
        """Route the laser to input `port` (0-indexed) via the firmware's `P<1..4>`.

        `port = -1` sends `P0`, the switch's open channel, which is dark.

        The Sercalo hangs off the board's Serial1, not off a port of its own, so port
        selection is a board command. The firmware waits out the mechanical settle and
        echoes `PORT <n>`; a missing or disagreeing echo means the mirror did not move,
        which would otherwise show up as a whole sweep silently taken at the wrong port."""
        self.ser.reset_input_buffer()
        self.ser.write(f"P{int(port) + 1}\n".encode())
        deadline = time.time() + self.cfg.timeout_s + 2.0
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if raw.startswith("PORT"):
                got = int(raw.split()[-1]) - 1
                if got != int(port):
                    raise PICError(f"switch went to port {got}, asked for {port}")
                return got
            if raw.startswith("ERR"):
                raise PICError(f"firmware refused port {port}: {raw!r}")
        raise PICError(f"no PORT echo for port {port} from {self.cfg.port}")

    def set_zero(self):
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
                    continue  # startup banner or a partial line
        raise PICError(f"timeout waiting for {n} values from {self.cfg.port}")


class MockPIC(PIC):
    """Hardware-free stand-in backed by a forward function ``f(v18) -> raw_pds``."""

    def __init__(self, forward=None, noise: float = 3e-4, config: PICConfig | None = None,
                 switch=None, **kw):
        super().__init__(config, **kw)
        self.forward = twin_forward(switch=switch) if forward is None else forward
        self.noise = noise
        self._rng = np.random.default_rng(0)

    def open(self):
        self.ser = "mock"
        return self

    def close(self):
        self.ser = None

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        v = self._prep_dac(voltages)
        y = np.asarray(self.forward(v), float).ravel()
        if self.noise:
            y = y + self._rng.normal(0.0, self.noise, y.shape)
        return np.clip(y, 0.0, None)


def twin_forward(calib=None, error=None, x=None, seed: int = 0, dark: float = 0.01,
                 switch=None):
    """A forward for :class:`MockPIC` that is the actual physics: heater volts -> phases
    through a calibration, phases -> field through the twin, intensity -> photodiode
    volts through the readout scale.

    Defaults to a *fabricated* instance (sampled Vpi, phi0 and coupler error, both seeded)
    rather than the nominal one, so a characterization run against the mock has something
    real to find instead of recovering the constants it was handed.

    Pass `switch` and the injected port follows whatever the optical switch is set to, so a
    per-port sweep against the mock exercises the same loop the bench will."""
    import torch

    from theory.calib import Calibration
    from theory.clements import NMODE as twin_nmode
    from theory.twin import MeshError, Twin

    calib = Calibration.sample(seed=seed) if calib is None else calib
    twin = Twin(MeshError.sample(seed=seed) if error is None else error)
    xt = None if x is None else torch.as_tensor(np.asarray(x), dtype=twin.dtype)

    def _input():
        if xt is not None or switch is None:
            return xt
        e = np.zeros(twin_nmode, complex)
        e[switch.selected or 0] = 1.0
        return torch.as_tensor(e, dtype=twin.dtype)

    def forward(v):
        ph = calib.phases(np.asarray(v, float))
        inten = twin.outputs(torch.as_tensor(ph, dtype=torch.float32),
                             _input()).detach().numpy()
        raw = np.full(NUM_ADC_RAW, dark)
        raw[list(OUT_PDS)] = calib.to_volts_response(inten)
        return raw

    forward.calib, forward.twin = calib, twin
    return forward


assert NUM_DAC == 16  # the firmware sketch and pic.layout must agree on the width
