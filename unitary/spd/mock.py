"""Hardware-free Vega board: a physical model behind a fake serial port.

Photon detections are Poisson: signal at `signal_hz` with Gaussian timing jitter around
`tof_ps`, dark counts at `dark_hz` spread uniformly over the TDC window. A vault fills with
`samples` stamps, so its acquisition time is Gamma(samples, 1/(signal+dark)), stamped in timer
cycles. Bytes leave at the UART's real rate (10 bits per byte), so vaults take real time on
the wire and a reader sees them split across reads exactly as it would on hardware.

Board behaviour it reproduces:
  * a command that arrives during acquisition discards that vault and prints
    "PARTIAL VAULT DISCARDED", the text that collides with the VAUL magic;
  * a command that arrives while a vault is on the wire is lost (kept in `.lost`), which is
    why the host must wait for the gap;
  * `timing=False` sends zero cycles and zero timer Hz, as the board does without a timer.

The reply format ("<cmd>\\r\\nOK\\r\\n") is a placeholder: the real console's is not documented.

`count_hz` switches it to the counting firmware: no vaults, one `PHOTON_COUNT=n` line per
FRAME_S with n ~ Poisson(count_hz * FRAME_S), as the dark stream measured on the bench is.
"""

from __future__ import annotations

import time

import numpy as np

from .vega import BAUD, FRAME_S, frame


class MockVega:
    """Duck-types the slice of `serial.Serial` that `Vega` uses."""

    def __init__(
        self,
        port="mock-spd",
        baud=BAUD,
        *,
        samples=2000,
        signal_hz=2e5,
        dark_hz=500.0,
        tof_ps=12_000.0,
        jitter_ps=45.0,
        window_ps=100_000,
        timer_hz=64_000_000,
        timing=True,
        seed=0,
        timeout=0.05,
        count_hz=None,
    ):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.samples, self.signal_hz, self.dark_hz = samples, signal_hz, dark_hz
        self.tof_ps, self.jitter_ps, self.window_ps = tof_ps, jitter_ps, window_ps
        self.timer_hz, self.timing = timer_hz, timing
        self.count_hz = count_hz
        self.rng = np.random.default_rng(seed)
        self.is_open = True
        self.lost: list[str] = []
        self.received: list[str] = []
        self._rx = bytearray()  # on the host side of the wire
        self._out = bytearray()  # queued in the board's UART
        self.in_vault = False
        self._cmd = b""
        self._t = time.monotonic()  # wire clock: when the next byte may leave
        self._start_acq(self._t)

    def _start_acq(self, t0):
        if self.count_hz is not None:
            self._acq_done = t0 + FRAME_S
            return
        self._acq_s = self.rng.gamma(self.samples, 1.0 / (self.signal_hz + self.dark_hz))
        self._acq_done = t0 + self._acq_s

    def _vault(self) -> bytes:
        n = self.samples
        sig = self.rng.random(n) < self.signal_hz / (self.signal_hz + self.dark_hz)
        t = np.where(
            sig,
            self.rng.normal(self.tof_ps, self.jitter_ps, n),
            self.rng.uniform(0, self.window_ps, n),
        )
        tof = np.clip(np.rint(t), 0, self.window_ps - 1)
        if not self.timing:
            return frame(tof)
        return frame(tof, round(self._acq_s * self.timer_hz), self.timer_hz)

    def _pump(self):
        now = time.monotonic()
        if self.count_hz is not None:
            while self._acq_done <= now:
                n = self.rng.poisson(self.count_hz * FRAME_S)
                self._out += f"PHOTON_COUNT={n}\r\n".encode()
                self._acq_done += FRAME_S
            self._rx += self._out
            self._out.clear()
            return
        while True:
            if not self._out:
                if self._acq_done > now:
                    self._t = now
                    return
                self._t = max(self._t, self._acq_done)
                self._out += self._vault()
                self.in_vault = True
            n = min(len(self._out), int((now - self._t) * self.baud / 10))
            if n <= 0:
                return
            self._rx += self._out[:n]
            del self._out[:n]
            self._t += n * 10 / self.baud
            if not self._out and self.in_vault:
                self.in_vault = False
                self._start_acq(self._t)

    @property
    def in_waiting(self) -> int:
        self._pump()
        return len(self._rx)

    def read(self, size=1) -> bytes:
        deadline = time.monotonic() + (self.timeout or 0)
        while True:
            self._pump()
            if self._rx or time.monotonic() >= deadline:
                out = bytes(self._rx[:size])
                del self._rx[:size]
                return out
            time.sleep(0.001)

    def write(self, data: bytes) -> int:
        self._pump()
        self._cmd += data
        *cmds, self._cmd = self._cmd.split(b"\r")
        for c in cmds:
            c = c.decode("ascii", errors="replace")
            if self.in_vault:
                self.lost.append(c)
                continue
            self.received.append(c)
            self._out += b"PARTIAL VAULT DISCARDED\r\n" + c.encode() + b"\r\nOK\r\n"
            self._start_acq(time.monotonic())
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._pump()
        self._rx.clear()

    def close(self):
        self.is_open = False
