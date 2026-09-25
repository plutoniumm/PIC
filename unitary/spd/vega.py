"""Vega TDC board: VAUL vault stream parser and serial device.

The board streams binary vaults of photon time-of-flight stamps and shares the same UART with
a text console. A vault on the wire is

    offset  size  field
    0       4     b"VAUL"
    4       4     uint32 LE  sample count, 1..MAX_SAMPLES
    8       8     uint64 LE  elapsed cycles of the acquisition timer (0 = timing unavailable)
    16      4     uint32 LE  timer frequency, Hz (0 = timing unavailable)
    20      4*n   uint32 LE  TOF, ps

Commands are ASCII lines ended by "\\r" and are only written while no vault is on the wire:
the host cannot tell where a command landed inside binary data, so it waits for the gap.

The counting firmware prints `PHOTON_COUNT=n` on the console every FRAME_S instead: the
detections in that frame, dark counts included. Those lines are taken off the console into
a queue of their own, and `Vega.count(seconds)` adds consecutive frames into one reading.
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

BAUD = 1_250_000
MAGIC = b"VAUL"
HEADER = 20
MAX_SAMPLES = 20000
CSV_HEADER = ("Timestamp_Unix", "Burst_ID", "TOF_ps")
FRAME_S = 0.020  # one PHOTON_COUNT line per frame; measured 20.2 ms apart over the UART
COUNT = re.compile(r"PHOTON_COUNT=(\d+)")

log = logging.getLogger("spd")


class SPDError(RuntimeError):
    pass


@dataclass
class Vault:
    burst: int
    cycles: int
    timer_hz: int
    tof_ps: np.ndarray
    t: float

    @property
    def count(self) -> int:
        return len(self.tof_ps)

    @property
    def seconds(self) -> float | None:
        return self.cycles / self.timer_hz if self.cycles and self.timer_hz else None

    @property
    def rate(self) -> float | None:
        """Timestamps/s. (n-1) intervals span the acquisition, as the original logger counts."""
        s = self.seconds
        return None if s is None else ((self.count - 1) / s if self.count > 1 else 0.0)


@dataclass
class Count:
    """Detections added over `frames` consecutive frames."""

    counts: int
    frames: int
    t: float

    @property
    def seconds(self) -> float:
        return self.frames * FRAME_S

    @property
    def rate(self) -> float:
        return self.counts / self.seconds


def frame(tof_ps, cycles: int = 0, timer_hz: int = 0) -> bytes:
    tof = np.asarray(tof_ps, dtype="<u4")
    return MAGIC + struct.pack("<IQI", len(tof), cycles, timer_hz) + tof.tobytes()


def _magic_tail(b: bytes) -> int:
    """Length of a VAUL prefix at the end of b, which the next read may complete."""
    for n in range(len(MAGIC) - 1, 0, -1):
        if b.endswith(MAGIC[:n]):
            return n
    return 0


class VaultParser:
    """Byte stream in, console text (str) and Vault objects out, in wire order.

    `busy` is True from the first byte of a VAUL until its payload is complete; that is the
    window in which a command must not be written.
    """

    def __init__(self):
        self.buf = b""
        self.bursts = 0
        self.busy = False

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []

        def text(b):
            if b:
                out.append(b.decode("utf-8", errors="replace"))

        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                k = _magic_tail(self.buf)
                text(self.buf[: len(self.buf) - k])
                self.buf = self.buf[len(self.buf) - k :]
                self.busy = False
                return out
            if i:
                text(self.buf[:i])
                self.buf = self.buf[i:]
            # "VAUL" also starts the console word "VAULT" ("PARTIAL VAULT DISCARDED"). The
            # original logger took every "VAULT" as text, which also eats a real vault whose
            # count has low byte 0x54 (84, 340, ...). A valid count is <= 20000, so its byte 6
            # is always 0, while text there is printable; wait for byte 6 and decide on it.
            if self.buf.startswith(b"VAULT"):
                if len(self.buf) < 7:
                    self.busy = True
                    return out
                if self.buf[6]:
                    text(self.buf[:5])
                    self.buf = self.buf[5:]
                    continue
            self.busy = True
            if len(self.buf) < HEADER:
                return out
            (n,) = struct.unpack_from("<I", self.buf, 4)
            if not 0 < n <= MAX_SAMPLES:
                log.warning("invalid sample count %d; resynchronizing", n)
                self.buf = self.buf[1:]
                continue
            cycles, hz = struct.unpack_from("<QI", self.buf, 8)
            end = HEADER + 4 * n
            if len(self.buf) < end:
                return out
            tof = np.frombuffer(self.buf, dtype="<u4", count=n, offset=HEADER).copy()
            self.buf = self.buf[end:]
            self.busy = False
            self.bursts += 1
            if not (cycles and hz):
                log.warning("vault %d: acquisition timing unavailable", self.bursts)
            out.append(Vault(self.bursts, cycles, hz, tof, time.time()))


def resolve(port: str) -> str:
    """A device path or a USB serial number (chip id) -> device path; anything else as given."""
    from serial.tools import list_ports

    for p in list_ports.comports():
        # Windows' FTDI driver appends a channel letter to the chip id (AU05XLI8 -> AU05XLI8A)
        if port == p.device or (
            p.serial_number and p.serial_number.rstrip("AB") == port.rstrip("AB")
        ):
            return p.device
    return port


def _rig_ids() -> set:
    try:
        from pic.config import USB_SERIAL
    except ImportError:
        return set()
    return set(USB_SERIAL.values())


def autodetect(exclude=None) -> str:
    """The one USB serial device that is not a known rig instrument; refuses to guess."""
    from serial.tools import list_ports

    exclude = _rig_ids() if exclude is None else set(exclude)
    ports = [p for p in list_ports.comports() if p.vid]
    cand = [p for p in ports if p.serial_number not in exclude]
    if len(cand) == 1:
        return cand[0].device
    seen = "\n".join(
        f"  {p.device}  vid={p.vid:04x} sn={p.serial_number} {p.description}" for p in ports
    )
    why = "no unknown USB serial device" if not cand else f"{len(cand)} candidates"
    raise SPDError(f"cannot pick the Vega port: {why}; pass --port <path or chip id>\n{seen}")


class Vega:
    """Vega board over serial. Vaults go to a queue, console text to `on_text`.

    with Vega("COM7", csv="tof.csv") as v:
        v.send("help")
        for vault in v.vaults(timeout=5):
            print(vault.count, vault.rate)
    """

    def __init__(self, port=None, *, baud=BAUD, csv=None, on_text=None, ser=None):
        self.port, self.baud, self.csv, self.ser = port, baud, csv, ser
        self.on_text = on_text or (lambda s: print(s, end="", flush=True))
        self.parser = VaultParser()
        self.q: queue.Queue = queue.Queue()
        self.frames: queue.Queue = queue.Queue()  # (t, n) per PHOTON_COUNT line
        self._line = ""
        self._safe = threading.Event()
        self._safe.set()
        self._stop = threading.Event()
        self._wlock = threading.Lock()
        self._thread = None

    def open(self):
        if self._thread:
            return self
        if self.ser is None:
            import serial

            dev = resolve(self.port) if self.port else autodetect()
            try:
                self.ser = serial.Serial(dev, self.baud, timeout=0.05)
            except serial.SerialException as e:
                raise SPDError(f"cannot open {dev}: {e}")
        self.ser.reset_input_buffer()
        if self.csv and not os.path.exists(self.csv):
            with open(self.csv, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx, daemon=True, name="vega-rx")
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(1.0)
            self._thread = None
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _rx(self):
        try:
            while not self._stop.is_set():
                data = self.ser.read(max(1, self.ser.in_waiting))
                if not data:
                    continue
                items = self.parser.feed(data)
                # Set/clear before handing items out, so a sender woken by a finished vault
                # never sees a stale busy flag.
                (self._safe.clear if self.parser.busy else self._safe.set)()
                for it in items:
                    if isinstance(it, str):
                        self._text(it)
                    else:
                        self._log(it)
                        self.q.put(it)
        except Exception as e:  # a dead port must surface in the consumer, not vanish
            self.q.put(e)
            self.frames.put(e)

    def _text(self, s: str):
        # line by line, since a count line can arrive split across reads; 50 of them a
        # second are data, not console, and never reach on_text
        *lines, self._line = (self._line + s).split("\n")
        for ln in lines:
            m = COUNT.fullmatch(ln.strip())
            if m:
                self.frames.put((time.time(), int(m.group(1))))
            else:
                self.on_text(ln + "\n")

    def count(self, seconds: float = 1.0, timeout: float = 1.0) -> Count:
        """Detections over the next `seconds`, added from whole frames. Frames that arrived
        before the call are dropped: a reading starts after whatever was just changed."""
        while not self.frames.empty():
            self.frames.get_nowait()
        n = max(1, round(seconds / FRAME_S))
        total = 0
        for _ in range(n):
            try:
                it = self.frames.get(timeout=timeout)
            except queue.Empty:
                raise SPDError(f"no PHOTON_COUNT frame within {timeout} s")
            if isinstance(it, Exception):
                raise SPDError(f"serial link lost: {it}") from it
            total += it[1]
        return Count(total, n, it[0])

    def _log(self, v: Vault):
        if self.csv:
            with open(self.csv, "a", newline="") as f:
                csv.writer(f).writerows([v.t, v.burst, int(x)] for x in v.tof_ps)

    def send(self, cmd: str, timeout: float | None = None):
        if not self._safe.wait(timeout):
            raise SPDError(f"vault still arriving after {timeout} s; {cmd!r} not sent")
        # ponytail: the board may start a vault between wait() and write(); the host cannot
        # see that, and the original logger has the same window. Closing it needs firmware.
        with self._wlock:
            self.ser.write((cmd.strip() + "\r").encode("ascii"))
            self.ser.flush()

    @property
    def busy(self) -> bool:
        return not self._safe.is_set()

    def get(self, timeout: float | None = None) -> Vault:
        try:
            it = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no vault within {timeout} s")
        if isinstance(it, Exception):
            raise SPDError(f"serial link lost: {it}") from it
        return it

    def vaults(self, timeout: float | None = None):
        """Yield vaults until none arrives for `timeout` s (forever if None)."""
        while True:
            try:
                yield self.get(timeout)
            except TimeoutError:
                return


def _selftest():
    import random
    import tempfile

    from .mock import MockVega

    rng = np.random.default_rng(1)
    f1 = frame([1, 2, 3, 4, 5], 1000, 1_000_000)
    f2 = frame(rng.integers(0, 2**32, 84, dtype=np.uint32), 5, 10)  # low count byte = "T"
    f3 = frame([7, 8], 0, 0)  # timing unavailable
    junk = b"VAUL\xff\xff\xff\xff" + bytes(range(100, 112))  # bad count, must resync
    stream = (
        b"boot ok\r\n"
        + f1
        + b"PARTIAL VAULT DISCARDED\r\n"
        + b"\xfe\x01VA"
        + f2
        + junk
        + f3
        + b"VAULT\r\n"
        + b"tail VA"
    )
    ref = [f1, f2, f3]

    def run(chunks):
        p, items = VaultParser(), []
        for c in chunks:
            items += p.feed(c)
        return (
            p,
            [x for x in items if isinstance(x, Vault)],
            "".join(x for x in items if isinstance(x, str)),
        )

    logging.getLogger("spd").setLevel(logging.ERROR)
    splits = [[stream], [stream[i : i + 1] for i in range(len(stream))]]
    for _ in range(200):
        cuts = sorted(random.Random(_).sample(range(1, len(stream)), 12))
        splits.append([stream[a:b] for a, b in zip([0] + cuts, cuts + [len(stream)])])
    for chunks in splits:
        p, vs, txt = run(chunks)
        assert [frame(v.tof_ps, v.cycles, v.timer_hz) for v in vs] == ref, [v.count for v in vs]
        assert [v.burst for v in vs] == [1, 2, 3]
        assert "boot ok" in txt and "PARTIAL VAULT DISCARDED" in txt and "VAULT\r\n" in txt
        assert txt.endswith("tail ") and p.buf == b"VA" and not p.busy
    assert vs[0].seconds == 1e-3 and vs[0].rate == 4000.0 and vs[2].seconds is None

    p = VaultParser()
    p.feed(f1[:3])
    assert p.busy is False and p.buf == b"VAU"  # a partial magic is not yet a vault
    p.feed(f1[3:10])
    assert p.busy
    assert len(p.feed(f1[10:])) == 1 and not p.busy

    # End to end through the device class and the physical mock.
    tof, sig, dark = 12_000.0, 2e5, 2e3
    mock = MockVega(samples=2000, signal_hz=sig, dark_hz=dark, tof_ps=tof, jitter_ps=40.0)
    texts = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tof.csv")
        with Vega(ser=mock, csv=path, on_text=texts.append) as v:
            vs = [v.get(5) for _ in range(3)]
            for i in range(4):
                v.send(f"ping {i}", timeout=5)
                time.sleep(0.02)
            vs += [v.get(5) for _ in range(2)]
        with open(path) as f:
            rows = sum(1 for _ in f)
    assert rows == 1 + v.parser.bursts * mock.samples, rows
    assert not mock.lost, mock.lost
    txt = "".join(texts)
    assert all(f"ping {i}" in txt for i in range(4)) and "PARTIAL VAULT DISCARDED" in txt
    t = np.concatenate([x.tof_ps for x in vs]).astype(float)
    peak = t[abs(t - tof) < 500]
    assert abs(np.median(peak) - tof) < 10 and 30 < peak.std() < 50, (np.median(peak), peak.std())
    assert abs(len(peak) / len(t) - sig / (sig + dark)) < 0.01
    for x in vs:
        assert abs(x.rate / (sig + dark) - 1) < 0.1, x.rate

    # Refusal path: a command written into a vault is lost on the board, not queued.
    mock2 = MockVega(samples=5000, signal_hz=1e7)
    deadline = time.monotonic() + 2
    while not mock2.in_vault and time.monotonic() < deadline:
        mock2.read(1)
    mock2.write(b"lost\r")
    assert mock2.lost == ["lost"]

    # Counting firmware: frames added into a reading, split lines rejoined, console kept.
    texts = []
    with Vega(ser=MockVega(count_hz=500.0, seed=3), on_text=texts.append) as v:
        rs = [v.count(1.0) for _ in range(4)]
        v.send("STATUS")
        time.sleep(0.1)
    assert all(r.frames == 50 and r.seconds == 1.0 for r in rs)
    rate = sum(r.counts for r in rs) / 4
    assert abs(rate - 500) < 4 * np.sqrt(500 / 4), rate  # Poisson, 4 sigma
    assert "STATUS" in "".join(texts) and "PHOTON_COUNT" not in "".join(texts)
    p = Vega(ser=MockVega(count_hz=0.0), on_text=texts.append)
    for c in ("PHOTON_CO", "UNT=7\r", "\nPHOTON_COUNT=", "2\r\n"):
        p._text(c)
    assert [p.frames.get_nowait()[1] for _ in range(2)] == [7, 2]
    print("spd selftest ok")
