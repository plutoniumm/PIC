"""AeroDiode PDMv5 laser driver over serial -- Windows-free, no vendor lib, no Wine.

Protocol + register map reverse-engineered from the vendor `PDMv5.dll` and CONFIRMED
against captured GUI traffic (the GUI's `write id=0x16 val=41d00000` decodes to exactly
26.0 C via SetTemperature, reg 0x16 F32). Frame layout:

    host->dev : [LEN][ADDR][CMD][DATA...][CHK]     CHK = (XOR(all prior bytes) - 1) & 0xFF
    dev->host : [LEN][STS ][DATA...][CHK]           STS 0x00 = ok
    LEN = total frame length incl LEN and CHK.  Register id = 2 bytes big-endian.
    Value = F32 big-endian (4 B) or U08 (1 B), per the DLL's per-register format code.

Commands: 0x01 read-address, 0x02 version, 0x10 write-instr, 0x11 read-instr,
          0x12 apply, 0x14 measure (live telemetry).  Currents are in mA.

This class READS freely; every emission call (set current / enable / TEC) is a plain
method, but the safe ramp / zero-before-enable / never-disable-blind policy lives in
laser.py -- drive through the Laser class, not by poking these directly.

VERBATIM COPY of 6x6/pic/devices/pdmv5.py -- the PDMv5 is the same physical
instrument on both rigs and its wire protocol is frozen. Fix a protocol bug in
both places or neither; do not let them drift.
"""

from __future__ import annotations
import math
import struct
import time

# What a vanishing USB serial node throws. serial.SerialException subclasses OSError, but on
# POSIX pyserial lets termios.error through unwrapped, and that is the one that actually
# surfaced when this adapter re-enumerated mid-sweep. termios does not exist on Windows --
# importing it unconditionally stopped this whole driver loading there -- and pyserial's
# Windows backend raises SerialException anyway, so OSError alone covers it.
try:
    import termios

    _LINK_ERRORS = (OSError, termios.error)
except ImportError:
    _LINK_ERRORS = (OSError,)

# name -> (register id, format).  SETTINGS use write-instr(0x10)+apply / read-instr(0x11).
SETTINGS = {
    "cw_max_current": (0x3E, "f32"),
    "max_avg_current": (0x1F, "f32"),
    "max_current": (0x2A, "f32"),
    "cw_current": (0x3F, "f32"),  # CW current setpoint (the ramp target)
    "cw_laser_status": (0x3C, "u8"),  # CW output enable
    "laser_status": (0x20, "u8"),  # master output enable
    "tec_status": (0x14, "u8"),  # TEC on/off
    "temperature": (0x16, "f32"),  # TEC temperature setpoint, deg C
    "operating_mode": (0x41, "u8"),  # 0 = ACC (const current), 1 = APC (const power)
    "cw_current_source": (0x3D, "u8"),  # 0 = INT, 1 = EXT
    "factory_max_current": (0x09, "f32"),
    "factory_max_avg_current": (0x0B, "f32"),
}

# name -> (measure id, format).  TELEMETRY via measure(0x14); a separate id namespace.
MEASURES = {
    "key": (0x00, "u8"),  # 1 = key switch unlocked
    "bnc_interlock": (0x01, "u8"),  # 1 = satisfied
    "ext_interlock": (0x02, "u8"),  # 1 = satisfied
    "diode_temperature": (0x14, "f32"),
    "temperature_consign": (0x0C, "f32"),
    "diode_cw_current": (0x1F, "f32"),  # actual measured diode current
    "diode_avg_current": (0x20, "f32"),
    "diode_voltage": (0x1E, "f32"),
    "compliance_voltage": (0x3D, "f32"),
    "input_voltage": (0x3C, "f32"),
    "bfm_optical_power": (0x2A, "f32"),  # built-in monitor photodiode -> is light out?
    "tec_current": (0x16, "f32"),
    "driver_enable": (0x32, "u8"),
    "mmd_enable": (0x33, "u8"),
    "liv_status": (0x03, "u32"),
    "alarms": (0x46, "u32"),  # fault bitmask -> why emission is refused
}

_CMD_READADDR, _CMD_VERSION = 0x01, 0x02
_CMD_WRITE, _CMD_READ, _CMD_APPLY, _CMD_MEASURE = 0x10, 0x11, 0x12, 0x14
BAUD = 125000


def _chk(body: bytes) -> int:
    x = 0
    for b in body:
        x ^= b
    return (x - 1) & 0xFF


def _frame(addr: int, cmd: int, data: bytes = b"") -> bytes:
    body = bytes([4 + len(data), addr, cmd]) + data
    return body + bytes([_chk(body)])


def _encode(fmt: str, value) -> bytes:
    # format early-exit: refuse non-numeric / non-finite / out-of-range before any bytes
    # reach the driver (struct.pack would happily encode a NaN setpoint; int&0xFF would
    # silently wrap an out-of-range enable byte).
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise PDMv5Error(f"non-numeric {fmt} value {value!r}")
    if not math.isfinite(x):
        raise PDMv5Error(f"refusing to encode non-finite {fmt} value {value!r}")
    if fmt == "f32":
        return struct.pack(">f", x)
    if fmt in ("u8", "u32"):
        hi = 0xFF if fmt == "u8" else 0xFFFFFFFF
        if not 0 <= int(x) <= hi:
            raise PDMv5Error(f"{fmt} value out of range 0..{hi}: {value!r}")
        return bytes([int(x)]) if fmt == "u8" else struct.pack(">I", int(x))
    raise ValueError(f"bad fmt {fmt}")


def _decode(fmt: str, data: bytes):
    if fmt == "f32":
        return struct.unpack(">f", data[:4])[0]
    if fmt == "u32":
        return struct.unpack(">I", data[:4])[0]
    if fmt == "u8":
        return data[0] if data else None
    raise ValueError(f"bad fmt {fmt}")


class PDMv5Error(RuntimeError):
    pass


class PDMv5:
    def __init__(self, port: str, addr: int | None = None, *, timeout: float = 0.3):
        self.port = port
        self.addr = addr  # None -> discover via read-address
        self.timeout = timeout
        self.ser = None

    def open(self):
        import serial

        self.ser = serial.Serial(self.port, BAUD, timeout=self.timeout)
        time.sleep(0.05)
        self.ser.reset_input_buffer()

        if self.addr is None:
            self.addr = self.read_address()

        return self

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _txn(self, cmd: int, data: bytes = b"", addr: int | None = None, retries: int = 5):
        a = self.addr if addr is None else addr
        pkt = _frame(a if a is not None else 0, cmd, data)

        # FTDI link is flaky and mangles the odd frame; commands are idempotent, so resend
        # on a missing/corrupt reply.
        last = None
        for attempt in range(retries + 1):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(pkt)
                self.ser.flush()
                resp = self._read_frame()
            except _LINK_ERRORS as e:
                # This adapter re-enumerates under load: the node stays in /dev, every call
                # raises "Device not configured" for a second or two, and then it comes back.
                # Left unhandled that kills an hour-long sweep at whatever minute it happens,
                # and -- worse -- it kills it inside Rig.close(), so the laser is never told
                # to switch off. Reopening is the whole recovery; it is the same port.
                last = f"link dropped ({type(e).__name__}: {e})"
                self._reopen()
                time.sleep(0.4 * (attempt + 1))
                continue

            if resp is not None:
                _, sts, body = resp
                if sts != 0x00:
                    raise PDMv5Error(f"device error status 0x{sts:02x} for cmd 0x{cmd:02x}")
                return body

            last = "no/again bad frame"
            time.sleep(0.05)

        raise PDMv5Error(f"no valid response to cmd 0x{cmd:02x} ({last})")

    def _reopen(self):
        """Close and reopen the port, swallowing whatever the dying handle throws."""
        import serial

        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        try:
            self.ser = serial.Serial(self.port, BAUD, timeout=self.timeout)
        except Exception:
            self.ser = None

    def _read_frame(self):
        """Read one [LEN][STS][DATA...][CHK] frame; return (len,sts,data) or None."""
        if self.ser is None:
            raise OSError("port is not open")
        head = self.ser.read(1)
        if not head:
            return None

        length = head[0]
        if length < 3 or length > 64:
            return None

        rest = self.ser.read(length - 1)
        if len(rest) != length - 1:
            return None

        frame = head + rest
        if _chk(frame[:-1]) != frame[-1]:
            return None

        return length, frame[1], frame[2:-1]

    def read_address(self) -> int:
        body = self._txn(_CMD_READADDR, addr=0x00)
        return body[0] if body else 1

    def version(self):
        return self._txn(_CMD_VERSION)

    def read_setting(self, name: str):
        rid, fmt = SETTINGS[name]
        body = self._txn(_CMD_READ, bytes([(rid >> 8) & 0xFF, rid & 0xFF]))

        return _decode(fmt, body)

    def measure(self, name: str):
        mid, fmt = MEASURES[name]
        body = self._txn(_CMD_MEASURE, bytes([(mid >> 8) & 0xFF, mid & 0xFF]))

        return _decode(fmt, body)

    def write_setting(self, name: str, value, *, apply: bool = True, verify: bool = True):
        rid, fmt = SETTINGS[name]
        try:
            payload = _encode(fmt, value)
        except PDMv5Error as e:
            raise PDMv5Error(f"refusing to write {name}: {e}")
        data = bytes([(rid >> 8) & 0xFF, rid & 0xFF]) + payload

        self._txn(_CMD_WRITE, data)
        if apply:
            self._txn(_CMD_APPLY)

        if not verify:
            return None

        got = self.read_setting(name)
        if fmt == "u8" and int(got) != (int(value) & 0xFF):
            raise PDMv5Error(f"write {name}={value} not honoured (read back {got})")
        if fmt == "f32" and abs(float(got) - float(value)) > max(0.5, 0.01 * abs(float(value))):
            raise PDMv5Error(f"write {name}={value} not honoured (read back {got})")

        return got

    def status(self) -> dict:
        """Read everything relevant. Safe: no writes, no emission."""
        s = {"address": self.addr}

        for n in (
            "key",
            "bnc_interlock",
            "ext_interlock",
            "driver_enable",
            "diode_cw_current",
            "diode_temperature",
            "temperature_consign",
        ):
            try:
                s[n] = self.measure(n)
            except Exception as e:
                s[n] = f"<err {e}>"

        for n in (
            "operating_mode",
            "cw_current_source",
            "tec_status",
            "temperature",
            "cw_laser_status",
            "laser_status",
            "cw_current",
            "cw_max_current",
            "max_avg_current",
            "max_current",
            "factory_max_current",
            "factory_max_avg_current",
        ):
            try:
                s[n] = self.read_setting(n)
            except Exception as e:
                s[n] = f"<err {e}>"

        return s


def _fmt_status(s: dict) -> str:
    lab = {
        "address": "device address",
        "key": "key switch (1=unlocked)",
        "bnc_interlock": "BNC interlock (1=ok)",
        "ext_interlock": "EXT interlock (1=ok)",
        "driver_enable": "driver enabled",
        "diode_cw_current": "measured diode current [mA]",
        "diode_temperature": "diode temp [C]",
        "temperature_consign": "temp setpoint readback [C]",
        "operating_mode": "operating mode (0=ACC,1=APC)",
        "cw_current_source": "CW source (0=INT,1=EXT)",
        "tec_status": "TEC on/off",
        "temperature": "TEC setpoint [C]",
        "cw_laser_status": "CW output enable",
        "laser_status": "master output enable",
        "cw_current": "CW current setpoint [mA]",
        "cw_max_current": "CW max-current ceiling [mA]",
        "max_avg_current": "max avg-current ceiling [mA]",
        "max_current": "max-current ceiling [mA]",
        "factory_max_current": "FACTORY max current [mA]",
        "factory_max_avg_current": "FACTORY max avg current [mA]",
    }

    out = ["  PDMv5 status:"]
    for k, v in s.items():
        vs = f"{v:.3f}" if isinstance(v, float) else str(v)
        out.append(f"    {lab.get(k, k):38} {vs}")

    return "\n".join(out)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="PDMv5 laser: read-only status dump")
    ap.add_argument("--port", default="/dev/cu.usbserial-AU05XLI8")
    a = ap.parse_args(argv)

    with PDMv5(a.port) as d:
        print(f"connected: addr={d.addr}")
        print(_fmt_status(d.status()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
