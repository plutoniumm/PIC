"""End-to-end check: drive the laser at a few powers, read the PIC photodiodes at each.

Confirms the whole optical chain -- laser -> fiber -> chip -> photodiodes -> Arduino ->
host. Prints a DARK (laser off) baseline then one row per power level; if the live PDs
climb above dark, light is reaching the detectors.

    python e2e.py                       # sweep +5/+10/+15 dBm, DACs at 0, read the 10 live PDs
    python e2e.py 3 9 15                # custom dBm levels
    python e2e.py --raw 20 60 100       # levels are raw setpoints, not dBm
    python e2e.py --dac 5:2,9:4         # set some DAC heaters before reading (default: all 0)
    python e2e.py --mock                # no hardware -- exercises the harness end to end

Two independent serial devices: the laser is the FTDI adapter, the PIC is the Arduino.
The PIC autodetect and the laser FTDI share the usbserial glob, so we find the laser
first and exclude its port before opening the PIC.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pic import PIC, PICError
from src.pic.config import NUM_DAC
from src.pic.interface import _PORT_PATTERNS
from laser.laser import Laser, LaserError
from laser.pdmv5 import PDMv5Error


def find_pic_port(explicit, exclude):
    if explicit:
        return explicit
    import glob

    cands = sorted({p for pat in _PORT_PATTERNS for p in glob.glob(pat)})
    cands = [p for p in cands if p not in exclude]
    # the Arduino is a CH340/usbmodem; a bare usbserial is almost always the laser FTDI
    pref = [p for p in cands if "usbmodem" in p or "wchusbserial" in p]
    return (pref or cands or [None])[0]


def build_vec(spec: str) -> np.ndarray:
    v = np.zeros(NUM_DAC)
    for part in filter(None, (s.strip() for s in spec.split(","))):
        ch, val = part.split(":")
        v[int(ch)] = float(val)
    return v


def _poke(laser, sp):
    # re-assert the setpoint so the laser's idle watchdog doesn't drop the master mid-read
    try:
        laser.dev.write_setting("cw_current", sp, verify=False)
    except PDMv5Error:
        pass


def _dwell(laser, sp, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        _poke(laser, sp)
        time.sleep(min(0.3, secs))


def read_live(pic, vec, live, laser, sp, repeats, settle):
    """Mean/std of the live PDs over `repeats` reads, re-poking the laser throughout."""
    if settle > 0:
        pic.measure_raw(vec)
        _dwell(laser, sp, settle)
    ys = []
    for _ in range(repeats):
        _poke(laser, sp)
        ys.append(np.asarray(pic.measure_raw(vec), float)[list(live)])
    Y = np.asarray(ys)
    return Y.mean(0), Y.std(0)


class _FakeDev:
    port = "mock-laser"

    def __init__(self):
        self.cur = 0.0

    def write_setting(self, name, val, **kw):
        if name == "cw_current":
            self.cur = float(val)

    def read_setting(self, name):
        return self.cur if name == "cw_current" else 1

    def measure(self, name):
        return 0.0


class _FakeLaser:
    setpoint_to_dbm = staticmethod(Laser.setpoint_to_dbm)
    dbm_to_setpoint = staticmethod(Laser.dbm_to_setpoint)

    def __init__(self):
        self.dev = _FakeDev()

    def open(self):
        return self

    def close(self):
        pass

    def hw_on(self):
        self.dev.cur = Laser.FLOOR_SP
        print("  [mock] laser ON at floor")

    def hw_off(self):
        self.dev.cur = 0.0
        print("  [mock] laser OFF")

    def set(self, dbm, raw=False):
        self.dev.cur = float(dbm) if raw else Laser.dbm_to_setpoint(dbm)[0]


def _mock_pic(laser):
    def forward(v):
        _, mW = Laser.setpoint_to_dbm(laser.dev.cur)
        base = 0.02 + 0.0005 * np.arange(14)
        gain = 0.01 * (1 + 0.3 * np.sin(np.arange(14)))
        return base + gain * max(mW, 0.0)

    from src.pic import MockPIC

    return MockPIC(forward, noise=1e-3).open()


def _print_table(live, dark, rows):
    head = (
        f"  {'level':>14}  "
        + " ".join(f"{'pd' + str(j):>6}" for j in live)
        + f"  {'dL2':>7}"
    )
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for label, y in [("DARK (off)", dark)] + rows:
        cells = " ".join(f"{x:6.3f}" for x in y)
        dl2 = float(np.linalg.norm(y - dark))
        print(f"  {label:>14}  {cells}  {dl2:7.4f}")


def _verdict(live, dark, rows):
    if not rows:
        return
    _, bright = rows[-1]
    delta = bright - dark
    ch = int(np.argmax(np.abs(delta)))
    rise = float(delta[ch])
    noise = 5e-3
    print()
    if np.linalg.norm(delta) > max(0.01, 5 * noise) and rise > 0:
        print(
            f"  LIGHT DETECTED: brightest level lifts pd{live[ch]} by {rise:+.3f} V "
            f"from dark. Chain works."
        )
    else:
        print(
            "  NO clear response -- PDs flat vs dark. Check fiber coupling / alignment, "
            "that the laser is actually lit (`./do laser status`), and the input DACs."
        )


def run(levels, *, raw, repeats, settle, dac, pic_port, laser_port, mock):
    vec = build_vec(dac)
    unit = "sp" if raw else "dBm"

    if mock:
        laser = _FakeLaser().open()
        pic = _mock_pic(laser)
        print("MOCK: no hardware; PDs are synthesised from the laser setpoint.")
    else:
        laser = Laser(port=laser_port).open()
        pic_port = find_pic_port(pic_port, {laser.dev.port})
        if not pic_port:
            laser.close()
            print(
                "ERROR: no PIC/Arduino serial port found (only the laser FTDI is present)."
                "\n  Plug in the Arduino, or pass --pic-port /dev/cu.usbmodemXXXX."
            )
            return 1
        pic = PIC(port=pic_port).open()
        print(f"laser: {laser.dev.port}   PIC: {pic_port}")

    rows = []
    try:
        # verify the PIC actually speaks the firmware (14 floats back) before trusting it
        pic.measure_raw(vec)

        laser.hw_off()  # guarantee a true dark baseline
        dark, _ = read_live(pic, vec, pic.live_pds, laser, 0.0, repeats, 0.0)

        laser.hw_on()
        for lvl in levels:
            laser.set(lvl, raw=raw)
            sp = float(laser.dev.read_setting("cw_current"))
            y, _ = read_live(pic, vec, pic.live_pds, laser, sp, repeats, settle)
            _, emW = Laser.setpoint_to_dbm(sp)
            label = f"{lvl:g} {unit}" if raw else f"{lvl:+g} dBm"
            print(f"  set {label:>10}  (sp {sp:5.1f}, ~{emW:5.1f} mW)")
            rows.append((label, y))
    finally:
        try:
            laser.hw_off()
        except Exception as e:
            print(f"  (laser off warning: {e})")
        laser.close()
        pic.close()  # zeros the DACs

    _print_table(pic.live_pds, dark, rows)
    _verdict(pic.live_pds, dark, rows)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="./test", description="end-to-end laser -> PIC photodiode read"
    )
    ap.add_argument("levels", nargs="*", type=float, default=[5.0, 10.0, 15.0])
    ap.add_argument(
        "--raw", action="store_true", help="levels are raw setpoints, not dBm"
    )
    ap.add_argument(
        "--repeats", type=int, default=5, help="PD reads averaged per level"
    )
    ap.add_argument(
        "--settle", type=float, default=0.5, help="dwell before each read [s]"
    )
    ap.add_argument(
        "--dac", default="", help="DAC heaters, e.g. 5:2,9:4 (default all 0)"
    )
    ap.add_argument("--pic-port", default=None)
    ap.add_argument("--laser-port", default=None)
    ap.add_argument(
        "--mock", action="store_true", help="no hardware; self-test the harness"
    )
    a = ap.parse_args(argv)

    levels = a.levels or [5.0, 10.0, 15.0]
    try:
        return run(
            levels,
            raw=a.raw,
            repeats=a.repeats,
            settle=a.settle,
            dac=a.dac,
            pic_port=a.pic_port,
            laser_port=a.laser_port,
            mock=a.mock,
        )
    except (LaserError, PICError, PDMv5Error) as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted -- laser and DACs driven safe on exit.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
