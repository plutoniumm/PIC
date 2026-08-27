"""Opening the instruments, and the one guarded laser session everything runs inside.

`laser_session` is ported from the 6x6 rig unchanged in substance: a background timer
hard-offs the laser after `duration_s` no matter what the measurement loop does, all
laser-serial access is under one lock so the watchdog can never collide with a keepalive,
and emission is verified by baselining the monitor photodiode off-versus-on rather than
trusting `laser_status`. That last point is not paranoia -- the driver reports LASER ON
while the front panel sits at READY and no light leaves the diode. Do not weaken any of it.

What is new here is the TEC. `telemetry()` returns the chip temperature alongside the
laser state, and a session refuses to open unless the substrate is inside its band, so a
run cannot silently record data taken while the chip was still settling.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

from .devices.laser import Laser
from .devices.mock import MockLaser
from .devices.tec import MockTEC, NoTEC, make_tec
from .interface import PIC, MockPIC, PICError, find_port


def _resolve_pic_port(explicit, laser_port):
    """The board's port, excluding the laser FTDI that `find_port`'s glob also matches."""
    if explicit:
        return explicit
    env = os.environ.get("PIC4_PORT")
    if env:
        return env
    _, cands = find_port(None)
    cands = [c for c in cands if c != laser_port]
    if not cands:
        raise PICError("no board serial port found (after excluding the laser). Pass "
                       "--pic-port or set $PIC4_PORT; is another process holding it?")
    return cands[0]


def open_devices(mock=False, laser_port=None, pic_port=None, tec="mock"):
    """Return (laser, pic, tec), all open. Real hardware unless mock=True."""
    if mock:
        return MockLaser().open(), MockPIC().open(), MockTEC().open()
    laser = Laser(port=laser_port).open()
    pic = PIC(port=_resolve_pic_port(pic_port, laser.dev.port)).open()
    return laser, pic, make_tec(tec).open()


class _Session:
    """Handle yielded by :func:`laser_session`; see there for the fields and methods."""


@contextmanager
def laser_session(laser, *, duration_s, power_dbm, tec=None, bfm_off=None, emit_eps=0.05,
                  read_pds=None, pd_eps=0.02,
                  require_stable=True):
    """Bounded, watchdog-guarded laser-on session. The only copy of the laser safety logic.

    On enter: check the chip is at temperature, baseline the monitor photodiode with the
    laser still off, turn on to `power_dbm`, and arm a background hard-off at `duration_s`.
    Yields a handle with `.sp .mA .bfm_off .bfm_on .emitted .deadline .chip_c`, plus
    `.keepalive(sleep_s=0)` (re-pokes the setpoint so the driver's idle self-disable cannot
    drop the beam), `.set_power(dbm)`, `.telemetry()` and `.expired()`. On every exit path
    -- normal, exception, or watchdog -- the timer is cancelled and the laser forced off.
    """
    if tec is not None and require_stable and not isinstance(tec, NoTEC):
        tec.wait_stable()

    lock = threading.Lock()
    stopped = threading.Event()
    wd = None

    def _bfm():
        return float(laser.dev.measure("bfm_optical_power"))

    def force_off():  # a watchdog must never die silently
        with lock:
            try:
                print(f"\n[watchdog] {duration_s:.0f}s reached -> forcing laser OFF")
                laser.off()
            except Exception as e:
                print(f"[watchdog] hw_off error: {e}")
        stopped.set()

    s = _Session()
    s.stopped = stopped
    s.tec = tec
    s.bfm_off = _bfm() if bfm_off is None else bfm_off  # laser still off here
    s.pds_off = None if read_pds is None else read_pds()   # dark baseline on the chip
    try:
        with lock:
            laser.on(power_dbm)  # open (idempotent) + enable + ramp, one call
            s.sp = float(laser.dev.read_setting("cw_current"))
            s.mA = laser.measured_mA()
        wd = threading.Timer(duration_s, force_off)
        wd.daemon = True
        wd.start()
        s.deadline = time.time() + duration_s
        s.bfm_on = _bfm()
        s.emitted = (s.bfm_on - s.bfm_off) > emit_eps
        # The beam monitor is the wrong sole witness on this unit: it reads near 3.5 V with
        # the laser off, so a real turn-on moves it by less than the threshold and a lit
        # chip reports as dark. The detectors on the far end of the fibre are the better
        # evidence -- if they brighten, light reached the chip, whatever the monitor says.
        if not s.emitted and read_pds is not None and s.pds_off is not None:
            import numpy as _np
            lit = float(_np.max(_np.asarray(read_pds(), float)
                                - _np.asarray(s.pds_off, float)))
            if lit > pd_eps:
                s.emitted = True
                s.emitted_via = f"chip detectors (+{1e3 * lit:.0f} mV); monitor saw only " \
                                f"{s.bfm_on - s.bfm_off:+.3f} V"
        s.chip_c = float("nan") if tec is None else tec.temperature()

        def keepalive(sleep_s: float = 0.0):
            with lock:
                laser.dev.write_setting("cw_current", s.sp, verify=False)
            if sleep_s:
                time.sleep(sleep_s)

        def set_power(dbm):
            """Change the live output power mid-session and update the keepalive target so
            it holds the new setpoint, letting one lit session sweep several levels."""
            with lock:
                laser.set(dbm)
                s.sp = float(laser.dev.read_setting("cw_current"))
                s.mA = laser.measured_mA()

        def telemetry():
            """(bfm optical power, diode temp, measured mA, chip temp) -- the instrument
            state to log alongside every measurement. The chip temperature is the one the
            6x6 rig could not record, and the one that decides whether a calibration from
            an earlier session still applies."""
            with lock:
                bfm = float(laser.dev.measure("bfm_optical_power"))
                temp = float(laser.dev.measure("diode_temperature"))
                mA = float(laser.measured_mA())
            chip = float("nan") if tec is None else tec.temperature()
            return bfm, temp, mA, chip

        s.keepalive = keepalive
        s.set_power = set_power
        s.telemetry = telemetry
        s.expired = lambda: stopped.is_set() or time.time() >= s.deadline
        yield s
    finally:
        if wd is not None:
            wd.cancel()
        with lock:
            if not stopped.is_set():
                laser.off()
