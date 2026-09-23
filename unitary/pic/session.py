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

Also new, and for the same reason: **a watchdog trip is fatal to the run, not just to the
laser.** The timer fires on a background thread, so a measurement loop that never asks
keeps reading -- and what it reads is dark current, which looks like a small number rather
than like a failure. On 2026-08-27 a matvec job sized at the flat 120 s default outran its
watchdog, was hard-offed at 120 s, and then measured the dark chip for the rest of the
night and printed a plausible answer. So `force_off` now marks the session tripped, every
read primitive calls `check_active()` first, and the next measurement raises
`WatchdogTripped`. A safety timeout that degrades into silent data corruption is worse
than no timeout. Loops that want to keep a partial result check `expired()` *before* the
read, which is how they can prove what they kept is pre-trip.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
import time
from contextlib import contextmanager

from .devices.laser import Laser
from .devices.mock import MockLaser
from .devices.tec import MockTEC, NoTEC, make_tec
from .config import BAUD_RATE, READY_BANNER, RESET_WAIT_S
from .interface import PIC, MockPIC, PICError, find_port


def _identify(port: str) -> str:
    """Boot banner -> which instrument is on this port: 'board', 'tec' or 'unknown'.

    Opening a CDC port asserts DTR and resets the Arduino, so this costs the TEC controller
    a PID restart and about a minute of reconvergence when it guesses wrong. That is the
    cheaper mistake. The expensive one is `cands[0]`, which this replaces: the TEC enumerates
    ahead of the board alphabetically and is the same Mega 2560, so the old first-candidate
    rule opened the temperature controller and addressed it as the mesh driver."""
    import serial

    banner = ""
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1.0) as ser:
            # the bootloader holds the line for over a second before the sketch runs, so a
            # single RESET_WAIT_S read races it and reports a live board as unknown
            deadline = time.time() + 3 * RESET_WAIT_S
            while time.time() < deadline and READY_BANNER not in banner:
                time.sleep(0.4)
                banner += ser.read(ser.in_waiting or 0).decode("ascii", "replace")
                if "LT8722" in banner or "target temperature" in banner:
                    break
    except Exception as e:  # a busy port is a diagnosis, not a mystery
        return f"unreadable ({type(e).__name__}: {e})"
    if READY_BANNER in banner:
        return "board"
    if "LT8722" in banner or "target temperature" in banner:
        return "tec"
    return "unknown"


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
        raise PICError(
            "no board serial port found (after excluding the laser). Pass "
            "--pic-port or set $PIC4_PORT; is another process holding it?"
        )
    if len(cands) == 1:
        return cands[0]
    # The remembered answer is TRUSTED rather than re-checked, because a second look cannot
    # confirm it. The board prints its banner from `setup()`, and macOS only pulses DTR --
    # and so only resets the Mega -- on the first open after the device enumerates. Probe it
    # twice in one run and the second open returns a stale ADC frame with no banner, which
    # reads exactly like "not the board". Probing also resets the TEC controller, whose
    # sketch runs a polarity test into the TEC element on every reset. So: discover once,
    # remember, and do not touch either instrument again. Delete pic_data/.board_port after
    # re-cabling.
    cache = Path(__file__).resolve().parent.parent / "pic_data" / ".board_port"
    try:
        remembered = cache.read_text().strip()
    except OSError:
        remembered = ""
    if remembered in cands:
        return remembered  # trusted, not re-probed: see above

    # Two passes, with a settle between. Both Arduinos sit behind one USB hub, and resetting
    # the TEC -- which opening its port does -- leaves the board's node briefly deaf: it
    # opens and reads clean but the banner never arrives. A second look after the bus
    # settles finds it. Probing the board first would dodge this, but which port that is
    # is exactly what we are trying to work out.
    seen = {}
    for attempt in range(2):
        if attempt:
            time.sleep(1.5)
        for c in cands:
            seen[c] = _identify(c)
            if seen[c] == "board":
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(c)
                except OSError:
                    pass  # a read-only tree costs a probe, not a run
                return c
    raise PICError(
        f"no board found among {cands} (identified {seen}). The TEC controller "
        f"is the same Mega 2560 and must not be driven as the board; set "
        f"$PIC4_PORT to the one whose banner is {READY_BANNER!r}."
    )


def open_devices(mock=False, laser_port=None, pic_port=None, tec="mock"):
    """Return (laser, pic, tec), all open. Real hardware unless mock=True."""
    if mock:
        return MockLaser().open(), MockPIC().open(), MockTEC().open()
    laser = Laser(port=laser_port).open()
    pic = PIC(port=_resolve_pic_port(pic_port, laser.dev.port)).open()
    return laser, pic, make_tec(tec).open()


class WatchdogTripped(RuntimeError):
    """A read was attempted after the session watchdog forced the laser off.

    Never caught to continue: the beam is off, so anything measured from here on is dark
    current. Catch it only to salvage data taken strictly before the trip."""


class _Session:
    """Handle yielded by :func:`laser_session`; see there for the fields and methods."""

    tripped = False

    def check(self):
        """Raise if this session's watchdog has already fired."""
        if self.tripped:
            raise WatchdogTripped(
                f"laser session watchdog fired at {self.duration_s:.0f} s and forced the "
                f"laser OFF -- every read after that is dark current, so this run is "
                f"aborted rather than completed with dark data. Size --duration from the "
                f"job (see pic.acquisition.estimate_seconds) or pass a larger one."
            )


# Sessions are process-wide singular on this rig -- one laser, one lock -- so the read
# primitives can consult the live one without every call site threading the handle down to
# it. `settled_read(pic, ...)` only ever gets the board, and it is the call that must not
# return dark volts.
_ACTIVE: list[_Session] = []
_REGISTRY_LOCK = threading.Lock()


def check_active():
    """Raise `WatchdogTripped` if any open session has been hard-offed. Cheap enough to
    call before every read, which is where it has to be: the timer runs on its own thread
    and a loop that does not ask never finds out."""
    with _REGISTRY_LOCK:
        live = list(_ACTIVE)
    for s in live:
        s.check()


@contextmanager
def laser_session(
    laser,
    *,
    duration_s,
    power_dbm,
    tec=None,
    bfm_off=None,
    emit_eps=0.05,
    read_pds=None,
    pd_eps=0.02,
    require_stable=True,
):
    """Bounded, watchdog-guarded laser-on session. The only copy of the laser safety logic.

    On enter: check the chip is at temperature, baseline the monitor photodiode with the
    laser still off, turn on to `power_dbm`, and arm a background hard-off at `duration_s`.
    Yields a handle with `.sp .mA .bfm_off .bfm_on .emitted .deadline .chip_c .tripped`,
    plus `.keepalive(sleep_s=0)` (re-pokes the setpoint so the driver's idle self-disable
    cannot drop the beam), `.set_power(dbm)`, `.telemetry()`, `.expired()` and `.check()`.
    On every exit path -- normal, exception, or watchdog -- the timer is cancelled and the
    laser forced off.

    `expired()` is the polite question, asked before a read so a loop can stop and keep
    what it has. `check()` is the guarantee, asked *by* the read: once the watchdog has
    fired there is no light, so a measurement raises rather than returning dark volts.
    """
    if tec is not None and require_stable and not isinstance(tec, NoTEC):
        print("waiting for the TEC to settle before lighting the laser", flush=True)
        tec.wait_stable()

    lock = threading.Lock()
    stopped = threading.Event()
    wd = None

    def _bfm():
        return float(laser.dev.measure("bfm_optical_power"))

    def force_off():  # a watchdog must never die silently
        # Marked before the beam is touched: `laser.off()` can block on the serial lock the
        # measurement loop is holding, and the reads it takes meanwhile are already dark.
        s.tripped = True
        with lock:
            try:
                print(
                    f"\n[watchdog] {duration_s:.0f}s reached -> forcing laser OFF; "
                    f"the run is aborted, not continued"
                )
                laser.off()
            except Exception as e:
                print(f"[watchdog] hw_off error: {e}")
        stopped.set()

    s = _Session()
    s.duration_s = float(duration_s)
    s.stopped = stopped
    s.tec = tec
    s.bfm_off = _bfm() if bfm_off is None else bfm_off  # laser still off here
    s.pds_off = None if read_pds is None else read_pds()  # dark baseline on the chip
    try:
        with lock:
            laser.on(power_dbm)  # open (idempotent) + enable + ramp, one call
            s.sp = float(laser.dev.read_setting("cw_current"))
            s.mA = laser.measured_mA()
        wd = threading.Timer(duration_s, force_off)
        wd.daemon = True
        wd.start()
        with _REGISTRY_LOCK:
            _ACTIVE.append(s)
        s.deadline = time.time() + duration_s
        s.bfm_on = _bfm()
        s.emitted = (s.bfm_on - s.bfm_off) > emit_eps
        # The beam monitor is the wrong sole witness on this unit: it reads near 3.5 V with
        # the laser off, so a real turn-on moves it by less than the threshold and a lit
        # chip reports as dark. The detectors on the far end of the fibre are the better
        # evidence -- if they brighten, light reached the chip, whatever the monitor says.
        if not s.emitted and read_pds is not None and s.pds_off is not None:
            import numpy as _np

            lit = float(_np.max(_np.asarray(read_pds(), float) - _np.asarray(s.pds_off, float)))
            if lit > pd_eps:
                s.emitted = True
                s.emitted_via = (
                    f"chip detectors (+{1e3 * lit:.0f} mV); monitor saw only "
                    f"{s.bfm_on - s.bfm_off:+.3f} V"
                )
        s.chip_c = float("nan") if tec is None else tec.temperature()

        # Rate-limited, because callers poke it per sample. `characterize` calls it once per
        # sweep level -- 252 times per channel -- and each call is a serial write to the
        # PDMv5, whose FTDI drops and re-enumerates under load. With the transport retrying
        # across a dropout that turned a 1.3 min channel into 20, which is how a 16 min sweep
        # became 4.6 hours. The job here is only to outrun the driver's idle self-disable, so
        # a few seconds is as good as a few milliseconds; the safety watchdog is a separate
        # thread and does not depend on this at all.
        KEEPALIVE_S = 5.0
        last = [0.0]

        def keepalive(sleep_s: float = 0.0, force: bool = False):
            now = time.time()
            if force or now - last[0] >= KEEPALIVE_S:
                last[0] = now
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
        # Deregistered first: the exit path itself zeroes the board and reads nothing, and
        # a session left in the registry would make every later measurement in the process
        # raise on a watchdog that is no longer guarding anything.
        with _REGISTRY_LOCK:
            if s in _ACTIVE:
                _ACTIVE.remove(s)
        if wd is not None:
            wd.cancel()
        with lock:
            if not stopped.is_set():
                laser.off()
