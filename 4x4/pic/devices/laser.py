"""AeroDiode PDMv5 laser control -- safe power on/off + smooth current ramp.

Self-contained: the only dependency is pyserial. The serial protocol + register map live
in pdmv5.py; this module is the safe operating policy on top of it.

    python -m laser hw 1      # power ON: verify key+interlocks, TEC off, enable at floor
    python -m laser set 10    # ramp to +10 dBm output (via the optical calibration)
    python -m laser set 5 -t 30  # ramp to +5 dBm, hold 30 s, then auto power-off
    python -m laser hw 0      # ramp down to 0, then disable output + TEC off
    python -m laser status    # full device readout   (run from the repo root)

As a Python object, one call does the happy path (autodetect + open + enable + ramp):

    laser = Laser()           # searches ports, ready to use
    laser.on(5)               # ON and ramped to +5 dBm
    laser.on(5, time=30)      # ON at +5 dBm, hold 30 s, then auto power-off (blocking)
    laser.off(); laser.close()
    # hw_on()/set()/hw_off() remain the primitives on() is built from.

SAFETY (first priority -- enforced here):
  * hw 1 REFUSES to enable unless the driver responds on serial (connected) AND the key
    switch is ON AND both interlocks are closed. If not, it tells you exactly what to fix.
  * The output is NEVER disabled while current is flowing: every off-path ramps to ~0 mA
    and *verifies* the measured current before dropping the enables.
  * set() only ramps an already-ON laser; it will not silently enable one (run hw 1 first).
  * ALWAYS run `hw 0` before turning the key off or unplugging.

The module refuses to turn on into a 0 mA setpoint (pulsed-diode driver), so hw 1 enables
at a small non-zero FLOOR current; everything after that ramps smoothly.

VERBATIM COPY of 6x6/pic/devices/laser.py -- the PDMv5 is the same physical
instrument on both rigs and its wire protocol is frozen. Fix a protocol bug in
both places or neither; do not let them drift.
"""

from __future__ import annotations
import time

from .pdmv5 import PDMv5, PDMv5Error, _fmt_status


class LaserError(RuntimeError):
    pass


class Laser:
    # "setpoint" is the raw drive register, NOT mA/mW: measured I ≈ 2.5*setpoint (cal below)
    FLOOR_SP = 5.0  # device won't enable into a 0 setpoint
    SP_MAX = 95.0  # current clamps at the 250 mA ceiling here (measured 245.6 mA)
    CEIL_MA = 250.0  # overcurrent ceilings, real mA (= diode avg rating)
    RATE_SP_S = 60.0
    DT_S = 0.1
    OFF_EPS_MA = 3.0  # "off" threshold; measured baseline ~1 mA
    SETTLE_S = 0.1
    ENABLE_TRIES = 15  # never disable between attempts (see _latch_master)
    LATCH_POLLS = 8

    # TEC uncalibrated on this unit (TECSlope=0, empty cal table); room is ~25 C, so run
    # open-loop with it OFF. Flip True only once the TEC is calibrated.
    USE_TEC = False

    # optical cal 2026-08-27 (external meter, 9 points sp 8..70): P[mW] = CAL_A*sp + CAL_B.
    # rms residual 0.067 mW, max 0.107 mW. Supersedes the 2026-07-08 fit (0.3437/-1.44),
    # which was 23% low in slope with an almost unchanged threshold -- the signature of a
    # measurement-scale change (meter/connector), not a diode that has aged. The absolute
    # scale is therefore only as good as the meter coupling was on the day; the shape is solid.
    CAL_A = 0.4223
    CAL_B = -1.890
    # A reproducible kink sits at sp 82: 29.80 mW measured against 32.74 extrapolated, the
    # same value approached from above and from below at identical drive current (213.4 mA),
    # so it is the diode's LI curve and not thermal hysteresis. Point-to-point efficiency
    # collapses to 69 uW/mA across 70->82 and recovers to 174 uW/mA across 82->95. One more
    # reason to keep the usable ceiling below it rather than fit through it.
    # Linear only to sp ~70. Beyond it the diode current clamps at the 250 mA ceiling and the
    # LI curve rolls off (sp 95 measured 35.41 mW against 38.23 extrapolated). The ceiling is
    # set at the end of the trustworthy line rather than at the most light the diode can make,
    # so `set <dBm>` never lands somewhere the calibration does not describe.
    PMAX_MW = 27.67
    PMAX_DBM = 14.4

    def __init__(
        self,
        port: str | None = None,
        *,
        sp_max: float = SP_MAX,
        rate_sp_s: float = RATE_SP_S,
        dt_s: float = DT_S,
    ):
        self.dev = PDMv5(port or self.autodetect())

        self.sp_max = float(sp_max)
        self.rate = abs(float(rate_sp_s)) or self.RATE_SP_S
        self.dt = max(1e-3, float(dt_s))

    @classmethod
    def dbm_to_setpoint(cls, dbm: float):
        """Target output dBm -> (raw setpoint, expected mW), clamped to the optical ceiling."""
        mW = min(10.0 ** (float(dbm) / 10.0), cls.PMAX_MW)
        sp = (mW - cls.CAL_B) / cls.CAL_A

        return max(cls.FLOOR_SP, min(sp, cls.SP_MAX)), mW

    @classmethod
    def setpoint_to_dbm(cls, sp: float):
        """Raw setpoint -> (expected dBm, expected mW) from the calibration."""
        import math

        mW = max(cls.CAL_A * float(sp) + cls.CAL_B, 1e-4)

        return 10.0 * math.log10(mW), mW

    @staticmethod
    def autodetect() -> str:
        import glob, re

        ports = sorted(glob.glob("/dev/cu.usbserial-*"))

        # the FTDI laser adapter has an alphanumeric serial (e.g. AU05XLI8); the CH340
        # Arduino shows a purely numeric id (1110/1120) -- never open that one.
        cand = [p for p in ports if not re.fullmatch(r"\d+", p.rsplit("-", 1)[-1])]
        if not cand:
            raise LaserError(
                "no AeroDiode FTDI serial port found -- is the laser plugged in?"
            )

        for p in cand:
            if "AU05XLI8" in p:
                return p
        return cand[0]

    def open(self):
        if getattr(self, "_serial_open", False):  # idempotent: on() can call it freely
            return self
        try:
            self.dev.open()
        except Exception as e:
            raise LaserError(f"laser not connected / cannot open serial: {e}")

        self._serial_open = True
        return self

    def close(self):
        self.dev.close()  # serial only -- does NOT change the laser's state
        self._serial_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def is_on(self) -> bool:
        return 1 in (
            self.dev.read_setting("laser_status"),
            self.dev.read_setting("cw_laser_status"),
        )

    def measured_mA(self) -> float:
        return float(self.dev.measure("diode_cw_current"))

    def preflight(self) -> dict:
        return {
            "key switch ON": self.dev.measure("key") == 1,
            "BNC interlock closed": self.dev.measure("bnc_interlock") == 1,
            "EXT interlock closed": self.dev.measure("ext_interlock") == 1,
        }

    def _require_ready(self):
        checks = self.preflight()
        bad = [k for k, ok in checks.items() if not ok]
        if not bad:
            return

        fixes = []
        if not checks["key switch ON"]:
            fixes.append("turn the KEY switch to ON")
        if not (checks["BNC interlock closed"] and checks["EXT interlock closed"]):
            fixes.append("close the interlock(s)")

        raise LaserError("cannot enable -- " + "; ".join(fixes) + f"   (unmet: {bad})")

    def ramp_to(self, target_sp: float) -> float:
        target = max(0.0, min(float(target_sp), self.sp_max))
        start = float(self.dev.read_setting("cw_current"))
        span = abs(target - start)

        if span < 1e-9:
            self.dev.write_setting("cw_current", target, verify=False)
            return target

        dur = span / self.rate
        print(
            f"  ramp setpoint {start:.1f} -> {target:.1f}  ({dur:.1f}s @ {self.rate:.0f}/s)"
        )

        t0 = time.monotonic()
        while True:
            f = (time.monotonic() - t0) / dur
            if f >= 1.0:
                self.dev.write_setting("cw_current", target, verify=False)
                break

            sp = start + (target - start) * f
            self.dev.write_setting("cw_current", sp, verify=False)
            time.sleep(self.dt)

        return target

    def _latch_master(self, floor: float) -> bool:
        # Enable gate (proven by A/B on a fresh key cycle): the driver won't bring the master
        # up until the host has ISSUED measure() calls -- a status-read handshake. Polling
        # laser_status alone hangs >60 s; measuring alarms+current each poll latches in a few s.
        # Also: floor setpoint first (won't turn on into 0); never disable to retry (restarts
        # the post-reset refractory); never re-write cw mid-poll (resets an in-progress latch).
        for _ in range(self.ENABLE_TRIES):
            self.dev.write_setting("cw_current", floor, verify=False)
            time.sleep(self.SETTLE_S)
            self.dev.write_setting("cw_laser_status", 1, verify=False)
            time.sleep(self.SETTLE_S)
            self.dev.write_setting("laser_status", 1, verify=False)

            for _ in range(self.LATCH_POLLS):
                self.dev.measure("alarms")  # handshake (mandatory -- see above)
                self.dev.measure("diode_cw_current")
                time.sleep(self.SETTLE_S)
                if self.dev.read_setting("laser_status") == 1:
                    return True

        return False

    def hw_on(self):
        print("hw 1: powering on")
        self._require_ready()
        print("  [OK] connected, key ON, interlocks closed")

        # 0 = ACC (constant-current) mode; 0 = internal current source
        self.dev.write_setting("operating_mode", 0, verify=False)
        self.dev.write_setting("cw_current_source", 0, verify=False)

        if not self.is_on():
            floor = max(1.0, min(self.FLOOR_SP, self.sp_max))
            if not self._latch_master(floor):
                self.dev.write_setting("cw_current", 0.0, verify=False)
                self.dev.write_setting("laser_status", 0, verify=False)
                self.dev.write_setting("cw_laser_status", 0, verify=False)
                raise LaserError(
                    "laser did not enable (master status stayed 0 after "
                    f"{self.ENABLE_TRIES} attempts) -- output left off"
                )

        for reg in (
            "factory_max_current",
            "max_current",
            "max_avg_current",
            "cw_max_current",
        ):
            self.dev.write_setting(reg, self.CEIL_MA, verify=False)
        print(f"  ceilings clamped to {self.CEIL_MA:.0f} mA")

        self.dev.write_setting("tec_status", 1 if self.USE_TEC else 0, verify=False)
        print(
            f"  TEC {'on' if self.USE_TEC else 'OFF (open-loop; room temp-controlled)'}"
        )

        _, emW = self.setpoint_to_dbm(self.dev.read_setting("cw_current"))
        print(
            f"  LASER ON at floor setpoint {self.dev.read_setting('cw_current'):.0f} "
            f"(~{emW:.1f} mW). Use `set <dBm>` to ramp."
        )

    def hw_off(self):
        print("hw 0: powering off")

        if self.is_on():
            self.ramp_to(0.0)
            self.dev.write_setting("cw_current", 0.0, verify=False)

            m = self.measured_mA()
            if m > self.OFF_EPS_MA:
                raise LaserError(
                    f"REFUSING to disable: current still {m:.1f} mA after ramp -- "
                    "output left enabled for safety, investigate before retrying"
                )

            self.dev.write_setting("laser_status", 0, verify=False)
            self.dev.write_setting("cw_laser_status", 0, verify=False)

        self.dev.write_setting("tec_status", 0, verify=False)
        print(
            "  LASER OFF -- current 0, output disabled, TEC off. Safe to turn the key off / unplug."
        )

    def set(self, dbm: float, *, raw: bool = False):
        if not self.is_on():
            raise LaserError("laser is OFF -- run `hw 1` first (the key must be ON)")

        if raw:
            sp = max(0.0, min(float(dbm), self.sp_max))
            self.ramp_to(sp)

            edbm, emW = self.setpoint_to_dbm(sp)
            print(
                f"  raw setpoint {sp:.1f}  (~{emW:.1f} mW / {edbm:+.1f} dBm expected)"
            )
            return

        want = float(dbm)
        sp, mW = self.dbm_to_setpoint(want)

        if want > self.PMAX_DBM:
            print(
                f"  clamping {want:+.1f} -> +{self.PMAX_DBM:.1f} dBm "
                f"(optical ceiling {self.PMAX_MW:.0f} mW, current-limited)"
            )

        self.ramp_to(sp)
        print(
            f"  {want:+.1f} dBm -> setpoint {sp:.1f}  "
            f"(~{mW:.1f} mW; measured I {self.measured_mA():.0f} mA)"
        )

    def on(self, power_dbm: float | None = None, *, time: float | None = None,
           raw: bool = False):
        """One-call power-up: open the serial if needed, safely enable the laser (at the
        non-zero floor), and -- if a level is given -- ramp to ``power_dbm`` (dBm, or a
        raw setpoint with ``raw=True``). With ``time=SEC`` it holds that level for SEC
        seconds then powers off (blocking; Ctrl-C powers off early). Returns self, so
        ``Laser().on(5)`` is the whole happy path.

        For a bounded on-time while a measurement loop runs *concurrently*, use
        ``template.laser_session`` instead -- its watchdog hard-offs from a background
        thread rather than blocking here."""
        self.open()  # idempotent
        self.hw_on()
        if power_dbm is not None:
            self.set(power_dbm, raw=raw)
        if time is not None:
            self.hold_then_off(time)  # this method (not on()) is what touches time.sleep
        return self

    def off(self):
        """Safe power-down (ramp to 0, verify current, disable) -- alias of hw_off()."""
        self.hw_off()
        return self

    def hold_then_off(self, hold_s: float):
        """Stay at the current setpoint for hold_s seconds, then power off. A Ctrl-C
        during the hold powers off early -- the laser is never left on by this path."""
        hold_s = float(hold_s)
        print(f"  holding for {hold_s:.0f}s, then auto power-off (Ctrl-C to stop early)")
        try:
            time.sleep(hold_s)
        except KeyboardInterrupt:
            print("\n  interrupted -- powering off early")
        self.hw_off()

    def status(self):
        print(_fmt_status(self.dev.status()))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="laser.py", description="AeroDiode PDMv5 laser control"
    )
    ap.add_argument(
        "--port", default=None, help="serial device (default: autodetect the FTDI)"
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_hw = sub.add_parser("hw", help="power on (1) / off (0)")
    p_hw.add_argument("state", type=int, choices=[0, 1])

    p_set = sub.add_parser("set", help="ramp to an output power in dBm")
    p_set.add_argument("dBm", type=float, help="target output power in dBm (e.g. 10)")
    p_set.add_argument(
        "--raw", action="store_true", help="treat the value as a raw setpoint, not dBm"
    )
    p_set.add_argument(
        "-t",
        "--time",
        type=float,
        default=None,
        metavar="SEC",
        help="hold for SEC seconds then auto power-off (default: stay on indefinitely)",
    )

    sub.add_parser("status", help="full device readout")

    a = ap.parse_args(argv)

    try:
        with Laser(port=a.port) as laser:
            if a.cmd == "hw":
                laser.hw_on() if a.state == 1 else laser.hw_off()
            elif a.cmd == "set":
                laser.set(a.dBm, raw=a.raw)
                if a.time is not None:
                    laser.hold_then_off(a.time)
            elif a.cmd == "status":
                laser.status()
    except (LaserError, PDMv5Error) as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
