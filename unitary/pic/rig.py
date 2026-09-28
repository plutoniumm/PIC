"""Rig: one object that runs the whole 4x4 hardware-in-the-loop.

Four swappable components -- laser, board, TEC, model -- each independently hardware or
mock, so every path in this package can be exercised with nothing plugged in.

    rig = Rig(laser="mock", board="mock", tec="mock", model="mock").open()
    with rig.session(duration_s=30, power_dbm=13):
        y = rig.measure(v)          # 18 DAC volts in, 8 raw photodiode volts back
        f = rig.realize(U)          # program a target unitary and measure what came out
    rig.close()
"""

from __future__ import annotations

from contextlib import contextmanager

import time

import numpy as np

from .config import (
    TEC_SETPOINT_C,
    VOLTAGE_MAX,
    VOLTAGE_MAX_CH,
    out_mask,
    ADC_REF_V,
    ADC_SAT_V,
    detector_mode,
)
from .devices.laser import Laser
from .log import ev
from .devices.mock import MockLaser
from .devices.switch import make_switch
from theory.clements import NMODE
from theory.layout import N_HEATERS
from .devices.tec import make_tec
from .interface import PIC, MockPIC, spd_board
from .session import _resolve_pic_port, check_active, laser_session

# The recorded `chip_c` is one instantaneous reading taken at session open, not the session
# mean, so this is wider than the 0.05 C band the loop holds -- and still two orders under
# the 2 C offset it exists to catch.
CALIB_TEMP_TOL_C = 0.25


class CalibrationTemperatureError(RuntimeError):
    pass


class SaturatedRead(RuntimeError):
    """A photodiode read at or above the ADC reference. Clipped, not noisy."""


class FirmwareMismatch(RuntimeError):
    pass


# The firmware prints its table to two decimals, so anything under half a least digit is
# formatting rather than disagreement.
FIRMWARE_VMAX_TOL_V = 0.011


def assert_firmware_vmax(board, host=VOLTAGE_MAX_CH, tol_v: float = FIRMWARE_VMAX_TOL_V):
    """Check the board's own clamp table against `config.VOLTAGE_MAX_CH`, channel by channel.

    Two copies of the same sixteen numbers exist -- here and as `const float VMAX[]` in
    `Arduino/pic4x4/pic4x4.ino` -- because the DAC has to be protected by the thing driving
    it, not by the host asking nicely. Duplication is fine; unchecked duplication is not.
    When the two diverge the host commands one voltage, the DAC outputs another, and every
    number downstream describes a heater state nobody chose, with no error raised anywhere.
    It had already happened: three channels read 0.00 in the firmware while the host
    believed they were open. The 6x6 lost months to the same shape of bug.

    The reply to `V` is `VMAX <16 floats>`. A board that answers with an ADC frame instead
    is running firmware from before the query existed -- and that firmware has just parsed
    `V` as a comma-separated DAC list and written garbage to every channel, which is why
    this zeroes the DACs before refusing rather than after.

    Mock and simulated boards hold no second copy, so there is nothing to disagree with and
    the check is skipped."""
    ser = getattr(board, "ser", None)
    if ser is None or isinstance(ser, str) or not hasattr(ser, "write"):
        return None
    host = np.asarray(host, float)
    ser.reset_input_buffer()
    ser.write(b"V\n")
    deadline = time.time() + getattr(board.cfg, "timeout_s", 3.0)
    seen = []
    while time.time() < deadline:
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if not raw:
            continue
        seen.append(raw)
        if raw.startswith("VMAX"):
            try:
                fw = np.array([float(x) for x in raw.split()[1:]], float)
            except ValueError:
                break
            if fw.size != host.size:
                raise FirmwareMismatch(
                    f"firmware reports {fw.size} VMAX entries, the host has {host.size}. "
                    f"NUM_DAC disagrees between pic/config.py and "
                    f"Arduino/pic4x4/pic4x4.ino; reflash the firmware."
                )
            bad = np.flatnonzero(np.abs(fw - host) > tol_v)
            if bad.size:
                rows = "\n".join(
                    f"    ch{int(i):<3} host {host[i]:.2f} V   " f"firmware {fw[i]:.2f} V"
                    for i in bad
                )
                raise FirmwareMismatch(
                    f"the firmware's per-channel clamp disagrees with "
                    f"pic.config.VOLTAGE_MAX_CH on {bad.size} channel(s):\n{rows}\n"
                    f"  The two tables are meant to be identical -- the host commands a "
                    f"voltage and the DAC outputs whichever is smaller, silently. Reflash "
                    f"Arduino/pic4x4/pic4x4.ino with the table in pic/config.py."
                )
            return fw
        if "," in raw and len(raw.split(",")) == board.cfg.num_adc_raw:
            break  # an ADC frame: old firmware, and it wrote
    # Whatever it did with `V`, the heaters are not where we think. Best effort, and its
    # own failure must not replace the message that says why we are here.
    try:
        board.set_zero()
        zeroed = "The DACs have been zeroed."
    except Exception as e:
        zeroed = f"The DACs could NOT be zeroed ({e}) -- power the board down."
    saw = f"answered {seen[-1]!r}" if seen else "did not answer at all"
    raise FirmwareMismatch(
        f"the board {saw} to the `V` clamp query, so it is running firmware from before "
        f"that query existed -- and that firmware has just read `V` as a DAC line and "
        f"written every channel from it. {zeroed} Reflash "
        f"Arduino/pic4x4/pic4x4.ino before running anything."
    )


def assert_calib_temperature(
    calib, setpoint_c: float = TEC_SETPOINT_C, tol_c: float = CALIB_TEMP_TOL_C
):
    """Refuse to run a calibration that was fitted at a different chip temperature.

    Every heater's phi0 is an optical path length, and path length is thermo-optic: a
    calibration taken at 25 C driven at 27 C puts the same uncorrected phase offset on the
    whole mesh, which no downstream correction models and no residual reveals -- programmed
    targets simply come out rotated. It happened: `TEC_SETPOINT_C` was edited to 27 without
    retaking `pic_data/calib.json`, and every hardware number after that carried it.

    The fix when this fires is to recalibrate at the setpoint, or to put the setpoint back
    to what the calibration was taken at. It is never to widen this gate. A cooler that
    cannot hold the setpoint is a hardware fault to report, not a constant to edit."""
    meta = getattr(calib, "meta", None) or {}
    chip_c = meta.get("chip_c")
    if not meta:
        return None  # nominal: no measurement, so nothing to disagree with
    if chip_c is None or not np.isfinite(float(chip_c)):
        ev(
            "session",
            "note",
            "the calibration records no chip temperature, so it cannot be checked against the "
            f"{setpoint_c:.2f} C setpoint. Re-run `python -m pic char --write` to anchor it.",
            "warn",
            set_c=setpoint_c,
        )
        return None
    chip_c = float(chip_c)
    if abs(chip_c - setpoint_c) > tol_c:
        # the operator's call, not a gate: say it once and carry on
        ev(
            "session",
            "note",
            f"calibration taken at {chip_c:.2f} C, TEC set to {setpoint_c:.2f} C "
            f"({abs(chip_c - setpoint_c):.2f} C apart); phi0 carries that as a uniform offset",
            "warn",
            calib_c=chip_c,
            set_c=setpoint_c,
        )
    return chip_c


def make_laser(spec, port=None):
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockLaser()
    if spec == "hw":
        return Laser(port=port)
    raise ValueError(f"laser={spec!r}; use 'hw', 'mock', or a Laser instance")


def make_board(spec, port=None, laser_port=None, switch=None, detectors="pd"):
    """``'hw'|'mock'|'sim'`` -> a board; a PIC instance passes through unchanged.

    'mock' is the idealised chip this package was designed against -- all 18 heaters live,
    Vpi near 1.5 V, a clean readout. 'sim' is the bench that exists: six wired channels, a
    3 V clamp, the measured loss table and the measured noise (`pic.sim`). Use 'mock' to ask
    whether an algorithm is right and 'sim' to ask whether it will survive the bench."""
    if not isinstance(spec, str):
        return spec
    if spec == "sim" and detectors == "spd":
        raise ValueError("the sim board models photodiode volts, not photon counts; use 'mock'")
    if spec == "mock":
        board = MockPIC(switch=switch)
    elif spec == "sim":
        from .sim import BenchPIC

        return BenchPIC(switch=switch)
    elif spec == "hw":
        board = PIC(port=_resolve_pic_port(port, laser_port))
    else:
        raise ValueError(f"board={spec!r}; use 'hw', 'mock', 'sim', or a PIC instance")
    # the one place the readout is chosen: everything downstream reads through this board
    return spd_board(board, mock=spec == "mock") if detectors == "spd" else board


class Rig:
    def __init__(
        self,
        laser="hw",
        board="hw",
        tec="mock",
        switch="mock",
        model=None,
        *,
        keep_laser: bool = False,
        calib=None,
        laser_port=None,
        pic_port=None,
        dynamic=False,
        detectors=None,
    ):
        self.laser = make_laser(laser, laser_port)
        self.tec = make_tec(tec)
        self.switch = make_switch(switch)
        self.keep_laser = bool(keep_laser)
        self._board_spec = board
        self._pic_port = pic_port
        # None is the bench's saved choice (Settings tab); a selftest pins it
        self.detectors = detector_mode() if detectors is None else detectors
        self.board = None  # opened lazily so the board port can exclude the laser's
        if isinstance(model, str):  # lazy so `import pic` stays torch-free
            from .model import make_model

            model = make_model(model)
        self.model = model
        self._calib = calib
        self._mask = out_mask()
        self._twin = None
        from .drift import DriftTracker

        self.drift = DriftTracker(self, enabled=dynamic)

    @property
    def calib(self):
        """The heater law in force. Falls back to the nominal one, which is enough to
        drive the chip but not to program a target accurately."""
        if self._calib is None:
            from dataclasses import replace

            from theory.calib import Calibration

            c = Calibration.load_or_nominal()
            if self.detectors == "spd":
                # an SPD read is already 0 dark and 1 at its max; the file's readout law is
                # the photodiodes', and is kept for them (`__main__._merge_calib`)
                c = replace(c, pd_gain=np.ones_like(c.pd_gain), pd_offset=np.zeros_like(c.pd_offset))
            self._calib = c
        return self._calib

    @property
    def twin(self):
        """The differentiable mesh model, used by the drift fit. Lazy: importing it pulls
        in torch, and `import pic` stays torch-free."""
        if self._twin is None:
            from theory.twin import Twin

            self._twin = Twin()
        return self._twin

    def open(self):
        self.laser.open()
        self._laser_was_on = bool(getattr(self.laser, "is_on", lambda: False)())
        if self._laser_was_on and not self.keep_laser:
            print(
                "  note: laser was already on; it will be turned off on exit "
                "(pass --keep-laser to leave it lit)"
            )
        self.tec.open()
        self.switch.open()
        laser_port = getattr(getattr(self.laser, "dev", None), "port", None)
        self.board = make_board(
            self._board_spec, self._pic_port, laser_port, self.switch, self.detectors
        )
        self.board.open()
        # Before anything is driven: the clamp table the DAC will actually enforce has to be
        # the one the host thinks it is commanding against.
        assert_firmware_vmax(self.board)
        # Asked once, here, while the DACs are still at their reset zeros: a firmware from
        # before the `C` query parses it as a DAC line and writes every channel from it, so
        # this is the one moment where finding out costs nothing.
        caps = self.board.capabilities()
        if not caps.get("sweep"):
            print(
                "  note: this firmware has no batched sweep, so a four-port sweep costs "
                "4 x repeats round trips instead of one. Reflash "
                "Arduino/pic4x4/pic4x4.ino to get it."
            )
        if hasattr(self.switch, "attach"):  # board-routed switch: it needs the open board
            self.switch.attach(self.board)
        return self

    def select_input(self, port: int) -> int:
        """Route the laser to one of the four input ports. See `pic.devices.switch` for why
        a characterization has to visit more than one."""
        return self.switch.select(port)

    def session(self, *, duration_s, power_dbm, calibrating=False, **kw):
        """Guarded laser session: background watchdog, emission verify, TEC gate.

        The calibration/setpoint temperature check sits here and not inside `laser_session`
        because it is a property of the rig rather than of the beam, but it runs at the same
        moment as the TEC settle gate so that every hardware entry point -- all of them open
        a session -- is covered by both."""
        # a calibration establishes the temperature the others are checked against
        if not calibrating:
            assert_calib_temperature(self.calib, self.tec.target)
        return self._session(duration_s, power_dbm, kw)

    @contextmanager
    def _session(self, duration_s, power_dbm, kw):
        # Heaters to 0 V on the way in and on the way out. A multi-round job used to leave the
        # last operating point on through its fit and through the next round's TEC wait, so
        # the die kept heating and the settle gate never passed.
        self._zero()
        try:
            with laser_session(
                self.laser,
                duration_s=duration_s,
                power_dbm=power_dbm,
                tec=self.tec,
                read_pds=lambda: self.outputs(np.zeros(N_HEATERS)),
                **kw,
            ) as s:
                yield s
        finally:
            self._zero()

    def hold_band(self, s, timeout_s: float = 600.0, resume: float = 0.5) -> float:
        """Before a heater state: if the die has left the TEC band, rest the heaters at 0 V and
        wait -- keeping the laser alive -- until it is back. Every state is then measured in
        band; a job that heats the die just takes longer. Returns the die temperature."""
        t = self.tec.temperature()
        if abs(t - self.tec.target) <= self.tec.tolerance:
            return t
        ev(
            "session",
            "paused",
            f"paused: die at {t:.2f} C, outside {self.tec.target:.2f} ± {self.tec.tolerance:.2f}; "
            "heaters at 0 V until it is back",
            "warn",
            die_c=t,
            set_c=self.tec.target,
            tol=self.tec.tolerance,
        )
        self._zero()
        t0 = time.time()
        # Hysteresis: pause at the band's edge, resume only well inside it. Resuming at the
        # edge let the next heater state push it straight back out -- a pause every few states.
        while abs(t - self.tec.target) > resume * self.tec.tolerance:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"die did not come back into band in {timeout_s:.0f} s")
            s.keepalive(sleep_s=2.0)
            t = self.tec.temperature()
        ev(
            "session",
            "resumed",
            f"resumed after {time.time() - t0:.0f} s at {t:.2f} C",
            "ok",
            die_c=t,
            secs=time.time() - t0,
        )
        return t

    def _zero(self):
        if self.board is not None:
            try:
                self.board.set_zero()
            except Exception:
                pass  # a board that cannot be reached is not driving anything either

    def measure(self, v) -> np.ndarray:
        """Set the 18 DAC volts, return the raw photodiode volts."""
        check_active()  # a read after the watchdog trip is dark current, not a measurement
        return self._checked(self.board.measure_raw(v))

    def _checked(self, raw) -> np.ndarray:
        # The ADC reference moved from AVCC to the 1.1 V bandgap to stop wasting 90% of the
        # converter, which leaves 2.35x headroom over the brightest read this rig has ever
        # taken (0.468 V) rather than 10x. A clipped read is not noisy, it is WRONG and it
        # looks like a perfectly good number, so it has to be caught here -- the alternative
        # was a firmware marker, which would have changed a wire protocol the 6x6 shares.
        if self.detectors == "spd":  # photon-count fractions: there is no ADC to clip
            return raw
        y = np.asarray(raw, float)
        hot = y >= ADC_SAT_V  # (4,) from a read, (4, 4) from a sweep
        if hot.any():
            pds = np.flatnonzero(hot if y.ndim == 1 else hot.any(1))
            raise SaturatedRead(
                f"photodiode(s) {pds.tolist()} read "
                f"{y[hot].round(4).tolist()} V against a "
                f"{ADC_REF_V:.2f} V ADC reference -- the converter is clipping and the value "
                f"is not a measurement. Lower --dbm, or move the firmware to "
                f"INTERNAL2V56 and set ADC_REF_V to match."
            )
        return raw

    @property
    def batched(self) -> bool:
        """Whether a four-port sweep costs one round trip or 4*repeats of them. The cost
        model and the laser watchdog are sized from this, so it asks the board rather than
        assuming, and it is the same gate `sweep_ports` applies."""
        return (
            self.board is not None
            and getattr(self.board, "batched", False)
            and getattr(self.switch, "port", None) in ("board", "mock")
        )

    def sweep_ports(self, cycles: int = 1, reads: int = 1):
        """A whole four-port sweep in ONE round trip -> raw T[pd, port], or None.

        None means "this rig cannot", and the caller falls back to the host loop: either the
        board is running firmware from before the `S` command, or the mirror is not on the
        board's Serial1 and the firmware cannot move it. Both are ordinary configurations,
        not faults, which is why this returns rather than raises.

        Everything `measure` guards is guarded here too: the watchdog is checked on both
        sides of a sweep that lasts seconds, and every entry goes through the same
        saturation gate. A batched read that skipped either would be the same measurement
        with the safety taken off."""
        board = self.board
        # The firmware drives the mirror off its own Serial1, so a switch on a host serial
        # port -- or no switch at all -- would leave every column read at one position and
        # the four look like a sweep. `batched` is that gate plus the firmware's own answer.
        if not self.batched:
            return None
        check_active()
        raw = board.sweep_raw(int(cycles), int(reads))
        check_active()  # a trip mid-sweep would leave the later columns dark
        self.switch._sel = NMODE - 1  # the firmware leaves the mirror on the last port
        return self._checked(np.asarray(raw, float))[self._mask]

    def outputs(self, v) -> np.ndarray:
        """Same, reduced to the four mesh outputs."""
        return self.measure(v)[self._mask]

    def program(self, U, vmax: float = VOLTAGE_MAX):
        """Target unitary -> (18 DAC volts, reachable mask). Closed form, no search.

        With `dynamic=True` the target phases are pre-distorted by the drift the tracker
        has inferred, so what lands on the chip is the target as the chip is *now* rather
        than as it was at calibration."""
        from theory.program import phases_for

        return self.calib.volts(self.drift.correct(phases_for(U)), vmax=vmax)

    def realize(self, U, vmax: float = VOLTAGE_MAX):
        """Program `U` onto the chip and read what the outputs did. Returns
        (volts, reachable mask, output photodiode volts)."""
        v, ok = self.program(U, vmax=vmax)
        return v, ok, self.outputs(v)

    def predict(self, v) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Rig has no model -- pass model= (see pic.model)")
        return self.model.predict(v)

    def status(self) -> dict:
        return {
            "laser_on": bool(getattr(self.laser, "is_on", lambda: False)()),
            "tec": self.tec.status(),
            "switch": self.switch.status(),
            "drift": self.drift.status(),
            "calib": self.calib.meta or "nominal (uncharacterized)",
        }

    def close(self):
        try:
            if self.board is not None:
                self.board.set_zero()
                self.board.close()
        finally:
            try:
                # Off by default, always. Leaving a lit diode behind with no watchdog on it
                # is the one failure mode that damages hardware unattended, and convenience
                # is not worth it -- `keep_laser` makes the exception explicit and opt-in.
                if not (self.keep_laser and getattr(self, "_laser_was_on", False)):
                    self.laser.off()
            finally:
                self.laser.close()
                self.tec.close()
                self.switch.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
