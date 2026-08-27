"""Staged end-to-end check of the 4x4 rig: not whether the chain works, but where it broke.

Every link is checked in the order light travels through it, and the first failure stops the
run. That ordering is the whole point: a photodiode reading nothing means something
different depending on whether the switch moved, and a stage run downstream of a broken link
produces plausible numbers about a chain that was already dead.

    ./do e2e --mock                       # no hardware; the bench model
    ./do e2e --mock --fault dead-pd       # ...driven into one of its refusal paths
    ./do e2e --explain                    # what each stage checks, without running anything
    ./do e2e --pic-port /dev/cu.usbmodem11101 \
             --laser-port /dev/cu.usbserial-AU05XLI8 \
             --tec-port /dev/cu.usbmodem11201

Four serial devices share two globs and the board is never guessed at. The FTDI adapter is
identifiable by name; the board and the TEC controller are both bare `usbmodem` numbers that
encode a USB location and nothing about the device, and opening the TEC to find out resets
its Arduino and throws away a minute of settling. So they are taken from the arguments or
from `$PIC4_PORT`, and if that leaves it ambiguous this says so instead of probing.

All laser work runs inside `pic.session.laser_session`, which arms a background hard-off
before the first lit read and cancels it on every exit path, and which verifies emission
optically rather than believing `laser_status`.

The hardware-free path is `pic.sim`, the bench as delivered -- the measured loss table, the
per-channel current-limited ceilings, the channels with no heater behind them -- plus the
per-detector dark noise measured on 2026-08-27, where PD0 scatters some 70x wider than its
neighbours and carries real signal at SNR 1-2 while they sit at 20-50. A bring-up check that
only passes against an idealised instrument proves nothing, so the mock is the pessimistic
model and `--fault` walks it through each refusal path in turn.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pic.acquisition import estimate_seconds
from pic.characterize import MIN_SNR
from pic.config import (
    ADC_AVG_N, ADC_BITS, ADC_REF_V, DEFAULT_SETTLE_S, NUM_ADC_RAW, NUM_DAC, OUT_PDS,
    READY_BANNER, SWITCH_SETTLE_S, TEC_SETPOINT_C, TEC_SETTLE_S, TEC_TOLERANCE_C,
    VOLTAGE_MAX_CH, WIRED_DACS,
)
from pic.devices.laser import Laser, LaserError
from pic.devices.mock import MockLaser
from pic.devices.pdmv5 import PDMv5Error
from pic.devices.switch import BoardSwitch, SwitchError
from pic.devices.tec import MockTEC, NoTEC, TECError, make_tec
from pic.interface import PIC, PICError, _PORT_PATTERNS, find_port
from pic.layout import LABEL_OF_DAC
from pic.session import laser_session
from theory.clements import NMODE

READ_QUANTUM_V = ADC_REF_V / (2**ADC_BITS - 1) / ADC_AVG_N  # the firmware averages AVG_N

NOISE_OUTLIER = 4.0     # x the median detector's sd before a channel is called an outlier
DEAD_DELTA_V = 2 * READ_QUANTUM_V   # a live detector moves further than this somewhere
PORT_SPREAD_DB = 4.0    # per-port coupling spread worth telling the operator about
DARK_READS = 40         # a 40-sample sd is good to ~11 percent, which is enough to rank on
LIT_READS = 8

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"

CHECKS = {
    "ports": "every serial device present, and which instrument each one is. Opens nothing.",
    "board": "the reset banner, then a full DAC-vector-in, ADC-vector-out round trip.",
    "tec": "the die is at its setpoint, waiting out the settle before judging it.",
    "switch": "all four input ports plus the dark position, each confirmed by its echo.",
    "laser": "key, both interlocks and the alarm word -- before anything is enabled.",
    "dark": "per-detector dark mean and sd with the laser off and the path open.",
    "emission": "the diode is really emitting, at the monitor and at the chip.",
    "response": "each input port lights the detectors, measured as SNR over the dark sd.",
    "heaters": "every wired channel moves the light, at its own current-limited ceiling.",
}

MEANS = {
    "ports": "No serial port could be given a role. Either nothing is plugged in, or the "
             "board and the TEC controller are both `usbmodem` and cannot be told apart by "
             "name. Pass --pic-port and --tec-port, or set $PIC4_PORT. Nothing was opened.",
    "board": "The port did not answer what pic4x4.ino answers. Either it is not the board "
             "(the TEC controller sits on an indistinguishable name and streams three-field "
             "CSV), the sketch on it is not pic4x4.ino, or another process holds the port. "
             "A host expecting 4 values waits forever against a firmware sending 8.",
    "tec": "The die temperature is unknown or uncontrolled, so nothing measured after this "
           "can be compared with anything measured on another day -- which is the one "
           "advantage this rig has over the 6x6. Check that the second Arduino is running "
           "Arduino/tec_pid/tec_pid.ino. Pass --tec-port none to proceed without it.",
    "switch": "The Sercalo did not go where it was told. It hangs off the board's Serial1, "
              "not off a port of its own, so this is the firmware's P command, that 9600 "
              "line, or the switch's power -- not a host serial fault. Every per-port "
              "measurement after this would silently have been taken at one port.",
    "laser": "The driver will not emit. Key and both interlocks gate the enable in "
             "hardware, and this stage refuses before anything is enabled rather than "
             "after. Turn the key, close the interlocks, and re-run.",
    "dark": "The detectors are not converting. Every channel reading one value with no "
            "dither at all means the ADC is not sampling: check A0..A3 and the PD-TIA "
            "supply before believing any optical measurement.",
    "emission": "`laser_status == 1` does not prove the diode is emitting, which is why "
                "this compares the monitor photodiode and the chip photodiodes off- "
                "versus-on instead of believing the register. Armed but dark means the "
                "setpoint is under threshold or the diode makes no light; light at the "
                "monitor but not at the chip means the fibre, the switch or the facet.",
    "response": "Light reaches the chip but not the detectors it should reach. A port with "
                "no light is that port's coupling or the switch mirror; a detector with no "
                "response is a dead channel. Either one invalidates every per-port "
                "characterization taken through it.",
    "heaters": "Nothing the host writes reaches a heater. The DAC config sequence in "
               "pic4x4.ino runs on every write, not once at boot; if that was 'optimised' "
               "into setup() the chips never come up and a whole sweep comes back with no "
               "drive on it. Check the SPI wiring and CS 10 / CS 9.",
}


@dataclass
class Stage:
    name: str
    status: str
    note: str
    code: str = ""


@dataclass
class Report:
    stages: list = field(default_factory=list)

    def add(self, st):
        self.stages.append(st)
        return st

    @property
    def failure(self):
        return next((s for s in self.stages if s.status == FAIL), None)

    def get(self, name):
        return next((s for s in self.stages if s.name == name), None)

    def table(self, width: int = 96) -> str:
        w = max(len(s.name) for s in self.stages)
        pad = " " * (w + 10)
        rows = [f"{'stage':<{w}}  {'result':<6}  what it saw", "-" * width]
        for s in self.stages:
            note = textwrap.wrap(s.note, max(20, width - len(pad))) or [""]
            rows.append(f"{s.name:<{w}}  {s.status:<6}  {note[0]}")
            rows += [pad + line for line in note[1:]]
        return "\n".join(rows)


def is_ftdi(port: str) -> bool:
    """The laser's FTDI adapter carries an alphanumeric serial; the CH340 boards show a
    purely numeric one. Same rule `Laser.autodetect` uses, applied to a list we already have."""
    return "usbserial" in port and not re.fullmatch(r"\d+", port.rsplit("-", 1)[-1])


def classify(found, pic=None, laser=None, tec=None):
    """Give every serial port a role. Returns (roles, unassigned).

    `roles['pic']` is None when the board cannot be identified without opening something,
    which is a refusal and not a fallback: the TEC controller answers on an indistinguishable
    name and opening it resets it."""
    found = list(found)
    roles = {"laser": laser, "pic": pic, "tec": tec}
    if roles["laser"] is None:
        ftdi = [p for p in found if is_ftdi(p)]
        roles["laser"] = ftdi[0] if len(ftdi) == 1 else None
    rest = [p for p in found if p not in set(filter(None, roles.values())) and not is_ftdi(p)]
    if roles["pic"] is None and len(rest) == 1:
        roles["pic"] = rest[0]
    rest = [p for p in rest if p != roles["pic"]]
    if roles["tec"] is None and len(rest) == 1:
        roles["tec"] = rest[0]
    return roles, [p for p in rest if p != roles["tec"]]


class BannerPIC(PIC):
    """`PIC` that keeps the reset banner instead of flushing it.

    `PIC.open` throws it away, reasonably -- nothing in normal operation needs it. A
    bring-up check does: the banner is the only positive identification that a port is the
    4x4 board rather than the TEC controller, and it costs nothing since the board resets on
    open anyway."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.banner = None
        self.chatter = None

    def open(self):
        import serial

        port, cands = find_port(self.cfg.port)
        if port is None:
            raise PICError(f"no serial port found; candidates were {cands}")
        self.cfg.port = port
        self.ser = serial.Serial(port, self.cfg.baud, timeout=self.cfg.timeout_s)
        deadline = time.time() + self.cfg.reset_wait_s + 3.0
        while time.time() < deadline:
            line = self.ser.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            if line.startswith(READY_BANNER):
                self.banner = line
                break
            self.chatter = line  # unsolicited traffic: almost certainly the TEC's telemetry
            break
        self.ser.reset_input_buffer()
        return self


class BenchBoard:
    """The hardware-free board: `pic.sim`'s bench model plus the three things this harness
    has to be able to watch fail.

    `pic.sim.BenchPIC` models the chip and the readout but knows nothing about the laser, the
    firmware's port command, or the fact that PD0 is a different detector from PD1. All three
    are the difference between a bring-up check that exercises its own logic and one that
    passes vacuously, so they live here rather than in the simulator, which is anchored on
    the archive rather than on today's bench.
    """

    ARCHIVE_LASER_DBM = 8.0  # what the sessions pic.sim's numbers come from were taken at

    # Measured 2026-08-27 at AVG_N = 16, laser off: PD0 scatters ~70x wider than the other
    # three. It is not dead -- it carries signal at SNR 1-2 where the others sit at 23-47 --
    # and averaging is the only lever, which is why AVG_N went from 5 to 16.
    PD_DARK_SIGMA_V = np.array([0.0440, 0.0006, 0.0006, 0.0006])
    # A channel that scatters +/- 44 mV and never reads negative is sitting on a TIA offset;
    # without one the ADC's floor would clip half of its distribution away and the measured
    # sd could not have been 44 mV in the first place. Two sigma is the smallest offset
    # consistent with the measurement, so it is the one assumed.
    PD_DARK_OFFSET_V = 2 * PD_DARK_SIGMA_V
    DEAD_PD_V = 0.0049  # a disconnected TIA input sits at an offset and stays there

    def __init__(self, laser=None, switch=None, *, dead_pd=None, stuck_port=None,
                 dark_port=None, bad_firmware=False, seed=0):
        from pic.sim import READ_SIGMA_V, SESSION_INPUT_DBM, BenchPIC

        self.pic = BenchPIC(switch=switch)
        self.cfg = self.pic.cfg
        self.sim = self.pic.sim
        self.laser = laser
        self.switch = switch
        self.banner = READY_BANNER
        self.chatter = None
        self.dead_pd = dead_pd
        self.stuck_port = stuck_port
        self.bad_firmware = bool(bad_firmware)
        self._rng = np.random.default_rng(seed)
        # the simulator already carries an electronic floor for every channel; what is added
        # here is only what today's measurement has beyond it, so the two are not counted twice
        self.excess_sigma = np.sqrt(np.maximum(self.PD_DARK_SIGMA_V**2 - READ_SIGMA_V**2, 0.0))
        # the laser's calibration is in dBm at the diode; pic.sim's input is dBm at the
        # switch, so the link loss is whatever reconciles them at the archive's laser power
        self.link_loss_db = self.ARCHIVE_LASER_DBM - SESSION_INPUT_DBM
        if dark_port is not None:
            self.sim.port_gain[int(dark_port)] = 1e-5

    def open(self):
        self.pic.open()
        return self

    def close(self):
        self.pic.close()

    def set_zero(self):
        self.measure_raw(np.zeros(NUM_DAC))

    def _launch_dbm(self) -> float:
        """Power at the switch, through the laser's own calibration -- so `--dbm` moves the
        photodiodes here the way it does on the bench, and a diode that is armed but under
        its lasing threshold delivers nothing at all."""
        if self.laser is None:
            return -np.inf  # no laser attached yet: the chip is dark, which is the truth
        emitting = getattr(self.laser, "emitting", None)
        if not (emitting() if emitting else self.laser.is_on()):
            return -np.inf
        sp = float(self.laser.dev.read_setting("cw_current"))
        return Laser.setpoint_to_dbm(sp)[0] - self.link_loss_db

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        if self.bad_firmware:
            raise PICError(f"timeout waiting for {NUM_ADC_RAW} values from mock-board")
        self.sim.input_dbm = self._launch_dbm()
        y = np.asarray(self.pic.measure_raw(voltages), float)
        y[list(OUT_PDS)] += (self.PD_DARK_OFFSET_V
                             + self._rng.normal(0.0, 1.0, len(OUT_PDS)) * self.excess_sigma)
        if self.dead_pd is not None:
            y[int(self.dead_pd)] = self.DEAD_PD_V
        return np.clip(np.round(y / READ_QUANTUM_V) * READ_QUANTUM_V, 0.0, ADC_REF_V)

    def measure(self, voltages, settle_s=None):
        return self.measure_raw(voltages)[list(OUT_PDS)]

    def select_port(self, port: int) -> int:
        """The firmware's `P` command, echo semantics included, so a `BoardSwitch` has
        something to talk to with nothing plugged in."""
        p = int(port)
        if not -1 <= p < NMODE:
            raise PICError(f"firmware refused port {p}: 'ERR port'")
        if self.stuck_port is not None and p != self.stuck_port:
            raise PICError(f"switch went to port {self.stuck_port}, asked for {p}")
        if p >= 0:
            self.sim.select(p)
        return p


class Chain:
    def __init__(self, a):
        self.a = a
        self.rep = Report()
        self.stack = contextlib.ExitStack()
        self.board = self.laser = self.tec = self.switch = self.s = None
        self.roles = {}
        self.tec_ok = False
        self.dark_mean = self.dark_sd = None
        self.lit = {}
        self.best_port = 0
        self.pd_rows = []
        self.heater_rows = []

    def sd_floor(self):
        """A standard deviation below the readout quantum is not a measurement of anything,
        so SNR is computed against the larger of the two."""
        return np.maximum(self.dark_sd, READ_QUANTUM_V)

    def _reads(self, v, n, settle_s=0.0):
        if self.s is not None:
            self.s.keepalive()
        self.board.measure_raw(v)
        if settle_s:
            time.sleep(settle_s)
        y = np.array([self.board.measure_raw(v) for _ in range(n)], float)
        return y.mean(0), y.std(0)

    def stage_ports(self):
        if self.a.mock:
            found = ["/dev/cu.usbmodem11101", "/dev/cu.usbmodem11201",
                     "/dev/cu.usbserial-AU05XLI8"]
        else:
            found = sorted({p for pat in _PORT_PATTERNS for p in glob.glob(pat)})
        if not found:
            return Stage("ports", FAIL, "no serial devices present", "none")

        pic_arg = self.a.pic_port or os.environ.get("PIC4_PORT")
        tec_arg = None if self.a.tec_port in (None, "none") else self.a.tec_port
        if self.a.mock and self.a.fault != "ambiguous-ports":
            # the mock hands itself the flags the bench needs, so --mock is a smoke test
            # rather than a standing demonstration that two usbmodems cannot be told apart;
            # --fault ambiguous-ports withholds them and exercises that refusal instead
            pic_arg = pic_arg or found[0]
            if self.a.tec_port is None:
                tec_arg = found[1]
        roles, spare = classify(found, pic=pic_arg, laser=self.a.laser_port, tec=tec_arg)
        if self.a.tec_port == "none":
            roles["tec"] = None
        self.roles = roles

        used = [p for p in roles.values() if p]
        if len(set(used)) != len(used):
            return Stage("ports", FAIL, f"two roles resolved to one port: {roles}", "collision")
        if roles["laser"] is None:
            return Stage("ports", FAIL,
                         f"no AeroDiode FTDI among {found}; pass --laser-port", "no-laser")
        if roles["pic"] is None:
            cands = [p for p in found if not is_ftdi(p)]
            return Stage("ports", FAIL,
                         f"board ambiguous among {cands}; pass --pic-port or set $PIC4_PORT",
                         "ambiguous")

        tec = roles["tec"] or ("skipped" if self.a.tec_port == "none" else "not found")
        note = (f"{len(found)} present; board {roles['pic']}, laser {roles['laser']}, "
                f"TEC {tec}")
        if spare:
            note += f"; unassigned {spare}"
        return Stage("ports", OK, note)

    def stage_board(self):
        if self.a.mock:
            f = self.a.fault
            self.board = BenchBoard(laser=None, switch=None,
                                    bad_firmware=f == "bad-firmware",
                                    dead_pd=2 if f == "dead-pd" else None,
                                    dark_port=1 if f == "dark-port" else None).open()
        else:
            self.board = BannerPIC(port=self.roles["pic"]).open()
        t0 = time.time()
        y = self.board.measure_raw(np.zeros(NUM_DAC))
        dt = time.time() - t0
        self.board.measure_raw(np.zeros(NUM_DAC))  # a second trip: one reply, not the last one
        if y.size != NUM_ADC_RAW or not np.all(np.isfinite(y)):
            return Stage("board", FAIL, f"reply was {y}", "width")
        note = (f"{NUM_DAC} volts -> {NUM_ADC_RAW} ADC volts in {dt * 1e3:.0f} ms, "
                f"{y.min():.3f}-{y.max():.3f} V")
        if self.board.banner:
            return Stage("board", OK, f'banner "{self.board.banner}"; {note}')
        extra = f"; port was already streaming {self.board.chatter!r}" if self.board.chatter else ""
        return Stage("board", WARN, f"no reset banner (already running?); {note}{extra}",
                     "no-banner")

    def stage_tec(self):
        if self.a.tec_port == "none":
            return Stage("tec", SKIP, "--tec-port none: die temperature unrecorded", "skipped")
        if self.a.mock:
            # starts at the setpoint, because the bench cooler has been holding 25 C for
            # hours by the time anyone runs this -- `warm-chip` is the die that has not
            self.tec = (NoTEC() if self.a.fault == "no-tec" else
                        MockTEC(tau_s=1e6 if self.a.fault == "warm-chip" else 0.05,
                                ambient_c=30.0 if self.a.fault == "warm-chip"
                                else TEC_SETPOINT_C))
        elif self.roles.get("tec") is None:
            return Stage("tec", FAIL, "no TEC controller port; pass --tec-port or "
                                      "--tec-port none", "no-port")
        else:
            self.tec = make_tec(self.roles["tec"])
        self.tec.open()
        t = self.tec.temperature()
        if not np.isfinite(t):
            return Stage("tec", FAIL, "no telemetry from the TEC controller", "silent")
        st = self.tec.status()
        drive = st.get("drive_v")
        tail = f", drive {drive:+.2f} V" if drive is not None else ""
        waited = 0.0
        if abs(t - self.tec.target) > TEC_TOLERANCE_C:
            # a measurement taken while the substrate is still moving is worse than none, so
            # give it the settling time before judging it rather than recording it as-is
            try:
                waited = self.tec.wait_stable(timeout_s=self.a.tec_wait)
            except TECError:
                waited = float(self.a.tec_wait)
            t = self.tec.temperature()
        err = abs(t - self.tec.target)
        note = f"{t:.2f} C, setpoint {self.tec.target:.2f}{tail}"
        if err <= TEC_TOLERANCE_C:
            self.tec_ok = True
            return Stage("tec", OK, note + (f", settled in {waited:.0f} s" if waited else ""))
        return Stage("tec", WARN, f"{note}; {err:.2f} C out of band after {waited:.0f} s -- "
                                  f"the die is not where the calibration was taken",
                     "out-of-band")

    def stage_switch(self):
        self.switch = BoardSwitch().attach(self.board)
        if self.a.mock:
            self.board.switch = self.switch
            self.board.pic.switch = self.switch
            self.board.stuck_port = 0 if self.a.fault == "stuck-switch" else None
        t0 = time.time()
        for k in range(NMODE):
            self.switch.select(k)
        self.switch.dark()
        if self.switch.position() is not None:
            return Stage("switch", FAIL, "still reports a live port after P0", "not-dark")
        return Stage("switch", OK,
                     f"P1..P{NMODE} and P0 (dark) all echoed in {time.time() - t0:.1f} s; the "
                     f"driver compares each echo, so a mirror that did not move raises here")

    def stage_laser(self):
        if self.a.mock:
            self.laser = MockLaser(key=0 if self.a.fault == "no-key" else 1,
                                   lase_sp=1e9 if self.a.fault == "no-emission" else None)
            self.board.laser = self.laser
        else:
            self.laser = Laser(port=self.roles["laser"])
        self.laser.open()
        pre = self.laser.preflight()
        bad = [k for k, v in pre.items() if not v]
        if bad:
            return Stage("laser", FAIL, f"unmet: {bad}", "interlock")
        alarms = int(self.laser.dev.measure("alarms"))
        note = (f"key ON, both interlocks closed, driver {self.laser.dev.version()}, "
                f"alarms 0x{alarms:x}")
        if self.laser.is_on():
            self.laser.off()  # a true dark baseline needs it off, and off is always allowed
            return Stage("laser", WARN, f"{note}; was already ON -- turned off for the dark "
                                        f"baseline, check no other process holds it", "was-on")
        return Stage("laser", OK, note)

    def stage_dark(self):
        self.dark_mean, self.dark_sd = self._reads(np.zeros(NUM_DAC), self.a.dark_reads)
        sd = self.dark_sd[list(OUT_PDS)]
        med = float(np.median(sd))
        loud = [j for j in OUT_PDS if self.dark_sd[j] > NOISE_OUTLIER * max(med, 1e-9)]
        note = (f"{self.a.dark_reads} reads, laser off and switch dark; sd "
                f"{sd.min() * 1e3:.2f}-{sd.max() * 1e3:.2f} mV, median {med * 1e3:.2f}")
        if np.all(sd <= 0):
            return Stage("dark", FAIL, f"{note}; no dither on any channel", "no-dither")
        if loud:
            ratio = max(self.dark_sd[j] / max(med, 1e-9) for j in loud)
            names = ", ".join(f"PD{j}" for j in loud)
            return Stage("dark", WARN,
                         f"{note}; {names} at {ratio:.0f}x the median detector", "outlier")
        return Stage("dark", OK, note)

    def stage_emission(self):
        # The watchdog has to outlast everything still to be measured with the beam on, so
        # the bound is the worst case: every wired channel re-probed at every port. It is
        # deliberately generous -- a hard-off mid-run is safe but wastes a bench session --
        # and it uses the package's own timing model so the two cannot drift apart.
        n_groups = 1 + NMODE + NMODE * (1 + len(WIRED_DACS))
        dur = (estimate_seconds(n_groups, self.a.settle, self.a.reads)
               + (1 + 2 * NMODE) * (SWITCH_SETTLE_S + 0.3))
        self.s = self.stack.enter_context(
            laser_session(self.laser, duration_s=dur, power_dbm=self.a.dbm,
                          tec=self.tec, require_stable=self.tec_ok,
                          bfm_off=None))
        self.switch.select(0)
        lit, _ = self._reads(np.zeros(NUM_DAC), self.a.reads)
        snr = (lit - self.dark_mean) / self.sd_floor()
        seen = bool(np.any(snr[list(OUT_PDS)] >= MIN_SNR))
        note = (f"bfm {self.s.bfm_off:.3f} -> {self.s.bfm_on:.3f}, sp {self.s.sp:.1f} "
                f"({self.s.mA:.0f} mA); PD sum {self.dark_mean[list(OUT_PDS)].sum():.3f} -> "
                f"{lit[list(OUT_PDS)].sum():.3f} V, best PD SNR {snr.max():.1f}")
        if self.s.emitted and seen:
            return Stage("emission", OK, note)
        if self.s.emitted and not seen:
            return Stage("emission", FAIL, f"{note}; monitor sees light, the chip does not",
                         "no-coupling")
        if seen:
            return Stage("emission", WARN, f"{note}; monitor says dark but the detectors "
                                           f"see light -- trust the detectors", "bfm-dark")
        return Stage("emission", FAIL, f"{note}; armed but dark", "no-emission")

    def stage_response(self):
        for k in range(NMODE):
            self.switch.select(k)
            self.lit[k] = self._reads(np.zeros(NUM_DAC), self.a.reads)[0]
        delta = {k: self.lit[k] - self.dark_mean for k in self.lit}
        snr = {k: delta[k] / self.sd_floor() for k in delta}
        # ports are compared on light, not on level: PD0 carries a TIA offset several times
        # its neighbours' whole signal, and a raw sum would rank the ports by that offset
        totals = np.array([delta[k][list(OUT_PDS)].sum() for k in range(NMODE)])
        self.best_port = int(np.argmax(totals))
        best_snr = np.max(np.stack([snr[k] for k in range(NMODE)]), axis=0)
        best_delta = np.max(np.stack([delta[k] for k in range(NMODE)]), axis=0)

        self.pd_rows = [(j, self.dark_mean[j], self.dark_sd[j], self.lit[self.best_port][j],
                         best_snr[j]) for j in OUT_PDS]
        dead = [j for j in OUT_PDS
                if best_snr[j] < 1.0 and best_delta[j] < DEAD_DELTA_V]
        darkport = [k for k in range(NMODE)
                    if max(snr[k][j] for j in OUT_PDS) < 3.0]
        db = 10 * np.log10(np.maximum(totals, 1e-12) / totals.max())
        spread = float(-db.min())
        note = (f"per-port signal {totals.min():.3f}-{totals.max():.3f} V, best port "
                f"{self.best_port}; {sum(best_snr[j] >= MIN_SNR for j in OUT_PDS)}/"
                f"{len(OUT_PDS)} detectors above SNR {MIN_SNR:.0f}")
        if dead:
            names = ", ".join(f"PD{j}" for j in dead)
            return Stage("response", FAIL,
                         f"{note}; {names} never moved by {DEAD_DELTA_V * 1e3:.1f} mV at "
                         f"any port", "dead-pd")
        if darkport:
            names = ", ".join(f"port {k}" for k in darkport)
            return Stage("response", FAIL, f"{note}; no light on {names}", "dark-port")
        if spread > PORT_SPREAD_DB:
            return Stage("response", WARN,
                         f"{note}; port {int(np.argmin(db))} sits {spread:.1f} dB below the "
                         f"best -- coupling on that port, not the mesh", "spread")
        return Stage("response", OK, f"{note}, spread {spread:.1f} dB")

    def stage_heaters(self):
        """Every wired channel to its own ceiling, at the port that shows it best.

        The ceiling is per channel and not the firmware's 3 V: the heaters come in two
        resistance groups and 3 V through the 60R group is 1.7x its current rating, so
        `VOLTAGE_MAX_CH` is the number to drive to and the driver does not enforce it.

        A channel silent at one port is re-probed at the others before being called silent.
        At a single port most heaters modulate nothing -- a first-column MZI sits upstream of
        two of the four rails, and an external phase on one is a global phase when only one
        port is lit -- so a single-port version of this stage would report half the board
        dead. Re-probing only the silent ones keeps that to a few extra reads.
        """
        base = np.zeros(NUM_DAC)
        wired = [d for d in sorted(WIRED_DACS) if VOLTAGE_MAX_CH[d] > 0]
        best = {}

        def probe(port, chans):
            self.switch.select(port)
            y0, sd0 = self._reads(base, self.a.reads, settle_s=self.a.settle)
            for d in chans:
                v = base.copy()
                v[d] = VOLTAGE_MAX_CH[d]
                y, sd1 = self._reads(v, self.a.reads, settle_s=self.a.settle)
                step = np.abs(y - y0)
                # against the *lit* scatter, not the dark floor. The laser's common-mode
                # jitter is 2 percent of whatever is on the detector, which is several times
                # a clean channel's dark sd -- measured against dark, every silent heater
                # here "modulated" and the stage reported 13 of 13 live on a chip with 6.
                denom = np.maximum(np.maximum(sd0, sd1), self.sd_floor())
                snr = float(np.max(step / denom))
                if snr > best.get(d, (0.0, 0.0, -1))[1]:
                    best[d] = (float(step.max()), snr, port)

        probe(self.best_port, wired)
        for k in [p for p in range(NMODE) if p != self.best_port]:
            silent = [d for d in wired if best[d][1] < MIN_SNR]
            if not silent:
                break
            probe(k, silent)
        self.board.set_zero()

        self.heater_rows = [(d, LABEL_OF_DAC[d]) + best[d] for d in wired]
        moved = [d for d in wired if best[d][1] >= MIN_SNR]
        note = (f"{len(moved)}/{len(wired)} wired channels modulate at their own ceiling "
                f"({min(VOLTAGE_MAX_CH[d] for d in wired):.1f}-"
                f"{max(VOLTAGE_MAX_CH[d] for d in wired):.1f} V)")
        if not moved:
            return Stage("heaters", FAIL, f"{note}; nothing the host writes reaches the chip",
                         "no-drive")
        silent = [LABEL_OF_DAC[d] for d in wired if d not in moved]
        if silent:
            return Stage("heaters", WARN, f"{note}; silent at every port: {silent}", "silent")
        return Stage("heaters", OK, note)

    def run(self):
        seq = [self.stage_ports, self.stage_board, self.stage_tec, self.stage_switch,
               self.stage_laser, self.stage_dark, self.stage_emission, self.stage_response,
               self.stage_heaters]
        try:
            for fn in seq:
                name = fn.__name__[len("stage_"):]
                try:
                    st = fn()
                except (PICError, LaserError, TECError, SwitchError, PDMv5Error,
                        OSError, ValueError) as e:
                    st = Stage(name, FAIL, f"{type(e).__name__}: {e}", "raised")
                self.rep.add(st)
                if st.status == FAIL:
                    for later in seq[seq.index(fn) + 1:]:
                        self.rep.add(Stage(later.__name__[len("stage_"):], SKIP,
                                           "not reached", "not-reached"))
                    break
        finally:
            self.close()
        return self.rep

    def close(self):
        self.stack.close()  # laser_session's own finally: watchdog cancelled, laser off
        for shut in (self._safe_board, self._safe_laser, self._safe_tec):
            try:
                shut()
            except Exception as e:
                print(f"  (cleanup: {e})")

    def _safe_board(self):
        if self.board is not None:
            self.board.set_zero()
            self.board.close()

    def _safe_laser(self):
        if self.laser is not None:
            self.laser.off()
            self.laser.close()

    def _safe_tec(self):
        if self.tec is not None:
            self.tec.close()


def print_report(chain, rep):
    a = chain.a
    if a.mock:
        head = "MOCK (pic.sim bench model, no hardware"
        head += f", fault {a.fault})" if a.fault != "none" else ")"
    else:
        head = f"HARDWARE at {a.dbm:+.1f} dBm"
    print(f"\n4x4 end-to-end bring-up -- {head}\n")
    print(rep.table())

    if chain.pd_rows:
        print("\nphotodiodes: dark noise with the laser off, response at the best input port.")
        print("SNR is the lit-minus-dark step in units of that detector's own dark sd, which "
              "is\nwhat pic.characterize gates its fringe fits on -- amplitude alone lets the "
              "noisiest\ndetector win every channel that has no real signal.\n")
        print(f"  {'pd':>3}  {'dark mean':>10}  {'dark sd':>9}  {'lit':>9}  {'SNR':>7}  note")
        med = float(np.median([r[2] for r in chain.pd_rows]))
        for j, dm, sd, lit, snr in chain.pd_rows:
            flags = []
            if sd > NOISE_OUTLIER * max(med, 1e-9):
                flags.append(f"{sd / max(med, 1e-9):.0f}x the median detector")
            if snr < MIN_SNR:
                flags.append("below the fit gate")
            print(f"  {j:>3}  {dm * 1e3:9.1f} mV  {sd * 1e3:6.2f} mV  {lit * 1e3:6.1f} mV  "
                  f"{snr:7.1f}  {'; '.join(flags)}")

    if chain.lit:
        d = {k: chain.lit[k] - chain.dark_mean for k in sorted(chain.lit)}
        totals = np.array([d[k][list(OUT_PDS)].sum() for k in sorted(d)])
        db = 10 * np.log10(np.maximum(totals, 1e-12) / totals.max())
        print("\ninput ports, heaters at 0 V, dark subtracted:\n")
        print("  " + f"{'port':>4}  " + "  ".join(f"{'pd' + str(j):>8}" for j in OUT_PDS)
              + f"  {'sum':>9}  {'vs best':>8}")
        for k in sorted(d):
            cells = "  ".join(f"{d[k][j] * 1e3:8.1f}" for j in OUT_PDS)
            print(f"  {k:>4}  {cells}  {totals[k] * 1e3:6.1f} mV  {db[k]:+7.1f} dB")

    if chain.heater_rows:
        print("\nwired heaters, one at a time to their own current-limited ceiling:\n")
        print(f"  {'dac':>3}  {'heater':<14}  {'ceiling':>8}  {'largest step':>13}  "
              f"{'SNR':>7}  {'port':>4}")
        for d, label, step, snr, port in chain.heater_rows:
            print(f"  {d:>3}  {label:<14}  {VOLTAGE_MAX_CH[d]:6.1f} V  {step * 1e3:10.1f} mV  "
                  f"{snr:7.1f}  {port:>4}")

    bad = rep.failure
    print()
    if bad is None:
        warns = [s for s in rep.stages if s.status == WARN]
        print("CHAIN OK -- laser, fibre, switch, chip, detectors and DAC drive all confirmed.")
        if warns:
            print(f"{len(warns)} stage(s) warned: {', '.join(s.name for s in warns)}. Nothing "
                  "is broken, but read those rows before trusting a measurement.")
    else:
        print(f"CHAIN BROKEN AT '{bad.name}' -- {bad.note}\n")
        print(MEANS[bad.name])
    return 0 if bad is None else 1


def explain(width: int = 92):
    print("stages, in the order light travels. The first failure stops the run, because a "
          "stage\nbelow a broken link only produces plausible numbers about a dead chain.\n")
    for name, why in MEANS.items():
        print(f"{name:<10}{CHECKS[name]}")
        for line in textwrap.wrap(why, width - 10):
            print(f"{'':<10}{line}")
        print()
    return 0


def run(a):
    if a.explain:
        return explain()
    if a.fault != "none" and not a.mock:
        print("--fault only makes sense with --mock: it drives the model into a refusal path.")
        return 2
    chain = Chain(a)
    rep = chain.run()
    return print_report(chain, rep)


def _mock_args(fault="none", **kw):
    a = dict(mock=True, fault=fault, explain=False, dbm=8.0, dark_reads=12, reads=4,
             settle=0.0, tec_wait=1.0, laser_port=None, tec_port=None, pic_port=None)
    return argparse.Namespace(**{**a, **kw})


def _selftest(verbose: bool = False):
    """Every stage, and every refusal path, with nothing plugged in.

    A harness that only ever runs against a working rig tests one branch of itself. Each
    fault below is injected into the physical model -- not stubbed over the harness -- so
    what is checked is that the stage which is supposed to catch it does, and that the ones
    after it are skipped rather than run against a broken chain."""
    import io

    expect = {
        "none": (None, None),
        "ambiguous-ports": ("ports", "ambiguous"),
        "bad-firmware": ("board", "raised"),
        "no-tec": ("tec", "silent"),
        "stuck-switch": ("switch", "raised"),
        "no-key": ("laser", "interlock"),
        "no-emission": ("emission", "no-emission"),
        "dead-pd": ("response", "dead-pd"),
        "dark-port": ("response", "dark-port"),
    }
    out = {}
    for fault, (want_stage, want_code) in expect.items():
        a = _mock_args(fault, dark_reads=12, reads=4)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            chain = Chain(a)
            rep = chain.run()
            code = print_report(chain, rep)
        bad = rep.failure
        got = (bad.name, bad.code) if bad else (None, None)
        assert got == (want_stage, want_code), f"{fault}: expected {(want_stage, want_code)}, got {got} ({rep.table()})"
        assert code == (0 if bad is None else 1), fault
        if bad is not None:
            after = rep.stages[[s.name for s in rep.stages].index(bad.name) + 1:]
            assert all(s.status == SKIP for s in after), f"{fault}: ran past the failure"
        out[fault] = got
        if verbose:
            print(buf.getvalue())
    out["faults"] = len(expect) - 1

    # the clean run has to be a *pass*, and it has to reproduce the two bench facts this
    # harness exists to surface: PD0's noise, and port 3's coupling
    with contextlib.redirect_stdout(io.StringIO()):
        chain = Chain(_mock_args("none", dark_reads=40, reads=6))
        rep = chain.run()
        warm = Chain(_mock_args("warm-chip", dark_reads=8, reads=3)).run()
    # a die off its setpoint is a data-quality warning, not a broken chain: the run must
    # continue and say so, because refusing here would be refusing to measure the rig at all
    assert warm.get("tec").status == WARN and warm.failure is None, warm.table()
    out["warm-chip"] = warm.get("tec").note
    sd = np.array([r[2] for r in chain.pd_rows])
    out["pd0_ratio"] = float(sd[0] / np.median(sd[1:]))
    out["pd_snr"] = [round(float(r[4]), 1) for r in chain.pd_rows]
    out["stages"] = {s.name: s.status for s in rep.stages}
    assert rep.failure is None, rep.table()
    assert out["pd0_ratio"] > 20, out["pd0_ratio"]
    assert chain.pd_rows[0][4] < MIN_SNR, "PD0 should sit under the fit gate on this bench"
    assert max(r[4] for r in chain.pd_rows[1:]) > MIN_SNR, "no clean detector saw light"
    assert rep.get("heaters").status in (OK, WARN)
    out["heaters"] = rep.get("heaters").note
    # The modulation gate against the simulator's ground truth. A false *positive* is a bug
    # and the assertion is one-sided for that reason: measured against the dark floor instead
    # of the lit scatter, the laser's 2 percent common-mode jitter made every silent channel
    # pass and this stage reported 13 live on a chip with 6. A false negative is not a bug --
    # a 4.6 V-Vpi heater derated to a 1.5 V ceiling covers 0.11 pi and honestly may not show.
    sim = chain.board.sim
    truth = {d for d, *_ in chain.heater_rows
             if np.isfinite(sim.vpi[sim.heater_of_dac[d]])}
    got = {r[0] for r in chain.heater_rows if r[3] >= MIN_SNR}
    assert got <= truth, f"heater gate: {sorted(got - truth)} have no heater on them"
    assert got, "heater gate found nothing on a board with six live channels"
    out["heater_gate"] = (len(got), len(truth), len(chain.heater_rows))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="./do e2e", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true",
                    help="no hardware: the pic.sim bench model, the measured detector noise, "
                         "and a laser that can be armed without emitting")
    ap.add_argument("--fault", default="none",
                    choices=["none", "ambiguous-ports", "bad-firmware", "no-tec",
                             "stuck-switch", "warm-chip", "no-key", "no-emission",
                             "dead-pd", "dark-port"],
                    help="with --mock, drive the model into one refusal path")
    ap.add_argument("--explain", action="store_true",
                    help="print what each stage checks and what its failure means; runs nothing")
    ap.add_argument("--pic-port", default=None, help="board serial port (or $PIC4_PORT)")
    ap.add_argument("--laser-port", default=None, help="AeroDiode FTDI serial port")
    ap.add_argument("--tec-port", default=None,
                    help="TEC controller serial port, or 'none' to run without it")
    ap.add_argument("--dbm", type=float, default=8.0,
                    help="laser output power for the lit stages; 8 dBm is what this chip's "
                         "best data used and leaves the TIA headroom")
    ap.add_argument("--dark-reads", type=int, default=DARK_READS)
    ap.add_argument("--reads", type=int, default=LIT_READS)
    ap.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                    help="thermo-optic settle before a heater read")
    ap.add_argument("--tec-wait", type=float, default=TEC_SETTLE_S,
                    help="seconds to let the die reach its setpoint before judging it")
    ap.add_argument("--selftest", action="store_true", help="run the harness against itself")
    a = ap.parse_args(argv)

    if a.selftest:
        r = _selftest()
        print(f"stages          {r['stages']}")
        print(f"refusal paths   {r['faults']} faults, each caught by its own stage")
        print(f"PD0 dark noise  {r['pd0_ratio']:.0f}x the median detector; per-PD SNR "
              f"{r['pd_snr']}")
        print(f"heater gate     {r['heater_gate'][0]} of {r['heater_gate'][1]} planted "
              f"channels found in {r['heater_gate'][2]} wired, no false positives")
        print(f"warm die        {r['warm-chip']}")
        print(f"heaters         {r['heaters']}")
        return 0
    try:
        return run(a)
    except KeyboardInterrupt:
        print("\ninterrupted -- laser driven off and DACs zeroed on the way out.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
