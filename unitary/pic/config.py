"""Hardware constants for the 4x4 chip: the firmware protocol, the DAC/ADC widths, and
the operating limits.

Everything here that is not yet confirmed against the delivered board is marked
PROVISIONAL. The 6x6 taught this the hard way -- three different voltage ranges lived in
three layers of that stack and disagreed silently -- so the host range, the firmware
clamp and the DAC reference are all stated here and the firmware sketch reads the same
numbers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import theory.calib as _theory_calib
from theory.calib import VOLTAGE_MAX, VPI_NOMINAL  # noqa: F401  (one source for both)
from theory.layout import N_HEATERS

BAUD_RATE = 115200

# USB serial numbers name each instrument on every OS; the device path (/dev/cu.*, /dev/tty*,
# COM*) changes with the OS and the socket. A replaced board gets a new number here.
USB_SERIAL = {"board": "1344A47423035130E988", "tec": "1344A474230351308829", "laser": "AU05XLI8"}
FTDI_VID = 0x0403
# Which detectors read the mesh outputs: "pd", the analog photodiode board, or "spd", the
# single-photon detectors in SPD_MAP_PATH. Set from the Settings tab; PD when never set.
DETECTORS = ("pd", "spd")
DETECTOR_PATH = "pic_data/detector_mode.json"
# {SPD chip id: output (PD slot) it sits on}, set with `python -m spd assign`. An SPD on USB
# that is not in it is refused rather than guessed.
SPD_MAP_PATH = "pic_data/spd_map.json"
_PINNED = []  # `pin_detectors`: a selftest's mode, which must not follow the bench's file


def detector_mode() -> str:
    import json

    if _PINNED:
        return _PINNED[-1]
    try:
        m = json.loads(open(DETECTOR_PATH).read())["mode"]
    except (OSError, ValueError, KeyError, TypeError):
        return "pd"
    return m if m in DETECTORS else "pd"


def set_detector_mode(mode: str) -> str:
    import json

    if mode not in DETECTORS:
        raise ValueError(f"detector mode {mode!r}; use one of {', '.join(DETECTORS)}")
    with open(DETECTOR_PATH, "w") as f:
        json.dump({"mode": mode}, f)
    return mode


@contextmanager
def pin_detectors(mode: str):
    """Run a block in `mode` whatever the Settings tab says, data paths included."""
    _PINNED.append(mode)
    try:
        yield mode
    finally:
        _PINNED.pop()


def data_path(p, mode=None) -> Path:
    """`p` as the detector mode in force owns it. Calibration, fits, transfer tables, DPNN
    checkpoints and the error log are all readout-specific, so SPD mode keeps its own under
    pic_data/spd/ and runs/spd/, and flipping back to PD finds PD's exactly as they were."""
    p = Path(p)
    if (mode or detector_mode()) != "spd":
        return p
    parts = list(p.parts)
    for k, x in enumerate(parts):
        if x in ("pic_data", "runs"):
            return Path(*parts[: k + 1], "spd", *parts[k + 1 :])
    return p


# theory must not import pic, so the calibration file's mode arrives through this hook
_theory_calib.data_path = data_path


def spd_map(path=None) -> dict[str, int]:
    import json

    try:
        d = json.loads(open(path or SPD_MAP_PATH).read())
        return {str(k): int(v) for k, v in d.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def port_of(x):
    """A chip id (USB serial number) or a role name -> its device path; anything else as given.
    Lets every --*-port flag take the id printed on the Tools tab, which does not change when
    a cable moves, instead of a path that does."""
    if not x:
        return x
    for dev, role, _ in usb_ports():
        if x == role or x == USB_SERIAL.get(role) or x == _serial_of(dev):
            return dev
    return x


def _chip(sn):
    """Windows' FTDI driver appends a channel letter to the chip id (AU05XLI8 -> AU05XLI8A)."""
    known = {*USB_SERIAL.values(), *spd_map()}
    if sn and sn not in known and sn[:-1] in known:
        return sn[:-1]
    return sn


def _serial_of(dev):
    from serial.tools import list_ports

    return next((_chip(p.serial_number) for p in list_ports.comports() if p.device == dev), None)


def usb_ports():
    """[(device, role, vid)] for every USB serial device; role from USB_SERIAL, else '?'."""
    from serial.tools import list_ports

    role = {v: k for k, v in USB_SERIAL.items()} | {k: "spd" for k in spd_map()}
    return sorted(
        (p.device, role.get(_chip(p.serial_number), "?"), p.vid)
        for p in list_ports.comports()
        if p.vid
    )


RESET_WAIT_S = 2.0  # the board resets when the serial port is opened
READY_BANNER = "pic4x4 ready"

NUM_DAC = 16  # DAC81416 channels; must match NUM_DAC in pic4x4.ino
NUM_ADC_RAW = 4  # A0..A3, one per mesh output through a PD-TIA (mrunal/PD_TIA_circuit.pdf)
OUT_PDS = (0, 1, 2, 3)  # the four mesh outputs, in rail order
MON_PDS = ()  # this block carries no tap monitors
NUM_OUT = len(OUT_PDS)

# Board wiring as rewired on 2026-09-22 (`mrunal/board_firmware_v2.md`). Sixteen DAC
# channels drive THIRTEEN heaters: H10, H9 and H4 each take a bonded pair of channels, and
# H18, H14 and H6 lost their driver. The previous map was one heater per channel and the
# identity; neither holds any more.
DAC_HEATER = (
    "H1",
    "H2",
    "H10",
    "H10",
    "H9",
    "H9",
    "H4",
    "H4",
    "H15",
    "H13",
    "H12",
    "H11",
    "H7",
    "H8",
    "H5",
    "H3",
)

# Channels physically shorted together onto one heater. Two DAC81416 outputs tied together
# and commanded to different voltages fight each other, so this is a damage constraint and
# not a modelling convenience: `PIC._sanitize` refuses a mismatched write and the firmware
# mirrors each group. Bonding is also the whole point of the rewire -- a pair sources twice
# the current, which is what finally takes a 57-ohm heater past 2 pi.
DAC_PAIRS = ((2, 3), (4, 5), (6, 7))
PAIR_OF_DAC = {d: g for g in DAC_PAIRS for d in g}

# Per channel, the resistance of the heater it drives; both channels of a pair carry the
# same heater and therefore the same number.
#
# H12 IS DISPUTED and deliberately takes the smaller value. `Arduino/pin_check` and every
# prior bench table give 62.3 ohm; the v2 firmware's own table gives 116.8, which is also
# exactly the value it gives H2 -- a copy-paste shape. Every other heater agrees between the
# two. At 40 mA the two readings differ by 4.67 V against 2.45 V, and guessing high would put
# 75 mA through it, so the low reading stands until `pic.resistance` settles it.
HEATER_OHMS = (
    114.1,
    116.8,
    56.4,
    56.4,
    57.2,
    57.2,
    57.8,
    57.8,
    114.1,
    114.9,
    62.3,
    118.8,
    116.1,
    114.1,
    117.1,
    113.5,
)
# Raised from the documents' 30 mA on a deliberate call: nothing in ananya/ or mrunal/ gives
# an absolute maximum, a derating curve or a source for the 30 -- it is asserted twice and
# never justified. Span goes as V^2, so 40 mA buys (40/30)^2 = 1.78x more phase on every
# channel -- enough to put several heaters past pi, which is the threshold that decides
# whether an MZI can reach every splitting ratio. The risk is lifetime, not immediate
# failure; nothing here is a datasheet limit. Put this back to 30 to revert everything.
# 50 mA since 2026-09-22, approved at the bench. The first characterization with a working
# DAC put Vpi at 4.73-5.33 V on every channel that fitted, against ceilings of 4.50-4.75 V at
# 40 mA: the mesh reaches only ~0.8 pi, the cosine never turns over, and seven of twelve
# channels came back unfittable for that reason rather than for want of signal. 50 mA puts
# 5.70 V on a 114 ohm heater, which clears Vpi. Risk is lifetime, not immediate failure.
HEATER_MAX_MA = 50.0

# Per-channel ceiling, from the channel's own heater and how many channels share it: a
# bonded pair sources 2 x HEATER_MAX_MA into one resistance, so V = I*n*R. Rounded down to
# 0.05 V so a rounding error cannot push a channel over the limit. Both channels of a pair
# land on the same number by construction, which is what makes a mirrored write safe.
#
# An unmeasured channel would be STAGED at V_UNMEASURED rather than held dark -- holding it
# dark was self-defeating, since no resistance clamped it to 0 V, so no characterization ever
# swept it, so its Vpi stayed nominal and `HeaterBox.from_calibration` dropped it for being
# unfitted. No channel needs that today: the three heaters that had no measured resistance
# are the three the rewire disconnected.
V_UNMEASURED = 1.5
# The DAC cannot exceed its own reference, and asking it to is not a clamp but a wrap:
# `pic4x4.ino` computes code = V * 65535 / DAC_REF as a 16-bit unsigned, so 5.90 V against a
# 5.0 V reference overflows to a LOW voltage with no error anywhere. 50 mA wants 5.7-5.9 V on
# the 114-119 ohm channels; the ceiling is the reference, which caps them near 44 mA. The
# bonded pairs reach their full 50 mA because half the resistance needs half the volts.
DAC_CEILING_V = 4.95  # DAC_REF 5.0 in the firmware, less a margin against rounding

VOLTAGE_MAX_CH = tuple(
    (
        V_UNMEASURED
        if r is None
        else min(
            DAC_CEILING_V,
            float(int(1e-3 * HEATER_MAX_MA * len(PAIR_OF_DAC.get(d, (d,))) * r / 0.05) * 0.05),
        )
    )
    for d, r in enumerate(HEATER_OHMS)
)


def mirror_pairs(v):
    """Copy each bonded group's first channel over the rest, in place, and return it.

    The free variable on this board is the DRIVE UNIT, not the DAC channel: `DAC_PAIRS`
    shorts two outputs onto one heater, so a draw that gives them separate values is not a
    richer sample, it is an impossible state that `PIC._prep_dac` refuses. Any code that
    builds a voltage vector by iterating channels needs this; code that builds one from
    `ACTIVE_DACS` or `REACHABLE_DACS` does not, since neither contains a partner.
    """
    import numpy as _np

    v = _np.asarray(v, float)
    for grp in DAC_PAIRS:
        v[..., list(grp)] = v[..., grp[0], None]
    return v


WIRED_DACS = tuple(i for i, v in enumerate(VOLTAGE_MAX_CH) if v > 0)  # all 16 channels

# One command scale for every channel. Callers work in "drive volts" 0..DRIVE_MAX_V and the
# scale opens each channel up to its own real ceiling, so nothing above has to carry a
# per-channel limit around. The 4.5 V channels get x3, the 2.25 V channels x1.5.
#
# Quantisation survives it: the DAC is 16 bits over 5 V, so an LSB is 76 uV and doubling it
# still leaves 39,300 steps across a 3 V channel. Phase goes as V^2, so the coarsest phase
# step sits at full drive and is pi*(2*76uV)*2*Vmax/Vpi^2 -- under 1e-4 rad. Not a limit.
DRIVE_MAX_V = 1.5
DRIVE_SCALE = tuple(v / DRIVE_MAX_V for v in VOLTAGE_MAX_CH)


def drive_to_volts(drive):
    """Uniform 0..DRIVE_MAX_V command -> real per-channel volts.

    This equalises the *command range*, not the phase range: span goes as (Vmax/Vpi)^2, so
    a 60R heater at 1.5 V still covers ~0.11 pi against a 120R heater's ~0.43 pi. Uniform
    drive makes the model and the surrogates rectangular; it does not buy reach."""
    d = np.clip(np.asarray(drive, float), 0.0, DRIVE_MAX_V)
    return d * np.asarray(DRIVE_SCALE, float)


def drive_volts_raw(drive):
    """The drive -> volts map with no ceiling applied, for reading stored data.

    `drive_to_volts` clips because it is on the way to the board, where the clamp is what
    keeps a 60-ohm heater off its current limit. A capture taken under a wider clamp table
    holds points above today's ceiling, and clipping those on the way into a model reports
    a voltage the chip never saw. Model features are what happened; the clamp is what is
    allowed. Nothing that commands hardware may use this."""
    return np.asarray(drive, float) * np.asarray(DRIVE_SCALE, float)


def volts_to_drive(volts):
    """Real per-channel volts -> the uniform command, for code that has to cross back.

    A dark channel has no inverse and maps to 0, which is the only voltage it can be at
    anyway. Volts above a channel's ceiling clamp, exactly as the driver clamps them."""
    s = np.asarray(DRIVE_SCALE, float)
    live = s > 0
    return np.clip(np.asarray(volts, float) / np.where(live, s, 1.0), 0.0, DRIVE_MAX_V) * live


VOLTAGE_MIN = 0.0
# The firmware clamp is per channel (`VMAX[]` in pic4x4.ino) and equals VOLTAGE_MAX_CH
# element for element -- `pic.rig.assert_firmware_vmax` checks that over the wire on every
# open, using the board's `V` query, because a scalar copy of a per-channel table is exactly
# what let three channels sit at 0 V in firmware while the host thought they were open.
# This is the widest of them, for callers that need one number to size an axis. It is NOT a
# per-channel limit and must never be used as one.
FIRMWARE_VMAX = max(VOLTAGE_MAX_CH)
DAC_REF_V = 5.0
DAC_BITS = 16
# Must match ADC_REF_V in Arduino/pic4x4/pic4x4.ino, which moved from AVCC to the 1.1 V
# bandgap: against 5 V the photodiodes used 9.4% of full scale and 6.6 of 10 bits, and at the
# MEDIAN table entry quantisation was 0.0063 relative against a 0.0035 noise floor -- 64% of
# entries were quantisation-limited. The bandgap is +-10% part to part, so this is a nominal
# scale and not a calibrated one; every quantity here is a ratio within one sweep
# (`to_transfer` divides each column by its own sum), so a scale error cancels. Clipping does
# not cancel, which is what ADC_SAT_V and `pic.rig.SaturatedRead` are for.
ADC_REF_V = 2.56
ADC_SAT_V = 0.99 * ADC_REF_V
ADC_BITS = 10

ADC_AVG_N = 16  # full ADC sweeps averaged per firmware reply; must match pic4x4.ino AVG_N

# The measured 0.12 s per read, split into the two halves that scale differently. That split
# is the whole case for the firmware's batched sweep: `ADC_FRAME_S` is real work and buys
# noise, `SERIAL_RTT_S` is pure latency and buys nothing, and until the sweep moved into the
# firmware every averaged frame paid one of each. AVG_N*NUM_ADC_RAW conversions at the
# ATmega2560's default prescaler (13 ADC clocks at 125 kHz = 104 us) is 6.7 ms of the 120,
# so 94 percent of a read was the host waiting on USB.
ADC_FRAME_S = ADC_AVG_N * NUM_ADC_RAW * 104e-6
SERIAL_RTT_S = 0.113  # 0.12 measured, less the conversions above
DEFAULT_TIMEOUT_S = 3.0
DEFAULT_SETTLE_S = 0.5  # thermo-optic settle before a read (mrunal/Setup.ino HEATER_DELAY_MS)

# 1x4 optical input switch (Sercalo) in front of U_IN1..4, on its own serial line. It selects
# one input port at a time, which is what makes a per-port characterization automatable.
SWITCH_BAUD = 9600
# Must match SWITCH_SETTLE_MS in Arduino/pic4x4/pic4x4.ino, which is where the wait actually
# happens: `BoardSwitch` drives the mirror through the firmware and sleeps nothing itself, so
# this reaches the bench only through the cost estimator and through the standalone
# host-driven `OpticalSwitch`. It sat at 1.0 -- mrunal/Setup.ino's SWITCH_DELAY_MS, the same
# unjustified second the firmware was carrying -- which made every duration estimate 7x
# pessimistic, and those estimates now size the laser watchdog. 0.18 and not 0.15 because a
# 100-state capture measured 4.8 s per state at repeats=3, which backs out to 0.24 s per move
# including the serial round trip.
SWITCH_SETTLE_S = 0.18

# Chip TEC. Unlike the 6x6 rig this one has a calibrated cooler, so the substrate sits at
# a fixed temperature instead of wandering -- which is what made drift the dominant error
# there. Hold the chip here and treat a reading outside the band as a stale measurement.
# 27 C, and nothing moves it. This is asserted on every TEC port open -- the Arduino resets
# on connect and forgets whatever the last process set -- so it is the only place the chip's
# operating point actually lives.
#
# 20 C was measured to hold tighter (sd 0.030 at drive 0.68 V) and was the operating point
# for part of 2026-08-27. It stopped being reachable: after several hours the loop railed at
# drive +2.00 V and sat flat at 26.5 C, 6.5 C above target, having slewed 25 -> 20 in 25
# seconds earlier the same evening. The TEC pumps heat into a sink whose own temperature had
# risen, so the achievable delta is measured from the sink and not from room air. 25 C was
# comfortable throughout -- until it was not: by the end of the session the loop could not
# reach 25 C either, railing at +2.00 V and sitting flat at 26.5 C.
#
# 27 C is where this bench actually lives. Measured: 27.00 C, error -0.00, sd 0.012 -- twice
# as steady as 25 C ever was -- at drive -0.02 V, which is to say the chip's own equilibrium
# with the loop holding rather than fighting. Near-zero drive leaves full authority in both
# directions for the ~2 W the heaters dissipate.
#
# 25 C and nothing else. `pic_data/calib.json` was fitted at 25 C, so every heater's phi0 is
# anchored there; operating anywhere else applies a uniform thermo-optic offset to the whole
# mesh that no downstream correction models. This constant and the calibration's `chip_c`
# have to agree, and moving one without retaking the other is what put a 2 C offset under
# every hardware number taken after 2026-08-27 22:30. If the cooler cannot hold 25, that is a
# hardware fault to report, not a constant to edit.
TEC_SETPOINT_C = 25.0
# Was 0.05 C (a ~4 sigma gate on the sd 0.012 hold). Widened to 0.5 C on 2026-09-23: the TEC
# power stage cannot pull the die below ~25.3 C (drive +1.63 V of 2.00 at idle, railed under
# any heater load), so 0.05 blocked every bench run. Restore 0.05 once the TEC holds again.
# 1.0 since 2026-09-25: at 2-3 A the TEC sits at its current limit and holds ~25.7 C at rest,
# limited by the heatsink, not the drive. Tighten once the hot side is cooled better.
TEC_TOLERANCE_C = 1.0
# A reset lands the chip near ambient, and pulling back down runs at ~0.5 C/min. The old
# 60 s gate was sized for holding a setpoint, not for reaching one from cold, and it failed a
# capture that was merely still on its way.
TEC_SETTLE_S = 300.0


@dataclass
class PICConfig:
    port: str | None = None
    baud: int = BAUD_RATE
    num_dac: int = NUM_DAC
    num_adc_raw: int = NUM_ADC_RAW
    out_pds: tuple = OUT_PDS
    voltage_min: float = VOLTAGE_MIN
    voltage_max: float = VOLTAGE_MAX
    timeout_s: float = DEFAULT_TIMEOUT_S
    settle_s: float = DEFAULT_SETTLE_S
    reset_wait_s: float = RESET_WAIT_S


def out_mask(num_adc_raw: int = NUM_ADC_RAW, out_pds=OUT_PDS) -> np.ndarray:
    """Boolean mask selecting the mesh-output photodiodes out of a raw ADC reply."""
    m = np.zeros(num_adc_raw, dtype=bool)
    m[list(out_pds)] = True
    return m


def dac_code(voltage: float, ref_v: float = DAC_REF_V, bits: int = DAC_BITS) -> int:
    """Host volts -> DAC code, the same conversion the firmware does."""
    return int(np.clip(voltage, 0.0, ref_v) * (2**bits - 1) / ref_v)
