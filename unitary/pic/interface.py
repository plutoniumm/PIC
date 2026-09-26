"""Serial interface to the 4x4 board: 18 DAC volts out, 8 mean photodiode volts back,
one round trip per call.

Same line protocol as the 6x6 firmware, so the driver is deliberately the same shape --
what changed is the widths and the mock, which here is a real physical model of the chip
rather than a bag of random fringes.
"""

from __future__ import annotations

import os
import time

import numpy as np

from theory.clements import NMODE
from theory.layout import MIRROR_OF

from .config import (
    ADC_FRAME_S,
    NUM_ADC_RAW,
    NUM_DAC,
    OUT_PDS,
    SERIAL_RTT_S,
    SWITCH_SETTLE_S,
    DAC_HEATER,
    VOLTAGE_MAX_CH,
    PICConfig,
    out_mask,
)


class PICError(RuntimeError):
    pass


def find_port(explicit: str | None = None) -> tuple[str | None, list[str]]:
    """Locate the board. Precedence: explicit argument, ``$PIC4_PORT``, its USB serial number,
    then every USB serial device that is not a known TEC or laser."""
    if explicit:
        return explicit, [explicit]
    env = os.environ.get("PIC4_PORT")
    if env:
        return env, [env]
    from .config import usb_ports

    ports = usb_ports()
    from .config import FTDI_VID

    # an unknown FTDI adapter is an SPD or a laser, never the board (an Arduino)
    cands = [d for d, r, _ in ports if r == "board"] or [
        d for d, r, v in ports if r == "?" and v != FTDI_VID
    ]
    return (cands[0] if cands else None), cands


class PIC:
    """Synchronous driver: one set-and-read round trip per call. A context manager."""

    def __init__(self, config: PICConfig | None = None, **kw):
        self.cfg = config or PICConfig(**kw)
        self.ser = None
        self._caps = None
        self._mask = out_mask(self.cfg.num_adc_raw, self.cfg.out_pds)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def open(self):
        import serial  # lazy so MockPIC needs no pyserial

        self._caps = None  # a different board may answer differently
        port, cands = find_port(self.cfg.port)
        if port is None:
            raise PICError(f"no serial port found; candidates were {cands}")
        self.cfg.port = port
        self.ser = serial.Serial(port, self.cfg.baud, timeout=self.cfg.timeout_s)
        time.sleep(self.cfg.reset_wait_s)  # the board resets on port open
        self.ser.reset_input_buffer()
        return self

    def close(self):
        if self.ser is not None and getattr(self.ser, "is_open", False):
            try:
                self.set_zero()
            finally:
                self.ser.close()
                self.ser = None

    def _prep_dac(self, voltages) -> np.ndarray:
        """Validate and clip before anything reaches the DACs. np.clip does not fix NaN,
        so a non-finite value would otherwise arrive at the firmware as the literal 'nan'
        and drive garbage into a heater."""
        try:
            v = np.asarray(voltages, float).ravel()
        except (ValueError, TypeError) as e:
            raise PICError(f"DAC voltages not numeric ({e})") from None
        if v.size != self.cfg.num_dac:
            raise PICError(f"expected {self.cfg.num_dac} voltages, got {v.size}")
        if not np.all(np.isfinite(v)):
            bad = np.where(~np.isfinite(v))[0][:8].tolist()
            raise PICError(f"non-finite DAC voltage(s) at index {bad}; refusing to send")
        # Per-channel, not the scalar ceiling. The firmware clamps too, and that is what
        # protects the hardware -- but if the host clips at 3 V and the firmware clips a
        # 60R channel to 1.5 V, every fit on that channel is against volts that never
        # landed, and the Vpi it produces is wrong in a way nothing downstream can see.
        v = np.clip(v, self.cfg.voltage_min, np.asarray(VOLTAGE_MAX_CH, float))
        # Bonded channels are shorted on the board, so a pair is one heater at one
        # voltage. The primary channel is that heater's address -- `REACHABLE_DACS` and
        # `ACTIVE_IDX` contain only primaries, so no caller ever names a partner -- and the
        # partner follows it here, at the last point before the wire. Mirroring rather than
        # refusing because a caller that leaves a partner at 0 is not disagreeing with
        # itself, it simply does not know about the bond, which is a board fact.
        #
        # A partner set to something else NONZERO is a different matter: that is a caller
        # with an opinion about a channel it cannot address, and it is refused, because
        # silently overwriting it would hide the bug. Two DAC81416 outputs tied together and
        # held apart drive current into each other; the firmware mirrors too, as a backstop.
        for partner, primary in MIRROR_OF.items():
            if v[partner] and abs(v[partner] - v[primary]) > 1e-9:
                raise PICError(
                    f"ch{partner} is bonded to ch{primary} (heater {DAC_HEATER[primary]}) "
                    f"but was given {v[partner]:.3f} V against {v[primary]:.3f} V; "
                    f"address the pair through ch{primary} only"
                )
            v[partner] = v[primary]
        return v

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        """Apply the DAC voltages, return all `num_adc_raw` photodiode volts. Retries a
        transient serial timeout so one dropped reply does not kill a long run."""
        v = self._prep_dac(voltages)
        line = ",".join(f"{x:.3f}" for x in v) + "\n"
        for attempt in range(retries + 1):
            self.ser.reset_input_buffer()
            self.ser.write(line.encode())
            try:
                return self._read_floats(self.cfg.num_adc_raw)
            except PICError:
                if attempt == retries:
                    raise
                time.sleep(0.15)

    def measure(self, voltages, settle_s: float | None = None) -> np.ndarray:
        """Return the four mesh-output photodiode volts, after a thermal settle."""
        settle_s = self.cfg.settle_s if settle_s is None else settle_s
        if settle_s and settle_s > 0:
            self.measure_raw(voltages)
            time.sleep(settle_s)
        return self.measure_raw(voltages)[self._mask]

    def select_port(self, port: int) -> int:
        """Route the laser to input `port` (0-indexed) via the firmware's `P<1..4>`.

        `port = -1` sends `P0`, the switch's open channel, which is dark.

        The Sercalo hangs off the board's Serial1, not off a port of its own, so port
        selection is a board command. The firmware waits out the mechanical settle and
        echoes `PORT <n>`; a missing or disagreeing echo means the mirror did not move,
        which would otherwise show up as a whole sweep silently taken at the wrong port."""
        self.ser.reset_input_buffer()
        self.ser.write(f"P{int(port) + 1}\n".encode())
        deadline = time.time() + self.cfg.timeout_s + 2.0
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if raw.startswith("PORT"):
                got = int(raw.split()[-1]) - 1
                if got != int(port):
                    raise PICError(f"switch went to port {got}, asked for {port}")
                return got
            if raw.startswith("ERR"):
                raise PICError(f"firmware refused port {port}: {raw!r}")
        raise PICError(f"no PORT echo for port {port} from {self.cfg.port}")

    def capabilities(self, refresh: bool = False) -> dict:
        """What the firmware can do beyond the frozen line protocol. `{}` means "only the
        protocol", which is a working board and not an error.

        Asked, not assumed, and asked separately from the `V` clamp query: a board can carry
        that table and still predate the batched sweep, so one capability does not imply the
        other. A firmware without `C` parses it as a DAC line -- zeroing every channel -- and
        replies with an ADC frame, which is the shape detected below. That is why `Rig.open`
        asks at open time, before anything is driven, and nowhere else."""
        if self._caps is not None and not refresh:
            return self._caps
        ser = self.ser
        if ser is None or isinstance(ser, str) or not hasattr(ser, "write"):
            self._caps = {}
            return self._caps
        ser.reset_input_buffer()
        ser.write(b"C\n")
        caps, deadline = {}, time.time() + self.cfg.timeout_s
        while time.time() < deadline:
            raw = ser.readline().decode("utf-8", "ignore").strip()
            if not raw:
                continue
            if raw.startswith("CAP"):
                for tok in raw.split()[1:]:
                    k, _, v = tok.partition("=")
                    try:
                        caps[k] = int(v)
                    except ValueError:
                        caps[k] = v
                break
            if len(raw.split(",")) == self.cfg.num_adc_raw:
                break  # an ADC frame: firmware from before `C`
        self._caps = caps
        return caps

    @property
    def batched(self) -> bool:
        return bool(self.capabilities().get("sweep"))

    def sweep_raw(
        self, cycles: int = 1, reads: int = 1, timeout_s: float | None = None
    ) -> np.ndarray:
        """A whole four-port sweep in ONE round trip -> raw volts `T[pd, port]`.

        The heaters are not written here and the settle is not paid here; the caller owns
        both, exactly as `pic.normalise.sweep` does, so this is a drop-in for that loop and
        not a different measurement. What it removes is 4*cycles*reads round trips and
        4*(cycles*reads - cycles) mirror settles.

        `reads` are frames averaged back to back at one mirror position -- 7 ms apart, so
        they average detector and ADC noise and nothing slower -- and `cycles` are complete
        visits to all four ports, which spreads the frames over the same seconds the host's
        `repeats` loop used to. They are not interchangeable; see `pic.acquisition.
        batch_sweep_seconds` for what each costs."""
        if not self.batched:
            raise PICError(
                "this firmware has no batched sweep; reflash "
                "Arduino/pic4x4/pic4x4.ino or use pic.normalise.sweep"
            )
        cycles, reads = max(1, int(cycles)), max(1, int(reads))
        caps = self.capabilities()
        lim = (
            caps.get("maxcycles", cycles),
            caps.get("maxreads", reads),
            caps.get("maxframes", cycles * reads),
        )
        if cycles > lim[0] or reads > lim[1] or cycles * reads > lim[2]:
            raise PICError(
                f"sweep {cycles}x{reads} exceeds the firmware's limits "
                f"(maxcycles {lim[0]}, maxreads {lim[1]}, maxframes {lim[2]})"
            )
        n = NMODE * self.cfg.num_adc_raw
        # The board is busy for the whole sweep and answers nothing until it is done, so the
        # read deadline has to be the sweep's own duration and not `timeout_s`. A default
        # 3 s timeout silently truncates any sweep past ~8 frames.
        if timeout_s is None:
            timeout_s = (
                3.0
                * (cycles * NMODE * (SWITCH_SETTLE_S + 0.02) + cycles * reads * NMODE * ADC_FRAME_S)
                + self.cfg.timeout_s
            )
        self.ser.reset_input_buffer()
        self.ser.write(f"S{cycles},{reads}\n".encode())
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if not raw:
                continue
            if raw.startswith("ERR"):
                raise PICError(f"firmware refused sweep {cycles}x{reads}: {raw!r}")
            if not raw.startswith("SWEEP"):
                continue
            parts = raw.split(None, 1)[1].split(",") if " " in raw else []
            if len(parts) != n:
                raise PICError(f"sweep returned {len(parts)} values, expected {n}")
            # port-major on the wire, as the mirror visits and as a session file stores it
            return np.array([float(p) for p in parts]).reshape(NMODE, self.cfg.num_adc_raw).T
        raise PICError(f"timeout waiting for a {cycles}x{reads} sweep from {self.cfg.port}")

    def set_zero(self):
        self.measure_raw(np.zeros(self.cfg.num_dac))

    def _read_floats(self, n: int) -> np.ndarray:
        deadline = time.time() + self.cfg.timeout_s
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "ignore").strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) == n:
                try:
                    return np.array([float(p) for p in parts])
                except ValueError:
                    continue  # startup banner or a partial line
        raise PICError(f"timeout waiting for {n} values from {self.cfg.port}")


# What a firmware carrying the batched sweep reports, mirrored here so a `--mock` run walks
# the same branch the bench does. Must match Arduino/pic4x4/pic4x4.ino.
MOCK_CAPS = {
    "sweep": 1,
    "ports": NMODE,
    "pins": NUM_ADC_RAW,
    "dac": NUM_DAC,
    "avg": 16,
    "maxcycles": 16,
    "maxreads": 64,
    "maxframes": 64,
    "switchms": 150,
}


def emulate_sweep(board, cycles: int, reads: int) -> np.ndarray:
    """The firmware's `S` command, done in Python against a simulated board.

    Same loop order as `readSweep` -- cycles outside, ports inside, `reads` frames averaged
    without moving the mirror -- so the two averaging axes stay distinguishable in a mock
    run. On a simulator whose only noise is per-read they are interchangeable and the
    distinction costs nothing; on the bench they are not, which is why the shape is kept."""
    acc = np.zeros((board.cfg.num_adc_raw, NMODE))
    v = board._last_v
    for _ in range(max(1, int(cycles))):
        for k in range(NMODE):
            if board.switch is not None:
                board.switch.select(k)
            acc[:, k] += np.mean([board.measure_raw(v) for _ in range(max(1, int(reads)))], axis=0)
    return acc / max(1, int(cycles))


class MockPIC(PIC):
    """Hardware-free stand-in backed by a forward function ``f(v18) -> raw_pds``."""

    def __init__(
        self, forward=None, noise: float = 3e-4, config: PICConfig | None = None, switch=None, **kw
    ):
        super().__init__(config, **kw)
        self.forward = twin_forward(switch=switch) if forward is None else forward
        self.noise = noise
        self.switch = switch
        self._rng = np.random.default_rng(0)
        self._last_v = np.zeros(self.cfg.num_dac)

    def open(self):
        self.ser = "mock"
        return self

    def close(self):
        self.ser = None

    def capabilities(self, refresh: bool = False) -> dict:
        return dict(MOCK_CAPS)

    def sweep_raw(self, cycles: int = 1, reads: int = 1, timeout_s=None) -> np.ndarray:
        return emulate_sweep(self, cycles, reads)

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        v = self._prep_dac(voltages)
        self._last_v = v  # the DACs hold a state; the sweep reads it back
        y = np.asarray(self.forward(v), float).ravel()
        if self.noise:
            y = y + self._rng.normal(0.0, self.noise, y.shape)
        return np.clip(y, 0.0, None)


# One SPD read: 10 PHOTON_COUNT frames. Poisson sets the noise, sigma_y = sqrt(y M t + D t)/(M t)
# on the 0..1 scale: at the bench's max M ~ 900/s and dark D ~ 17/s that is 0.075 at full
# scale and 0.026 at y = 0.1, against 0.2 s a read -- a four-port sweep stays under a second,
# and a caller that needs less noise averages repeats, as it did with the photodiodes.
SPD_READ_S = 0.2
# The mock's detectors, one per output, each with its own dark and max (counts/s) as four real
# ones would have. Bright enough that one 20 ms frame is a usable read, so a mock pipeline
# runs in seconds; `spd_board` reads the mock for one frame, the bench for SPD_READ_S.
MOCK_SPDS = {
    "MOCKSPD0": (0, 25.0, 60e3),
    "MOCKSPD1": (1, 17.0, 90e3),
    "MOCKSPD2": (2, 60.0, 120e3),
    "MOCKSPD3": (3, 12.0, 45e3),
}


class MissingOutputs(PICError, ValueError):
    """The detectors in hand do not cover the outputs a computation needs. A ValueError too,
    so a front end shows it as the refusal it is rather than as a crash."""


class SPDBoard:
    """The board with its analog readout replaced by single-photon detectors: the seam.

    Heaters, switch and everything else still go to `board`. A read comes back in the same
    shape the photodiodes gave -- one value per ADC slot -- as (rate - dark) / (max - dark)
    per detector, NaN in a slot with no detector on it. Every caller that reads through the
    board, the rig, `settled_read` or the session's emission check gets SPD values without
    knowing about SPDs; the ones that need all four outputs call `need_outputs`."""

    batched = False  # the firmware sweep reads the ADC, not the SPDs

    def __init__(self, board, spds, slots: dict, law: dict, seconds: float = SPD_READ_S):
        self.board, self.spds, self.slots, self.law, self.seconds = board, spds, slots, law, seconds
        self._I = np.zeros(NMODE)  # the mock chip's intensity, for mock detectors to count

    def __getattr__(self, k):
        if k == "board":  # not yet set: no recursion through the delegate
            raise AttributeError(k)
        return getattr(self.board, k)

    measure = PIC.measure

    @property
    def outputs(self) -> list[int]:
        return sorted(self.slots.values())

    def open(self):
        self.board.open()
        self.spds.open()
        return self

    def close(self):
        try:
            self.board.close()
        finally:
            self.spds.close()

    def sweep_raw(self, *a, **kw):
        raise PICError("an SPD readout has no firmware sweep; use pic.normalise.sweep")

    def measure_raw(self, voltages, retries: int = 3) -> np.ndarray:
        self.board.measure_raw(voltages, retries)
        if hasattr(self.board, "forward"):  # mock: what light the model puts on each output
            f = self.board.forward
            raw = np.asarray(f(self.board._last_v), float)[list(OUT_PDS)]
            cal = getattr(f, "calib", None)
            self._I = raw if cal is None else cal.to_intensity(raw)
        return self.read()

    def read(self) -> np.ndarray:
        """Counts from now for `seconds`: frames from before the call are dropped by `count`,
        so a read never mixes in light from the heater state before it."""
        from spd.array import normalise

        y = np.full(self.cfg.num_adc_raw, np.nan)
        for sid, c in self.spds.count(self.seconds).items():
            y[OUT_PDS[self.slots[sid]]] = normalise(c.rate, self.law[sid])
        return y


def spd_board(board, mock: bool) -> SPDBoard:
    """`board` behind the SPDs in pic_data/spd_map.json. On the bench every SPD on USB must be
    mapped and have a dark and max stored; the mock puts `MOCK_SPDS` on their outputs,
    counting the mock chip's own light, and never reads pic_data."""
    from spd.array import SPD, SPDs, check_law, discover, load_max
    from spd.vega import FRAME_S, SPDError

    from .config import SPD_MAP_PATH, spd_map

    if mock:
        slots = {i: k for i, (k, _, _) in MOCK_SPDS.items()}
        law = {i: (d, m) for i, (_, d, m) in MOCK_SPDS.items()}
        sb = SPDBoard(board, None, slots, law, seconds=FRAME_S)
        rate = lambda i: lambda: law[i][0] + (law[i][1] - law[i][0]) * float(sb._I[slots[i]])
        sb.spds = SPDs(
            [SPD.mock(i, seed=k, count_hz=rate(i)) for i, k in slots.items()],
            on_text=lambda sid, t: None,
        )
        return sb
    ids, where = discover(), spd_map()
    if not ids:
        raise MissingOutputs("SPD mode, but no SPD on USB; plug one in or switch to PD in Settings")
    bad = [i for i in ids if i not in where]
    if bad:
        raise MissingOutputs(
            f"SPD {', '.join(bad)} is on USB but not in {SPD_MAP_PATH}; "
            "`python -m spd assign <id> PD<n>` with the output it sits on"
        )
    try:
        law = check_law(ids, load_max())
    except SPDError as e:
        raise MissingOutputs(str(e)) from None
    return SPDBoard(board, SPDs(ids, on_text=lambda sid, t: None), {i: where[i] for i in ids}, law)


def need_outputs(board, what: str, outputs=range(NMODE)):
    """Refuse `what` unless the detectors cover `outputs`. The photodiode board always does.
    Takes the board or anything holding it as `.board` (the rig)."""
    board = board if isinstance(board, SPDBoard) else getattr(board, "board", None)
    if not isinstance(board, SPDBoard):
        return
    have = board.outputs
    miss = [k for k in outputs if k not in have]
    if miss:
        on = ", ".join(f"PD{k}" for k in have) or "no output"
        raise MissingOutputs(
            f"{what} needs outputs {', '.join(f'PD{k}' for k in outputs)}; the SPDs cover {on} only"
        )


def twin_forward(calib=None, error=None, x=None, seed: int = 0, dark: float = 0.01, switch=None):
    """A forward for :class:`MockPIC` that is the actual physics: heater volts -> phases
    through a calibration, phases -> field through the twin, intensity -> photodiode
    volts through the readout scale.

    Defaults to a *fabricated* instance (sampled Vpi, phi0 and coupler error, both seeded)
    rather than the nominal one, so a characterization run against the mock has something
    real to find instead of recovering the constants it was handed.

    Pass `switch` and the injected port follows whatever the optical switch is set to, so a
    per-port sweep against the mock exercises the same loop the bench will."""
    import torch

    from theory.calib import Calibration
    from theory.clements import NMODE as twin_nmode
    from theory.twin import MeshError, Twin

    calib = Calibration.sample(seed=seed) if calib is None else calib
    twin = Twin(MeshError.sample(seed=seed) if error is None else error)
    xt = None if x is None else torch.as_tensor(np.asarray(x), dtype=twin.dtype)

    def _input():
        if xt is not None or switch is None:
            return xt
        e = np.zeros(twin_nmode, complex)
        e[switch.selected or 0] = 1.0
        return torch.as_tensor(e, dtype=twin.dtype)

    def forward(v):
        ph = calib.phases(np.asarray(v, float))
        inten = twin.outputs(torch.as_tensor(ph, dtype=torch.float32), _input()).detach().numpy()
        raw = np.full(NUM_ADC_RAW, dark)
        raw[list(OUT_PDS)] = calib.to_volts_response(inten)
        return raw

    forward.calib, forward.twin = calib, twin
    return forward


assert NUM_DAC == 16  # the firmware sketch and pic.layout must agree on the width
