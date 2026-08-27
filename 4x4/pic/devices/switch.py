"""The 1x4 optical input switch in front of U_IN1..4.

A Sercalo SC/mSC 1xN on its own serial line, spoken to with `SET <1..4>` at 9600 baud
(`mrunal/Setup.ino` drives it from the Arduino's Serial1; the host can drive it directly
on a USB adapter instead). `SET 0` opens the path, `POS` reports where the mirror really is,
and every command is answered -- see `mrunal/Sercalo Optical Switch.pdf` sections 7.1,
10.8 and 10.9.

This is a bigger deal than a convenience. Characterizing a mesh through one input port
leaves phases unobservable -- an external phase on a first-column MZI is a global phase when
only that port is lit, so the heater modulates nothing however long you sweep it. Being able
to select the port under program control turns a four-position fibre re-plug into another
loop index, and takes heater identification from 10 of 12 to 12 of 12 on the mock instrument.

What the switch cannot do is light two ports at once. The remaining case, a superposition
across two inputs, needs an external splitter on the bench.

It costs 3.1 dB of insertion loss (`mrunal/Loss_Analysis (1).pdf`), which is in the link
budget and not a problem: the received signal sits about 39 dB above the PD-TIA noise floor.
"""

from __future__ import annotations

import time

from ..config import SWITCH_BAUD, SWITCH_SETTLE_S
from theory.clements import NMODE


class SwitchError(RuntimeError):
    pass


class OpticalSwitch:
    """Selects which of the four input ports is lit. Ports are 0-indexed here and sent
    1-indexed, matching the unit's own numbering."""

    def __init__(self, port: str, baud: int = SWITCH_BAUD, timeout_s: float = 2.0,
                 settle_s: float = SWITCH_SETTLE_S):
        self.port, self.baud, self.timeout_s, self.settle_s = port, baud, timeout_s, settle_s
        self.ser = None
        self._sel = None

    @property
    def selected(self) -> int | None:
        return self._sel

    def open(self):
        import serial

        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _command(self, line: str) -> str:
        """Send one ASCII command and return its reply.

        The reply is not optional. Section 7.1 of the datasheet: the device answers every
        command, and the caller must not send another until the current one has finished.
        Fire-and-forget leaves the answer in the input buffer, where the *next* command
        reads it as its own -- so a run drifts one reply behind itself and every port
        confirmation after the first is a lie."""
        if self.ser is None:
            raise SwitchError("switch not open")
        self.ser.reset_input_buffer()
        self.ser.write(f"{line}\n".encode())
        reply = self.ser.readline().decode("utf-8", "ignore").strip()
        if not reply:
            raise SwitchError(f"no reply to {line!r} from the switch on {self.port}")
        return reply

    def select(self, port: int):
        """Route the input to `port` (0..3), confirm it, and wait out the mechanical settle.

        The unit numbers its channels from 1 and reserves 0 for "route nowhere"; `dark`
        sends that. Everything above the driver counts input ports from 0, like the mesh
        rails, so the +1 lives here and only here."""
        port = int(port)
        if not 0 <= port < NMODE:
            raise SwitchError(f"input port {port} outside 0..{NMODE - 1}")
        reply = self._command(f"SET {port + 1}")
        if reply.split()[-1] != str(port + 1):
            raise SwitchError(f"switch refused port {port + 1}: {reply!r}")
        time.sleep(self.settle_s)
        self._sel = port
        return port

    def dark(self):
        """Open the optical path -- channel 0 routes the common port to nothing. The
        honest way to take a dark reading without touching the laser."""
        self._command("SET 0")
        time.sleep(self.settle_s)
        self._sel = None
        return None

    def position(self) -> int | None:
        """Ask the unit where it actually is, rather than where we last told it to go."""
        p = int(self._command("POS").split()[-1])
        return None if p == 0 else p - 1

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def status(self) -> dict:
        return {"port": self._sel, "open": self.ser is not None}


class MockSwitch(OpticalSwitch):
    """Hardware-free stand-in. Settles instantly and remembers the selection, which is all
    the mock forward needs in order to inject through the right column of U."""

    def __init__(self, settle_s: float = 0.0, **kw):
        super().__init__(port="mock", settle_s=settle_s, **kw)

    def open(self):
        self.ser = "mock"
        self._sel = 0
        return self

    def close(self):
        self.ser = None

    def select(self, port: int):
        port = int(port)
        if not 0 <= port < NMODE:
            raise SwitchError(f"input port {port} outside 0..{NMODE - 1}")
        self._sel = port
        return port

    def dark(self):
        self._sel = None
        return None

    def position(self):
        return self._sel


class BoardSwitch(OpticalSwitch):
    """The switch as this rig has it: no serial port of its own, driven from the board's
    Serial1 by the firmware. Selection is therefore a board command, so the board must be
    open before this can move -- `Rig.open` attaches it."""

    def __init__(self, board=None, **kw):
        kw.pop("settle_s", None)
        super().__init__(port="board", settle_s=0.0, **kw)
        self.board = board

    def attach(self, board):
        self.board = board
        self.ser = "board"
        return self

    def open(self):
        return self

    def close(self):
        self.ser = None

    def select(self, port: int):
        port = int(port)
        if not 0 <= port < NMODE:
            raise SwitchError(f"input port {port} outside 0..{NMODE - 1}")
        if self.board is None:
            raise SwitchError("the switch is driven by the board and no board is open yet")
        self._sel = self.board.select_port(port)
        return self._sel

    def dark(self):
        """Route the common port to nothing -- an optical zero that leaves the laser and
        the photodiode baselines exactly where they are, unlike turning the laser off."""
        if self.board is None:
            raise SwitchError("the switch is driven by the board and no board is open yet")
        self.board.select_port(-1)
        self._sel = None
        return None

    def position(self):
        return self._sel


class NoSwitch(OpticalSwitch):
    """A bench with the fibre plugged straight into one port. `select` refuses anything but
    that port, loudly, rather than letting a per-port sweep silently repeat itself."""

    def __init__(self, port_index: int = 0, **kw):
        super().__init__(port="none", **kw)
        self._sel = int(port_index)

    def open(self):
        self.ser = "none"
        return self

    def close(self):
        pass

    def select(self, port: int):
        if int(port) != self._sel:
            raise SwitchError(f"no optical switch on this rig; the fibre is in port "
                              f"{self._sel} and moving it is a manual step")
        return self._sel

    def dark(self):
        raise SwitchError("no optical switch on this rig; turn the laser down instead")

    def position(self):
        return self._sel


def make_switch(spec, **kw):
    """``'mock'|'none'|'hw'|<port>`` -> a switch; an instance passes through unchanged."""
    if not isinstance(spec, str):
        return spec
    if spec == "mock":
        return MockSwitch(**kw)
    if spec in ("none", "off"):
        return NoSwitch(**kw)
    if spec in ("hw", "board"):
        return BoardSwitch(**kw)
    return OpticalSwitch(spec, **kw)
