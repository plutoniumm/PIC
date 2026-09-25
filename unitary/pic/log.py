"""One structured event per line: `@ev {"c":…,"s":…,"lvl":…,"m":…,…fields}`.

Every job is a subprocess whose stdout the UI server reads, so a printed line is the
transport. The prefix lets `events.py` parse these as JSON and show any other line raw;
nothing else about a module has to know the UI exists. `m` stays a sentence a person can
read in a terminal, the fields are what the server keys off.

Each component is a small state machine; `s` is the state it just entered.

    component  states
    laser      off -> connecting -> on -> ramping -> set -> stopping -> off;
               interlock | unreachable (lit = the last of on|off is on)
    tec        stable (session gate passed); setpoint (retargeted by the UI)
    board      reserved: its prints are still raw
    switch     reserved: its prints are still raw
    session    note (calib temperature) -> waiting (TEC gate) -> lit (emitted true|false)
               -> paused <-> resumed -> closed; tripped (watchdog); keepalive_failed
    job        started -> summary (k/n) -> wrote -> done | failed | interrupted
    heater     prescan (k/n) -> queued (pd, port chosen) -> sweeping -> fit (ok true|false);
               recal (was_pi, phi0_pi, how; k/n)
    pd         noise
    table      progress (k/n) -> wrote | discarded (mock)
    dpnn       round (k/n) -> points -> fit (r2, k/n) -> done | interrupted; partial (trip)
    sync       done | declined | failed
    server     started, note, laser, tec

Levels are info | ok | warn | error. Progress events carry `k` and `n`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import sys
import math

COMPONENTS = frozenset(
    "laser tec board switch session job heater pd table dpnn sync server".split()
)
LEVELS = ("info", "ok", "warn", "error")
PREFIX = "@ev "


def _clean(v):
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:  # numpy scalar
        v = v.item()
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else None  # JSON has no NaN
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
        return [_clean(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    return v


@dataclass
class Event:
    """The one shape of every event: who, which state, how bad, a short message, and the data
    as fields -- kept apart from the message so the Logs tab can show numbers as numbers."""

    component: str
    state: str
    msg: str = ""
    lvl: str = "info"
    fields: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.component in COMPONENTS, self.component
        assert self.lvl in LEVELS, self.lvl

    def wire(self) -> str:
        e = {"c": self.component, "s": self.state, "lvl": self.lvl, "m": self.msg}
        if self.fields:
            e["f"] = {k: _clean(v) for k, v in self.fields.items()}
        return PREFIX + json.dumps(e, separators=(",", ":"), default=str)


def ev(component: str, state: str, msg: str = "", lvl: str = "info", **fields):
    line = Event(component, state, msg, lvl, fields).wire()
    # a person at a terminal reads a sentence; a pipe (the UI's) reads the JSON
    isatty = getattr(sys.stdout, "isatty", lambda: False)
    print(human(line) if isatty() else line, flush=True)


def human(line: str) -> str:
    """An `@ev` line as a sentence for a terminal; anything else unchanged."""
    if not line.startswith(PREFIX):
        return line
    try:
        e = json.loads(line[len(PREFIX) :])
    except ValueError:
        return line
    f = e.get("f") or {k: v for k, v in e.items() if k not in ("c", "s", "lvl", "m")}
    extra = " ".join(f"{k}={v}" for k, v in f.items())
    return f"[{e.get('c')}] {e.get('s')}: {e.get('m', '')}" + (f"  ({extra})" if extra else "")


def _selftest():
    import contextlib
    import io

    import numpy as np

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev("heater", "fit", "H1 ok", "ok", pad="H1", vpi=np.float64(5.2), k=np.int64(3), x=[np.nan])
    ln = buf.getvalue()
    assert ln.startswith(PREFIX) and ln.count("\n") == 1, ln
    e = json.loads(ln[len(PREFIX) :])
    assert e == {
        "c": "heater",
        "s": "fit",
        "lvl": "ok",
        "m": "H1 ok",
        "f": {"pad": "H1", "vpi": 5.2, "k": 3, "x": [None]},  # data apart from the message
    }, e
    for bad in (("nope", "x"), ("job", "x", "", "loud")):  # an unknown component or level
        try:
            ev(*bad)
        except AssertionError:
            continue
        raise RuntimeError(f"accepted {bad}")
    return e


if __name__ == "__main__":
    print(f"log: {_selftest()}")
