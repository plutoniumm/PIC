"""Every printed line, from the server and from every job, as an event.

A line `@ev {json}` (see `pic.log`) is already one: it is parsed and kept whole,

    {"t": 1790316154.88, "src": "fastchar", "job": "20260925-1219-fastchar", "c": "heater",
     "s": "fit", "lvl": "ok", "kind": "heater", "text": "H4:phi2 PD1 ... OK", ...fields}

Any other line is shown as it was printed: {"lvl": "raw", "kind": "raw", "text": line}. A
Python traceback is many lines and one fact, so it becomes a single error event whose text
is the final `SomeError: message` line and whose `detail` is the whole thing.

`kind` is what the Logs tab groups and draws by: laser, progress, wait, or the component.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from pathlib import Path

PREFIX = "@ev "
_ERR = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Tripped|Interrupt)\b(:|$)|^ABORTED\b")
_PROGRESS = {"progress", "prescan", "round"}
_WAIT = {"waiting", "paused"}


def parse(ln: str) -> dict:
    """One printed line -> the event's own fields (no t or src)."""
    if ln.startswith(PREFIX):
        try:
            e = json.loads(ln[len(PREFIX) :])
            c, s, text = e.pop("c"), e.pop("s"), str(e.pop("m", "")).strip()
            f = e.pop("f", None) or {k: e.pop(k) for k in list(e) if k != "lvl"}
        except (ValueError, KeyError, AttributeError, TypeError):
            pass  # a mangled event is still worth seeing, raw
        else:
            kind = "laser" if c == "laser" else "progress" if s in _PROGRESS else c
            kind = "wait" if s in _WAIT else kind
            # fields kept together for display, and also flat for the page's older readers
            flat = {k: v for k, v in f.items() if k not in ("c", "s", "lvl", "text", "kind")}
            return {
                **flat,
                "c": c,
                "s": s,
                "lvl": e.get("lvl", "info"),
                "kind": kind,
                "text": text,
                "fields": f,
            }
    return {"lvl": "raw", "kind": "raw", "text": ln.strip()}


class EventLog:
    def __init__(self, path, keep: int = 3000):
        self.path = Path(path)
        self.buf: deque = deque(maxlen=keep)
        self.lock = threading.Lock()
        self._part: dict = {}  # source -> unfinished line
        self._tb: dict = {}  # source -> traceback lines being gathered
        try:  # the tail of the last session, so the tab is not empty after a restart
            for ln in self.path.read_text().splitlines()[-keep:]:
                self.buf.append(json.loads(ln))
        except (OSError, ValueError):
            pass

    def feed(self, src: str, chunk: str, **tags):
        """Any amount of text from `src`; complete lines become events, each carrying `tags`
        (a job's id, so its own events can be picked out of the shared stream)."""
        with self.lock:
            text = self._part.pop(src, "") + chunk
            *lines, rest = text.split("\n")
            if rest:
                self._part[src] = rest
            for ln in lines:
                self._line(src, ln.rstrip("\r"), tags)

    def _line(self, src, ln, tags):
        tb = self._tb.get(src)
        if ln.startswith("Traceback (most recent call last)") or (
            tb is not None and ln[:1] in (" ", "\t", "")
        ):
            self._tb.setdefault(src, []).append(ln)
            return
        if tb is not None:
            del self._tb[src]
            if not ln.strip() or ln.startswith("During handling"):
                self._tb[src] = tb + [ln]
                return
            if _ERR.search(ln.strip()):  # the line that ends a traceback is the error itself
                err = {"lvl": "error", "kind": "error", "text": ln.strip()}
                return self._emit(src, err, tags, "\n".join(tb + [ln]))
            # a traceback cut off (the process died mid-print): its own event, and this line
            # is an ordinary line again rather than a fake error
            cut = {"lvl": "error", "kind": "error", "text": "traceback cut off"}
            self._emit(src, cut, tags, "\n".join(tb))
        if ln.strip():
            self._emit(src, parse(ln), tags)

    def _emit(self, src, fields, tags, detail=None):
        e = {"t": round(time.time(), 3), "src": src, **tags, **fields}
        if detail:
            e["detail"] = detail
        self.buf.append(e)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(e) + "\n")
        except OSError:
            pass

    def since(self, t: float = 0.0, n: int | None = 500, **match):
        """Events after `t`, the last `n` of them, with every `match` field equal."""
        with self.lock:
            out = [
                e for e in self.buf if e["t"] > t and all(e.get(k) == v for k, v in match.items())
            ]
        return out[-n:] if n else out


def _selftest(tmp):
    log = EventLog(Path(tmp) / "e.jsonl")
    log.feed(
        "job", '@ev {"c":"heater","s":"prescan","lvl":"info","m":"prescan 3/12","k":3,"n":12}\n'
    )
    log.feed(
        "job", '@ev {"c":"laser","s":"on","lvl":"info","m":"LASER ON at floor"}\npar', job="j1"
    )
    log.feed(
        "job", 'tial line\nTraceback (most recent call last):\n  File "x.py", line 1\n', job="j1"
    )
    log.feed(
        "job",
        "    boom()\npic.devices.tec.TECError: chip did not reach 25\n"
        '@ev {"c":"session","s":"paused","lvl":"warn","m":"paused: die at 26"}\n'
        '@ev {"c":"job","s":"wrote","lvl":"ok","m":"wrote x","path":"x"}\n@ev {broken\n',
        job="j1",
    )
    ev = log.since()
    got = [(e["lvl"], e["kind"]) for e in ev]
    assert got == [
        ("info", "progress"),
        ("info", "laser"),
        ("raw", "raw"),
        ("error", "error"),
        ("warn", "wait"),
        ("ok", "job"),
        ("raw", "raw"),
    ], got
    assert ev[0]["k"] == 3 and ev[0]["c"] == "heater" and ev[0]["text"] == "prescan 3/12"
    assert "m" not in ev[0] and ev[5]["path"] == "x"
    assert ev[2]["text"] == "partial line"  # a line split across two writes is one event
    assert ev[3]["text"].startswith("pic.devices.tec.TECError") and "boom()" in ev[3]["detail"]
    assert ev[6]["text"] == "@ev {broken"  # a mangled event is shown, not dropped
    assert [e["kind"] for e in log.since(job="j1")] == [
        "laser",
        "raw",
        "error",
        "wait",
        "job",
        "raw",
    ]
    assert EventLog(Path(tmp) / "e.jsonl").since()[-2]["kind"] == "job"  # survives a restart
    log.feed("srv", 'Traceback (most recent call last):\n  File "y"\n--- ui.py started\n')
    last = log.since()[-2:]
    assert last[0]["text"] == "traceback cut off" and last[1]["lvl"] == "raw", last
    return len(ev)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        print(f"events: {_selftest(d)} events, structured parsed, raw kept, traceback folded")
