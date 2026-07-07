"""A tiny plaintext language to drive the PIC's DAC heaters (and record the PDs).

Run with `python run.py <file.pic>`. A file is one or more *blocks* separated by a
line of `---` (or `done`); each block builds a 64-channel DAC pattern, then pulses it
on the hardware `iters` times every `loop` seconds, recording the 14 photodiodes per
pulse to one CSV. Blocks run top to bottom, so several experiments live in one file.

Grammar -- one statement per line, or several comma-separated on a line:

  V_5 = 2                      set DAC channel 5 to 2 V      (or: set V_5 to 2)
  V_5 = 2, V_9 = 4             several at once
  for H in heaters: V = 0.05*H set every heater; bare V means V_H, H is the index
  loop = 4                     pulse period, seconds         (or: pulse every 4s)
  iters = 10                   number of pulses              (or: repeat 10 times)
  settle = 0.5                 extra dwell before each read, seconds (optional)
  name warmup                  CSV name for this block       (or: name = warmup)
  sweep V_5 from 0 to 4 step 0.5   run the block once per value, one CSV each
  ---   /   done               end of block
  # comment                    line comment (to end of line)
  /* comment */                block comment (spans lines; may also sit inline)

Collections (for / sweep ... in): heaters | channels | 0..63 | 0..63 step 2 | [1,3,5]
Values are volts; an optional `V`/`volt` unit is allowed (`V_5 = 2 V`). Phase units
(`rad`, `deg`) are not wired yet -- they need the heater->phase map, so they warn.

The output is forward-looking: the 14 ADC columns are only meaningful once the
photodiodes are wired; until then they record whatever is floating.
"""

from __future__ import annotations
import os
import re
import signal
import sys
import threading
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pic import PIC, MockPIC, find_port, NUM_DAC, NUM_ADC_RAW, VOLTAGE_MIN

VMAX = 5.0  # DAC reference ceiling; host clip (firmware converts on a 0-5 V scale)
SETTING_KEYS = {"loop", "iters", "settle"}
UNIT_VOLT = {"v", "volt", "volts"}
UNIT_PHASE = {"rad", "radian", "radians", "deg", "degree", "degrees", "phase"}
SAFE = {"abs": abs, "min": min, "max": max, "round": round,
        "int": int, "float": float, "pi": np.pi}

_PAIR = re.compile(r"([A-Za-z_]\w*)\s*=\s*(.+?)(?=\s+[A-Za-z_]\w*\s*=|$)")
_FOR = re.compile(r"^for\s+(?:each\s+)?(\w+)\s+in\s+(.+?)\s*:\s*(.*)$", re.I)
_SWEEP = re.compile(r"^sweep\s+(\S+)\s+(.+)$", re.I)
_EVERY = re.compile(r"^(?:pulse\s+)?every\s+([\d.]+)\s*s(?:ec(?:onds?)?)?\.?$", re.I)
_REPEAT = re.compile(r"^repeat\s+(\d+)(?:\s+times?)?\.?$", re.I)
_SET = re.compile(r"^set\s+(.+?)\s+to\s+(.+)$", re.I)
_SEP = re.compile(r"^(?:-{3,}|done|end)$", re.I)
_CHAN = re.compile(r"^[Vv]_(\d+)$")
_UNIT = re.compile(r"\s+([A-Za-z]+)\s*$")
_RANGE = re.compile(r"^(\S+)\s*\.\.\s*(\S+)(?:\s+step\s+(\S+))?$")
_FROMTO = re.compile(r"^from\s+(\S+)\s+to\s+(\S+)(?:\s+step\s+(\S+))?$", re.I)

_ABORT = threading.Event()  # set by Ctrl-C or a typed kill word -> stop now + zero DACs
_KILL_WORDS = {"exit", "quit", "q", "stop"}


class ScriptError(Exception):
    pass


def _ev(expr, scope=None):
    env = dict(SAFE, **(scope or {}))
    return eval(expr.strip(), {"__builtins__": {}}, env)  # noqa: S307 (trusted local script)


def _split_unit(rhs):
    """Pull a trailing unit word off an expression: '2 V' -> ('2', 'v'). A bare
    trailing variable like the `H` in '4*H' is not a unit, so it is left alone."""
    m = _UNIT.search(rhs)
    if not m:
        return rhs, None
    u = m.group(1).lower()
    if u in UNIT_VOLT or u in UNIT_PHASE:
        return rhs[: m.start()], u
    return rhs, None


def _frange(a, b, step):
    """Inclusive numeric range, a..b stepping by `step` (defaults to 1)."""
    step = step or 1
    n = int(round((b - a) / step)) if step else 0
    return [round(a + i * step, 10) for i in range(n + 1)]


def parse_collection(s):
    s = s.strip()
    if s.lower() in ("heaters", "channels", "all"):
        return list(range(NUM_DAC))
    if s.startswith("["):
        return [_ev(x) for x in s.strip("[] ").split(",") if x.strip()]
    m = _RANGE.match(s)
    if m:
        a, b = _ev(m.group(1)), _ev(m.group(2))
        st = _ev(m.group(3)) if m.group(3) else 1
        return _frange(a, b, st)
    raise ScriptError(f"can't read collection: {s!r}")


def _channel(token):
    m = _CHAN.match(token.strip())
    if not m:
        raise ScriptError(f"expected a channel like V_5, got {token!r}")
    return int(m.group(1))


class Block:
    def __init__(self):
        self.assign = {}   # channel -> volts (the base configuration)
        self.settings = {}  # loop / iters / settle
        self.name = None
        self.sweeps = []   # (channel, [values], label)

    def _set_pair(self, key, rhs, scope, in_for):
        k = key.lower()
        if k in SETTING_KEYS:
            self.settings[k] = int(_ev(rhs)) if k == "iters" else float(_ev(rhs))
            return
        expr, unit = _split_unit(rhs)
        if unit in UNIT_PHASE:
            print(f"  ! {key}: '{unit}' phase units not wired yet "
                  f"(needs heater->phase map); treating as volts")
        val = float(_ev(expr, scope))
        if k == "v":
            if not in_for or not scope:
                raise ScriptError("bare 'V' is only valid inside a 'for' loop; "
                                  "use V_<n> at the top level")
            self.assign[int(scope[list(scope)[-1]])] = val  # innermost loop index
        elif _CHAN.match(key):
            self.assign[_channel(key)] = val
        else:
            print(f"  ! ignoring unknown setting {key!r}")

    def apply_pairs(self, text, scope=None, in_for=False):
        for key, rhs in _PAIR.findall(text):
            self._set_pair(key, rhs.rstrip(", ").strip(), scope, in_for)

    def runs(self):
        """Expand sweeps into concrete runs: (name, vec[64], settings)."""
        base = self.name or "block"
        dims = self.sweeps
        out = []
        for combo in product(*[vals for _, vals, _ in dims]):
            a = dict(self.assign)
            parts = []
            for (ch, _, label), v in zip(dims, combo):
                a[ch] = v
                parts.append(f"{label}={v:g}")
            name = base + ("_" + "_".join(parts) if parts else "")
            out.append((name, self._vec(a), self.settings))
        return out

    def _vec(self, assign):
        v = np.zeros(NUM_DAC)
        for ch, val in assign.items():
            if not 0 <= ch < NUM_DAC:
                print(f"  ! channel {ch} out of range 0..{NUM_DAC - 1}; skipped")
                continue
            c = float(np.clip(val, VOLTAGE_MIN, VMAX))
            if c != val:
                print(f"  ! V_{ch}={val} clipped to {c} V")
            v[ch] = c
        return v


def _strip_block_comments(text):
    """Drop /* ... */ block comments, preserving line numbers for error messages."""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def parse(text):
    """Split into indentation-aware rows, then consume statements into blocks."""
    rows = []  # (indent, stripped_line, lineno)
    for i, raw in enumerate(_strip_block_comments(text).splitlines(), 1):
        body = raw.split("#", 1)[0].expandtabs(4)
        if not body.strip():
            continue
        rows.append((len(body) - len(body.lstrip(" ")), body.strip(), i))

    blocks, blk, idx = [], Block(), 0
    while idx < len(rows):
        indent, line, lineno = rows[idx]
        if indent == 0 and _SEP.match(line):
            blocks.append(blk)
            blk = Block()
            idx += 1
            continue
        try:
            idx = _consume(rows, idx, blk, None)
        except ScriptError:
            raise
        except Exception as e:  # noqa: BLE001 -- surface any parse failure with its line
            raise ScriptError(f"line {lineno}: {line!r}: {e}") from e
    blocks.append(blk)
    return [b for b in blocks if b.assign or b.sweeps]


def _consume(rows, idx, blk, scope):
    """Process one statement -- which may be an indented `for` suite -- and return
    the index of the next statement. Recurses for the suite body and nested loops."""
    indent, line, lineno = rows[idx]
    m = _FOR.match(line)
    if m:
        var, coll, inline = m.groups()
        values = parse_collection(coll)
        if inline.strip():  # single-line form: `for H in heaters: V = 4*H`
            for v in values:
                _consume([(indent + 1, inline.strip(), lineno)], 0, blk,
                         dict(scope or {}, **{var: v}))
            return idx + 1
        end = idx + 1  # suite form: body is the following more-indented lines
        while end < len(rows) and rows[end][0] > indent:
            end += 1
        body = rows[idx + 1:end]
        if not body:
            raise ScriptError(f"line {lineno}: 'for' suite is empty")
        for v in values:
            sc = dict(scope or {}, **{var: v})
            j = 0
            while j < len(body):
                j = _consume(body, j, blk, sc)
        return end
    _exec_stmt(line, blk, scope)
    return idx + 1


def _exec_stmt(line, blk, scope=None):
    low = line.lower()
    if low.startswith("name"):
        blk.name = line[4:].lstrip(" =").strip().strip("\"'")
        return
    m = _SWEEP.match(line)
    if m:
        target, rest = m.groups()
        ch, rest = _channel(target), rest.strip()
        mm = _FROMTO.match(rest)
        if mm:
            a, b = _ev(mm.group(1)), _ev(mm.group(2))
            st = _ev(mm.group(3)) if mm.group(3) else 0.5
            vals = _frange(a, b, st)
        elif low.startswith("sweep") and re.search(r"\b(in|over)\b", rest, re.I):
            vals = parse_collection(re.split(r"\b(?:in|over)\b", rest, 1, re.I)[-1])
        else:
            raise ScriptError(f"bad sweep: {rest!r}")
        blk.sweeps.append((ch, vals, target))
        return
    m = _EVERY.match(line)
    if m:
        blk.settings["loop"] = float(m.group(1))
        return
    m = _REPEAT.match(line)
    if m:
        blk.settings["iters"] = int(m.group(1))
        return
    blk.apply_pairs(_SET.sub(r"\1 = \2", line), scope=scope, in_for=scope is not None)


def _watch_stdin():
    """Daemon thread: typing a kill word (exit/quit/q/stop) + Enter aborts the run."""
    try:
        for line in sys.stdin:
            if line.strip().lower() in _KILL_WORDS:
                print("\n[exit] stopping -- zeroing DACs...", flush=True)
                _ABORT.set()
                return
            if _ABORT.is_set():
                return
    except Exception:  # noqa: BLE001 -- stdin closed/unavailable; nothing to watch
        return


def _connect(mock, port):
    if mock:
        fwd = lambda v: 0.04 + 0.03 * np.sin(np.arange(NUM_ADC_RAW) + v.sum())
        return MockPIC(fwd, noise=5e-4, voltage_max=VMAX).open()
    p, cands = find_port(port)
    if not p:
        raise ScriptError(f"no serial port found (looked at {cands or 'none'})")
    print(f"port {p}")
    return PIC(port=p, voltage_max=VMAX).open()


def _shutdown(pic, zero_at_end):
    if zero_at_end:
        pic.close()  # PIC.close() zeros the DACs
        return
    ser = getattr(pic, "ser", None)
    if ser not in (None, "mock"):
        try:
            ser.close()
        except Exception:  # noqa: BLE001
            pass
    pic.ser = None  # drop the handle without auto-zeroing, so the DACs hold


def _execute(pic, name, vec, settings, outdir, write_csv):
    loop = settings.get("loop", 0.0)
    iters = max(1, int(settings.get("iters", 1)))
    settle = settings.get("settle", 0.0)
    nz = {i: round(float(x), 3) for i, x in enumerate(vec) if x}
    eta = loop * (iters - 1)
    print(f"\n[{name}] {len(nz)} channels set, {iters} pulse(s)"
          + (f" every {loop:g}s (~{eta:g}s)" if loop and iters > 1 else "")
          + f"  {dict(list(nz.items())[:8])}" + (" ..." if len(nz) > 8 else ""))
    rows, t0 = [], time.time()
    for i in range(iters):
        if _ABORT.is_set():
            break
        if settle:
            pic.measure_raw(vec)
            if _ABORT.wait(settle):  # wakes immediately on abort
                break
        adc = np.asarray(pic.measure_raw(vec), float)
        rows.append([i, round(time.time() - t0, 3)] + [round(float(x), 4) for x in adc])
        if iters > 1:
            print(f"  pulse {i + 1}/{iters}  adc[:4]={np.round(adc[:4], 3).tolist()}")
        if loop and i < iters - 1 and _ABORT.wait(loop):
            break
    if write_csv and rows:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{name}.csv")
        header = "iter,t_s," + ",".join(f"pd{j}" for j in range(len(rows[0]) - 2))
        np.savetxt(path, np.array(rows), fmt="%g", delimiter=",", header=header, comments="")
        print(f"  -> {path} ({len(rows)} rows)")
    return _ABORT.is_set()


def run_script(path, mock=False, dry=False, write_csv=True, port=None,
               zero_at_end=False, outdir="runs"):
    blocks = parse(open(path).read())
    runs = [r for b in blocks for r in b.runs()]
    if not runs:
        print("no runnable blocks found.")
        return 1
    total = sum(s.get("loop", 0) * (max(1, int(s.get("iters", 1))) - 1) for _, _, s in runs)
    print(f"{len(blocks)} block(s) -> {len(runs)} run(s), ~{total:g}s total"
          + (" [DRY]" if dry else "") + (" [MOCK]" if mock else ""))
    if dry:
        for name, vec, s in runs:
            nz = {i: round(float(x), 3) for i, x in enumerate(vec) if x}
            print(f"  {name}: {len(nz)} ch, iters={int(s.get('iters', 1))}, "
                  f"loop={s.get('loop', 0):g}s  {dict(list(nz.items())[:6])}")
        return 0
    if write_csv:
        print("note: ADC columns are floating until the photodiodes are wired.")
    _ABORT.clear()
    try:
        prev_sigint = signal.signal(signal.SIGINT, lambda *_: _ABORT.set())
    except ValueError:  # not the main thread -> Ctrl-C handler unavailable, typed exit still works
        prev_sigint = None
    threading.Thread(target=_watch_stdin, daemon=True).start()
    print("running -- press Ctrl-C or type 'exit' to stop early (DACs zero on exit).")
    pic = _connect(mock, port)
    try:
        for name, vec, s in runs:
            if _execute(pic, name, vec, s, outdir, write_csv):
                break
    finally:
        aborted = _ABORT.is_set()
        _shutdown(pic, zero_at_end or aborted)
        if prev_sigint is not None:
            signal.signal(signal.SIGINT, prev_sigint)
    if aborted:
        print("\n[exit] stopped early. DACs zeroed.")
        return 130
    print(f"\ndone. DACs {'zeroed' if zero_at_end else 'held'} at exit.")
    return 0


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    port = next((a.split("=", 1)[1] for a in argv if a.startswith("--port=")), None)
    outdir = next((a.split("=", 1)[1] for a in argv if a.startswith("--out=")), "runs")

    if not args:
        print("usage: python run.py <file.pic> "
              "[--mock] [--dry] [--no-csv] [--zero] [--port=DEV] [--out=DIR]")
        return 1

    return run_script(args[0], mock="--mock" in flags, dry="--dry" in flags,
                      write_csv="--no-csv" not in flags, port=port,
                      zero_at_end="--zero" in flags, outdir=outdir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
