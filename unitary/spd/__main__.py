"""python -m spd [selftest | count | assign | dark | max | log | send CMD] [--port X] [--mock]

    python -m spd count -s 1          # every SPD on USB: its slot (or unmapped), counts, rate, 0..1
    python -m spd assign DQ00QQ2C PD1 # this SPD sits on output 1 (`none` unmaps it)
    python -m spd dark -s 5           # laser off: store every connected SPD's dark
    python -m spd max DQ00QQ2C -s 5   # its output at its brightest: store its max, then its dark
                                      # with the switch parked. One output at a time.
"""

import argparse
import json
import re
import sys
import threading
import time

from .array import MAX_PATH, SPDs, discover, load_max, normalise, save_max
from .array import _selftest as _array_selftest
from .vega import SPDError, Vega, _selftest


def _open(a, **kw):
    ser = None
    if a.mock:
        from .mock import MockVega

        ser = MockVega()
    return Vega(a.port, ser=ser, **kw)


def _stdin_commands(v):
    for line in sys.stdin:
        if line.strip():
            if v.busy:
                print("[command] waiting for the current vault to finish...")
            v.send(line)
            print(f"[command] sent: {line.strip()}")


def main(argv=None):
    # on a parent so they work before or after the subcommand, as on `python -m pic`
    def flags(p, default=None):
        p.add_argument("--port", default=default, help="device path or USB serial number")
        p.add_argument(
            "--mock", action="store_true", default=default or False, help="physical mock"
        )

    ap = argparse.ArgumentParser(prog="python -m spd", description="Vega SPD/TDC readout")
    flags(ap)
    # the subcommands' copies default to SUPPRESS, so they never overwrite one given earlier
    common = argparse.ArgumentParser(add_help=False)
    flags(common, argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", parents=[common])
    for name, what, s in (
        ("count", "add PHOTON_COUNT frames per SPD; repeats until ctrl-c", 1.0),
        ("max", f"store each SPD's max, and its dark if a board can park the switch, in {MAX_PATH}", 5.0),
        ("dark", f"store each SPD's rate as it is now as its dark in {MAX_PATH}", 5.0),
    ):
        p = sub.add_parser(name, help=what, parents=[common])
        p.add_argument("ids", nargs="*", help="chip ids (default: every SPD on USB)")
        p.add_argument("-s", "--seconds", type=float, default=s, help="integration time")
        p.add_argument("-n", type=int, help="stop after N readings")
    p = sub.add_parser("assign", help="put an SPD on an output slot", parents=[common])
    p.add_argument("id", help="chip id, 8 characters as `count` prints it")
    p.add_argument("slot", help="PD0..PD3, or none to unmap")
    p = sub.add_parser(
        "log", help="print vault summaries; stdin lines are sent as commands", parents=[common]
    )
    p.add_argument("--csv", help="append Timestamp_Unix,Burst_ID,TOF_ps rows here")
    p.add_argument("-n", type=int, help="stop after N vaults")
    p = sub.add_parser("send", help="send one console command and print replies", parents=[common])
    p.add_argument("text")
    p.add_argument("--wait", type=float, default=1.0, help="seconds to listen for the reply")
    a = ap.parse_args(argv)

    if a.cmd in (None, "selftest"):
        _selftest()
        _array_selftest()
        _assign_selftest()
        return 0
    try:
        if a.cmd == "assign":
            print(assign(a.id, a.slot))
            return 0
        if a.cmd in ("count", "max", "dark"):
            return _count(a)
        if a.cmd == "send":
            with _open(a) as v:
                v.send(a.text, timeout=10)
                time.sleep(a.wait)
            return 0
        with _open(a, csv=a.csv) as v:
            threading.Thread(target=_stdin_commands, args=(v,), daemon=True).start()
            for i, x in enumerate(v.vaults(), 1):
                rate = f"{x.rate:,.0f}/s over {x.seconds:.6f} s" if x.seconds else "timing n/a"
                print(f"[vault {x.burst}] {x.count:,} stamps, {rate}")
                if a.n and i >= a.n:
                    break
    except (SPDError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def assign(chip: str, slot: str, path=None) -> str:
    """Record which output `chip` sits on. One SPD per slot: a second one on a taken slot is
    refused, so a mix-up is fixed on purpose (`none` first) rather than by overwriting."""
    from pic.config import SPD_MAP_PATH, USB_SERIAL, spd_map

    if not re.fullmatch(r"[A-Z0-9]{8}", chip):
        raise ValueError(f"{chip!r} is not an FTDI chip id (8 characters, A-Z 0-9, as `count` prints)")
    if chip in USB_SERIAL.values():
        raise ValueError(f"{chip} is a rig instrument, not an SPD")
    m = re.fullmatch(r"PD([0-3])|(none)", slot, re.I)
    if not m:
        raise ValueError(f"slot {slot!r}: use PD0, PD1, PD2, PD3 or none")
    path = path or SPD_MAP_PATH
    cur = spd_map(path)
    cur.pop(chip, None)
    if m.group(1) is not None:
        k = int(m.group(1))
        other = [i for i, v in cur.items() if v == k]
        if other:
            raise ValueError(f"PD{k} is {other[0]}'s; `python -m spd assign {other[0]} none` first")
        cur[chip] = k
    with open(path, "w") as f:
        json.dump(dict(sorted(cur.items(), key=lambda kv: kv[1])), f, indent=1)
    return f"{chip} -> {'PD' + m.group(1) if m.group(1) else 'unmapped'}  ({path})"


def _assign_selftest():
    import tempfile
    from pathlib import Path

    from pic.config import spd_map

    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "map.json")
        assign("DQ00QQ2C", "PD1", p)
        assign("AB12CD34", "pd3", p)
        for chip, slot, why in (
            ("AB12CD34", "PD1", "PD1 is DQ00QQ2C's"),
            ("dq00qq2c", "PD0", "not an FTDI chip id"),
            ("DQ00QQ2", "PD0", "not an FTDI chip id"),
            ("AU05XLI8", "PD0", "rig instrument"),
            ("EF56GH78", "PD4", "use PD0"),
        ):
            try:
                assign(chip, slot, p)
                raise AssertionError(f"assigned {chip} {slot}")
            except ValueError as e:
                assert why in str(e), e
        assign("AB12CD34", "PD2", p)  # moving an SPD frees its old slot
        assign("DQ00QQ2C", "none", p)
        assert spd_map(p) == {"AB12CD34": 2}, spd_map(p)


def _park_dark(mock):
    """The board with its switch on the dark channel, or (None, why). Opening the board resets
    it, so every heater is at 0 V after this; that is why the max is read first."""
    try:
        if mock:
            from pic.devices.switch import MockSwitch
            from pic.interface import MockPIC

            sw = MockSwitch().open()
            b = MockPIC(switch=sw).open()
            sw.dark()
        else:
            from pic.interface import PIC

            b = PIC().open()
            b.select_port(-1)
        return b, None
    except Exception as e:  # no board, or another process holds it
        return None, f"{type(e).__name__}: {e}"


def _store(spds, a, what, mock):
    c = dict(sorted(spds.count(a.seconds).items()))
    if not mock:  # a mock never writes into pic_data
        save_max(**{what: {i: x.rate for i, x in c.items()}})
    for i, x in c.items():
        print(f"{i}  {'max' if what == 'rates' else 'dark'} {x.rate:,.1f}/s  ({x.counts} in {x.seconds:g} s)")


def _count(a):
    # counting only listens: nothing is written to a detector, so its settings stay put
    from pic.config import spd_map

    kw = {"mock": 2, "count_hz": 40.0} if a.mock else {}
    if a.cmd == "max" and not a.ids and not a.mock:
        # each output has its own brightest state, so a max is taken for the lit one only
        ids = discover()
        if len(ids) != 1:
            raise SPDError(f"{len(ids)} SPDs on USB: name the one whose output is at its brightest")
        a.ids = ids
    where = spd_map()
    slot = lambda i: f"PD{where[i]}" if i in where else "unmapped"
    with SPDs(a.ids or None, on_text=lambda sid, t: None, **kw) as spds:
        if a.cmd == "count":
            by = {v: i for i, v in where.items()}
            on = set(spds.ids)
            print("  ".join(f"PD{k} {by.get(k, 'none')}" + ("" if by.get(k, "") in on or k not in by
                            else " (not on USB)") for k in range(4)))
        if a.cmd == "dark":
            _store(spds, a, "darks", a.mock)
            return 0
        if a.cmd == "max":
            _store(spds, a, "rates", a.mock)
            board, why = _park_dark(a.mock)
            if board is None:
                print(f"dark not re-measured, no board to park the switch ({why}); "
                      "run `python -m spd dark` with the laser off")
                return 0
            try:
                time.sleep(0.5)  # the mirror's settle, then only dark frames
                _store(spds, a, "darks", a.mock)
                print("switch left on its dark channel, heaters at 0 V")
            finally:
                board.close()
            return 0
        maxes, k = load_max(), 0
        while not a.n or k < a.n:
            c = dict(sorted(spds.count(a.seconds).items()))
            ok = lambda i: maxes.get(i) and None not in maxes[i] and maxes[i][1] > maxes[i][0]
            print(
                "  ".join(
                    f"{i} {slot(i)} {x.counts} ({x.rate:,.1f}/s"
                    + (f", {normalise(x.rate, maxes[i]):.3f}" if ok(i) else "")
                    + ")"
                    for i, x in c.items()
                ),
                flush=True,
            )
            k += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
