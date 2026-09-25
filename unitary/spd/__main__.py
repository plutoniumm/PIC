"""python -m spd [selftest | count | max | log | send CMD] [--port X] [--mock]

    python -m spd count -s 1        # every SPD by chip id: counts, rate, and 0..1 once a max is stored
    python -m spd max -s 5          # store the current rate as each SPD's max (mesh set brightest)
"""

import argparse
import sys
import threading
import time

from .array import MAX_PATH, SPDs, load_max, save_max
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
        ("max", f"store each SPD's rate as its max in {MAX_PATH}", 5.0),
    ):
        p = sub.add_parser(name, help=what, parents=[common])
        p.add_argument("ids", nargs="*", help="chip ids (default: every SPD on USB)")
        p.add_argument("-s", "--seconds", type=float, default=s, help="integration time")
        p.add_argument("-n", type=int, help="stop after N readings")
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
        return 0
    try:
        if a.cmd in ("count", "max"):
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
    except SPDError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def _count(a):
    # counting only listens: nothing is written to a detector, so its settings stay put
    kw = {"mock": 2, "count_hz": 40.0} if a.mock else {}
    with SPDs(a.ids or None, on_text=lambda sid, t: None, **kw) as spds:
        if a.cmd == "max":
            c = dict(sorted(spds.count(a.seconds).items()))
            if not a.mock:  # a mock never writes into pic_data
                save_max({i: x.rate for i, x in c.items()})
            for i, x in c.items():
                print(f"{i}  max {x.rate:,.1f}/s  ({x.counts} in {x.seconds:g} s)")
            return 0
        maxes, k = load_max(), 0
        while not a.n or k < a.n:
            c = dict(sorted(spds.count(a.seconds).items()))
            print(
                "  ".join(
                    f"{i} {x.counts} ({x.rate:,.1f}/s"
                    + (f", {x.rate / maxes[i]:.3f}" if maxes.get(i) else "")
                    + ")"
                    for i, x in c.items()
                ),
                flush=True,
            )
            k += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
