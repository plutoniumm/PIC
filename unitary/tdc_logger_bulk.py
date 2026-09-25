"""Log every SPD on USB to one CSV, printing a summary per burst.

    python tdc_logger_bulk.py                  # every SPD found by chip id
    python tdc_logger_bulk.py 1344A... 1344B...   # just these
    python tdc_logger_bulk.py --mock 2         # two simulated SPDs

Type a command and Enter to send it to every SPD (each waits for its gap between vaults),
or `<chip id>: <command>` for one.
"""

import argparse
import threading

from spd import SPDs


def commands(spds):
    while True:
        try:
            line = input().strip()
        except EOFError:
            return
        if not line:
            continue
        sid, _, cmd = line.partition(": ")
        ids = [sid] if cmd and sid in spds.ids else None
        spds.send(cmd if ids else line, ids=ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*", help="SPD chip ids (default: every SPD on USB)")
    ap.add_argument("--csv", default="tdc_continuous_vaults.csv")
    ap.add_argument("--mock", type=int, default=0, metavar="N", help="N simulated SPDs")
    a = ap.parse_args()
    with SPDs(a.ids or None, mock=a.mock, csv=a.csv) as spds:
        print(f"[spd] logging {', '.join(spds.ids)} -> {a.csv}")
        threading.Thread(target=commands, args=(spds,), daemon=True).start()
        try:
            for sid, v in spds.vaults():
                rate = f"{v.rate:,.0f}/s over {v.seconds:.6f} s" if v.rate else "timing unavailable"
                print(f"[{sid}] burst {v.burst}: {v.count:,} stamps, {rate}")
        except KeyboardInterrupt:
            print("\n[spd] stopped")


if __name__ == "__main__":
    main()
