"""PIC command-line entry: live photodiode monitor.

    python -m src.pic measure              # live dashboard in the terminal
    python -m src.pic measure out.csv      # append the raw stream to a CSV file
    python -m src.pic measure --mock       # simulator, no hardware

Each reading is `t_s` (seconds since start) followed by the 14 raw photodiode
voltages PD0..PD13; the heaters are held at 0 V for the duration. Streams until
Ctrl-C. Straight to a terminal (no file, stdout is a TTY) it runs as a live
dashboard: readings scroll between four pinned rows -- header and running per-PD
max on top, running average of the last 10 and the peak of that average on the
bottom. Redirected or to a file it emits plain CSV instead. With both devices
plugged in, pass --port or set $PIC_PORT (the port glob also matches the laser
FTDI)."""

from __future__ import annotations
import argparse
import shutil
import signal
import sys
import time
from collections import deque

import numpy as np

from .interface import PIC, MockPIC, find_port
from .config import NUM_DAC, NUM_ADC_RAW, DAMAGED_PDS


def _mock_forward():
    """Smooth 64->14 stand-in with the dead PDs pinned at the readout floor."""
    rng = np.random.default_rng(1)
    W = rng.normal(0.0, 0.10, (NUM_ADC_RAW, NUM_DAC))
    b = rng.uniform(0.0, 2 * np.pi, NUM_ADC_RAW)
    dead = [i in DAMAGED_PDS for i in range(NUM_ADC_RAW)]

    def f(v):
        v = np.clip(np.asarray(v, float), 0.0, 2.0)
        y = 0.02 + 0.04 * (1 + np.sin(W @ (v / 2.0) + b))
        y[dead] = 0.006
        return y

    return f


def open_pic(args):
    if args.mock:
        print("rig: MOCK (no hardware)", file=sys.stderr)
        return MockPIC(_mock_forward(), noise=8e-4).open()
    port, cands = find_port(args.port)
    if not port:
        sys.exit(
            f"no serial device found (looked at {cands or 'the usual /dev names'}); "
            "pass --port or --mock"
        )
    print(f"rig: {port}", file=sys.stderr)
    return PIC(port=port).open()


def _throttle(period, tic):
    """Sleep so the loop runs no faster than one read per `period` seconds."""
    if period > 0:
        rest = period - (time.monotonic() - tic)
        if rest > 0:
            time.sleep(rest)


def measure(args):
    zeros = np.zeros(NUM_DAC)
    with open_pic(args) as pic:
        w = shutil.get_terminal_size((100, 24))
        # dashboard has 4 fixed rows (header+max top, avg+maxavg bottom); need room to scroll
        dashboard = args.file is None and sys.stdout.isatty() and w.lines >= 6
        if dashboard:
            _measure_dashboard(pic, zeros, args)
        else:
            _measure_stream(pic, zeros, args)


def _measure_stream(pic, zeros, args):
    """Plain CSV: one row per read to a file or a redirected stdout."""
    cols = ["t_s"] + [f"PD{i}" for i in range(NUM_ADC_RAW)]
    to_file = args.file is not None
    sink = open(args.file, "a", buffering=1) if to_file else sys.stdout
    n = 0
    sink.write(",".join(cols) + "\n")
    sink.flush()
    if to_file:
        print(f"streaming PDs -> {args.file} (Ctrl-C to stop)", file=sys.stderr)
    t0 = time.monotonic()
    try:
        while True:
            tic = time.monotonic()
            y = pic.measure_raw(zeros)
            sink.write(f"{tic - t0:.3f}," + ",".join(f"{v:.3f}" for v in y) + "\n")
            sink.flush()
            n += 1
            if to_file and n % 20 == 0:
                print(
                    f"  {n} rows, t={tic - t0:.1f}s, PD5={y[5]:.3f} V", file=sys.stderr
                )
            _throttle(args.period, tic)
    except (KeyboardInterrupt, BrokenPipeError):
        dt = time.monotonic() - t0
        try:
            print(f"\nstopped: {n} rows over {dt:.1f}s", file=sys.stderr)
        except BrokenPipeError:
            pass
    finally:
        if to_file:
            sink.close()


def _measure_dashboard(pic, zeros, args):
    """Live terminal view with four pinned rows and a scrolling middle. Top two
    rows are fixed: the column header and the running per-PD max. Bottom two are
    fixed: the running average of the last 10 reads and the peak that average has
    reached. Data scrolls between them via a VT100 scroll region (rows 3..N-2)."""
    ESC = "\x1b"
    out = sys.stdout
    win = shutil.get_terminal_size((100, 24))
    rows, cw = win.lines, win.columns

    def region():  # scroll region = everything but the two fixed rows top and bottom
        out.write(f"{ESC}[3;{max(rows - 2, 3)}r")

    def on_resize(*_):
        nonlocal rows, cw
        w = shutil.get_terminal_size((100, 24))
        rows, cw = w.lines, w.columns
        region()
        out.flush()

    try:
        signal.signal(signal.SIGWINCH, on_resize)
    except (ValueError, AttributeError, OSError):
        pass  # not the main thread / platform without SIGWINCH

    clip = lambda s: s[: cw - 1]
    fmt = lambda label, vec: f"{label:>8}" + "".join(f"{v:7.3f}" for v in vec)
    header = "    t_s " + "".join(f"{'PD' + str(i):>7}" for i in range(NUM_ADC_RAW))
    last = deque(maxlen=10)
    mx = np.full(NUM_ADC_RAW, -np.inf)  # running per-PD max over all reads
    mxavg = np.full(NUM_ADC_RAW, -np.inf)  # peak of the last-10 running average
    n = 0
    t0 = time.monotonic()

    def paint(rownum, label, vec, style):  # redraw a fixed row without moving the data cursor
        line = clip(fmt(label, vec)).ljust(cw - 1)
        out.write(f"{ESC}7{ESC}[{rownum};1H{ESC}[2K{style}{line}{ESC}[0m{ESC}8")

    def draw_fixed(avg):
        paint(2, "max", mx, f"{ESC}[1m")  # bold, just under the header
        paint(rows - 1, f"avg{len(last)}", avg, f"{ESC}[7m")  # reverse-video bar
        paint(rows, "maxavg", mxavg, f"{ESC}[1m")  # bold, very bottom

    hint = "   (Ctrl-C to stop)"
    top = header + hint if len(header + hint) <= cw - 1 else header
    out.write(f"{ESC}[2J{ESC}[H")  # clear screen, cursor home
    out.write(clip(top) + "\n")  # row 1: fixed column header
    region()  # scroll region = rows 3..N-2
    out.write(f"{ESC}[3;1H")  # drop cursor into the region
    out.flush()
    try:
        while True:
            tic = time.monotonic()
            y = np.asarray(pic.measure_raw(zeros), float)
            last.append(y)
            np.maximum(mx, y, out=mx)
            avg = np.mean(np.stack(last), axis=0)
            np.maximum(mxavg, avg, out=mxavg)
            n += 1
            row = f"{tic - t0:8.2f}" + "".join(f"{v:7.3f}" for v in y)
            out.write(clip(row) + "\n")
            draw_fixed(avg)
            out.flush()
            _throttle(args.period, tic)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        except (ValueError, AttributeError, OSError):
            pass
        out.write(f"{ESC}[r{ESC}[{rows};1H\n")  # reset scroll region, park cursor
        out.flush()
        dt = time.monotonic() - t0
        print(f"stopped: {n} reads over {dt:.1f}s", file=sys.stderr)


# Subcommands whose whole argument tail is parsed by the lib module's own CLI.
# Imported lazily (inside the handler) so `python -m pic measure` never pulls torch.
_FORWARD = {
    "ising": ("pic.compute.ising", "main"),
    "matvec": ("pic.compute.matvec", "main"),
    "bringup": ("pic.bringup", "main"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _FORWARD:
        import importlib

        mod, fn = _FORWARD[argv[0]]
        return getattr(importlib.import_module(mod), fn)(argv[1:]) or 0

    ap = argparse.ArgumentParser(prog="python -m pic")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("measure", help="stream live photodiode voltages until Ctrl-C")
    p.add_argument(
        "file", nargs="?", default=None, help="CSV file to append to (default: stdout)"
    )
    p.add_argument("--port", default=None, help="serial device (default: autodetect)")
    p.add_argument("--mock", action="store_true", help="use the simulator, no hardware")
    p.add_argument(
        "--period",
        type=float,
        default=0.0,
        metavar="SEC",
        help="minimum seconds between reads (default: as fast as the hardware allows)",
    )
    # Decorative entries so `-h` lists the forwarded runners; real dispatch is above.
    sub.add_parser("ising", help="hardware Ising / Gram runner (pic.compute.ising -h)")
    sub.add_parser("matvec", help="signed intensity matvec runner (pic.compute.matvec -h)")
    sub.add_parser("bringup", help="hardware bring-up + settling tests (pic.bringup -h)")

    args = ap.parse_args(argv)
    if args.cmd == "measure":
        measure(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # a second Ctrl-C during cleanup -> exit quietly
        raise SystemExit(130)
