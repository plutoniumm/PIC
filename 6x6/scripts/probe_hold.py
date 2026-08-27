"""Drive one block of DAC channels to a fixed pattern and HOLD it, so the pins can be
probed with a multimeter (laser off, electrical map-check).

Opening the serial port resets the Mega, so this keeps the port open and re-asserts the
vector every couple seconds for the whole hold window -- the DACs stay energized until
this process is stopped. Channels outside the block are held at 0 V.

    python scripts/probe_hold.py --start 0 --count 16      # ch0..15 alternate 1/2 V
    python scripts/probe_hold.py --start 64 --count 16 --v1 1 --v2 2

Pattern is alternating v1,v2 across the block: even offset -> v1, odd offset -> v2.
"""

from __future__ import annotations

import argparse
import time

import serial

PORT = "/dev/cu.usbserial-1110"
NUM_DAC = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--v1", type=float, default=1.0)
    ap.add_argument("--v2", type=float, default=2.0)
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--seconds", type=float, default=900.0)
    a = ap.parse_args()

    vec = [0.0] * NUM_DAC
    for off in range(a.count):
        ch = a.start + off
        if 0 <= ch < NUM_DAC:
            vec[ch] = a.v1 if off % 2 == 0 else a.v2
    line = (",".join(f"{x:.2f}" for x in vec) + "\n").encode()

    s = serial.Serial(a.port, 115200, timeout=2)
    time.sleep(2.2)  # board resets on open
    s.reset_input_buffer()
    s.write(line)
    time.sleep(0.3)
    reply = s.readline().decode("utf-8", "ignore").strip()
    setpoints = {a.start + off: vec[a.start + off] for off in range(a.count)}
    print(f"APPLIED block {a.start}..{a.start + a.count - 1}: {setpoints}", flush=True)
    print(f"firmware reply ok: {len(reply.split(',')) == 14}", flush=True)

    t0 = time.time()
    while time.time() - t0 < a.seconds:
        s.write(line)
        s.reset_input_buffer()
        time.sleep(2)
    s.close()


if __name__ == "__main__":
    main()
