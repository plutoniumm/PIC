"""One-channel-at-a-time influence census -- the empirical DAC->heater ground truth.

For every channel: read a fresh all-zero baseline, drive the channel, read again.
EVERY raw repeat (no averaging) plus laser telemetry is streamed to the CSV the moment
it is measured and flushed per channel, so a crash loses at most one channel. Output
goes to the tracked pic_data/census/ so characterisation data survives this machine.

    python scripts/probe_channels.py                    # all 128 ch, 1.5 V, +5 dBm
    python scripts/probe_channels.py --channels 0-15,63 --repeats 8

Analyse afterwards by grouping on (ch, phase): delta = mean(drive) - mean(base).
"""

import argparse, csv, os, time
import numpy as np
from laser.laser import Laser
from src.pic import PIC
from src.census import CENSUS_HEADER
from template import laser_session, _resolve_pic_port


def parse_channels(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=128, help="DAC channels on the firmware")
    ap.add_argument("--channels", default=None, help='subset, e.g. "0-15,60,63" (default all)')
    ap.add_argument("--drive", type=float, default=1.5)
    ap.add_argument("--levels", default=None,
                    help='comma list of drive voltages per channel (fringe sweep); default = just --drive')
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--dbm", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=900, help="laser watchdog [s]")
    ap.add_argument("--port", default=os.environ.get("PIC_PORT"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-map", default=None,
                    help="dac_heater_map CSV: hold its characterized heaters at their V0 "
                         "(transparent) as the baseline, to route light past them")
    a = ap.parse_args()
    chans = parse_channels(a.channels, a.n)
    levels = [float(x) for x in a.levels.split(",")] if a.levels else [a.drive]

    base_vec = np.zeros(a.n)
    if a.base_map:
        for r in csv.DictReader(l for l in open(a.base_map) if not l.startswith("#")):
            if r.get("V0"):
                base_vec[int(r["dac"])] = float(r["V0"])
        print(f"baseline: {int((base_vec > 0).sum())} characterized heaters held at V0 "
              f"(transparent); swept channel starts from 0", flush=True)
    out = a.out or time.strftime("pic_data/census/census_%Y%m%d_%H%M%S.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    laser = Laser().open()
    pic = PIC(port=_resolve_pic_port(a.port, laser.dev.port), num_dac=a.n).open()
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(CENSUS_HEADER)

    def block(v, ch, phase, vlevel, tel):
        Y = []
        for r in range(a.repeats):
            y = pic.measure_raw(v)
            Y.append(y)
            w.writerow([f"{time.time():.2f}", ch, phase, r, vlevel, a.dbm,
                        f"{tel[0]:.3f}", f"{tel[1]:.2f}", f"{tel[2]:.2f}"]
                       + [f"{x:.4f}" for x in y])
        return np.mean(Y, axis=0)

    try:
        with laser_session(laser, duration_s=a.duration, power_dbm=a.dbm) as ls:
            print(f"emitted={ls.emitted}  bfm {ls.bfm_off:.3f}->{ls.bfm_on:.3f}  "
                  f"mA={ls.mA:.1f}  -> {out}", flush=True)
            pic.measure(base_vec, settle_s=1.5)
            for ch in chans:
                ls.keepalive()
                tel = ls.telemetry()
                b = base_vec.copy()
                b[ch] = 0.0  # sweep this channel from 0; other characterized heaters hold V0
                pic.measure(b, settle_s=a.settle)
                base = block(b, ch, "base", 0.0, tel)
                D = []
                for lv in levels:
                    v = b.copy()
                    v[ch] = lv
                    pic.measure(v, settle_s=a.settle)
                    D.append(block(v, ch, "drive", lv, tel) - base)
                f.flush()
                d = np.abs(np.array(D)).max(axis=0) * 1000  # per-PD max over levels
                top = np.argsort(-d)[:3]
                print(f"ch {ch:3d}: max|d|={d.max():5.1f} mV  top "
                      + ", ".join(f"pd{i}:{d[i]:.1f}" for i in top), flush=True)
    finally:
        f.flush()
        f.close()
        pic.set_zero()
        pic.close()
        laser.close()
    print(f"wrote {out} ({len(chans)} channels x 2 phases x {a.repeats} repeats)")


if __name__ == "__main__":
    main()
