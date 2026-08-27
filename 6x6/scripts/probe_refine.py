"""Stage-2 fine refinement of the per-heater fringe extrema.

The coarse census (scripts/probe_channels.py, 0.25 V steps) localizes each heater's
null (min transmission) and peak (max pass) to a 0.25 V bin. This refines only the
channels whose extremum is *interior* (not pinned at 0 or 2 V): around the coarse
level Vc it re-sweeps [Vc - half, Vc + half] at a fine step, laser lit, streaming every
raw read to pic_data/census/ just like stage 1.

    # after the coarse run finishes:
    python scripts/probe_refine.py --coarse pic_data/census/census_YYYYMMDD_HHMMSS.csv
    python scripts/probe_refine.py --coarse <csv> --target null --step 0.05

`Vc +/- half` follows the user's bracket: half defaults to 0.125 (= coarse_step/2), so a
coarse pick of 0.75 refines 0.625..0.875 in 0.05 V steps.
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np

from laser.laser import Laser
from src.pic import PIC
from template import laser_session, _resolve_pic_port

NPD = 14
VMIN, VMAX = 0.0, 2.0


def load_coarse(path):
    """(ch -> {'levels': sorted[v], 'delta': array[len(levels), 14]}) from a census CSV."""
    base = defaultdict(list)      # ch -> [pd14] rows at phase base
    drive = defaultdict(lambda: defaultdict(list))  # ch -> v -> [pd14] rows
    with open(path) as f:
        for r in csv.DictReader(f):
            ch = int(r["ch"])
            pds = [float(r[f"pd{i}"]) for i in range(NPD)]
            if r["phase"] == "base":
                base[ch].append(pds)
            else:
                drive[ch][float(r["drive_v"])].append(pds)
    out = {}
    for ch, dv in drive.items():
        b = np.mean(base[ch], axis=0) if base[ch] else np.zeros(NPD)
        levels = sorted(dv)
        delta = np.array([np.mean(dv[v], axis=0) - b for v in levels])
        out[ch] = {"levels": np.array(levels), "delta": delta}
    return out


def extrema(coarse_ch, target):
    """Interior coarse levels to refine for one channel: dominant PD's null and/or peak.
    Returns list of (Vc, kind); empty if the extremum is pinned at a scan boundary."""
    levels, delta = coarse_ch["levels"], coarse_ch["delta"]
    pd = int(np.argmax(np.abs(delta).max(axis=0)))     # dominant PD
    trace = delta[:, pd]
    picks = []
    want = {"null": [np.argmin], "peak": [np.argmax], "both": [np.argmin, np.argmax]}[target]
    for fn in want:
        i = int(fn(trace))
        if 0 < i < len(levels) - 1:                     # interior only
            picks.append((float(levels[i]), "null" if fn is np.argmin else "peak", pd))
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coarse", required=True, help="stage-1 census CSV")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--half", type=float, default=0.125, help="bracket half-width (coarse_step/2)")
    ap.add_argument("--target", choices=["null", "peak", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--dbm", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=1100)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--port", default=os.environ.get("PIC_PORT"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    coarse = load_coarse(a.coarse)
    plan = {}   # ch -> list of (Vc, kind, pd)
    for ch, c in coarse.items():
        picks = extrema(c, a.target)
        if picks:
            plan[ch] = picks
    n_sweeps = sum(len(v) for v in plan.values())
    print(f"{len(plan)}/{len(coarse)} channels have an interior extremum to refine "
          f"({n_sweeps} brackets); the rest are monotonic in 0-2 V.", flush=True)
    if not plan:
        return

    out = a.out or time.strftime("pic_data/census/refine_%Y%m%d_%H%M%S.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    laser = Laser().open()
    pic = PIC(port=_resolve_pic_port(a.port, laser.dev.port), num_dac=a.n).open()
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "ch", "kind", "dom_pd", "vc", "phase", "rep", "drive_v",
                "dbm", "bfm", "temp_c", "mA"] + [f"pd{i}" for i in range(NPD)])

    def block(v, ch, kind, pd, vc, phase, vlevel, tel):
        Y = []
        for r in range(a.repeats):
            y = pic.measure_raw(v)
            Y.append(y)
            w.writerow([f"{time.time():.2f}", ch, kind, pd, f"{vc:.3f}", phase, r,
                        f"{vlevel:.3f}", a.dbm, f"{tel[0]:.3f}", f"{tel[1]:.2f}",
                        f"{tel[2]:.2f}"] + [f"{x:.4f}" for x in y])
        return np.mean(Y, axis=0)

    try:
        with laser_session(laser, duration_s=a.duration, power_dbm=a.dbm) as ls:
            print(f"emitted={ls.emitted}  bfm {ls.bfm_off:.3f}->{ls.bfm_on:.3f}  -> {out}",
                  flush=True)
            zero = np.zeros(a.n)
            pic.measure(zero, settle_s=1.5)
            for ch, picks in sorted(plan.items()):
                for vc, kind, pd in picks:
                    ls.keepalive()
                    tel = ls.telemetry()
                    lo = max(VMIN, vc - a.half)
                    hi = min(VMAX, vc + a.half)
                    fine = np.round(np.arange(lo, hi + 1e-9, a.step), 3)
                    pic.measure(zero, settle_s=a.settle)
                    base = block(zero, ch, kind, pd, vc, "base", 0.0, tel)
                    D = []
                    for lv in fine:
                        v = zero.copy()
                        v[ch] = lv
                        pic.measure(v, settle_s=a.settle)
                        D.append(block(v, ch, kind, pd, vc, "drive", float(lv), tel) - base)
                    f.flush()
                    D = np.array(D)[:, pd]
                    star = fine[int(np.argmin(D) if kind == "null" else np.argmax(D))]
                    print(f"ch {ch:3d} {kind:4s} pd{pd:<2d}: coarse {vc:.2f} -> "
                          f"fine {star:.2f} V  (Δ {D.min() * 1000:+.1f}..{D.max() * 1000:+.1f} mV)",
                          flush=True)
    finally:
        f.flush(); f.close()
        pic.set_zero(); pic.close(); laser.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
