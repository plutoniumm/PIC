"""Rapid drift re-anchor: warm-start from each heater's stored null and locally re-find it.

Drift on this chip is offset-only same-day (phi0 moves, the fringe shape phi2/Vpi holds),
so we don't re-sweep -- we start at the stored Vnull and do a small parabolic / gradient
minimisation of the dominant PD around it. A few measurements per heater vs a full sweep.

    python scripts/reanchor.py --limit 12                 # quick test on 12 heaters
    python scripts/reanchor.py --base-map pic_data/dac_heater_map_b128.csv --update

Compares each heater's re-found null to its stored value (= the drift since characterization),
and reports speed. With --update, writes the new Vnull back into the config.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from laser.laser import Laser
from src.pic import PIC
from template import laser_session, _resolve_pic_port

VMAX = 4.0


def reanchor_one(pic, ch, pd, v_start, base, delta=0.15, maxiter=5, settle=0.4):
    """Minimise pd(V) near v_start (the null). Warm-started parabolic search with a
    gradient fallback. Returns (new_null, n_measurements)."""
    n = 0

    def meas(v):
        nonlocal n
        n += 1
        x = base.copy()
        x[ch] = float(np.clip(v, 0.0, VMAX))
        pic.measure_raw(x)          # apply + let it settle (firmware already averages 10 sweeps)
        time.sleep(settle)
        return pic.measure_raw(x)[pd]

    v = float(np.clip(v_start, 0.0, VMAX))
    for _ in range(maxiter):
        lo, mid, hi = meas(v - delta), meas(v), meas(v + delta)
        denom = lo - 2 * mid + hi
        if denom > 1e-9 and mid <= lo and mid <= hi:           # bracketed convex minimum
            step = 0.5 * delta * (lo - hi) / denom
            v = float(np.clip(v + step, 0.0, VMAX))
            if abs(step) < 0.02:
                break
            delta = max(delta * 0.6, 0.03)
        else:                                                   # not bracketed: step downhill
            v = float(np.clip(v - delta * np.sign(hi - lo), 0.0, VMAX))
    return v, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--limit", type=int, default=None, help="only the first N characterized heaters")
    ap.add_argument("--base-map", default=None, help="hold characterized heaters at V0 (light-routing)")
    ap.add_argument("--dbm", type=float, default=15.0)
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--port", default=None)
    ap.add_argument("--update", action="store_true", help="write re-found Vnull back to the config")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    chans = [c for c in cfg["channels"] if c["Vnull"] is not None and c["fringe_pd"] is not None]
    if a.limit:
        chans = chans[:a.limit]

    base = np.zeros(128)
    if a.base_map:
        for r in csv.DictReader(l for l in open(a.base_map) if not l.startswith("#")):
            if r.get("V0"):
                base[int(r["dac"])] = float(r["V0"])

    laser = Laser().open()
    pic = PIC(port=_resolve_pic_port(a.port, laser.dev.port), num_dac=128).open()
    results = []
    try:
        # size the watchdog to the workload (~7s/heater) but cap under a 10-min host
        # timeout; the expiry-break below stops the loop cleanly if it is ever reached, so
        # heaters are NEVER re-nulled against a dark (watchdog-off) chip.
        dur = min(max(120, len(chans) * 8 + 60), 540)
        with laser_session(laser, duration_s=dur, power_dbm=a.dbm) as ls:
            print(f"emitted={ls.emitted}  re-anchoring {len(chans)} heaters "
                  f"(base-map={'on' if a.base_map else 'off'}, watchdog {dur:.0f}s)", flush=True)
            t0 = time.time()
            for c in chans:
                if ls.expired():
                    print(f"session expired -- stopping after {len(results)}/{len(chans)} "
                          f"heaters; the rest keep their old nulls (not dark-measured)", flush=True)
                    break
                ls.keepalive()
                old = float(c["Vnull"])
                new, nmeas = reanchor_one(pic, c["dac"], int(c["fringe_pd"]), old, base,
                                          settle=a.settle)
                results.append((c["dac"], int(c["fringe_pd"]), old, new, new - old, nmeas))
                flag = "  <-- big, suspect" if abs(new - old) > 0.5 else ""
                print(f"ch {c['dac']:3d} pd{c['fringe_pd']:<2} null {old:.3f} -> {new:.3f}  "
                      f"drift {new - old:+.3f} V  ({nmeas} meas){flag}", flush=True)
            dt = time.time() - t0
    finally:
        pic.set_zero()
        pic.close()
        laser.close()

    drift = np.array([r[4] for r in results])
    small = np.abs(drift) <= 0.5
    print(f"\n{len(results)} heaters re-anchored in {dt:.1f}s "
          f"({dt / max(len(results), 1):.2f}s/heater, {np.mean([r[5] for r in results]):.1f} meas each)")
    print(f"drift (|.|<=0.5V, n={small.sum()}): mean {drift[small].mean():+.3f} V  "
          f"std {drift[small].std():.3f} V  max|.| {np.abs(drift[small]).max():.3f} V")
    full_sweep_s = len(results) * 9 * (a.settle + 0.3)  # rough: a full 0-4V/0.5V re-sweep
    print(f"vs a full re-sweep of the same set (~{full_sweep_s:.0f}s): {full_sweep_s / dt:.1f}x faster")

    if a.update:
        by = {r[0]: round(r[3], 3) for r in results if abs(r[4]) <= 0.5}
        nv0 = 0
        for c in cfg["channels"]:
            if c["dac"] not in by:
                continue
            vn = by[c["dac"]]
            c["Vnull"] = vn
            # keep V0 consistent: peak and null are half a fringe apart, so V0^2 = vn^2 +/-
            # Vpi^2 -- pick the physical root nearest the old V0 (drift is offset-only, so
            # the shape Vpi holds and V0 just tracks the null).
            vpi, v0 = c.get("Vpi"), c.get("V0")
            if vpi and v0 is not None:
                cands = [np.sqrt(vn ** 2 + vpi ** 2)]
                if vn ** 2 > vpi ** 2:
                    cands.append(np.sqrt(vn ** 2 - vpi ** 2))
                cands = [x for x in cands if 0.0 <= x <= VMAX]
                if cands:
                    c["V0"] = round(float(min(cands, key=lambda x: abs(x - v0))), 3)
                    nv0 += 1
        json.dump(cfg, open(a.config, "w"), indent=2)
        print(f"updated {len(by)} Vnull (+{nv0} V0) in {a.config}")


if __name__ == "__main__":
    main()
