"""Parallel (lockstep) characterization: drive several heaters at the SAME swept voltage and
recover each one's fringe from its own dominant PD in ONE sweep -- the top/bottom
independent-path speedup. Validates by also sweeping each heater SOLO under the same frontier
and checking the parallel-recovered V0/Vnull agree (no cross-path confounding).

    python scripts/parallel_char.py --channels 85,109,0,1 --port /dev/cu.usbserial-1110
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from laser.laser import Laser
from src.pic import PIC
from template import laser_session, _resolve_pic_port
from src.census import robust_fit
from src.sweep_analysis import fringe_extrema


def sweep(pic, base, drive_chans, levels, repeats, settle):
    P = []
    for v in levels:
        x = base.copy()
        for ch in drive_chans:
            x[ch] = v
        pic.measure(x, settle_s=settle)
        P.append(np.mean([pic.measure_raw(x) for _ in range(repeats)], axis=0))
    return np.array(P)


def extract(V, P, pd):
    fit = robust_fit(V, P[:, pd])
    if fit is None:
        return None, None
    ex = fringe_extrema(fit, vmax=5.0)
    return ex["v_at_max"], ex["v_at_min"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--levels", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--dbm", type=float, default=15.0)
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--port", default=None)
    a = ap.parse_args()
    chans = [int(x) for x in a.channels.split(",")]
    V = np.array([0.0] + [float(x) for x in a.levels.split(",")])
    cfg = json.load(open(a.config)); by = {c["dac"]: c for c in cfg["channels"]}
    base0 = np.zeros(128)
    for c in cfg["channels"]:
        if c.get("V0") is not None:
            base0[c["dac"]] = c["V0"]

    laser = Laser().open()
    pic = PIC(port=_resolve_pic_port(a.port, laser.dev.port), num_dac=128).open()
    try:
        with laser_session(laser, duration_s=400, power_dbm=a.dbm) as ls:
            print(f"emitted={ls.emitted}  frontier holds {int((base0>0).sum())} heaters at V0", flush=True)
            bp = base0.copy()
            for ch in chans:
                bp[ch] = 0.0
            t0 = time.time(); Ppar = sweep(pic, bp, chans, V, a.repeats, a.settle); t_par = time.time() - t0
            print(f"parallel sweep done ({t_par:.1f}s); now {len(chans)} solo sweeps...", flush=True)
            t0 = time.time(); Psolo = {}
            for ch in chans:
                bs = base0.copy(); bs[ch] = 0.0
                Psolo[ch] = sweep(pic, bs, [ch], V, a.repeats, a.settle)
            t_solo = time.time() - t0
    finally:
        pic.set_zero(); pic.close(); laser.close()

    print(f"\nparallel: {t_par:.1f}s for {len(chans)} heaters  |  solo: {t_solo:.1f}s  "
          f"-> {t_solo/t_par:.1f}x faster\n")
    print(f"{'dac':>4} {'pd':>3} {'V0_par':>7} {'V0_solo':>8} {'Vnull_par':>10} {'Vnull_solo':>11} {'dVnull':>7} {'ok':>4}")
    for ch in chans:
        pd = by[ch]["fringe_pd"]
        v0p, vnp = extract(V, Ppar, pd)
        v0s, vns = extract(V, Psolo[ch], pd)
        d = abs((vnp if vnp is not None else 0) - (vns if vns is not None else 0))
        print(f"{ch:>4} pd{pd:<2} {v0p or -1:>7.2f} {v0s or -1:>8.2f} {vnp or -1:>10.2f} "
              f"{vns or -1:>11.2f} {d:>7.2f} {'YES' if d < 0.4 else 'no':>4}")


if __name__ == "__main__":
    main()
