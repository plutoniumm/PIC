"""Outside-in, depth-ordered heater re-characterization with a transparent frontier.

Deep "meat" heaters read dark (fringe swing < 5 mV) because the MZIs upstream of them
are not set to pass light, so the light never reaches them. This orchestrator fixes that
by characterizing with a TRANSPARENT FRONTIER that grows from the input inward:

  * order the still-dark drivable heaters by mesh depth (input-side first) then row
    (outermost top/bottom edge first, centre last), split into independent top/bottom
    halves of the splitting tree;
  * for each target, hold every ALREADY-characterized heater at its V0 (opening a lit path
    from the input to the target), sweep the target 0->4 V, fit its dominant-PD fringe, and
    merge the result into the config BEFORE moving deeper -- so every new column is probed
    through the columns already opened.

Light flows input-coupler (high GDS-x) -> splitter tree -> encode -> V -> Sigma -> U -> PDs
(low GDS-x); geometry is `pic_data/heater_map.csv` (net == geometric heater H0..119).

    python scripts/rechar_outsidein.py --plan          # derive+print+save schedule, no hw
    python scripts/rechar_outsidein.py --mock           # full loop, no hardware
    python scripts/rechar_outsidein.py --dbm 15         # on hardware (laser + PIC)

Reuses scripts.census_fringes (robust_fit / analyze) for the fits and the merge rules of
scripts.merge_fringes_to_config. Raw reads stream to a pic_data/census/ CSV for provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pic import PIC, MockPIC, NUM_ADC_RAW
from src.pic.config import DAMAGED_PDS, VPI_NOMINAL
from src.census import CENSUS_HEADER, analyze, apply_fringe, stamp

HEATER_MAP = "pic_data/heater_map.csv"
SCHEDULE_OUT = "pic_data/rechar_schedule.json"
CENTER_Y = 1961.5  # splitting-tree axis (input-coupler y); top/bottom halves are independent
MAX_DBM = 15.0     # PD-damage ceiling -- average, never push power past this


def load_geom(path=HEATER_MAP):
    """net (== geometric heater H) -> (gds_x, gds_y, stage). Higher gds_x is more input-side."""
    return {int(r["H"]): (float(r["gds_x"]), float(r["gds_y"]), r["stage"])
            for r in csv.DictReader(open(path))}


def half_of(gy):
    return "top" if gy > CENTER_Y else "bot"


def derive_schedule(cfg, geom):
    """Outside-in schedule from config + geometry. Returns (groups, unreachable).

    A group = still-dark drivable targets at one (half, column-depth) frontier, ordered
    outermost-row-first. Groups run input-depth-first so each target's same-half upstream
    is already characterized when it is probed. `hold_upstream` lists the path-opening
    heaters (same half, more input-side) to hold at V0 -- split into already-characterized
    and to-be (scheduled in an earlier group)."""
    blocked = set(cfg["summary"]["dead"]) | set(cfg["summary"]["short"])
    xs = sorted({v[0] for v in geom.values()}, reverse=True)  # input-first column depths
    depth_rank = {x: i for i, x in enumerate(xs)}

    def characterized(c):
        return c.get("Vnull") is not None

    def drivable(c):
        return c.get("elec") == "ok" and c["dac"] not in blocked

    init_char = set()  # dacs characterized in the starting config, with geometry
    for c in cfg["channels"]:
        if characterized(c) and c.get("net") in geom:
            init_char.add(c["dac"])

    targets, unreachable = [], []
    for c in cfg["channels"]:
        if characterized(c) or not drivable(c):
            continue
        net = c.get("net")
        if net is None or net not in geom:
            unreachable.append({"dac": c["dac"], "stage": c.get("stage"),
                                "reason": "no net -- not wired to a mesh position"})
            continue
        gx, gy, stage = geom[net]
        targets.append({"dac": c["dac"], "net": net, "stage": stage, "gx": gx, "gy": gy,
                        "half": half_of(gy), "depth_rank": depth_rank[gx]})

    # geometry of every net-bearing channel (for upstream lookups)
    net_geom = {}  # dac -> (half, gx)
    for c in cfg["channels"]:
        if c.get("net") in geom:
            gx, gy, _ = geom[c["net"]]
            net_geom[c["dac"]] = (half_of(gy), gx)

    # group by (half, depth_rank); order groups depth-first then top-before-bottom
    buckets = defaultdict(list)
    for t in targets:
        buckets[(t["depth_rank"], t["half"])].append(t)
    order = sorted(buckets, key=lambda k: (k[0], 0 if k[1] == "top" else 1))

    groups, will_be_char = [], set(init_char)
    for gi, (drank, half) in enumerate(order):
        ts = sorted(buckets[(drank, half)], key=lambda t: -abs(t["gy"] - CENTER_Y))
        gx = ts[0]["gx"]
        up = [d for d in will_be_char if net_geom[d][0] == half and net_geom[d][1] > gx]
        already = sorted(d for d in up if d in init_char)
        tobe = sorted(d for d in up if d not in init_char)
        groups.append({
            "group": gi, "half": half, "stage": ts[0]["stage"],
            "gds_x": gx, "depth_rank": drank,
            "targets": [{"dac": t["dac"], "net": t["net"], "gds_y": t["gy"],
                         "dist_from_center": round(abs(t["gy"] - CENTER_Y), 1)} for t in ts],
            "hold_upstream_characterized": already,
            "hold_upstream_to_be": tobe,
        })
        will_be_char |= {t["dac"] for t in ts}
    return groups, unreachable


def print_schedule(groups, unreachable):
    n_t = sum(len(g["targets"]) for g in groups)
    print(f"\noutside-in schedule: {len(groups)} groups, {n_t} targets, "
          f"{len(unreachable)} unreachable\n")
    print(f"{'grp':>3} {'half':>4} {'stage':>10} {'gds_x':>8} {'depth':>6} "
          f"{'#tgt':>4} {'#hold':>6}  targets (dac:net, outermost first)")
    for g in groups:
        nhold = len(g["hold_upstream_characterized"]) + len(g["hold_upstream_to_be"])
        tg = " ".join(f"{t['dac']}:{t['net']}" for t in g["targets"])
        print(f"{g['group']:>3} {g['half']:>4} {g['stage']:>10} {g['gds_x']:>8.1f} "
              f"{g['depth_rank']:>4}/{max(x['depth_rank'] for x in groups):>1} "
              f"{len(g['targets']):>4} {nhold:>6}  {tg}")
    if unreachable:
        print("\nUNREACHABLE (no lit path exists under any transparent setting):")
        for u in unreachable:
            print(f"  dac {u['dac']:>3}  {u['stage']:<8}  {u['reason']}")


def save_schedule(groups, unreachable, path=SCHEDULE_OUT):
    doc = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "center_y": CENTER_Y,
            "ordering": "depth (input-side first) major, outermost-row minor; halves independent",
            "n_groups": len(groups),
            "n_targets": sum(len(g["targets"]) for g in groups),
            "n_unreachable": len(unreachable),
            "note": "orchestrator holds ALL characterized heaters at V0; hold_upstream is the "
                    "path-opening subset (same half, more input-side) for documentation.",
        },
        "groups": groups,
        "unreachable": unreachable,
    }
    json.dump(doc, open(path, "w"), indent=2)
    print(f"\nsaved schedule -> {path}")


def base_from_cfg(cfg, geom, num_dac):
    """All currently-characterized heaters held at their V0 (the transparent frontier)."""
    v = np.zeros(num_dac)
    for c in cfg["channels"]:
        if c.get("Vnull") is not None and c.get("V0") is not None:
            v[c["dac"]] = float(c["V0"])
    return v


def merge_rec(cfg, by_dac, ch, rec, src):
    """Fold one reliable fringe into the config (shared apply_fringe rule) and recount."""
    apply_fringe(by_dac[ch], rec, src)
    cfg["summary"]["characterized"] = sum(
        1 for x in cfg["channels"] if x.get("Vnull") is not None)


def frontier_mock_forward(cfg, geom, seed=0, num_dac=128, num_pd=NUM_ADC_RAW):
    """Frontier-gated forward for --mock: each net-bearing heater has a fringe on one
    dominant PD, but its amplitude is GATED by its same-half upstream heaters being held
    near their V0. So a deep target is DARK at the all-zero base and only lights up once
    the transparent frontier holds its upstream at V0 -- proving the ordering matters.
    Gate references the live cfg V0 (clipped to the 0..4 V the DAC reaches), so it opens
    exactly on the value the orchestrator holds, and grows as fits are merged in."""
    rng = np.random.default_rng(seed)
    live_pd = [i for i in range(num_pd) if i not in DAMAGED_PDS]
    phi2 = (np.pi / VPI_NOMINAL**2) * rng.uniform(0.7, 1.4, (num_pd, num_dac))
    phi0 = rng.uniform(0, 2 * np.pi, (num_pd, num_dac))
    amp = np.full((num_pd, num_dac), 0.004)
    baseDC = 0.25 + 0.03 * rng.random(num_pd)
    for c in cfg["channels"]:
        if c.get("net") in geom:
            amp[int(rng.choice(live_pd)), c["dac"]] = rng.uniform(0.03, 0.06)

    half = {c["dac"]: half_of(geom[c["net"]][1]) for c in cfg["channels"] if c.get("net") in geom}
    gx = {c["dac"]: geom[c["net"]][0] for c in cfg["channels"] if c.get("net") in geom}
    up_dacs = {d: [m for m in gx if half[m] == half[d] and gx[m] > gx[d]] for d in gx}
    by_dac = {c["dac"]: c for c in cfg["channels"]}
    net_dacs = list(gx)
    sigma = 0.8

    def forward(v):
        v = np.asarray(v, float)
        y = baseDC.copy()
        for d in net_dacs:
            g = 1.0
            for u in up_dacs[d]:
                cu = by_dac[u]
                if cu.get("Vnull") is None or cu.get("V0") is None:
                    continue  # uncharacterized upstream can't be held -> not part of the gate
                center = min(max(float(cu["V0"]), 0.0), 4.0)
                g *= np.exp(-((v[u] - center) / sigma) ** 2)
                if g < 1e-6:
                    break
            if g < 1e-6:
                continue
            y += amp[:, d] * np.cos(phi2[:, d] * v[d] ** 2 + phi0[:, d]) * g
        return np.clip(y, 0.0, None)

    return forward


def sweep_target(pic, base, dac, levels, repeats, settle_s, w, dbm, tel):
    """Sweep one target 0..levels through the held base; stream every raw read to the CSV
    (probe_channels schema, so census_fringes can read it back). `tel` is (bfm, temp, mA).
    Returns the rows written, so this one target can be fit from an isolated slice."""
    rows = []

    def block(v, phase, lvl):
        Y = []
        for r in range(repeats):
            y = pic.measure_raw(v)
            Y.append(y)
            row = [f"{time.time():.2f}", dac, phase, r, lvl, dbm,
                   f"{tel[0]:.3f}", f"{tel[1]:.2f}", f"{tel[2]:.2f}"] + [f"{x:.4f}" for x in y]
            w.writerow(row)
            rows.append(row)
        return np.mean(Y, axis=0)

    b = base.copy()
    b[dac] = 0.0
    pic.measure(b, settle_s=settle_s)
    block(b, "base", 0.0)
    for lv in levels:
        v = b.copy()
        v[dac] = lv
        pic.measure(v, settle_s=settle_s)
        block(v, "drive", lv)
    return rows


def fit_one(rows, tmp_csv, vmax=4.0):
    """Fit a single target from its own rows by analysing an isolated one-channel slice
    (census_fringes.analyze reuse, without re-fitting the whole growing census)."""
    with open(tmp_csv, "w", newline="") as tf:
        tw = csv.writer(tf)
        tw.writerow(CENSUS_HEADER)
        tw.writerows(rows)
    recs = analyze(tmp_csv, vmax=vmax, min_vis=0.005, min_swing=5.0)
    return recs[0] if recs else None


def estimate_seconds(n_targets, levels, repeats, settle_s):
    per = (len(levels) + 1) * (settle_s + repeats * 0.15 + 0.1)
    return n_targets * per * 1.3 + 30.0


def run(a):
    cfg = json.load(open(a.config))
    geom = load_geom()
    num_dac = cfg["firmware"]["num_dac"]
    groups, unreachable = derive_schedule(cfg, geom)
    print_schedule(groups, unreachable)
    save_schedule(groups, unreachable)
    if a.plan:
        return 0
    if a.limit:
        groups = groups[:a.limit]
    n_targets = sum(len(g["targets"]) for g in groups)

    if a.mock:
        tmp = tempfile.mkdtemp(prefix="rechar_mock_")
        census_out = a.out or os.path.join(tmp, "census.csv")
        out_config = a.out_config or os.path.join(tmp, "config.json")
    else:
        census_out = a.out or time.strftime("pic_data/census/rechar_%Y%m%d_%H%M%S.csv")
        out_config = a.out_config or a.config
    os.makedirs(os.path.dirname(census_out), exist_ok=True)

    levels = [float(x) for x in a.levels.split(",")]
    settle = 0.0 if a.mock else a.settle  # MockPIC has no thermal dynamics to wait on
    dbm = min(a.dbm, MAX_DBM)
    if a.dbm > MAX_DBM:
        print(f"note: clamping {a.dbm} -> {MAX_DBM} dBm (PD-damage ceiling)")
    duration = max(a.duration, estimate_seconds(n_targets, levels, a.repeats, a.settle))

    if a.mock:
        from laser.mock import MockLaser
        from template import laser_session
        laser = MockLaser().open()
        pic = MockPIC(frontier_mock_forward(cfg, geom, num_dac=num_dac),
                      noise=3e-4, num_dac=num_dac).open()
        _mock_demo(pic, cfg, geom, groups, levels)
    else:
        from laser.laser import Laser
        from template import laser_session, _resolve_pic_port
        laser = Laser().open()
        pic = PIC(port=_resolve_pic_port(a.port, laser.dev.port), num_dac=num_dac).open()

    by_dac = {c["dac"]: c for c in cfg["channels"]}
    src = stamp(census_out)
    tmp_csv = os.path.join(os.path.dirname(census_out) or ".", ".rechar_fit_slice.csv")
    f = open(census_out, "w", newline="")
    w = csv.writer(f)
    w.writerow(CENSUS_HEADER)
    done = 0
    print(f"\nrunning {len(groups)} groups / {n_targets} targets  @ {dbm:+.0f} dBm  "
          f"levels={levels}  -> {census_out}")
    try:
        with laser_session(laser, duration_s=duration, power_dbm=dbm) as ls:
            print(f"emitted={ls.emitted}  bfm {ls.bfm_off:.3f}->{ls.bfm_on:.3f}  mA={ls.mA:.1f}")
            for g in groups:
                base = base_from_cfg(cfg, geom, num_dac)  # frontier re-read -> grows per group
                nchar = int((base > 0).sum())
                print(f"\ngroup {g['group']:>2} [{g['half']} {g['stage']} x={g['gds_x']:.0f}] "
                      f"{len(g['targets'])} target(s); holding {nchar} heaters at V0")
                for t in g["targets"]:
                    dac = t["dac"]
                    ls.keepalive()
                    rows = sweep_target(pic, base, dac, levels, a.repeats, settle,
                                        w, dbm, ls.telemetry())
                    f.flush()
                    rec = fit_one(rows, tmp_csv, a.vmax)
                    sw = rec["swing_mV"] if rec else float("nan")
                    if rec and rec.get("reliable"):
                        merge_rec(cfg, by_dac, dac, rec, src)
                        json.dump(cfg, open(out_config, "w"), indent=2)
                        done += 1
                        print(f"  dac {dac:>3} (net {t['net']:>3}): swing {sw:5.1f} mV  "
                              f"-> V0={rec['V0']} Vnull={rec['Vnull']} pd{rec['pd']}  CHARACTERIZED")
                    else:
                        print(f"  dac {dac:>3} (net {t['net']:>3}): swing {sw:5.1f} mV  "
                              f"-> still dark (no reliable fringe)")
    finally:
        f.flush()
        f.close()
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        pic.set_zero()
        pic.close()
        laser.close()
    print(f"\ndone: characterized {done}/{n_targets} targets; config now "
          f"{cfg['summary']['characterized']} characterized  -> {out_config}")
    return 0


def _mock_demo(pic, cfg, geom, groups, levels):
    """Counterfactual proving the frontier logic: take the deepest target that has any
    already-characterized upstream, and compare its dominant-PD swing swept through an
    all-zero base (upstream dark) vs. through the transparent frontier (upstream at V0)."""
    cand = None
    for g in reversed(groups):
        if g["hold_upstream_characterized"]:
            cand = g["targets"][0]["dac"]
            break
    if cand is None:
        return
    num_dac = cfg["firmware"]["num_dac"]

    def swing(base):
        b = base.copy()
        b[cand] = 0.0
        rows = [pic.measure_raw(b)]
        for lv in levels:
            v = b.copy()
            v[cand] = lv
            rows.append(pic.measure_raw(v))
        P = np.array(rows)
        return (P.max(0) - P.min(0)).max() * 1000

    dark = swing(np.zeros(num_dac))
    lit = swing(base_from_cfg(cfg, geom, num_dac))
    print(f"[mock frontier demo] dac {cand}: swing dark-base {dark:.1f} mV  vs  "
          f"frontier-base {lit:.1f} mV  ({lit / max(dark, 1e-6):.0f}x)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--plan", action="store_true", help="derive+print+save schedule, no hardware")
    ap.add_argument("--mock", action="store_true", help="run the whole loop with no hardware")
    ap.add_argument("--dbm", type=float, default=15.0, help=f"laser power (<= {MAX_DBM:.0f})")
    ap.add_argument("--repeats", type=int, default=4, help="reads averaged per level")
    ap.add_argument("--settle", type=float, default=0.4)
    ap.add_argument("--levels", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0")
    ap.add_argument("--vmax", type=float, default=4.0,
                    help="fit reachability/reliability ceiling; set to max(levels) for a >4 V sweep")
    ap.add_argument("--port", default=os.environ.get("PIC_PORT"))
    ap.add_argument("--duration", type=float, default=0.0, help="laser watchdog floor [s]")
    ap.add_argument("--limit", type=int, default=0, help="only the first N groups (short test)")
    ap.add_argument("--out", default=None, help="census CSV (raw-read provenance)")
    ap.add_argument("--out-config", default=None, help="config to persist to (default = --config)")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
