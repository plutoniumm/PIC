"""Shared characterization primitives for the PIC-B census/fringe pipeline.

One home for the logic the census scripts used to each carry a copy of: the multi-start
fringe fit (``robust_fit``), census-CSV loading (``load_census``), the reliability-gated
per-channel fit (``analyze`` + ``digest``), the config-merge rule (``apply_fringe``), the
run-stamp parser (``stamp``), and the raw-stream column layout (``CENSUS_HEADER``).

The scripts in ``scripts/`` are thin CLIs over this module:
  * ``census_fringes.py``   -> fit a census CSV to operating points (analyze/digest)
  * ``merge_fringes_to_config.py`` -> fold reliable fits into pic_b_config.json (apply_fringe)
  * ``rechar_outsidein.py`` -> outside-in orchestrator (analyze + apply_fringe + CENSUS_HEADER)
  * ``parallel_char.py``    -> lockstep parallel sweep validator (robust_fit)

Pure functions over arrays/CSVs; no hardware.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .characterize import fit_fringe
from .sweep_analysis import fringe_extrema

NPD = 14
VPI_SEEDS = (0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5)  # multi-start: fringe period varies per heater

# raw-read stream layout written by probe_channels / rechar_outsidein and read by load_census
CENSUS_HEADER = ["t", "ch", "phase", "rep", "drive_v", "dbm", "bfm", "temp_c", "mA"] \
    + [f"pd{i}" for i in range(NPD)]


def robust_fit(v, p):
    """fit_fringe is sensitive to its phi2 seed (the V^2 law has many local minima over a
    wide sweep). Try several Vpi seeds and keep the lowest-rmse physical fit."""
    best = None
    for vpi0 in VPI_SEEDS:
        fit = fit_fringe(v, p, vpi0=vpi0)
        if fit is None:
            continue
        if best is None or fit["rmse"] < best["rmse"]:
            best = fit
    return best


def stamp(path: str) -> str:
    """The YYYYMMDD_HHMMSS run stamp parsed from a census/fringes filename (its thermal frame)."""
    m = re.search(r"(\d{8}_\d{6})", Path(path).name)
    return m.group(1) if m else Path(path).stem


def load_census(path):
    """ch -> (levels[nv], P[nv, 14]) with v=0 taken from the base phase."""
    base = defaultdict(list)
    drive = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            ch = int(r["ch"])
            pds = [float(r[f"pd{i}"]) for i in range(NPD)]
            (base[ch] if r["phase"] == "base" else drive[ch][float(r["drive_v"])]).append(pds)
    out = {}
    for ch, dv in drive.items():
        levels = [0.0] + [v for v in sorted(dv) if v > 0]
        rows = [np.mean(base[ch], axis=0) if base[ch] else np.zeros(NPD)]
        rows += [np.mean(dv[v], axis=0) for v in levels[1:]]
        out[ch] = (np.array(levels), np.array(rows))
    return out


def analyze(path, vmax=2.0, min_vis=0.15, min_swing=5.0):
    """Per channel: fit its most-responsive PD's fringe and read off V0/Vnull/Vpi, flagging
    ``reliable`` when the fit is physical, well-fit, and turns over inside the swept band."""
    data = load_census(path)
    recs = []
    for ch, (v, P) in sorted(data.items()):
        ptp = P.max(0) - P.min(0)
        pd = int(np.argmax(ptp))
        swing = float(ptp[pd]) * 1000
        fit = robust_fit(v, P[:, pd])
        rec = {"ch": ch, "pd": pd, "swing_mV": round(swing, 1)}
        if fit is None:
            rec.update(ok=False)
            recs.append(rec)
            continue
        ex = fringe_extrema(fit, vmax=vmax)
        vis, rmse_mV, vpi = fit["visibility"], fit["rmse"] * 1000, fit["Vpi"]
        # Trust is SNR-based, not contrast-based: a light-routed baseline (many heaters held
        # at V0) floods every PD with DC, so a real strong fringe can still show tiny
        # visibility. So gate on absolute swing (signal), rmse small vs swing (well fit), and
        # a Vpi that actually turns over in range (rejects monotonic/degenerate fits). min_vis
        # only screens sign-degenerate fits (negative visibility); default 0.15 for dark-baseline runs.
        reliable = (min_vis <= vis <= 1.05 and rmse_mV <= 0.2 * swing + 1
                    and 0.5 <= vpi <= vmax + 1 and swing >= min_swing)
        rec.update(
            ok=True,
            reliable=reliable,
            Vpi=round(vpi, 3),
            phi0=round(fit["phi0"], 3),
            visibility=round(vis, 3),
            rmse_mV=round(rmse_mV, 2),
            V0=round(ex["v_at_max"], 3) if ex["v_at_max"] is not None else None,
            Vnull=round(ex["v_at_min"], 3) if ex["v_at_min"] is not None else None,
            V0_reach=ex["max_reachable"],
            Vnull_reach=ex["min_reachable"],
        )
        recs.append(rec)
    return recs


def digest(recs, vmax=2.0):
    """Terse scannable digest of ``analyze`` records to stdout."""
    good = [r for r in recs if r.get("ok") and r["swing_mV"] >= 5]
    rel = [r for r in good if r["reliable"]]
    print(f"channels fit: {len(recs)}   usable swing (>=5 mV): {len(good)}   "
          f"RELIABLE fringe: {len(rel)}\n")
    print(f"{'ch':>3} {'pd':>3} {'swing':>6} {'Vpi':>5} {'V0':>5} {'Vnull':>6} "
          f"{'reach':>7} {'vis':>5} {'rmse':>5}")
    for r in sorted(rel, key=lambda r: -r["swing_mV"]):
        reach = ("V0" if r["V0_reach"] else "--") + "/" + ("Vn" if r["Vnull_reach"] else "--")
        print(f"{r['ch']:>3} {r['pd']:>3} {r['swing_mV']:>6.1f} {r['Vpi']:>5.2f} "
              f"{(r['V0'] if r['V0'] is not None else float('nan')):>5.2f} "
              f"{(r['Vnull'] if r['Vnull'] is not None else float('nan')):>6.2f} "
              f"{reach:>7} {r['visibility']:>5.2f} {r['rmse_mV']:>5.1f}")
    nreach = sum(r["Vnull_reach"] for r in rel)
    q = [r for r in good if not r["reliable"]]
    print(f"\nof reliable: null in range (<=2 V): {nreach}/{len(rel)}   "
          f"null past clamp (fit-extrapolated): {len(rel) - nreach}")
    print(f"{len(q)} channels have swing but no trustworthy fringe (monotonic / null far "
          f"past 2 V): {sorted(r['ch'] for r in q)}")


def apply_fringe(channel: dict, rec: dict, src: str) -> None:
    """Fold one reliable fringe record into a config channel (in place). ``rec`` carries
    numeric ``V0/Vnull/Vpi/swing_mV`` (or None) and an integer ``pd``. Callers gate on
    ``reliable`` and on dead/short before calling this."""
    channel["V0"] = rec.get("V0")
    channel["Vnull"] = rec.get("Vnull")
    channel["Vpi"] = rec.get("Vpi")
    channel["fringe_pd"] = int(rec["pd"])
    channel["swing_mV"] = rec.get("swing_mV")
    channel["fringe_src"] = src
