"""Merge a census_fringes CSV into pic_b_config.json (in place, surgical).

For every RELIABLE fringe, fill the channel's operating points (V0/Vnull/Vpi/fringe_pd)
and swing/provenance. By default only fills channels that are still uncharacterized
(Vnull is None) -- earliest-wins, so the initial thermal frame is preserved and later
passes just extend coverage. --force overwrites already-characterized channels too.

Dead/short channels are never written. Updates summary.characterized. Provenance
(fringe_src) is the YYYYMMDD_HHMMSS stamp parsed from the fringes filename.

    python scripts/merge_fringes_to_config.py pic_data/census/fringes_20260730_225742.csv
    python scripts/merge_fringes_to_config.py fringes_*.csv --force
"""

from __future__ import annotations

import argparse
import csv
import json

from src.census import apply_fringe, stamp  # noqa: F401  (stamp re-exported for back-compat)


def fnum(s):
    s = (s or "").strip()
    return float(s) if s not in ("", "None") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fringes", nargs="+", help="one or more fringes_*.csv (earliest arg wins per channel)")
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--force", action="store_true", help="also overwrite already-characterized channels")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    by_dac = {c["dac"]: c for c in cfg["channels"]}
    blocked = set(cfg["summary"]["dead"]) | set(cfg["summary"]["short"])

    filled, skipped_present, skipped_blocked, skipped_unreliable = 0, 0, 0, 0
    for path in a.fringes:
        src = stamp(path)
        for r in csv.DictReader(open(path)):
            ch = int(r["ch"])
            c = by_dac.get(ch)
            if c is None or ch in blocked:
                skipped_blocked += 1
                continue
            if r.get("reliable", "").strip() != "True":
                skipped_unreliable += 1
                continue
            if c.get("Vnull") is not None and not a.force:
                skipped_present += 1
                continue
            apply_fringe(c, {"V0": fnum(r.get("V0")), "Vnull": fnum(r.get("Vnull")),
                             "Vpi": fnum(r.get("Vpi")), "pd": int(r["pd"]),
                             "swing_mV": fnum(r.get("swing_mV"))}, src)
            filled += 1

    characterized = sum(1 for c in cfg["channels"] if c.get("Vnull") is not None)
    cfg["summary"]["characterized"] = characterized
    json.dump(cfg, open(a.config, "w"), indent=2)

    print(f"filled {filled} channels  (skipped: {skipped_present} already-characterized, "
          f"{skipped_unreliable} unreliable, {skipped_blocked} dead/short/unknown)")
    print(f"config characterized now: {characterized} / {cfg['summary']['drivable']} drivable  -> {a.config}")


if __name__ == "__main__":
    main()
