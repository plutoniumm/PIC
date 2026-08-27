"""Fuse the picpin snake structure with the 2026-07-23 PIC-B census into one
128-channel DAC <-> heater scaffold.

Mechanism (user's "snake on A, mirror-shunted to B"): the DAC daughterboard has two
96-way headers; each header pin drives ONE mirror-shorted net (a Section-A heater shorted
to its Section-B twin -- `vendor_a`/`vendor_b`). picpin decoded both header pinouts into
pic_data/connector_pin_map.csv. The 4-DAC/64-ch board filled header A; the 8-DAC/128-ch
board fills both:

    chips 0-3 (ch 0-63)   -> header A heater pins, snake/pin order  -> DAC 0..59  (+4 spare)
    chips 4-7 (ch 64-127) -> header B heater pins, snake/pin order  -> DAC 64..123 (+4 spare)

The chip->header assignment is INFERRED, anchored by the census: ch0 is a strong input
encoder driving PD8/10/12 (header-A enc.split net H15), and the dead chip 4 (ch64-79) is
header B's first block (ribbon suspect). Placement WITHIN a column still needs the live
laser procedure -- this table is the scaffold that procedure fills/corrects.

Census truth (which channels actually drive, and which PDs they move) is overlaid from
pic_data/census/fringe_fits_b128.csv so training can mask the ~60 dead channels today.

Usage:  python scripts/map_b128.py [--write]
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIC_DATA = ROOT / "pic_data"
CONN = PIC_DATA / "connector_pin_map.csv"
FRINGE = PIC_DATA / "census" / "fringe_fits_b128.csv"
OUT = PIC_DATA / "dac_heater_map_b128.csv"

CHANS_PER_HEADER = 64          # 4 DAC chips per header
FOOT_THR_MV = 8.0             # a channel "moves" a PD if its fringe swing clears this

# 2026-07-30 electrical map-check: every channel driven 1 V (even) / 2 V (odd), one 16-ch
# block at a time, probed at the pin with a multimeter, laser off. Firmware CS corrected to
# {10,9,8,7,5,4,3,2} first -- with the old {..6,5,4,3} the whole B half was CS-shifted and
# read dead. This measures DAC-drive presence, NOT which heater the channel reaches.
ELEC_DEAD = {39, 51, 62, 98, 100, 111, 118, 120, 124, 125, 126}   # ~0 V at the pin
ELEC_WEAK = {99}                                                   # ~1.5-1.6 V into a 2 V set
ELEC_SHORT = {112, 113, 114, 115}                                  # all sit ~1.2-1.3 V, tied


def elec_status(dac: int) -> str:
    if dac in ELEC_DEAD:
        return "dead"
    if dac in ELEC_WEAK:
        return "weak"
    if dac in ELEC_SHORT:
        return "short"
    return "ok"


def load_header_pins():
    """header -> [(pin, net, stage, vendor_a, vendor_b)] ordered by pin number,
    heater pins only (GND/PD skipped, just as the snake skips them)."""
    rows = [r for r in csv.DictReader(l for l in CONN.open() if not l.startswith("#"))
            if r["kind"] == "heater"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["connector"]].append(
            (int(r["pin"]), int(r["heater_gds"]), r["stage"],
             int(r["vendor_a"]), int(r["vendor_b"])))
    for h in by:
        by[h].sort()
    return by


def load_census():
    """ch -> {pd: swing_mV} from the fringe fits."""
    foot = collections.defaultdict(dict)
    if not FRINGE.exists():
        return foot
    for r in csv.DictReader(FRINGE.open()):
        foot[int(r["ch"])][int(r["pd"])] = float(r["swing_mV"])
    return foot


def latest_fringes():
    """Newest census_fringes output (fringes_*.csv), or None."""
    fs = sorted((PIC_DATA / "census").glob("fringes_*.csv"))
    return fs[-1] if fs else None


def load_operating_points(paths):
    """ch -> operating points (V0, Vnull, Vpi) for reliable fits, unioned across `paths`.
    EARLIER files win per channel: the session heats up as it runs, so the earlier (cooler,
    less-drifted) fit is preferred; a later run only fills channels the earlier one missed.
    `fringe_src` records which run each point came from (its thermal frame)."""
    op = {}
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        src = Path(path).stem.replace("fringes_", "")
        for r in csv.DictReader(open(path)):
            if str(r.get("reliable")) != "True":
                continue
            ch = int(r["ch"])
            if ch in op:  # earlier run already has a reliable fit -> keep it (less drift)
                continue
            op[ch] = {"V0": r["V0"], "Vnull": r["Vnull"], "Vpi": r["Vpi"],
                      "fringe_pd": r["pd"], "fringe_src": src}
    return op


def build(fringes_paths=None):
    hdr = load_header_pins()
    foot = load_census()
    op = load_operating_points(fringes_paths or [latest_fringes()])
    rows = []
    for hi, header in enumerate(("A", "B")):
        base = hi * CHANS_PER_HEADER
        for ordinal, (pin, net, stage, va, vb) in enumerate(hdr[header]):
            dac = base + ordinal
            rows.append(dict(dac=dac, chip=dac // 16, header=header, pin=pin,
                             net=net, stage=stage, vendor_a=va, vendor_b=vb))
        for spare in range(base + len(hdr[header]), base + CHANS_PER_HEADER):
            rows.append(dict(dac=spare, chip=spare // 16, header=header, pin="",
                             net="", stage="spare", vendor_a="", vendor_b=""))
    rows.sort(key=lambda r: r["dac"])
    for r in rows:
        pds = {p: s for p, s in foot.get(r["dac"], {}).items() if s >= FOOT_THR_MV}
        r["census_npds"] = len(pds)
        r["census_swing_mV"] = round(sum(foot.get(r["dac"], {}).values()), 1)
        r["census_live"] = int(len(pds) > 0)
        r["census_pds"] = " ".join(str(p) for p in sorted(pds, key=lambda p: -pds[p]))
        r["elec"] = elec_status(r["dac"])
        fr = op.get(r["dac"], {})
        r["V0"] = fr.get("V0", "")
        r["Vnull"] = fr.get("Vnull", "")
        r["Vpi"] = fr.get("Vpi", "")
        r["fringe_pd"] = fr.get("fringe_pd", "")
        r["fringe_src"] = fr.get("fringe_src", "")
    return rows


def digest(rows):
    real = [r for r in rows if r["stage"] != "spare"]
    live = [r for r in real if r["census_live"]]
    print(f"mapped nets: {len(real)}/120   spare channels: {len(rows) - len(real)}")
    print(f"census-live channels: {len(live)}/{len(real)}\n")

    ok = [r for r in rows if r["elec"] == "ok"]
    print(f"2026-07-30 electrical map-check (all 128 DAC channels):")
    print(f"  drive OK: {len(ok)}/128   dead: {sorted(ELEC_DEAD)}")
    print(f"  weak: {sorted(ELEC_WEAK)}   shorted-cluster: {sorted(ELEC_SHORT)}\n")

    print(f"{'stage':<11}{'nets':>5}{'live':>6}  live channels (dac)")
    for stage in ("enc.split", "enc.single", "V", "sigma", "U"):
        st = [r for r in real if r["stage"] == stage]
        lv = [r for r in st if r["census_live"]]
        print(f"{stage:<11}{len(st):>5}{len(lv):>6}  {[r['dac'] for r in lv]}")

    print("\nper DAC chip (live / mapped):")
    by = collections.defaultdict(lambda: [0, 0])
    for r in real:
        by[r["chip"]][0] += r["census_live"]
        by[r["chip"]][1] += 1
    for k in sorted(by):
        note = "  <- DEAD (ribbon suspect)" if by[k][0] == 0 else ""
        print(f"  chip{k} ch{16 * k:>3}-{16 * k + 15:<3} header {'A' if k < 4 else 'B'}: "
              f"{by[k][0]:2d}/{by[k][1]:<2d}{note}")

    print("\ninput column (enc.split/enc.single) -- your outer=direct, middle=broad test:")
    inp = [r for r in real if r["stage"].startswith("enc")]
    for r in sorted(inp, key=lambda r: (r["header"], r["net"])):
        kind = ("--"if not r["census_live"] else
                "DIRECT" if r["census_npds"] <= 2 else
                "broad" if r["census_npds"] >= 8 else "mid")
        print(f"  dac{r['dac']:3d} net H{r['net']:<3} {r['stage']:<10} "
              f"npds={r['census_npds']:2d} swing={r['census_swing_mV']:6.0f}mV "
              f"pds=[{r['census_pds']}]  {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fringes", nargs="+", default=None,
                    help="one or more census_fringes CSVs for V0/Vnull/Vpi, EARLIEST FIRST "
                         "(earlier run wins per channel; default: newest fringes_*.csv)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = build(a.fringes)
    digest(rows)
    nop = sum(1 for r in rows if r["V0"])
    print(f"\nheaters with a reliable fringe (V0/Vpi operating points): {nop}")
    if a.write:
        cols = ["dac", "chip", "header", "pin", "net", "stage", "vendor_a", "vendor_b",
                "elec", "census_live", "census_npds", "census_swing_mV", "census_pds",
                "V0", "Vnull", "Vpi", "fringe_pd", "fringe_src"]
        with OUT.open("w", newline="") as f:
            f.write(
                "# 128-ch DAC <-> heater scaffold for PIC B. PROVISIONAL, verified=0.\n"
                "# structure: scripts/map_b128.py from picpin (mechanism) + census overlay.\n"
                "# `net` = mirror-shorted net 0..119 (= heater_gds); `stage` from geometry.\n"
                "# chip->header assignment inferred (ch0 input-encoder anchor); within-column\n"
                "# placement pending the live column-by-column probe.\n"
                "# `elec` = 2026-07-30 multimeter map-check (ok/dead/weak/short) after the CS\n"
                "# fix {10,9,8,7,5,4,3,2}. `census_*` = the OLD 07-23 fringe_fits (wrong CS).\n"
                "# V0/Vnull/Vpi = 2026-07-30 0-4V optical fringe fit (scripts/census_fringes,\n"
                "# reliable fits only): V0 = max-pass volts, Vnull = null volts, Vpi = sqrt(pi/phi2).\n"
                "# Operate a heater by normalizing between V0 and Vnull. Blank = no reliable\n"
                "# fringe at this coupling (dim / dead).\n")
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in cols} for r in rows)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
