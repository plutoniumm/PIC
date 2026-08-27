"""Decode NUS/picpin.drawio into a DAC-channel -> heater map.

The drawio has two pages of the same schematic:
  page 1  heaters in the vendor's full numbering H0..H239 (240 physical heaters)
  page 2  same drawing, but every Section-A heater relabelled H0..H119 --
          the 120 electrically independent drive nets (A/B are mirror-shorted)

Both pages carry the two 96-way board connectors. Each connector pin is
labelled either "Ha - Hb PINS" (a shorted mirror pair, vendor numbering),
"GND (Hx-Hy)", "PDn" or "NC". The 64-channel DAC board is wired to the top
connector: channel n lands on the n-th *heater* pin, GND/PD pins skipped.
29 of those runs are drawn as explicit edges; this script rebuilds all 64 and
checks them against the drawn ones.

Usage:  python scripts/picpin_map.py [--write]
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAWIO = ROOT / "NUS" / "picpin.drawio"
OUT_DIR = ROOT / "pic_data"

MIRROR_Y = 1220.5  # screen y of the Section A | Section B mirror axis
TOP = dict(rail=(845, 862), box=(755, 800), xmax=2480)
BOT = dict(rail=(1545, 1562), box=(1650, 1700), xmax=2640)


def _text(v: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", v or "")).strip()


@dataclass
class Cell:
    x: float
    y: float
    w: float
    h: float
    value: str
    style: str
    ident: str

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


def load_page(idx: int) -> list[Cell]:
    root = ET.parse(DRAWIO).getroot()
    page = root.findall("diagram")[idx]
    out = []
    for c in page.findall(".//mxCell"):
        g = c.find("mxGeometry")
        if g is None or g.get("x") is None:
            continue
        out.append(
            Cell(
                float(g.get("x")),
                float(g.get("y")),
                float(g.get("width") or 0),
                float(g.get("height") or 0),
                _text(c.get("value")),
                c.get("style") or "",
                c.get("id"),
            )
        )
    return out


def rail_pins(cells: list[Cell], band: tuple[float, float], xmax: float) -> list[float]:
    """x centres of a 96-way connector's pins, ordered pin 1 (rightmost) .. 96."""
    lo, hi = band
    xs = [c.cx for c in cells
          if lo <= c.y < hi and 0 < c.cx < xmax and c.style.startswith("text")]
    xs.sort(reverse=True)
    assert len(xs) == 96, f"expected 96 rail pins, got {len(xs)}"
    return xs


PAIR_RE = re.compile(r"^H(\d+)\s*-\s*H?(\d+)\s*PINS$")
GND_RE = re.compile(r"^GND\s*\(\s*H?(\d+)\s*-\s*H?(\d+)\s*\)?$")
PD_RE = re.compile(r"^PD(\d+)$")


# Connector B pin 45 is labelled "H36 - H151"; H36 sits on connector A pin 13 and
# H151's mirror partner in block H132-H155 (pairs sum to 287) is H136.
TYPOS = {"H36 - H151 PINS": "H136 - H151 PINS"}


def classify(label: str) -> tuple[str, tuple]:
    label = re.sub(r"\s+", " ", label).strip()
    label = TYPOS.get(label, label)
    if m := PAIR_RE.match(label):
        return "heater", (int(m.group(1)), int(m.group(2)))
    if m := GND_RE.match(label):
        return "gnd", (int(m.group(1)), int(m.group(2)))
    if m := PD_RE.match(label):
        return "pd", (int(m.group(1)),)
    if label.startswith("GND"):
        return "gnd", ()
    if label == "NC":
        return "nc", ()
    return "other", ()


def connector(cells: list[Cell], spec: dict) -> list[dict]:
    """pin number -> what the drawio says is on it."""
    xs = rail_pins(cells, spec["rail"], spec["xmax"])
    lo, hi = spec["box"]
    boxes = [c for c in cells if lo <= c.y < hi and c.x > 0 and c.style.startswith("rounded=1")]
    rows = []
    for pin, px in enumerate(xs, start=1):
        near = min(boxes, key=lambda c: abs(c.cx - px))
        if abs(near.cx - px) > 14:
            rows.append(dict(pin=pin, label="", kind="unlabelled", arg=()))
            continue
        kind, arg = classify(near.value)
        rows.append(dict(pin=pin, label=near.value, kind=kind, arg=arg))
    return rows


def heater_positions(cells: list[Cell]) -> dict[int, list[tuple[float, float]]]:
    pos = defaultdict(list)
    for c in cells:
        m = re.fullmatch(r"H(\d+)", c.value)
        if m and 1000 < c.y < 1440 and 850 < c.x < 1950:
            pos[int(m.group(1))].append((c.x, c.y))
    return pos


def section_of(y: float) -> str:
    return "A" if y < MIRROR_Y else "B"


def build_numbering() -> tuple[dict[int, int], dict[int, tuple[float, float]]]:
    """vendor heater id -> (new 0..119 id if it is a Section-A heater), and positions."""
    old = heater_positions(load_page(0))
    new = heater_positions(load_page(1))
    at = {}  # (x, y) -> new id, Section A only
    for nid, places in new.items():
        for x, y in places:
            if section_of(y) == "A":
                at[(round(x), round(y))] = nid
    old2new, oldpos = {}, {}
    for oid, places in old.items():
        for x, y in places:
            oldpos.setdefault(oid, (x, y))
            key = (round(x), round(y))
            if key in at:
                old2new[oid] = at[key]
    return old2new, oldpos


def gds_join() -> tuple[dict[int, int], dict[int, str]]:
    """drawio H0..H119 -> the same physical heater's id in pic_data/heater_map.csv.

    Both numberings walk the same 15 Section-A columns with the same populations
    (16,8,6,8,6,8,6,8,12,6,8,6,8,6,8), so the columns pair off by x rank; inside a
    column heater_map.csv sorts by descending gds_y (top to bottom) while the
    drawio keeps the vendor's serpentine, so 8 of the 15 columns come out flipped.
    """
    pts = [(x, y, h) for h, pl in heater_positions(load_page(1)).items()
           for x, y in pl if section_of(y) == "A"]
    pts.sort(key=lambda t: -t[0])
    cols: list[list] = []
    for x, y, h in pts:
        if not cols or abs(cols[-1][0][0] - x) > 20:
            cols.append([])
        cols[-1].append((x, y, h))

    gcol = defaultdict(list)
    with (OUT_DIR / "heater_map.csv").open() as f:
        for r in csv.DictReader(f):
            gcol[float(r["gds_x"])].append((-float(r["gds_y"]), int(r["H"]), r["stage"]))

    to_gds, stage = {}, {}
    for col, gx in zip(cols, sorted(gcol, reverse=True)):
        drawio = [h for _, _, h in sorted(col, key=lambda t: t[1])]
        geom = sorted(gcol[gx])
        assert len(drawio) == len(geom), f"column population mismatch at x={gx}"
        for d, (_, g, st) in zip(drawio, geom):
            to_gds[d], stage[d] = g, st
    return to_gds, stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write CSVs into pic_data/")
    ap.add_argument("--promote", action="store_true",
                    help="also overwrite pic_data/dac_heater_map.csv (the map wiring.py loads)")
    args = ap.parse_args()

    page2 = load_page(1)
    top = connector(page2, TOP)
    bot = connector(page2, BOT)
    old2new, oldpos = build_numbering()
    to_gds, stage = gds_join()

    pairs_top = [r for r in top if r["kind"] == "heater"]
    pairs_bot = [r for r in bot if r["kind"] == "heater"]
    seen = defaultdict(list)
    for side, rows in (("A", pairs_top), ("B", pairs_bot)):
        for r in rows:
            seen[tuple(sorted(r["arg"]))].append((side, r["pin"]))
    dupes = {k: v for k, v in seen.items() if len(v) > 1}

    print(f"connector A (top): {len(pairs_top)} heater pins, "
          f"{sum(r['kind'] == 'gnd' for r in top)} gnd, {sum(r['kind'] == 'pd' for r in top)} pd")
    print(f"connector B (bot): {len(pairs_bot)} heater pins, "
          f"{sum(r['kind'] == 'gnd' for r in bot)} gnd, {sum(r['kind'] == 'pd' for r in bot)} pd")
    print(f"distinct pairs: {len(seen)}   duplicated across connectors: {len(dupes)}")

    # DAC channels walk the top connector's heater pins in pin order.
    dac_rows = []
    for n, r in enumerate(pairs_top):
        if n >= 64:
            break
        a, b = r["arg"]
        sec = {section_of(oldpos[h][1]): h for h in (a, b) if h in oldpos}
        h_a, h_b = sec.get("A"), sec.get("B")
        dac_rows.append(dict(
            dac=n, chip=n // 16, ch=n % 16, pin=r["pin"], label=r["label"],
            vendor_a=h_a, vendor_b=h_b, heater=old2new.get(h_a),
        ))

    drawn = drawn_edges()
    bad = [d for d in dac_rows if d["dac"] in drawn and drawn[d["dac"]] != tuple(sorted(
        (d["vendor_a"], d["vendor_b"])))]
    print(f"drawn DAC->pin edges: {len(drawn)}   disagreements with reconstruction: {len(bad)}")
    for d in bad:
        print("  MISMATCH", d)

    print(f"DAC channels driving heaters: {len(dac_rows)} of 64 "
          f"(spare: {', '.join(f'{c}.{h}' for c, h in [(n // 16, n % 16) for n in range(len(dac_rows), 64)])})")

    flipped = sorted(h for h in to_gds if to_gds[h] != h)
    print(f"drawio H0..H119 vs pic_data/heater_map.csv: {120 - len(flipped)} identical, "
          f"{len(flipped)} differ (8 columns are serpentine-reversed)")

    print("\n dac  chip.ch  pinA   sec-A / sec-B    stage       H(drawio)  H(gds)")
    for d in dac_rows:
        h = d["heater"]
        print(f" {d['dac']:3d}   {d['chip']}.{d['ch']:<2d}   {d['pin']:4d}   "
              f"H{d['vendor_a']:<4d}/ H{d['vendor_b']:<4d}  {stage[h]:<10}  H{h:<9d} H{to_gds[h]}")

    print("\nconnector B heater pins (no DAC board wired yet):")
    for r in pairs_bot:
        a, b = r["arg"]
        sec = {section_of(oldpos[h][1]): h for h in (a, b) if h in oldpos}
        h = old2new[sec["A"]]
        print(f"  pin {r['pin']:3d}   H{sec['A']:<4d}/ H{sec['B']:<4d}  {stage[h]:<10}  "
              f"H{h:<9d} H{to_gds[h]}")

    if not args.write:
        return

    def row_for(r):
        a, b = r["arg"]
        sec = {section_of(oldpos[h][1]): h for h in (a, b) if h in oldpos}
        h = old2new[sec["A"]]
        return sec["A"], sec["B"], h, to_gds[h], stage[h]

    header = (
        "# Generated by scripts/picpin_map.py from NUS/picpin.drawio -- do not hand-edit.\n"
        "# Schematic-derived, NOT hardware-verified. `heater_gds` is the id used by\n"
        "# pic_data/heater_map.csv and src.pic.layout; `heater_drawio` is the page-2\n"
        "# label, which runs the vendor serpentine and so is reversed inside 8 of the\n"
        "# 15 Section-A columns. `vendor_a`/`vendor_b` are the mirror-shorted pair in\n"
        "# the vendor's H0..H239 numbering (Section A / Section B); one pin drives both.\n"
    )

    p = OUT_DIR / "dac_heater_map_picpin.csv"
    with p.open("w", newline="") as f:
        f.write(header)
        w = csv.writer(f)
        w.writerow(["dac", "dac_chip", "dac_ch", "connector", "pin", "heater_drawio",
                    "heater_gds", "stage", "vendor_a", "vendor_b", "drawn_edge"])
        for d in dac_rows:
            h = d["heater"]
            w.writerow([d["dac"], d["chip"], d["ch"], "A", d["pin"], h, to_gds[h],
                        stage[h], d["vendor_a"], d["vendor_b"], int(d["dac"] in drawn)])
    print(f"\nwrote {p}")

    p = OUT_DIR / "connector_pin_map.csv"
    with p.open("w", newline="") as f:
        f.write(header)
        w = csv.writer(f)
        w.writerow(["connector", "pin", "kind", "label", "heater_drawio", "heater_gds",
                    "stage", "vendor_a", "vendor_b"])
        for side, rows in (("A", top), ("B", bot)):
            for r in rows:
                extra = row_for(r) if r["kind"] == "heater" else ("", "", "", "", "")
                va, vb, h, hg, st = extra
                w.writerow([side, r["pin"], r["kind"], r["label"], h, hg, st, va, vb])
    print(f"wrote {p}")

    if args.promote:
        promote(dac_rows, to_gds, stage, drawn)


# Marginal-eta^2 notes from the pooled 22-July 100k set (permutation null ~0.0011);
# kept only where they say something the schematic does not.
DATA_NOTES = {
    0: "CONFLICT: silent in 75k data (2x null) though wired here",
    1: "supported: strongest channel (1342x null), top reference LO modulator",
    14: "supported: very strong (798x null)",
    15: "supported: very strong (980x null)",
    48: "supported: strong (168x null)",
    63: "CONFLICT: diagram says unconnected, but alive at 26x null in 75k data",
}


def promote(dac_rows, to_gds, stage, drawn) -> None:
    """Rewrite pic_data/dac_heater_map.csv from the schematic (still verified=0)."""
    p = OUT_DIR / "dac_heater_map.csv"
    with p.open("w", newline="") as f:
        f.write(
            "# DAC channel -> geometric heater index (H0..H119; see pic_data/heater_map.csv).\n"
            "#\n"
            "# SOURCE: NUS/picpin.drawio, decoded by scripts/picpin_map.py --promote.\n"
            "# STATUS: schematic-derived, NOT hardware-verified -> verified=0 on every row.\n"
            "#\n"
            "# The board's two 96-way connectors each carry 60 of the chip's 120 heater nets\n"
            "# (each net = one mirror-shorted Section-A/Section-B pair). The 64-channel DAC\n"
            "# board is wired to connector A only: channel n lands on the n-th heater pin,\n"
            "# GND and PD pins skipped. 29 of those runs are drawn as edges in the drawio and\n"
            "# all 29 agree with this reconstruction; the rest follow by sequential fill.\n"
            "# Connector A turns out to be exactly the inner half of every column -- the 60\n"
            "# heaters nearest the A|B mirror axis -- so DAC 60..63 have nothing to drive.\n"
            "#\n"
            "# Reaching the other 60 nets needs a second DAC board on connector B; see\n"
            "# pic_data/connector_pin_map.csv for its pinout.\n"
            "#\n"
            "# TO VERIFY: python scripts/dac_heater_probe.py sweep  (64 ch x 9 levels, laser lit)\n"
            "# Then set verified=1 on the rows it settles. Note the CONFLICT rows below first.\n"
            "#\n"
            "# Leave H blank to mark a channel as deliberately unassigned. `#` comments are ignored.\n"
        )
        w = csv.writer(f)
        w.writerow(["dac", "H", "verified", "stage", "evidence"])
        by_dac = {d["dac"]: d for d in dac_rows}
        for n in range(64):
            note = DATA_NOTES.get(n, "")
            if n not in by_dac:
                w.writerow([n, "", 0, "", "; ".join(filter(None, [
                    "unconnected: connector A has only 60 heater pins", note]))])
                continue
            d = by_dac[n]
            h = to_gds[d["heater"]]
            prov = "drawn edge" if n in drawn else "sequential fill"
            w.writerow([n, h, 0, stage[d["heater"]],
                        "; ".join(filter(None, [f"picpin.drawio pin A{d['pin']} ({prov})", note]))])
    print(f"wrote {p}  (was the identity placeholder)")


def drawn_edges() -> dict[int, tuple[int, int]]:
    """DAC index -> vendor pair, read off the edges actually drawn in the file."""
    root = ET.parse(DRAWIO).getroot()
    page = root.findall("diagram")[1]
    cells = {c.get("id"): c for c in page.findall(".//mxCell")}
    out = {}
    for c in page.findall(".//mxCell"):
        if c.get("edge") != "1":
            continue
        src, tgt = cells.get(c.get("source")), cells.get(c.get("target"))
        if src is None or tgt is None:
            continue
        m = re.fullmatch(r"DAC(\d)(\d\d)A", _text(src.get("value")))
        if not m:
            continue
        kind, arg = classify(_text(tgt.get("value")))
        if kind == "heater":
            out[int(m.group(1)) * 16 + int(m.group(2))] = tuple(sorted(arg))
    return out


if __name__ == "__main__":
    main()
