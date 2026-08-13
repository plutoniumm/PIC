"""Chip layout as a renderer-independent scene graph.

Where every heater, photodiode, MZI and rail sits, as data rather than draw calls, so a
renderer can paint measurements onto the real structure (`scripts/pic_state_diagram.py`).
Geometry is stylised but faithful to the GDS trace in `pic_data/netlist_6x6P.json`; the
heater numbering H0..H119 is the same descending-x, descending-y order that
`pic_data/heater_map.csv` records, and `verify_against_netlist()` asserts the counts.

`scripts/pic_clean_schematic.py` still carries its own copy of these stylised coordinates
for the static PNG. The two agree today and both derive their *facts* from the netlist,
but the drawing constants are duplicated: change one and change the other, or fold the
schematic onto `build_scene()`.

    from pic.layout import build_scene
    scene = build_scene()          # {"bbox", "items", "heaters", "pds", "stages"}

Items are primitives (`line`, `rect`, `text`, `header`) plus semantic nodes (`heater`,
`pd`, `cut`, `coupler`) that each renderer styles for itself -- the scene carries
position and identity, not appearance policy.

Optical flow runs LEFT to RIGHT here (input coupler -> splitter tree -> encode -> V ->
sigma -> U -> readout PDs), which is mirrored from the GDS x axis where light travels
from high x to low x.
"""

from __future__ import annotations

import json
import os

BLUE, RED, GREEN, GOLD, PURPLE = "#1f6fb2", "#c0392b", "#2e8b57", "#b8860b", "#7d3c98"
GREY, LOC = "#4d4d4d", "#43b07a"
LO_DASH = (4, 2)

YS = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # 6 signal rails (top -> bottom)
Y_REFT, Y_REFB = 7.0, 0.0  # 2 reference rails (outer, straight)
MW = 1.18  # MZI glyph width
YTOP = 8.6

# rail index -> (monitor PD, homodyne PD), from the GDS PD-feeder trace
SIG_ROUTE = [(0, 2, 1), (1, 4, 3), (2, 6, 5), (3, 7, 8), (4, 9, 10), (5, 11, 12)]
HG = 0.30  # PD pair half-separation about its signal rail

NETLIST = "pic_data/netlist_6x6P.json"
HEATER_STAGES = [  # (H_lo, H_hi inclusive, stage) -- matches heater_map.csv
    (0, 15, "enc.split"),
    (16, 23, "enc.single"),
    (24, 65, "V"),
    (66, 77, "sigma"),
    (78, 119, "U"),
]


def stage_of(h: int) -> str:
    for lo, hi, s in HEATER_STAGES:
        if lo <= h <= hi:
            return s
    raise ValueError(f"heater {h} out of range")


class Scene:
    """Accumulates primitives and keeps an index of the interactive nodes."""

    def __init__(self):
        self.items: list[dict] = []
        self.heaters: list[dict] = []
        self.pds: list[dict] = []
        self.stages: list[dict] = []
        self._h = 0

    def line(self, pts, c=GREY, w=1.25, dash=None, z=3):
        it = {"t": "line", "pts": [[round(x, 4), round(y, 4)] for x, y in pts],
              "c": c, "w": w, "z": z}
        if dash:
            it["dash"] = list(dash)
        self.items.append(it)

    def seg(self, x0, x1, y, c=GREY, w=1.25, z=1):
        self.line([(x0, y), (x1, y)], c, w, z=z)

    def rect(self, x, y, w, h, c, z=4):
        self.items.append({"t": "rect", "x": round(x, 4), "y": round(y, 4),
                           "w": round(w, 4), "h": round(h, 4), "c": c, "z": z})

    def text(self, x, y, s, c="#111", sz=8, ha="center", va="center", z=9, box=False):
        self.items.append({"t": "text", "x": round(x, 4), "y": round(y, 4), "s": s,
                           "c": c, "sz": sz, "ha": ha, "va": va, "z": z, "box": box})

    def heater(self, x, y, up=True):
        h = self._h
        self._h += 1
        node = {"t": "heater", "h": h, "x": round(x, 4), "y": round(y, 4),
                "up": bool(up), "stage": stage_of(h), "z": 7}
        self.items.append(node)
        self.heaters.append(node)
        return h

    def pd(self, x, y, idx, role):
        node = {"t": "pd", "i": idx, "x": round(x, 4), "y": round(y, 4),
                "role": role, "z": 6}
        self.items.append(node)
        self.pds.append(node)

    def cut(self, x, y, up=True):
        self.items.append({"t": "cut", "x": round(x, 4), "y": round(y, 4),
                           "up": bool(up), "z": 5})

    def coupler(self, x, y):
        self.items.append({"t": "coupler", "x": round(x, 4), "y": round(y, 4), "z": 5})

    def header(self, x0, x1, s, c):
        it = {"t": "header", "x0": round(x0, 4), "x1": round(x1, 4), "y": YTOP,
              "s": s, "c": c, "z": 2}
        self.items.append(it)
        self.stages.append(it)


def _mzi(sc: Scene, x, yt, yb, color=BLUE):
    """Canonical 2x2 MZI across rails yt > yb (2 heaters)."""
    cy = (yt + yb) / 2.0
    arm = 0.21
    x0, x1 = x, x + MW
    xc1, xc2 = x0 + 0.26, x1 - 0.26
    xa, xb = x0 + 0.46, x1 - 0.46
    yA, yB = cy + arm, cy - arm
    sc.line([(x0, yt), (xc1, cy), (xa, yA), (xb, yA), (xc2, cy), (x1, yt)], color, 1.4)
    sc.line([(x0, yb), (xc1, cy), (xa, yB), (xb, yB), (xc2, cy), (x1, yb)], color, 1.4)
    for xc in (xc1, xc2):
        sc.rect(xc - 0.05, cy - 0.075, 0.10, 0.15, color)
    sc.heater((xa + xb) / 2, yA, up=True)
    sc.heater((xa + xb) / 2, yB, up=False)
    return x1


def _edge_drop(sc: Scene, x, y, up=True, color=BLUE):
    """Edge MZI that dumps one output to a cut-off; the rail continues (2 heaters)."""
    s = 1 if up else -1
    x0, x1 = x, x + MW
    xa, xb = x0 + 0.34, x1 - 0.34
    y_dump = y + s * 0.42
    y_sig = y - s * 0.13
    sc.line([(x0, y), (x0 + 0.22, y)], color, 1.4)
    sc.line([(x0 + 0.22, y), (xa, y_dump)], color, 1.4)
    sc.line([(x0 + 0.22, y), (xa, y_sig)], color, 1.4)
    sc.line([(xa, y_dump), (xb, y_dump)], color, 1.4)
    sc.line([(xa, y_sig), (xb, y_sig)], color, 1.4)
    sc.line([(xb, y_sig), (x1 - 0.22, y), (x1, y)], color, 1.4)
    sc.rect(x0 + 0.20, y - 0.05, 0.08, 0.10, color)
    sc.cut(xb, y_dump, up=up)
    hi, lo = ((xa + xb) / 2, y_dump), ((xa + xb) / 2, y_sig)
    if up:  # emit the higher-y heater first, so numbering follows descending y
        sc.heater(*hi, up=True)
        sc.heater(*lo, up=False)
    else:
        sc.heater(*lo, up=True)
        sc.heater(*hi, up=False)
    return x1


def _mzi1(sc: Scene, x, y, color=GOLD, gap=0.42, label=None):
    """1x2-type MZI on a single rail (2 heaters)."""
    x0, x1 = x, x + MW
    xa, xb = x0 + 0.32, x1 - 0.32
    yt, yb = y + gap / 2, y - gap / 2
    sc.line([(x0, y), (x0 + 0.20, y), (xa, yt)], color, 1.4)
    sc.line([(x0, y), (x0 + 0.20, y), (xa, yb)], color, 1.4)
    sc.line([(xa, yt), (xb, yt)], color, 1.4)
    sc.line([(xa, yb), (xb, yb)], color, 1.4)
    sc.line([(xb, yt), (x1 - 0.20, y), (x1, y)], color, 1.4)
    sc.line([(xb, yb), (x1 - 0.20, y), (x1, y)], color, 1.4)
    sc.heater((xa + xb) / 2, yt, up=True)
    sc.heater((xa + xb) / 2, yb, up=False)
    if label:
        sc.text(x0 - 0.07, y, label, color, 8, ha="right", box=True)
    return x1


def _mesh(sc: Scene, xcols, label, color=BLUE):
    """6 signal rails, columns alternating 3 and 4 MZIs; 4-columns edge-drop top+bottom."""
    for ci, xc in enumerate(xcols):
        if ci % 2 == 1:
            _edge_drop(sc, xc, YS[0], up=True, color=color)
            _mzi(sc, xc, YS[1], YS[2], color)
            _mzi(sc, xc, YS[3], YS[4], color)
            _edge_drop(sc, xc, YS[5], up=False, color=color)
        else:
            _mzi(sc, xc, YS[0], YS[1], color)
            _mzi(sc, xc, YS[2], YS[3], color)
            _mzi(sc, xc, YS[4], YS[5], color)
    for ci in range(len(xcols) - 1):
        for y in YS:
            sc.seg(xcols[ci] + MW, xcols[ci + 1], y)
    sc.header(xcols[0], xcols[-1] + MW, label, color)


def _tree(sc: Scene, x0, y0, x1, targets):
    if len(targets) == 1:
        sc.line([(x0, y0), (x1, targets[0])], PURPLE, 1.3)
        return
    mid = len(targets) // 2
    top, bot = targets[:mid], targets[mid:]
    yt, yb = sum(top) / len(top), sum(bot) / len(bot)
    xm = (x0 + x1) / 2
    sc.line([(x0, y0), (xm, y0)], PURPLE, 1.3)
    sc.line([(xm, yb), (xm, yt)], PURPLE, 1.3)
    _tree(sc, xm, yt, x1, top)
    _tree(sc, xm, yb, x1, bot)


def build_scene() -> dict:
    """Assemble the whole 6x6P chip. Heater numbering is assigned in draw order and is
    asserted to match `heater_map.csv`'s stage boundaries."""
    sc = Scene()

    x_in = 0.0
    x_s0, x_s1 = 0.7, 3.5
    x_es, x_eo = 3.9, 5.6
    step = 1.55
    x_V = [6.6 + step * i for i in range(6)]
    x_Vend = x_V[-1] + MW
    x_S = x_Vend + 0.55
    x_U0 = x_S + MW + 0.7
    x_U = [x_U0 + step * i for i in range(6)]
    x_Uend = x_U[-1] + MW
    x_rd = x_Uend + 0.5
    x_tap = x_rd + 1.0
    x_lobus = x_tap + 0.9
    x_comb = x_lobus + 0.55
    x_pd = x_comb + 1.7
    all_y = [Y_REFT] + YS + [Y_REFB]

    # input coupler + 1 -> 8 splitter tree
    sc.items.append({"t": "input", "x": x_in, "y": 3.5, "z": 6})
    sc.seg(x_in, x_s0, 3.5, PURPLE, 1.6)
    _tree(sc, x_s0, 3.5, x_s1, all_y)
    sc.header(x_s0, x_s1, "splitter tree", PURPLE)

    # encode: every rail gets a split-MZI (H0-15) then a single shifter (H16-23)
    for y in all_y:
        sc.seg(x_s1, x_es, y)
        sc.seg(x_es + MW, x_V[0], y)
    for y in all_y:
        _mzi1(sc, x_es, y, color=PURPLE)
    for y in all_y:
        sc.heater(x_eo, y, up=True)
    sc.header(x_es, x_eo + 0.2, "encode", PURPLE)

    # reference arms: straight, edge to edge, no interaction
    for y in (Y_REFT, Y_REFB):
        sc.seg(x_es + MW, x_lobus, y, "#808080", 1.3)
        sc.text(x_V[0] + 0.2, y + (0.14 if y == Y_REFT else -0.14),
                "reference arm (homodyne LO), no interaction", "#737373", 7,
                ha="left", va="bottom" if y == Y_REFT else "top")

    _mesh(sc, x_V, "V mesh")

    for y in YS:  # V -> sigma -> U routing (signal rails only)
        sc.seg(x_Vend, x_S, y)
        sc.seg(x_S + MW, x_U0, y)
    for k, y in enumerate(YS):
        _mzi1(sc, x_S, y, color=GOLD, label=f"σ{k+1}")
    sc.header(x_S, x_S + MW, "Σ", GOLD)

    _mesh(sc, x_U, "U mesh")

    # readout: each U output -> 1x2 tap -> monitor PD + (combine with LO) -> homodyne PD
    pdy = {}
    for ri, mon, homo in SIG_ROUTE:
        y = YS[ri]
        if ri <= 2:
            pdy[homo], pdy[mon] = y + HG, y - HG
        else:
            pdy[mon], pdy[homo] = y + HG, y - HG
    pdy[0], pdy[13] = YS[0] + HG + 0.55, YS[5] - HG - 0.55
    sc.header(x_rd, x_pd + 0.7, "readout: taps + homodyne PDs", GREEN)

    xin = x_pd - 0.25  # vertical routing channel just before the PD bank
    sc.line([(x_lobus, YS[2] - 0.2), (x_lobus, Y_REFT)], LOC, 1.1, LO_DASH, z=2)
    sc.line([(x_lobus, Y_REFB), (x_lobus, YS[3])], LOC, 1.1, LO_DASH, z=2)
    sc.text(x_lobus - 0.07, (Y_REFT + YS[0]) / 2, "LO", LOC, 7, ha="right")
    for refy, k in ((Y_REFT, 0), (Y_REFB, 13)):
        sc.line([(x_lobus, refy), (xin, refy), (xin, pdy[k]), (x_pd - 0.05, pdy[k])],
                LOC, 0.9, LO_DASH, z=2)
        sc.pd(x_pd, pdy[k], k, "lo")

    for ri, mon, homo in SIG_ROUTE:
        y = YS[ri]
        my, hy = pdy[mon], pdy[homo]
        sc.seg(x_Uend, x_tap, y, GREEN, 1.2, z=3)
        sc.rect(x_tap, y - 0.05, 0.07, 0.10, GREEN)
        sc.line([(x_tap + 0.07, y), (x_tap + 0.30, y), (x_tap + 0.30, my),
                 (x_pd - 0.05, my)], GREEN, 0.9)
        sc.pd(x_pd, my, mon, "mon")
        sc.line([(x_tap + 0.07, y), (x_comb - 0.13, y)], GREEN, 1.2)
        sc.line([(x_lobus, y - 0.09), (x_comb - 0.13, y - 0.09)], LOC, 0.9, LO_DASH)
        sc.coupler(x_comb, y)
        sc.line([(x_comb + 0.13, y), (xin, y), (xin, hy), (x_pd - 0.05, hy)], GREEN, 1.1)
        sc.pd(x_pd, hy, homo, "homo")

    assert sc._h == 120, f"expected 120 heaters, laid out {sc._h}"
    assert len(sc.pds) == 14, f"expected 14 PDs, laid out {len(sc.pds)}"
    for h in sc.heaters:  # draw order must agree with heater_map.csv's stage blocks
        assert h["stage"] == stage_of(h["h"])

    return {
        # right margin leaves room for the live PD bars + value readouts
        "bbox": [x_in - 1.4, -0.55, x_pd + 2.75, YTOP + 0.5],
        "items": sorted(sc.items, key=lambda i: i.get("z", 0)),
        "heaters": sc.heaters,
        "pds": sc.pds,
        "stages": sc.stages,
        "colors": {"blue": BLUE, "red": RED, "green": GREEN, "gold": GOLD,
                   "purple": PURPLE, "lo": LOC},
    }


def verify_against_netlist(root: str = ".") -> dict:
    """Cross-check the stylised layout's counts against the GDS-traced netlist."""
    path = os.path.join(root, NETLIST)
    d = json.load(open(path))
    n_h = sum(1 for c in d["components"] if c.get("is_heater"))
    n_pd = sum(1 for c in d["components"] if c["role"] == "pd")
    sc = build_scene()
    return {"netlist_heaters": n_h, "scene_heaters": len(sc["heaters"]),
            "netlist_pds": n_pd, "scene_pds": len(sc["pds"]),
            "ok": n_h == len(sc["heaters"]) and n_pd == len(sc["pds"])}


if __name__ == "__main__":
    s = build_scene()
    print(f"items={len(s['items'])} heaters={len(s['heaters'])} pds={len(s['pds'])}")
    print("bbox", s["bbox"])
    print("verify:", verify_against_netlist())
