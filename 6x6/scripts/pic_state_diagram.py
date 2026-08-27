"""PIC B heater-state diagram: the clean 6x6P schematic, every heater coloured by its
`analytic` operating role (knob / weak knob / held-transparent) layered over its electrical
fault class, both read from `pic_data/pic_b_config.json`. Operational model: heaters with no
reachable null are held fully transparent (max-through @ V0); the characterized ones are the
adjustable knobs.

Geometry is NOT re-derived here -- it is `pic.layout.build_scene()`, the shared
renderer-independent scene graph. Each scene heater's geometric id
`h` (0..119) is joined to the config's `net` field (== heater_gds) to pull its state.

    python scripts/pic_state_diagram.py            # writes pic_structure_state.png

Re-run after re-characterization (rebuild the config first with scripts/build_config.py).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.pic.layout import build_scene

CONFIG = "pic_data/pic_b_config.json"
OUT = "pic_structure_state.png"

# heater state -> (fill, edge, base label). Legend appends the live count. Legend order.
# Driven by the config `analytic` field (ok/weak/dead/spare -> knob/weak_knob/transparent/
# spare) layered over the electrical fault classes, which take precedence. `transparent` is
# the key new visual: heaters with no reachable null, held at V0 (max-through), out of the way.
STATES = {
    "knob": ("#2e8b57", "#1c5c39", "adjustable knob (analytic ok)"),
    "weak_knob": ("#f1c40f", "#b8960a", "weak knob (analytic weak: 41/42/47)"),
    "transparent": ("#7ec8e3", "#1b6f8f", "dead -> held transparent @ V0 (max-through)"),
    "short": ("#e8821a", "#a75808", "shorted / mirror-tied"),
    "elec_dead": ("#c0392b", "#7d1f14", "electrically dead (no response)"),
    "elec_weak": ("#b8860b", "#7a5c07", "electrically weak (dac 99)"),
    "spare": ("#b6b6b6", "#8a8a8a", "off-mesh spare (N/A)"),
}
GREY = "#4d4d4d"
YTOP = 8.6

# PD tiers from the config's photodiodes block (strong / weak / dead).
PDCOL = {"strong": "#1a9850", "weak": "#e8821a", "dead": "#9aa0a6"}


def categorize(cfg: dict):
    """Return (cat, spare_na): `cat(channel) -> state` and the set of off-mesh spare DACs.
    Electrical faults come first (config summary lists) so a fault always wins; among the
    electrically-ok heaters the config `analytic` field decides knob / weak_knob /
    transparent / spare. dac 99 is analytic ok but elec weak, so it draws elec_weak."""
    s = cfg["summary"]
    dead, short, weak = set(s["dead"]), set(s["short"]), set(s["weak"])
    spare_na = {c["dac"] for c in cfg["channels"] if c.get("analytic") == "spare"}
    amap = {"ok": "knob", "weak": "weak_knob", "dead": "transparent", "spare": "spare"}

    def cat(c: dict) -> str:
        d = c["dac"]
        if d in dead:
            return "elec_dead"
        if d in short:
            return "short"
        if d in weak:
            return "elec_weak"
        return amap.get(c.get("analytic"), "knob" if c["Vnull"] is not None else "transparent")

    return cat, spare_na


def _dash(it):
    return (0, tuple(it["dash"])) if "dash" in it else "solid"


def draw_line(ax, it):
    xs = [p[0] for p in it["pts"]]
    ys = [p[1] for p in it["pts"]]
    ax.plot(xs, ys, color=it["c"], lw=it["w"], ls=_dash(it), zorder=it.get("z", 3),
            solid_capstyle="round")


def draw_rect(ax, it):
    ax.add_patch(mp.Rectangle((it["x"], it["y"]), it["w"], it["h"],
                              facecolor=it["c"], edgecolor="none", zorder=it.get("z", 4)))


def draw_text(ax, it):
    bbox = dict(boxstyle="round,pad=0.12", fc="white", ec="none") if it.get("box") else None
    ax.text(it["x"], it["y"], it["s"], color=it["c"], fontsize=it["sz"],
            ha=it["ha"], va=it["va"], zorder=it.get("z", 9), bbox=bbox)


def draw_header(ax, it):
    ax.plot([it["x0"], it["x1"]], [it["y"], it["y"]], color=it["c"], lw=2,
            zorder=it.get("z", 2))
    ax.text((it["x0"] + it["x1"]) / 2, it["y"] + 0.12, it["s"], ha="center", va="bottom",
            fontsize=10, color=it["c"], weight="bold", zorder=10)


def draw_cut(ax, it, color="#c0392b"):
    """Radiating terminator glyph at a dumped port."""
    s = 1 if it["up"] else -1
    cx, cy = it["x"] + 0.14, it["y"] + s * 0.10
    ax.add_patch(mp.Circle((cx, cy), 0.085, fill=False, edgecolor=color, lw=1.3, zorder=5))
    ax.plot([cx - 0.06, cx + 0.06], [cy - 0.06, cy + 0.06], color=color, lw=1.1, zorder=6)


def draw_coupler(ax, it, color="#2e8b57"):
    x, y = it["x"], it["y"]
    ax.add_patch(mp.Rectangle((x - 0.11, y - 0.17), 0.22, 0.34, fill=False,
                              edgecolor=color, lw=1.3, zorder=5))
    ax.plot([x - 0.11, x + 0.11], [y - 0.17, y + 0.17], color=color, lw=0.7, zorder=5)
    ax.plot([x - 0.11, x + 0.11], [y + 0.17, y - 0.17], color=color, lw=0.7, zorder=5)


def draw_input(ax, it, color="#7d3c98"):
    ax.add_patch(mp.RegularPolygon((it["x"], it["y"]), 4, radius=0.17,
                                   facecolor=color, edgecolor="none", zorder=6))
    ax.text(it["x"], it["y"] + 0.42, "6x6P\ninput", ha="center", va="bottom",
            fontsize=8, color=color)


def draw_pd(ax, it, tier):
    c = PDCOL[tier]
    x, y, idx = it["x"], it["y"], it["i"]
    ax.add_patch(mp.RegularPolygon((x, y), 3, radius=0.16, orientation=-1.57,
                                   facecolor=c, edgecolor="none", zorder=6))
    ax.add_patch(mp.Rectangle((x + 0.15, y - 0.16), 0.045, 0.32, facecolor=c,
                              edgecolor="none", zorder=6))
    ax.text(x + 0.36, y, f"PD{idx}", ha="left", va="center", fontsize=7,
            color="0.55" if tier == "dead" else "0.1")


def draw_heater(ax, node, state):
    x, y, up = node["x"], node["y"], node["up"]
    fill, edge, _ = STATES[state]
    if state == "elec_dead":
        d = 0.075
        ax.plot([x - d, x + d], [y - d, y + d], color=fill, lw=1.7, zorder=8)
        ax.plot([x - d, x + d], [y + d, y - d], color=fill, lw=1.7, zorder=8)
    elif state == "transparent":  # held fully transparent -- distinct cyan disc
        ax.add_patch(mp.Circle((x, y), 0.072, facecolor=fill, edgecolor=edge, lw=1.1, zorder=8))
    else:
        hatch = "///" if state == "elec_weak" else None
        ax.add_patch(mp.Rectangle((x - 0.058, y - 0.05), 0.116, 0.10, facecolor=fill,
                                  edgecolor=edge, lw=0.6, hatch=hatch, zorder=8))
    dy = 0.12 if up else -0.12
    ax.text(x, y + dy, f"H{node['h']}", ha="center", va="bottom" if up else "top",
            fontsize=4.3, color="#5a5a5a", zorder=9)


def main():
    cfg = json.load(open(CONFIG))
    by_net = {c["net"]: c for c in cfg["channels"] if c["net"] is not None}
    scene = build_scene()

    # join check: every scene heater must find a config net, and vice-versa.
    scene_ids = {h["h"] for h in scene["heaters"]}
    missing_pos = sorted(n for n in by_net if n not in scene_ids)
    missing_state = sorted(h for h in scene_ids if h not in by_net)
    if missing_pos:
        print(f"WARNING: {len(missing_pos)} config nets have no layout position: {missing_pos}")
    if missing_state:
        print(f"WARNING: {len(missing_state)} layout heaters have no config net: {missing_state}")

    cat, spare_na = categorize(cfg)
    ch = cfg["channels"]
    states = {h["h"]: cat(by_net[h["h"]]) for h in scene["heaters"] if h["h"] in by_net}
    onmesh = Counter(states.values())                       # 120 heaters actually drawn
    offmesh = Counter(cat(c) for c in ch if c["net"] is None)  # 8 spares, no position

    # legend counts, config-derived over all 128 channels (task categories; may overlap:
    # dac 99 is analytic ok yet drawn elec_weak).
    counts = {
        "knob": sum(c.get("analytic") == "ok" for c in ch),
        "weak_knob": sum(c.get("analytic") == "weak" for c in ch),
        "transparent": sum(c.get("analytic") == "dead" for c in ch),
        "short": len(cfg["summary"]["short"]),
        "elec_dead": len(cfg["summary"]["dead"]),
        "elec_weak": len(cfg["summary"]["weak"]),
        "spare": len(spare_na),
    }

    pd_tier = {}
    pdb = cfg["photodiodes"]
    for i in range(14):
        pd_tier[i] = ("dead" if i in pdb["dead"] else
                      "strong" if i in pdb["strong"] else "weak")

    fig, ax = plt.subplots(figsize=(24, 9.6))

    dispatch = {"line": draw_line, "rect": draw_rect, "text": draw_text,
                "header": draw_header, "cut": draw_cut, "coupler": draw_coupler,
                "input": draw_input}
    for it in scene["items"]:  # already z-sorted by build_scene
        t = it["t"]
        if t == "heater":
            draw_heater(ax, it, states.get(it["h"], "knob"))
        elif t == "pd":
            draw_pd(ax, it, pd_tier[it["i"]])
        elif t in dispatch:
            dispatch[t](ax, it)

    # link the shorted DAC 112-115 cluster (nets tied together on chip 7).
    short_dacs = cfg["summary"]["short"]
    short_nets = [c["net"] for c in cfg["channels"] if c["dac"] in short_dacs and c["net"] is not None]
    pos = {h["h"]: (h["x"], h["y"]) for h in scene["heaters"]}
    pts = [pos[n] for n in short_nets if n in pos]
    if len(pts) >= 2:
        pts_sorted = sorted(pts, key=lambda p: p[1])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, color=STATES["short"][0], lw=1.4, ls=(0, (3, 2)), zorder=6)
        cx = sum(xs) / len(xs)
        cy = max(ys) + 0.55
        ax.annotate(f"DAC {short_dacs[0]}-{short_dacs[-1]}\nshorted together",
                    xy=(cx, max(ys)), xytext=(cx, cy), ha="center", va="bottom",
                    fontsize=7.5, color=STATES["short"][1], weight="bold",
                    arrowprops=dict(arrowstyle="-", color=STATES["short"][0], lw=1.0))

    bx = scene["bbox"]
    ax.set_xlim(bx[0] - 0.3, bx[2] + 0.4)
    ax.set_ylim(bx[1] - 1.5, bx[3] + 0.8)
    ax.axis("off")
    s = cfg["summary"]
    date = cfg["meta"]["generated"][:10]
    ax.set_title(
        f"PIC B heater state - {s['characterized']}/{s['drivable']} drivable characterized   ({date})\n"
        "PROVISIONAL, verified=0 - schematic + electrical + optical, not hardware-confirmed "
        "(heater colour = state, joined config net -> geometric heater id)",
        fontsize=12.5)
    ax.text((bx[0] + bx[2]) / 2, bx[3] + 0.55,
            "operational model: dead heaters held fully transparent (max-through @ V0); "
            "characterized heaters are the adjustable knobs",
            ha="center", va="center", fontsize=10, style="italic", color=STATES["transparent"][1],
            bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6fb", ec=STATES["transparent"][0], lw=0.8))

    # unreachable spares have no mesh position: draw them as an off-mesh N/A bank.
    sx, sy = bx[0] + 0.2, YTOP - 1.0
    ax.text(sx, sy + 0.4, "off-mesh spares", fontsize=8.5, weight="bold",
            color=STATES["spare"][1], ha="left")
    for j, d in enumerate(sorted(spare_na)):
        yy = sy - j * 0.42
        ax.add_patch(mp.Rectangle((sx, yy - 0.13), 0.26, 0.26, facecolor=STATES["spare"][0],
                                  edgecolor=STATES["spare"][1], hatch="////", lw=0.7))
        ax.text(sx + 0.38, yy, f"DAC {d}  N/A", fontsize=7.5, color="#555",
                ha="left", va="center")

    # legend: heater states (with live counts) + PD tiers
    def lbl(k):
        return f"{STATES[k][2]}: {counts[k]}"

    def sq(k):
        return Line2D([0], [0], marker="s", color="w", markerfacecolor=STATES[k][0],
                      markeredgecolor=STATES[k][1], markersize=10, label=lbl(k))

    handles = [
        sq("knob"),
        sq("weak_knob"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATES["transparent"][0],
               markeredgecolor=STATES["transparent"][1], markersize=11, markeredgewidth=1.3,
               label=lbl("transparent")),
        Line2D([0], [0], marker="x", color=STATES["elec_dead"][0], lw=0,
               markersize=9, markeredgewidth=1.7, label=lbl("elec_dead")),
        sq("short"),
        sq("elec_weak"),
        sq("spare"),
        Line2D([0], [0], color=GREY, lw=1.4, label="waveguide / rail"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=PDCOL["strong"],
               markersize=11, label="PD: strong"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=PDCOL["weak"],
               markersize=11, label="PD: weak"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.4, frameon=False,
              bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")
    print(f"legend counts: {counts}")
    print(f"on-mesh drawn: {dict(onmesh)}   off-mesh: {dict(offmesh)}")
    print(f"join: {len(missing_pos)} nets without pos, {len(missing_state)} heaters without net")


if __name__ == "__main__":
    main()
