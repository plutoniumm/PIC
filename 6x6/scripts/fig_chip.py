"""Section A figure: the PIC-B chip state map -> slides/images/chip_state.pdf

Paper-proportioned adaptation of `scripts/pic_state_diagram.py` (which writes a 24-inch
slide-sized PNG; left untouched). Same data, same geometry, redrawn at 7 in for a full-width
paper figure: per-heater "H<id>" labels dropped (illegible below ~4 pt), waveguides desaturated
to a recessive grey so the state colours carry all the ink, headline title removed (the caption
carries it). Heaters and photodiodes are drawn as point-sized markers rather than
data-coordinate patches, so the schematic can be stretched vertically for legibility without
distorting the glyphs.

Reads:
  pic_data/pic_b_config.json  -- per-channel `elec` fault class, `analytic` operating role,
                                 the `summary` counts, the `photodiodes` strong/weak tiers
  pic.layout.build_scene()    -- the shared renderer-independent scene graph (positions only)

Each scene heater's geometric id (0..119) joins to the config `net` field to pull its state.
State precedence follows pic_state_diagram.py: an electrical fault (dead / shorted / weak)
always wins over the optical `analytic` role -- DAC 99 is analytic-ok but electrically weak.

    PYTHONPATH=. python scripts/fig_chip.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pic.layout import build_scene

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BAD = "#e34948"   # status: reserved, marks the electrically dead channels
WIRE = "#c6c5c1"  # recessive waveguide grey

# state -> (marker, fill, edge, legend label); legend order.
STATES = {
    "knob":        ("s", BLUE,      "#104281", "characterized knob"),
    "weak_knob":   ("s", YELLOW,    "#8a5f00", "weak fringe"),
    "transparent": ("o", AQUA,      "#0d6b4a", "drivable, uncharacterized"),
    "short":       ("s", ORANGE,    "#8f3a14", "shorted / mirror-tied"),
    "elec_dead":   ("x", BAD,       BAD,       "electrically dead"),
    "elec_weak":   ("s", "#f7dfae", "#8a5f00", "electrically weak"),
    "spare":       ("s", "#c9c8c4", MUTED,     "off-mesh spare"),
}


def categorize(cfg: dict):
    """`cat(channel) -> state`, plus the off-mesh spare DAC set."""
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
        return amap.get(c.get("analytic"),
                        "knob" if c["Vnull"] is not None else "transparent")

    return cat, spare_na


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--out", default="slides/images/chip_state.pdf")
    a = ap.parse_args(argv)

    cfg = json.load(open(a.config))
    by_net = {c["net"]: c for c in cfg["channels"] if c["net"] is not None}
    scene = build_scene()

    plt.rcParams.update({
        "font.size": 7, "legend.fontsize": 6.0,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    cat, spare_na = categorize(cfg)
    ch = cfg["channels"]
    states = {h["h"]: cat(by_net[h["h"]]) for h in scene["heaters"] if h["h"] in by_net}
    counts = {
        "knob": sum(c.get("analytic") == "ok" for c in ch),
        "weak_knob": sum(c.get("analytic") == "weak" for c in ch),
        "transparent": sum(c.get("analytic") == "dead" for c in ch),
        "short": len(cfg["summary"]["short"]),
        "elec_dead": len(cfg["summary"]["dead"]),
        "elec_weak": len(cfg["summary"]["weak"]),
        "spare": len(spare_na),
    }
    pdb = cfg["photodiodes"]

    fig, ax = plt.subplots(figsize=(7.0, 3.15))

    # background structure: waveguides, rails, taps, couplers, dumped ports
    for it in scene["items"]:
        t = it["t"]
        if t == "line":
            ax.plot([p[0] for p in it["pts"]], [p[1] for p in it["pts"]], color=WIRE,
                    lw=min(it["w"] * 0.45, 0.6),
                    ls=(0, tuple(it["dash"])) if "dash" in it else "solid",
                    zorder=it.get("z", 3), solid_capstyle="round")
        elif t == "rect":
            ax.add_patch(mp.Rectangle((it["x"], it["y"]), it["w"], it["h"],
                                      facecolor="#a9a8a4", edgecolor="none", zorder=4))
        elif t == "coupler":
            ax.add_patch(mp.Rectangle((it["x"] - 0.13, it["y"] - 0.18), 0.26, 0.36,
                                      fill=False, edgecolor="#a9a8a4", lw=0.5, zorder=5))
        elif t == "cut":
            ax.plot([it["x"] + 0.14], [it["y"] + (0.12 if it["up"] else -0.12)], "o",
                    mfc="none", mec="#a9a8a4", mew=0.5, ms=2.6, zorder=5)
        elif t == "input":
            ax.plot([it["x"]], [it["y"]], "D", color=MUTED, ms=4.0, zorder=6)
            ax.text(it["x"], it["y"] + 0.45, "6$\\times$6P\ninput", ha="center", va="bottom",
                    fontsize=5.0, color=MUTED, linespacing=1.2)
        elif t == "header":
            ax.plot([it["x0"], it["x1"]], [it["y"], it["y"]], color=MUTED, lw=0.9, zorder=2)
            ax.text((it["x0"] + it["x1"]) / 2, it["y"] + 0.16,
                    it["s"].split(":")[0], ha="center", va="bottom", fontsize=5.8,
                    color=INK, zorder=10)
        elif t == "text":
            ax.text(it["x"], it["y"], it["s"], color=MUTED, fontsize=4.6, ha=it["ha"],
                    va=it["va"], zorder=9)

    # heaters, batched per state so each is one marker collection
    for k, (mk, fill, edge, _) in STATES.items():
        pts = [(h["x"], h["y"]) for h in scene["heaters"] if states.get(h["h"]) == k]
        if not pts:
            continue
        xs, ys = zip(*pts)
        if mk == "x":
            ax.plot(xs, ys, "x", color=fill, mew=0.9, ms=3.0, ls="none", zorder=8)
        else:
            ax.plot(xs, ys, mk, mfc=fill, mec=edge, mew=0.4, ms=3.1, ls="none", zorder=8)

    # photodiodes: tier in ink/grey so the colour budget stays with the heaters
    for it in scene["items"]:
        if it["t"] != "pd":
            continue
        strong = it["i"] in pdb["strong"]
        ax.plot([it["x"]], [it["y"]], ">", mfc=INK if strong else "white",
                mec=INK if strong else MUTED, mew=0.5, ms=3.6, zorder=6)
        ax.text(it["x"] + 0.30, it["y"], f"PD{it['i']}", ha="left", va="center",
                fontsize=4.8, color=INK if strong else MUTED, zorder=9)

    # the shorted DAC 112-115 cluster, tied together on chip 7
    short_dacs = cfg["summary"]["short"]
    pos = {h["h"]: (h["x"], h["y"]) for h in scene["heaters"]}
    pts = sorted((pos[c["net"]] for c in ch
                  if c["dac"] in short_dacs and c["net"] in pos), key=lambda p: p[1])
    if len(pts) >= 2:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=ORANGE, lw=0.6, ls=(0, (2.5, 1.8)), zorder=6)
        ax.annotate(f"DAC {short_dacs[0]}–{short_dacs[-1]} shorted",
                    xy=(sum(xs) / len(xs), max(ys)),
                    xytext=(sum(xs) / len(xs) + 1.6, max(ys) + 0.95), ha="center",
                    va="bottom", fontsize=5.2, color="#8f3a14",
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))

    bx = scene["bbox"]
    ax.set_xlim(bx[0] - 0.4, bx[2] + 1.4)
    ax.set_ylim(bx[1] - 0.4, bx[3] + 0.5)
    ax.axis("off")

    def h(k):
        mk, fill, edge, lab = STATES[k]
        if mk == "x":
            return Line2D([0], [0], marker="x", color=fill, lw=0, ms=4.2, mew=1.1,
                          label=f"{lab}: {counts[k]}")
        return Line2D([0], [0], marker=mk, color="w", markerfacecolor=fill,
                      markeredgecolor=edge, markersize=4.6, label=f"{lab}: {counts[k]}")

    handles = [h(k) for k in STATES] + [
        Line2D([0], [0], marker=">", color="w", markerfacecolor=INK, markeredgecolor=INK,
               markersize=4.6, label=f"PD, strong: {len(pdb['strong'])}"),
        Line2D([0], [0], marker=">", color="w", markerfacecolor="white",
               markeredgecolor=MUTED, markersize=4.6, label=f"PD, weak: {len(pdb['weak'])}"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.055),
                    ncol=5, frameon=False, handletextpad=0.3, columnspacing=1.4,
                    labelspacing=0.4)
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.tight_layout(pad=0.25)
    for ext in ("pdf", "png"):
        p = a.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print("wrote", p)
    print("counts:", counts, "| on-mesh:", dict(Counter(states.values())))


if __name__ == "__main__":
    main()
