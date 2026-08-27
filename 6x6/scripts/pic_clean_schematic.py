"""Clean, fully-labelled schematic of the 6x6P PIC (stylised but faithful to the GDS trace)."""

from __future__ import annotations
import json, csv, math, collections
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D

NET = "pic_data/netlist_6x6P.json"
BLUE, RED, GREEN, GOLD, PURPLE = "#1f6fb2", "#c0392b", "#2e8b57", "#b8860b", "#7d3c98"
HLBL = "#7a1f12"
DEAD = {0, 2, 7, 11}
# PD learnability classes — data-driven (signal size + held-out black-box R² on the 100k set):
#   dead = output never leaves the ~5 mV floor; unlearnable = real swing but no input explains it
#   (drift-dominated); good = inputs determine it (R²≥0.4).  See scripts/pd_learnability.py.
PD_CLASS = {
    0: "dead",
    1: "good",
    2: "dead",
    3: "good",
    4: "unlearnable",
    5: "good",
    6: "unlearnable",
    7: "dead",
    8: "unlearnable",
    9: "unlearnable",
    10: "good",
    11: "dead",
    12: "good",
    13: "good",
}
PDCOL = {"good": "#1a9850", "unlearnable": "#e8821a", "dead": "#9aa0a6"}
# Section-B tiers, 2026-07-23 post-rewire (pic_data/census/census_b128_postcycle.csv +
# fringe sweep sweep_b128_fringes.csv; 68/128 DAC channels live, chip 4 dead — all empirical,
# no schematic map). green = swing ≥25 mV from ≥2 channels over the 0–2 V sweep; orange =
# responds but weak/narrow. No PD is fully dead on B. Channel↔heater identity is UNKNOWN
# pending the empirical map, so heater glyphs carry no per-channel state.
PD_CLASS_B = {
    p: ("good" if p in {3, 5, 6, 7, 8, 9, 10, 12} else "unlearnable") for p in range(14)
}
CORRECTED = False  # set by main(): False = plain A-labelled schematic, True = Section-B state

YS = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # 6 signal rails (top->bottom)
Y_REFT, Y_REFB = 7.0, 0.0  # 2 reference rails (outer, straight)
MW = 1.18  # MZI glyph width
YTOP = 8.6


def build_heater_map():
    d = json.load(open(NET))
    heat = [c for c in d["components"] if c.get("is_heater")]

    def stage(x):
        if x > 14800:
            return "enc.split"
        if x > 14500:
            return "enc.single"
        if x > 12150:
            return "V"
        if x > 11600:
            return "sigma"
        return "U"

    for h in heat:
        h["stage"] = stage(h["x"])
    heat.sort(key=lambda c: (-c["x"], -c["y"]))
    for i, h in enumerate(heat):
        h["H"] = i
    with open("pic_data/heater_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["H", "stage", "gds_x", "gds_y"])
        for h in heat:
            w.writerow([h["H"], h["stage"], round(h["x"], 2), round(h["y"], 2)])
    return heat, collections.Counter(h["stage"] for h in heat)


class HCounter:
    def __init__(self):
        self.i = 0

    def next(self):
        v = self.i
        self.i += 1
        return v


def rail(ax, x0, x1, y, c="0.30", lw=1.25, z=1):
    ax.plot([x0, x1], [y, y], color=c, lw=lw, zorder=z, solid_capstyle="round")


def heater(ax, x, y, label, up=True):
    ax.add_patch(
        mp.Rectangle(
            (x - 0.05, y - 0.038),
            0.10,
            0.076,
            facecolor=RED,
            edgecolor="none",
            zorder=7,
        )
    )
    dy = 0.115 if up else -0.115
    ax.text(
        x,
        y + dy,
        f"H{label}",
        ha="center",
        va="bottom" if up else "top",
        fontsize=4.7,
        color=HLBL,
        zorder=8,
    )


def cut(ax, x, y, up=True, color=RED):
    """A radiating cut-off (∅) at the end of a stub."""
    s = 1 if up else -1
    cx, cy = x + 0.20, y + s * 0.34
    ax.plot([x, cx - 0.05], [y, cy - s * 0.08], color=color, lw=1.6, zorder=4)
    ax.add_patch(
        mp.Circle((cx, cy), 0.105, fill=False, edgecolor=color, lw=1.6, zorder=5)
    )
    ax.plot(
        [cx - 0.075, cx + 0.075], [cy - 0.075, cy + 0.075], color=color, lw=1.3, zorder=6
    )
    for ang in (0.35, 0.95, 1.55):
        ax.plot(
            [cx + 0.105 * math.cos(ang), cx + 0.23 * math.cos(ang)],
            [cy + s * 0.105 * math.sin(ang), cy + s * 0.23 * math.sin(ang)],
            color=color,
            lw=0.9,
            zorder=5,
        )


def mzi(ax, x, yt, yb, hc, color=BLUE):
    """Canonical 2x2 MZI on rails yt>yb (2 heaters)."""
    cy = (yt + yb) / 2.0
    arm = 0.21
    x0, x1 = x, x + MW
    xc1, xc2 = x0 + 0.26, x1 - 0.26
    xa, xb = x0 + 0.46, x1 - 0.46
    yA, yB = cy + arm, cy - arm
    ax.plot(
        [x0, xc1, xa, xb, xc2, x1],
        [yt, cy, yA, yA, cy, yt],
        color=color,
        lw=1.4,
        zorder=3,
    )
    ax.plot(
        [x0, xc1, xa, xb, xc2, x1],
        [yb, cy, yB, yB, cy, yb],
        color=color,
        lw=1.4,
        zorder=3,
    )
    for xc in (xc1, xc2):
        ax.add_patch(
            mp.Rectangle(
                (xc - 0.05, cy - 0.075),
                0.10,
                0.15,
                facecolor=color,
                edgecolor="none",
                zorder=4,
            )
        )
    heater(ax, (xa + xb) / 2, yA, hc.next(), up=True)
    heater(ax, (xa + xb) / 2, yB, hc.next(), up=False)
    return x1


def edge_drop(ax, x, y, hc, up=True, color=BLUE):
    """Edge MZI on a signal rail: coupler -> two heated arms -> second coupler; one output
    continues on the rail, the spare port is terminated at a cut-off AFTER the recombiner.
    (GDS: 84 2x2 MMIs = exactly 2 per mesh MZI -- heaters only ever sit between couplers.)"""
    s = 1 if up else -1
    x0, x1 = x, x + MW
    xa, xb = x0 + 0.36, x1 - 0.44
    xc = xb + 0.24  # second coupler
    yU = y + s * 0.30  # outer arm
    yD = y - s * 0.13  # inner arm
    ax.plot([x0, x0 + 0.20], [y, y], color=color, lw=1.4, zorder=3)  # enter
    ax.plot([x0 + 0.20, xa], [y, yU], color=color, lw=1.4, zorder=3)
    ax.plot([x0 + 0.20, xa], [y, yD], color=color, lw=1.4, zorder=3)
    ax.plot([xa, xb], [yU, yU], color=color, lw=1.4, zorder=3)  # outer arm
    ax.plot([xa, xb], [yD, yD], color=color, lw=1.4, zorder=3)  # inner arm
    ax.plot([xb, xc], [yU, y], color=color, lw=1.4, zorder=3)  # recombine
    ax.plot([xb, xc], [yD, y], color=color, lw=1.4, zorder=3)
    for xr in (x0 + 0.20, xc):
        ax.add_patch(
            mp.Rectangle(
                (xr - 0.04, y - 0.075),
                0.08,
                0.15,
                facecolor=color,
                edgecolor="none",
                zorder=4,
            )
        )
    ax.plot([xc + 0.04, x1], [y, y], color=color, lw=1.4, zorder=3)  # continue
    cut(ax, xc + 0.06, y + s * 0.12, up=up)  # 2nd coupler's spare port -> terminated
    h1 = ((xa + xb) / 2, yU)
    h2 = ((xa + xb) / 2, yD)
    if up:  # emit higher-y first
        heater(ax, *h1, hc.next(), up=True)
        heater(ax, *h2, hc.next(), up=False)
    else:
        heater(ax, *h2, hc.next(), up=True)
        heater(ax, *h1, hc.next(), up=False)
    return x1


def mzi1(ax, x, y, hc, color=GOLD, gap=0.42, label=None):
    """1x2-type MZI on one rail (2 heaters)."""
    x0, x1 = x, x + MW
    xa, xb = x0 + 0.32, x1 - 0.32
    yt, yb = y + gap / 2, y - gap / 2
    ax.plot([x0, x0 + 0.20, xa], [y, y, yt], color=color, lw=1.4, zorder=3)
    ax.plot([x0, x0 + 0.20, xa], [y, y, yb], color=color, lw=1.4, zorder=3)
    ax.plot([xa, xb], [yt, yt], color=color, lw=1.4, zorder=3)
    ax.plot([xa, xb], [yb, yb], color=color, lw=1.4, zorder=3)
    ax.plot([xb, x1 - 0.20, x1], [yt, y, y], color=color, lw=1.4, zorder=3)
    ax.plot([xb, x1 - 0.20, x1], [yb, y, y], color=color, lw=1.4, zorder=3)
    heater(ax, (xa + xb) / 2, yt, hc.next(), up=True)
    heater(ax, (xa + xb) / 2, yb, hc.next(), up=False)
    if label:
        ax.text(
            x0 - 0.07,
            y,
            label,
            ha="right",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
            zorder=9,
        )
    return x1


def coupler(ax, x, y, color=GREEN):
    """2x2 readout coupler (signal x reference LO -> homodyne pair)."""
    ax.add_patch(
        mp.Rectangle(
            (x - 0.13, y - 0.19),
            0.26,
            0.38,
            fill=False,
            edgecolor=color,
            lw=1.5,
            zorder=5,
        )
    )
    ax.plot([x - 0.13, x + 0.13], [y - 0.19, y + 0.19], color=color, lw=0.8, zorder=5)
    ax.plot([x - 0.13, x + 0.13], [y + 0.19, y - 0.19], color=color, lw=0.8, zorder=5)


def pd(ax, x, y, idx, role=None):
    """PD glyph coloured by class — A learnability classes plain, Section-B census corrected."""
    cls = (PD_CLASS_B if CORRECTED else PD_CLASS)[idx]
    c = PDCOL[cls]
    ax.add_patch(
        mp.RegularPolygon(
            (x, y),
            3,
            radius=0.16,
            orientation=-1.57,
            facecolor=c,
            edgecolor="none",
            zorder=6,
        )
    )
    ax.add_patch(
        mp.Rectangle(
            (x + 0.15, y - 0.16), 0.045, 0.32, facecolor=c, edgecolor="none", zorder=6
        )
    )
    ax.text(
        x + 0.36,
        y,
        f"PD{idx}",
        ha="left",
        va="center",
        fontsize=7,
        color="0.55" if cls == "dead" else "0.1",
    )


def header(ax, x0, x1, text, color):
    ax.plot([x0, x1], [YTOP, YTOP], color=color, lw=2, zorder=2)
    ax.text(
        (x0 + x1) / 2,
        YTOP + 0.12,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
        weight="bold",
    )


def mesh(ax, xcols, label, hc, color=BLUE):
    """6 signal rails, columns alternating 3·4 MZIs; 4-columns add an edge-drop on the top & bottom rail."""
    for ci, xc in enumerate(xcols):
        if ci % 2 == 1:  # 4-MZI column
            edge_drop(ax, xc, YS[0], hc, up=True, color=color)
            mzi(ax, xc, YS[1], YS[2], hc, color)
            mzi(ax, xc, YS[3], YS[4], hc, color)
            edge_drop(ax, xc, YS[5], hc, up=False, color=color)
        else:  # 3-MZI column
            mzi(ax, xc, YS[0], YS[1], hc, color)
            mzi(ax, xc, YS[2], YS[3], hc, color)
            mzi(ax, xc, YS[4], YS[5], hc, color)
    for ci in range(len(xcols) - 1):  # inter-column connectors
        for y in YS:
            rail(ax, xcols[ci] + MW, xcols[ci + 1], y)
    header(ax, xcols[0], xcols[-1] + MW, label, color)


def main(corrected=False):
    global CORRECTED
    CORRECTED = corrected
    heat, counts = build_heater_map()
    print("heater_map.csv stage counts:", dict(counts), "total", len(heat))

    fig, ax = plt.subplots(figsize=(24, 9.2))
    hc = HCounter()

    x_in = 0.0
    x_s0, x_s1 = 0.7, 3.5
    x_es = 3.9  # split encoders
    x_eo = 5.6  # single shifters
    step = 1.55
    x_V = [6.6 + step * i for i in range(6)]
    x_Vend = x_V[-1] + MW
    x_S = x_Vend + 0.55
    x_U0 = x_S + MW + 0.7
    x_U = [x_U0 + step * i for i in range(6)]
    x_Uend = x_U[-1] + MW
    x_rd = x_Uend + 0.5  # signal rails arrive
    x_tap = x_rd + 1.0  # signal taps (1x2 split)
    x_lobus = x_tap + 0.9  # reference LO bus
    x_comb = x_lobus + 0.55  # homodyne combiners
    x_pd = x_comb + 1.7  # PD bank (14, y-order)
    all_y = [Y_REFT] + YS + [Y_REFB]

    # splitter tree 1 -> 8
    ax.add_patch(
        mp.RegularPolygon(
            (x_in, 3.5), 4, radius=0.17, facecolor=PURPLE, edgecolor="none", zorder=6
        )
    )
    ax.text(
        x_in,
        3.5 + 0.42,
        "6x6P\ninput",
        ha="center",
        va="bottom",
        fontsize=8,
        color=PURPLE,
    )
    rail(ax, x_in, x_s0, 3.5, PURPLE, 1.6)

    def tree(x0, y0, x1, targets):
        if len(targets) == 1:
            ax.plot([x0, x1], [y0, targets[0]], color=PURPLE, lw=1.3, zorder=3)
            return
        mid = len(targets) // 2
        top, bot = targets[:mid], targets[mid:]
        yt, yb = sum(top) / len(top), sum(bot) / len(bot)
        xm = (x0 + x1) / 2
        ax.plot([x0, xm], [y0, y0], color=PURPLE, lw=1.3, zorder=3)
        ax.plot([xm, xm], [yb, yt], color=PURPLE, lw=1.3, zorder=3)
        tree(xm, yt, x1, top)
        tree(xm, yb, x1, bot)

    tree(x_s0, 3.5, x_s1, all_y)
    header(ax, x_s0, x_s1, "splitter tree (1→2→4→8)", PURPLE)

    # encode: 8 rails get a split-MZI then a single shifter
    for y in all_y:
        rail(ax, x_s1, x_es, y, "0.30")
        rail(ax, x_es + MW, x_V[0], y, "0.30")
    for y in all_y:
        mzi1(ax, x_es, y, hc, color=PURPLE)  # H0-15
    for y in all_y:
        heater(ax, x_eo, y, hc.next(), up=True)  # H16-23
    header(ax, x_es, x_eo + 0.2, "encode: 8 split-MZIs + 8 shifters", PURPLE)

    # reference arms: straight, edge-to-edge, no interaction
    for y, va in ((Y_REFT, "bottom"), (Y_REFB, "top")):
        rail(ax, x_es + MW, x_lobus, y, "0.5", 1.3)
        ax.text(
            x_V[0] + 0.2,
            y + (0.14 if y == Y_REFT else -0.14),
            "reference arm (homodyne LO) — no interaction",
            ha="left",
            va=va,
            fontsize=7,
            color="0.45",
        )

    mesh(ax, x_V, "V mesh — 3·4·3·4·3·4 MZIs  (4-cols edge-drop → ∅)", hc)

    # V -> Sigma -> U routing (signal rails only)
    for y in YS:
        rail(ax, x_Vend, x_S, y)
        rail(ax, x_S + MW, x_U0, y)
    for k, y in enumerate(YS):
        mzi1(ax, x_S, y, hc, color=GOLD, label=f"σ{k+1}")  # H66-77
    header(ax, x_S, x_S + MW, "Σ (6 attenuators)", GOLD)

    mesh(ax, x_U, "U mesh — 3·4·3·4·3·4 MZIs  (4-cols edge-drop → ∅)", hc)

    # readout: each U output -> 1x2 tap -> monitor PD + (combiner with LO) -> homodyne PD.
    # rail i -> (monitor PD, homodyne PD), per the PD-feeder trace:
    sig_route = [(0, 2, 1), (1, 4, 3), (2, 6, 5), (3, 7, 8), (4, 9, 10), (5, 11, 12)]
    hg = 0.30
    pdy = {}  # each PD pair straddles its signal rail (group flip)
    for ri, mon, homo in sig_route:
        y = YS[ri]
        if ri <= 2:
            pdy[homo], pdy[mon] = y + hg, y - hg
        else:
            pdy[mon], pdy[homo] = y + hg, y - hg
    pdy[0], pdy[13] = YS[0] + hg + 0.55, YS[5] - hg - 0.55
    header(
        ax,
        x_rd,
        x_pd + 0.7,
        "readout — 6 taps → monitor + homodyne PDs  (LO = ref arms)",
        GREEN,
    )

    LOC, LOS = "#43b07a", (
        0,
        (4, 2),
    )  # LO lines: lighter green, dashed (so crossings read as crossovers)
    xin = x_pd - 0.25  # vertical routing channel just before the bank

    # LO buses + the two LO-monitor PDs
    ax.plot(
        [x_lobus, x_lobus], [YS[2] - 0.2, Y_REFT], color=LOC, lw=1.1, ls=LOS, zorder=2
    )
    ax.plot([x_lobus, x_lobus], [Y_REFB, YS[3]], color=LOC, lw=1.1, ls=LOS, zorder=2)
    ax.text(
        x_lobus - 0.07,
        (Y_REFT + YS[0]) / 2,
        "LO",
        ha="right",
        va="center",
        fontsize=7,
        color=LOC,
    )
    for refy, k in ((Y_REFT, 0), (Y_REFB, 13)):
        ax.plot(
            [x_lobus, xin, xin, x_pd - 0.05],
            [refy, refy, pdy[k], pdy[k]],
            color=LOC,
            lw=0.9,
            ls=LOS,
            zorder=2,
        )
        pd(ax, x_pd, pdy[k], k, role="lo")

    for ri, mon, homo in sig_route:
        y = YS[ri]
        my, hy = pdy[mon], pdy[homo]
        rail(ax, x_Uend, x_tap, y, GREEN, 1.2)  # signal in
        ax.add_patch(
            mp.Rectangle(
                (x_tap, y - 0.05),
                0.07,
                0.10,
                facecolor=GREEN,
                edgecolor="none",
                zorder=4,
            )
        )
        # monitor: short step to its PD row, straight across
        ax.plot(
            [x_tap + 0.07, x_tap + 0.30, x_tap + 0.30, x_pd - 0.05],
            [y, y, my, my],
            color=GREEN,
            lw=0.9,
            zorder=3,
        )
        pd(ax, x_pd, my, mon, role="mon")
        # signal → combiner (⊗ LO) → homodyne PD
        ax.plot([x_tap + 0.07, x_comb - 0.13], [y, y], color=GREEN, lw=1.2, zorder=3)
        ax.plot(
            [x_lobus, x_comb - 0.13],
            [y - 0.09, y - 0.09],
            color=LOC,
            lw=0.9,
            ls=LOS,
            zorder=3,
        )  # LO into combiner
        coupler(ax, x_comb, y)
        ax.plot(
            [x_comb + 0.13, xin, xin, x_pd - 0.05],
            [y, y, hy, hy],
            color=GREEN,
            lw=1.1,
            zorder=3,
        )
        pd(ax, x_pd, hy, homo, role="homo")

    ax.set_xlim(x_in - 1.2, x_pd + 1.9)
    ax.set_ylim(-1.4, YTOP + 1.0)
    ax.axis("off")
    if corrected:
        title = (
            "6x6P PIC — Section-B measured state (post-rewire census + 0–2 V fringe sweep, 2026-07-23).  "
            "all 14 PDs respond, none dead;  PD colour = measured tier.  H labels are geometric only — "
            "the DAC-channel → heater map is being derived empirically."
        )
        unl_label = "PD: weak / narrow response (few driving channels)"
    else:
        title = (
            "6x6P PIC — clean schematic, all 120 heaters labelled (H0–H119, GDS order).  "
            "input → splitter+encode → V → Σ → U → homodyne readout."
        )
        unl_label = "PD: unlearnable (drift-dominated)"
    ax.set_title(title, fontsize=12.5)
    leg = [
        Line2D([0], [0], color=PURPLE, lw=2, label="input / splitter / encode"),
        Line2D([0], [0], color=BLUE, lw=2, label="mesh MZI (2×2)"),
        Line2D([0], [0], color=GOLD, lw=2, label="Σ attenuator (1×2 MZI)"),
        Line2D([0], [0], color="0.50", lw=2, label="reference arm (straight)"),
        Line2D([0], [0], color=GREEN, lw=2, label="readout / LO"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="w",
            markeredgecolor=RED,
            markersize=9,
            label="cut-off → ∅ (wire ends)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=RED,
            markersize=8,
            label="heater Hn",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=PDCOL["good"],
            markersize=10,
            label="PD: good (input-determined)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=PDCOL["unlearnable"],
            markersize=10,
            label=unl_label,
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=PDCOL["dead"],
            markersize=10,
            label="PD: dead (none on B)" if corrected else "PD: dead / drift sensor",
        ),
    ]
    ax.legend(
        handles=leg,
        loc="lower center",
        ncol=5,
        fontsize=8.3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.tight_layout()
    out = "pic_structure_corrected.png" if corrected else "pic_structure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (heaters drawn: {hc.i})")
    assert hc.i == 120, f"expected 120 heaters, drew {hc.i}"


if __name__ == "__main__":
    main(corrected=False)
    main(corrected=True)
