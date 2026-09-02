"""Rig diagrams for the 4x4: what the light goes through, and what drives it.

    ./do figs                  # all of them, into figs/
    ./do figs mesh chain       # just those

Hardware-free, and deliberately not hand-drawn. Geometry comes from `theory.clements` and
`theory.layout`, electrical limits from `pic.config`, and every measured number from
`pic_data/calib.json` and `pic_data/char_results.json` -- so re-running this after a
re-characterization redraws the chip as it is now rather than as it was. The one thing
hardcoded here is the dark-noise spread, which no committed file carries yet.

`mesh.png` speaks the 6x6 rig's schematic language (`../6x6/scripts/pic_state_diagram.py`,
`../6x6/pic/layout.py`): continuous waveguide rails that visibly converge and recombine at
each MZI, every heater a small state-coloured square drawn where it physically sits, live
counts in a one-row legend at the bottom. Each square also carries a symmetric whisker whose
length is how badly its fringe fit closed -- a long bar is a heater you should not trust.
Nothing floats in a detached panel -- if a heater has no place on the chip it goes in the
off-mesh bank and says so.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pic.config import (
    ADC_AVG_N, ADC_BITS, BAUD_RATE, DAC_BITS, DAC_REF_V, HEATER_MAX_MA, HEATER_OHMS,
    NUM_ADC_RAW, SWITCH_BAUD, TEC_SETPOINT_C, TEC_TOLERANCE_C, VOLTAGE_MAX_CH,
)
from theory.clements import COLUMN, MESH, NCOL, NMODE, NMZI
from theory.layout import HEATERS, N_HEATERS, REF_RAIL, THETA_DAC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figs")
CALIB = os.path.join(HERE, "pic_data", "calib.json")
CHARS = os.path.join(HERE, "pic_data", "char_results.json")

BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, VIOLET = "#e34948", "#4a3aa7"
INK, INK2, MUT, GRID = "#0b0b0b", "#52514e", "#8a897f", "#d9d8d2"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "figure.dpi": 150, "savefig.dpi": 170,
    "text.color": INK, "axes.titlecolor": INK,
})

# state -> (fill, edge, legend label). Order is the legend order. Semantics ported from the
# 6x6's heater-state diagram: green is the adjustable knob, amber the weak one, a red cross
# the channel that cannot be driven at all.
STATES = {
    "char":   (GREEN,  "#0d6b4a", "characterized knob — Vπ fitted"),
    "weak":   (AMBER,  "#8a5f00", "swept and modulating — fit rejected"),
    "capped": (AMBER,  "#8a5f00", f"≈60 Ω: {HEATER_MAX_MA:.0f} mA cap — no fringe found"),
    "uncal":  (VIOLET, "#2b2170", "drivable, never swept"),
    "dark":   (RED,    "#8f1f18", "resistance unconfirmed — held at 0 V"),
}
PD_OK, PD_BAD = GREEN, ORANGE

# Fit-quality whisker. sqrt(1 - r²) is the residual as a fraction of the fringe amplitude, so
# a whisker is "how much of this heater's response the fit failed to explain" and a perfect
# fit draws nothing. Never swept draws nothing either -- the square's colour says which.
ERR_FULL, ERR_MIN, ERR_CAP = 0.34, 0.012, 0.055

ROLE_NAME = {"theta": "θ", "phi": "φ", "out_phase": "α", "aux": "—"}

# Dark-run scatter, 2026-08-27, per PD. Only PD0's was recorded -- it is a different detector
# from the other three in every way that matters, which is why the firmware averages
# ADC_AVG_N sweeps and why no fit is ever accepted on PD0 alone. Which PD gets flagged is
# read off the measured dark offsets, not remembered here.
PD_DARK_SIGMA_MV = {0: 26.0}


def _load():
    calib = json.load(open(CALIB))
    chars = {int(k): v for k, v in json.load(open(CHARS)).items()}
    date = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(CALIB)))
    return calib, chars, date


def state_of(dac: int, chars: dict) -> str:
    """Why a channel is or is not a usable knob, most specific reason first."""
    if VOLTAGE_MAX_CH[dac] == 0.0:
        return "dark"
    r = chars.get(dac)
    if r is not None and r["ok"]:
        return "char"
    if fit_error(dac, chars) is not None:
        return "weak"
    if HEATER_OHMS[dac] is not None and HEATER_OHMS[dac] < 90:
        return "capped"
    return "uncal"


def fit_error(dac: int, chars: dict):
    """Residual as a fraction of fringe amplitude, or None when the channel was never swept.
    A refused sweep leaves r² NaN, which is not the same claim as a bad fit."""
    r = chars.get(dac)
    if r is None:
        return None
    r2 = r.get("r2")
    if not isinstance(r2, (int, float)) or not math.isfinite(r2):
        return None
    return math.sqrt(max(0.0, 1.0 - float(r2)))


def ceiling_span(calib, dac: int) -> float:
    """Phase span this channel reaches at its own ceiling, in units of π, from calib.json.

    Not `char_results`' `span_pi`: `pic.characterize.fit_fringe` is called without `vmax`, so
    that field is `(VOLTAGE_MAX / Vπ)²` against the flat 3 V ceiling for every channel — it
    is neither the swept top nor what the driver now allows, and quoting it put MZI4 on this
    drawing at 0.55π on a chip that reaches 1.37π. Both operands here are keyed by DAC
    channel, which is the only ordering `VOLTAGE_MAX_CH` and `calib["vpi"]` are in."""
    return (VOLTAGE_MAX_CH[dac] / calib["vpi"][dac]) ** 2


def role_note(h, chars, calib) -> str:
    """One line saying what this channel is worth, in its own terms."""
    st = state_of(h.h, chars)
    if st == "char":
        r = chars[h.h]
        return (f"Vπ {calib['vpi'][h.h]:.2f} V · span {ceiling_span(calib, h.h):.2f}π · "
                f"r² {r['r2']:.4f}")
    if st == "weak":
        r = chars[h.h]
        return (f"swept, fit rejected: {1000 * r['amplitude']:.1f} mV, r² {r['r2']:.3f} — "
                f"real modulation, under the accept gate")
    if st == "capped":
        return f"{HEATER_OHMS[h.h]:.0f} Ω → {VOLTAGE_MAX_CH[h.h]:.2f} V cap: no usable fringe"
    if st == "dark":
        return "resistance never confirmed — the driver holds it at 0 V"
    if h.role == "out_phase":
        return "output trimmer: a diagonal screen cannot change |Ux|², so nothing can fit it"
    if h.role == "phi":
        return "φ is a global phase under one lit port — needs light in two at once"
    return "wired and drivable; no role assigned yet"


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p, HERE), f"({os.path.getsize(p) // 1024} KB)")


def box(ax, x, y, w, h, fc="white", ec=INK2, lw=1.2, dash=None, z=3, r=0.012):
    p = mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                          fc=fc, ec=ec, lw=lw, zorder=z)
    if dash:
        p.set_linestyle(dash)
    ax.add_patch(p)
    return p


def rail(ax, x0, x1, y, c=INK2, lw=1.3, z=1):
    ax.plot([x0, x1], [y, y], color=c, lw=lw, zorder=z, solid_capstyle="round")


def heater(ax, x, y, dac, chars, up=True, tag=True, size=0.085, sz=7.0):
    """A heater is a small square on the arm it heats; colour is state, whisker is fit
    quality, tag is DAC and pad.
    Returns the state so the caller counts what was actually drawn, not what it meant to."""
    st = state_of(dac, chars)
    fc, ec, _ = STATES[st]
    if st == "dark":
        d = size * 1.05
        ax.plot([x - d, x + d], [y - d, y + d], color=fc, lw=1.9, zorder=8)
        ax.plot([x - d, x + d], [y + d, y - d], color=fc, lw=1.9, zorder=8)
    else:
        ax.add_patch(mp.Rectangle((x - size, y - size * 0.85), 2 * size, size * 1.70,
                                  facecolor=fc, edgecolor=ec, lw=0.9, zorder=8))
    e = fit_error(dac, chars)
    bar = ERR_FULL * e if e is not None else 0.0
    if bar >= ERR_MIN:
        # grown outward from the square's edge, so a near-perfect fit cannot look like a
        # strikethrough and a missing whisker always means "no fit", never "zero error".
        y0 = size * 0.85 + 0.035
        for s in (1, -1):
            ax.plot([x, x], [y + s * y0, y + s * (y0 + bar)], color=ec, lw=1.5, zorder=7)
            ax.plot([x - ERR_CAP, x + ERR_CAP], [y + s * (y0 + bar)] * 2, color=ec, lw=1.5,
                    zorder=7)
    else:
        bar = 0.0
    if tag:
        dy = 0.15 + bar
        ax.text(x, y + (dy if up else -dy), f"DAC{dac}·{HEATERS[dac].pad}", ha="center",
                va="bottom" if up else "top", fontsize=sz, color=INK2, zorder=9)
    return st


def mzi(ax, x, yt, yb, w, arm, xc, xa, col=BLUE):
    """The 6x6's canonical 2x2 glyph: both rails converge into a coupler, split into two
    arms, and recombine. Returns the upper arm's midpoint, where the internal phase sits."""
    cy = (yt + yb) / 2.0
    x1 = x + w
    xc1, xc2 = x + xc, x1 - xc
    xA, xB = x + xa, x1 - xa
    yA, yB = cy + arm, cy - arm
    ax.plot([x, xc1, xA, xB, xc2, x1], [yt, cy, yA, yA, cy, yt], color=col, lw=1.5, zorder=3)
    ax.plot([x, xc1, xA, xB, xc2, x1], [yb, cy, yB, yB, cy, yb], color=col, lw=1.5, zorder=3)
    for xk in (xc1, xc2):
        ax.add_patch(mp.Rectangle((xk - 0.055, cy - 0.085), 0.11, 0.17, facecolor=col,
                                  edgecolor="none", zorder=4))
    return (xA + xB) / 2.0, yA


def pd_glyph(ax, x, y, idx, ok):
    c = PD_OK if ok else PD_BAD
    ax.add_patch(mp.RegularPolygon((x, y), 3, radius=0.20, orientation=-1.5708,
                                   facecolor=c, edgecolor="none", zorder=6))
    ax.add_patch(mp.Rectangle((x + 0.19, y - 0.20), 0.055, 0.40, facecolor=c,
                              edgecolor="none", zorder=6))
    ax.text(x + 0.36, y + 0.09, f"PD{idx} → A{idx}", ha="left", va="bottom", fontsize=8.4,
            color=INK, zorder=9)
    return c


def fig_mesh(calib, chars, date):
    """The optical path: switch -> 4 rails -> 6 MZIs in 4 columns -> output screen -> 4 PDs.

    Every claim on the drawing is read back out of the source of truth: rail pairs from
    `clements.MESH`, columns from `clements.COLUMN`, which DAC drives which phase from
    `layout.HEATERS`. The asserts below are the point -- an MZI drawn on the wrong rail
    pair would still look like a plausible mesh."""
    ncol = max(COLUMN) + 1
    assert ncol == NCOL and len(MESH) == len(COLUMN) == NMZI
    for c in range(ncol):
        rails = [r for k in range(NMZI) if COLUMN[k] == c for r in MESH[k]]
        assert len(rails) == len(set(rails)), f"column {c} puts two MZIs on one rail"
    by_role: dict[str, dict[int, object]] = {}
    for h in HEATERS:
        by_role.setdefault(h.role, {})[h.index] = h
    assert {k: h.h for k, h in by_role["theta"].items()} == dict(THETA_DAC)
    assert REF_RAIL not in by_role["out_phase"], "the reference rail must carry no trimmer"

    DY, MW, ARM, XCP, XAP = 1.40, 1.65, 0.29, 0.36, 0.64
    X0, PITCH, X_RAIL0, X_PHI = 4.05, 2.95, 2.45, 0.45
    Y = [(NMODE - 1 - r) * DY for r in range(NMODE)]
    XC = [X0 + PITCH * c for c in range(ncol)]
    x_alpha = XC[-1] + MW + 1.20
    x_pd = x_alpha + 1.70
    y_top = Y[0] + 1.05
    y_mid = (Y[0] + Y[-1]) / 2.0

    fig, ax = plt.subplots(figsize=(17.6, 6.8))
    ax.axis("off")
    ax.set_xlim(-1.55, x_pd + 3.25)
    ax.set_ylim(-0.62, y_top)

    nrm = calib["meta"]["normalisation"]
    port_v = nrm["port_full_v"]                     # per INPUT port, drawn on the input rails
    port_db = [10 * math.log10(p / max(port_v)) for p in port_v]
    worst = port_db.index(min(port_db))
    dark_v, full_v = nrm["dark_v"], nrm["full_v"]   # per PD, drawn at the detectors
    # which detector gets flagged is the measured dark floor, not a remembered channel number
    noisy_pd = max(range(NMODE), key=lambda r: dark_v[r])

    box(ax, 0.45, y_mid - 0.98, 1.35, 1.96, fc="#f4f6fa", ec=VIOLET, lw=1.5)
    ax.text(1.125, y_mid + 0.76, "Sercalo\n1×4", ha="center", va="top", fontsize=10,
            color=VIOLET, fontweight="bold")
    ax.text(1.125, y_mid - 0.10, f'"SET 1..4"\n{SWITCH_BAUD} bd\n3.1 dB IL\nSET 0 = dark',
            ha="center", va="top", fontsize=7.2, color=INK2)
    ax.annotate("", xy=(0.45, y_mid), xytext=(-1.05, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.text(-0.30, y_mid + 0.12, "laser 1550 nm", ha="center", va="bottom", fontsize=8.5,
            color=ORANGE)

    # rails are drawn only where no glyph sits, so the light path is one unbroken line.
    occupied = {r: [] for r in range(NMODE)}
    for k in range(NMZI):
        for r in MESH[k]:
            occupied[r].append((XC[COLUMN[k]], XC[COLUMN[k]] + MW))
    for r in range(NMODE):
        ax.plot([1.80, 2.20, X_RAIL0], [y_mid, Y[r], Y[r]], color=VIOLET, lw=1.3, zorder=3)
        x = X_RAIL0
        for a, b in sorted(occupied[r]):
            rail(ax, x, a, Y[r])
            x = b
        rail(ax, x, x_pd - 0.28, Y[r])
        ax.text(X_RAIL0 + 0.06, Y[r] + 0.11, f"IN{r}", ha="left", va="bottom", fontsize=8.4,
                color=INK2)
        ax.text(X_RAIL0 + 0.06, Y[r] - 0.11, f"{port_db[r]:+.1f} dB", ha="left", va="top",
                fontsize=7.4, color=RED if r == worst else MUT)

    counts = {k: 0 for k in STATES}
    best_vpi = min((r["vpi"] for r in chars.values() if r["ok"]), default=None)
    for c in range(ncol):
        ks = [k for k in range(NMZI) if COLUMN[k] == c]
        for k in ks:
            m, n = MESH[k]
            x = XC[c]
            hx, hy = mzi(ax, x, Y[m], Y[n], MW, ARM, XCP, XAP)
            th = by_role["theta"][k]
            counts[heater(ax, hx, hy, th.h, chars)] += 1
            ph = by_role["phi"][k]
            counts[heater(ax, x - X_PHI, Y[m], ph.h, chars, up=False)] += 1

            st = state_of(th.h, chars)
            if st == "char":
                note = (f"Vπ {calib['vpi'][th.h]:.2f} V   "
                        f"span {ceiling_span(calib, th.h):.2f}π")
            elif st == "weak":
                note = f"fit rejected   r² {chars[th.h]['r2']:.3f}"
            elif st == "capped":
                # even at the most favourable Vπ this rig has measured, span goes as (V/Vπ)².
                reach = (VOLTAGE_MAX_CH[th.h] / best_vpi) ** 2
                note = (f"{HEATER_OHMS[th.h]:.0f} Ω → {VOLTAGE_MAX_CH[th.h]:.2f} V cap   "
                        f"span ≤ {reach:.2f}π")
            else:
                note = "not swept"
            # every MZI labels above its own upper rail, so the placement rule is one rule.
            ax.text(x + MW / 2, Y[m] + 0.52, f"MZI{k + 1}  rails ({m},{n})", ha="center",
                    va="bottom", fontsize=8.6, color=INK, fontweight="bold", zorder=9)
            ax.text(x + MW / 2, Y[m] + 0.30, note, ha="center", va="bottom", fontsize=7.2,
                    color=STATES[st][1], zorder=9)

    for r in range(NMODE):
        if r == REF_RAIL:
            ax.text(x_alpha, Y[r] + 0.13, "no trimmer — global phase", ha="center",
                    va="bottom", fontsize=7.2, color=MUT, zorder=9)
            continue
        counts[heater(ax, x_alpha, Y[r], by_role["out_phase"][r].h, chars, up=True)] += 1

    for r in range(NMODE):
        ok = r != noisy_pd
        col = pd_glyph(ax, x_pd, Y[r], r, ok)
        sig = PD_DARK_SIGMA_MV.get(r)
        tail = f"   σ ≈ {sig:.0f} mV" if not ok and sig is not None else ""
        ax.text(x_pd + 0.36, Y[r] - 0.09,
                f"full {1000 * full_v[r]:.0f} mV · dark {1000 * dark_v[r]:.1f} mV{tail}",
                ha="left", va="top", fontsize=7.2, color=col if not ok else MUT, zorder=9)

    aux = [h for h in HEATERS if h.role == "aux"]
    if aux:
        ax.text(-1.45, y_top - 0.10, "off-mesh", ha="left", va="top", fontsize=8.4,
                color=MUT, fontweight="bold")
        for j, h in enumerate(aux):
            yy = y_top - 0.56 - 0.44 * j
            counts[heater(ax, -1.32, yy, h.h, chars, tag=False)] += 1
            ax.text(-1.13, yy, f"DAC{h.h}·{h.pad}\nno role assigned", ha="left", va="center",
                    fontsize=6.8, color=MUT, linespacing=1.45)
    assert sum(counts.values()) == N_HEATERS, "every channel must be drawn exactly once"

    # a state nobody is in is not information; drawing it would only invite a stale claim.
    handles = [Line2D([], [], marker="s", color="w", markerfacecolor=f, markeredgecolor=e,
                      markersize=10, label=f"{lbl}  ({counts[k]})")
               for k, (f, e, lbl) in STATES.items() if k != "dark" and counts[k]]
    if counts["dark"]:
        handles.append(Line2D([], [], marker="x", color=STATES["dark"][0], lw=0, markersize=9,
                              markeredgewidth=1.9,
                              label=f"{STATES['dark'][2]}  ({counts['dark']})"))
    handles += [
        ax.errorbar([], [], yerr=[], color=INK2, lw=1.5, capsize=3.5, marker="s",
                    markersize=7, markerfacecolor="white", markeredgecolor=INK2,
                    label="bar = √(1−r²) of the fringe fit: longer is noisier, none = never swept"),
        Line2D([], [], color=INK2, lw=1.4, label="waveguide / rail"),
        Line2D([], [], color=BLUE, lw=1.6, label="MZI: 2 couplers, 2 arms"),
        Line2D([], [], marker="v", color="w", markerfacecolor=PD_OK, markersize=11,
               label="PD: strong"),
        Line2D([], [], marker="v", color="w", markerfacecolor=PD_BAD, markersize=11,
               label=f"PD: weak (PD{noisy_pd})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.065), ncol=4,
              frameon=False, fontsize=8.4, handlelength=1.6, columnspacing=2.2,
              labelspacing=0.5)

    n_tun = sum(state_of(THETA_DAC[k], chars) == "char" for k in range(NMZI))
    ax.set_title(
        f"4x4 Clements mesh, Quanfluence PIC1A (Ligentec AN800 SiN) — "
        f"{n_tun} of {NMZI} MZI internal phases characterized   ({date})\n"
        f"heater colour = state; θ sits between the couplers, φ on the upper input arm, "
        f"α on the output rails. DAC k drives θ of MZI {NMZI}−k (measured); "
        f"the φ/α assignment is PROVISIONAL.",
        fontsize=12.5, linespacing=1.5)
    fig.tight_layout()
    save(fig, "mesh.png")


def fig_chain(calib, chars, date):
    """Host to die: three serial links, two Arduinos, and where the light joins the loop."""
    fig, ax = plt.subplots(figsize=(16.0, 8.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def node(x, cy, w, title, lines, col, h=None, tsz=10.5, lsz=7.7):
        n = lines.count("\n") + 1 if lines else 0
        h = h if h is not None else 0.058 + 0.0245 * n
        y = cy - h / 2
        box(ax, x, y, w, h, fc="white", ec=col, lw=1.6, r=0.008)
        ax.text(x + w / 2, y + h - 0.026, title, ha="center", va="top", fontsize=tsz,
                color=col, fontweight="bold", zorder=5)
        ax.text(x + w / 2, y + h - 0.062, lines, ha="center", va="top", fontsize=lsz,
                color=INK2, zorder=5, linespacing=1.45)

    def link(p0, p1, label, col=INK2, lw=1.6, dash=None, t=0.5, dx=0.0, dy=0.014, sz=7.6):
        kw = dict(arrowstyle="-|>", color=col, lw=lw, shrinkA=2, shrinkB=2)
        if dash:
            kw["linestyle"] = dash
        ax.annotate("", xy=p1, xytext=p0, arrowprops=kw, zorder=2)
        ax.text(p0[0] + t * (p1[0] - p0[0]) + dx, p0[1] + t * (p1[1] - p0[1]) + dy, label,
                ha="center", va="center", fontsize=sz, color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))

    box(ax, 0.010, 0.200, 0.215, 0.730, fc="white", ec=VIOLET, lw=1.8, r=0.008)
    ax.text(0.1175, 0.912, "host   ·   ./do", ha="center", va="top", fontsize=12,
            color=VIOLET, fontweight="bold")
    layers = [
        (0.720, "theory/", "clements · twin · calib\nprogram · drift · layout\n"
                           "pure maths, no serial", BLUE),
        (0.500, "pic/", "interface · devices · rig\nacquisition · characterize\n"
                        "normalise · sim · drift", GREEN),
        (0.280, "learn/", "unitary_fit (52 par)\ndpnn · prune · train_hw", ORANGE),
    ]
    for y, name, sub, col in layers:
        box(ax, 0.026, y, 0.183, 0.155, fc=col + "14", ec=col, lw=1.1, r=0.006)
        ax.text(0.1175, y + 0.128, name, ha="center", va="top", fontsize=10,
                color=col, fontweight="bold")
        ax.text(0.1175, y + 0.098, sub, ha="center", va="top", fontsize=7.0, color=INK2,
                linespacing=1.45)
    for y0 in (0.435, 0.655):
        ax.annotate("", xy=(0.1175, y0 + 0.057), xytext=(0.1175, y0 + 0.008),
                    arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.3))
    ax.text(0.1175, 0.258, "imports run one way only", ha="center", va="top", fontsize=7.2,
            color=MUT, style="italic")

    node(0.295, 0.845, 0.185, "AeroDiode PDMv5",
         "diode laser, its own TEC off\nshared with the 6x6 rig —\none holder at a time", ORANGE)
    node(0.295, 0.525, 0.185, "Arduino Mega  ·  pic4x4.ino",
         f"one CSV line in, one out\n{BAUD_RATE} baud\n \n"
         f"SPI 10 MHz mode 1, CS 10\nDAC config on every write\n \n"
         f"{NUM_ADC_RAW} × ADC {ADC_BITS}-bit, {ADC_AVG_N}× avg\n"
         f'Serial1 → switch: "P<0..4>"', GREEN)
    node(0.295, 0.158, 0.185, "Arduino Mega  ·  tec_pid.ino",
         "10 k NTC → Steinhart\nPID Kp 5, one update / s\nSPI to the LT8722\n"
         "streams temp,set,drive", BLUE)

    live_v = [v for v in VOLTAGE_MAX_CH if v > 0]
    n_dark = len(VOLTAGE_MAX_CH) - len(live_v)
    node(0.565, 0.775, 0.170, "DAC81416",
         f"{len(VOLTAGE_MAX_CH)} ch · {DAC_BITS} bit / {DAC_REF_V:.0f} V ref\n"
         f"ceiling V = {HEATER_MAX_MA:.0f} mA × R\n"
         f"{min(live_v):.2f}–{max(live_v):.2f} V"
         + (f" · {n_dark} × 0 V" if n_dark else ""), GREEN)
    node(0.565, 0.530, 0.170, "Sercalo 1×4 switch",
         f'{SWITCH_BAUD} bd · "SET 1..4"\n3.1 dB insertion loss\nSET 0 = optical dark', VIOLET)
    node(0.565, 0.320, 0.170, "PD-TIA × 4",
         "≈41 kΩ InGaAs\none per mesh output", GREEN)
    node(0.565, 0.158, 0.170, "LT8722 H-bridge",
         "±2 V into ±1 A\nPeltier under the die", BLUE)

    n_live = sum(v > 0 for v in VOLTAGE_MAX_CH)
    node(0.800, 0.490, 0.180, "PIC1A die",
         f" \n4×4 Clements mesh\n{NMZI} MZIs, {NCOL} columns\n \n18 heater pads,\n"
         f"{N_HEATERS} with a driver,\n{n_live} with a safe ceiling\n \n"
         f"{NMODE} inputs · {NMODE} outputs\nunitary in and out",
         INK2, h=0.520, tsz=12)

    link((0.225, 0.845), (0.295, 0.845), "USB · FTDI")
    link((0.225, 0.525), (0.295, 0.525), "USB serial")
    link((0.225, 0.158), (0.295, 0.158), "USB serial")

    link((0.480, 0.800), (0.565, 0.560), "fibre", col=ORANGE, lw=2.4, t=0.36, dx=-0.016)
    link((0.480, 0.610), (0.565, 0.740), "SPI", col=GREEN, t=0.80, dx=0.012, dy=0.020)
    link((0.480, 0.510), (0.565, 0.510), "Serial1", col=VIOLET, dash=(0, (4, 2)))
    link((0.565, 0.330), (0.480, 0.430), "A0–A3", col=GREEN, dy=-0.026)
    link((0.480, 0.158), (0.565, 0.158), "SPI", col=BLUE)

    link((0.735, 0.545), (0.800, 0.640), f"{NMODE} inputs", col=ORANGE, lw=2.4, t=0.34,
         dy=0.024)
    link((0.735, 0.760), (0.800, 0.700), f"{N_HEATERS} heaters", col=GREEN, dy=0.026)
    link((0.800, 0.400), (0.735, 0.330), f"{NMODE} outputs", col=ORANGE, lw=2.4, dy=-0.026)
    link((0.735, 0.158), (0.800, 0.245), "heat", col=BLUE, dash=(0, (4, 2)), dy=0.024)

    ax.text(0.890, 0.212,
            f"held at {TEC_SETPOINT_C:.0f}.00 ± {TEC_TOLERANCE_C:.2f} °C —\n"
            "the thing the 6x6 rig never had",
            ha="center", va="top", fontsize=7.8, color=BLUE, linespacing=1.45)

    ax.text(0.0, 0.052,
            "Four serial devices over two overlapping port globs: board, laser (FTDI, matched "
            "by name), TEC controller, switch. Pass explicit ports; one process per port.\n"
            "`laser_status == 1` does not prove emission — every session baselines "
            "bfm_optical_power off-versus-on before it believes a lit read.",
            ha="left", va="top", fontsize=8.4, color=INK2, linespacing=1.7)

    ax.set_title(f"4x4 rig control chain — host to die   ({date})", fontsize=13.5)
    fig.tight_layout()
    save(fig, "chain.png")


def fig_channels(calib, chars, date):
    """All 16 DAC channels against the ceiling each one is allowed, and why."""
    fig, ax = plt.subplots(figsize=(13.6, 8.4))
    ax.set_xlim(-1.72, 6.75)
    ax.set_ylim(-1.15, N_HEATERS - 0.35)
    ax.axis("off")

    ax.plot([0, 0], [-0.55, N_HEATERS - 0.55], color=INK2, lw=1.0)
    # One reference line per resistance group, at that group's own ceiling. The old fixed
    # 1.5/3.0 V pair was the flat design clamp; under V = I_max·R no channel sits at either.
    ohm = [(VOLTAGE_MAX_CH[d], r) for d, r in enumerate(HEATER_OHMS) if r is not None]
    for v in (max(v for v, r in ohm if r < 90), max(v for v, r in ohm if r >= 90)):
        ax.plot([v, v], [-0.55, N_HEATERS - 0.55], color=GRID, lw=1.0, ls=(0, (3, 3)))
        ax.text(v, -0.72, f"{v:.2f} V", ha="center", va="top", fontsize=8.5, color=MUT)
    ax.text(0, -0.72, "0", ha="center", va="top", fontsize=8.5, color=MUT)

    counts = {k: 0 for k in STATES}
    for d in range(N_HEATERS):
        y = N_HEATERS - 1 - d
        h = HEATERS[d]
        st = state_of(d, chars)
        counts[st] += 1
        fc, ec, _ = STATES[st]
        vmax = VOLTAGE_MAX_CH[d]
        ax.add_patch(mp.Rectangle((0, y - 0.28), max(vmax, 0.012), 0.56, facecolor=fc,
                                  edgecolor=ec, lw=0.9, zorder=4))
        ax.text(-1.68, y, f"DAC{d:<2d}", ha="left", va="center", fontsize=9.6, color=INK,
                fontweight="bold")
        ax.text(-1.20, y, h.pad, ha="left", va="center", fontsize=9.6, color=INK)
        role = (f"{ROLE_NAME[h.role]} MZI{h.index + 1}" if h.role in ("theta", "phi")
                else f"α rail {h.index}" if h.role == "out_phase" else "no role")
        ax.text(-0.72, y, role, ha="left", va="center", fontsize=8.6,
                color=INK if h.role == "theta" else MUT)
        r = HEATER_OHMS[d]
        ax.text(-0.06, y, f"{r:.0f} Ω" if r is not None else "R ?", ha="right", va="center",
                fontsize=8.4, color=MUT)
        ax.text(max(vmax, 0.012) + 0.08, y, role_note(h, chars, calib), ha="left", va="center",
                fontsize=8.2, color=ec if st in ("char", "dark") else INK2)

    handles = [mp.Patch(facecolor=f, edgecolor=e, label=f"{lbl}  ({counts[k]})")
               for k, (f, e, lbl) in STATES.items() if counts[k]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.40, -0.005),
              frameon=False, fontsize=8.8, ncol=2, handlelength=1.5, columnspacing=2.4)

    ax.set_title(
        f"DAC81416 channel map, and what each channel is allowed to do   ({date})\n"
        f"{N_HEATERS} drivers for 18 pads — H16 and H17 have none. Bar length is the "
        f"per-channel voltage ceiling.", fontsize=12.5, linespacing=1.5)
    fig.tight_layout()
    save(fig, "channels.png")


FIGS = {"mesh": fig_mesh, "chain": fig_chain, "channels": fig_channels}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(FIGS)
    bad = [w for w in want if w not in FIGS]
    if bad:
        raise SystemExit(f"unknown figure {bad}; choose from {', '.join(FIGS)}")
    data = _load()
    for w in want:
        FIGS[w](*data)
