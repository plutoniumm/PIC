"""Section III figure: characterised channels of both mesh sections vs their sweep limits.

    PYTHONPATH=. python scripts/fig_census.py

One panel: every reliably fringe-characterised channel of PIC-A (full-range 0-5 V sweeps,
Anagha/PIC Data_pranav) and PIC-B (census under the 4 V firmware clamp), plotted as V_pi
against the single-channel PD swing, with each section's sweep limit as a vertical line.
A point at/right of its own section's line cannot reach its null within the swept range,
so its operating point is read off the fit (the paper's "extrapolated" case). The
full-range PIC-A sweep closes every fringe; the clamped PIC-B sweep does not.

Reads pic_data/pic_b_config.json (per-channel Vpi/swing) and pic_data/fringe_fits_pica.csv
(fits of the raw PIC-A sweeps by src.census.robust_fit, reliability-gated, vmax=5 V).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--pica", default="pic_data/fringe_fits_pica.csv")
    ap.add_argument("--out", default="slides/images/census.pdf")
    a = ap.parse_args(argv)

    cfg = json.load(open(a.config))
    vclamp = float(cfg["firmware"]["v_clamp"])
    # distributions, not the full census: the first 64 channels by index per section
    car = sorted((c for c in cfg["channels"] if c.get("Vpi") is not None),
                 key=lambda c: c["dac"])[:64]
    vpi_b = np.array([c["Vpi"] for c in car], float)
    sw_b = np.array([c["swing_mV"] for c in car], float)

    rows = sorted((r for r in csv.DictReader(open(a.pica)) if r["reliable"] == "True"),
                  key=lambda r: int(r["ch"]))[:64]
    vpi_a = np.array([float(r["Vpi"]) for r in rows])
    sw_a = np.array([float(r["swing_mV"]) for r in rows])
    VMAX_A = 5.0

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.9))

    ax.plot(vpi_b, sw_b, "o", color=BLUE, ms=4.4, mec="white", mew=0.6, zorder=4,
            label=f"PIC-B, swept to {vclamp:.0f} V ({len(vpi_b)})")
    ax.plot(vpi_a, sw_a, "s", color=ORANGE, ms=4.0, mec="white", mew=0.6, zorder=3,
            label=f"PIC-A, swept to {VMAX_A:.0f} V ({len(vpi_a)})")
    ax.axvline(vclamp, color=BLUE, lw=1.0, ls="--", zorder=2)
    ax.axvline(VMAX_A, color=ORANGE, lw=1.0, ls="--", zorder=2)
    ax.text(vclamp - 0.08, 6.2, "B limit", color=BLUE, fontsize=7, ha="right")
    ax.text(VMAX_A + 0.08, 6.2, "A limit", color=ORANGE, fontsize=7, ha="left")

    ax.set_yscale("log")
    ax.set_xlim(0, max(vpi_b.max(), VMAX_A) * 1.08)
    ax.set_ylim(4, 900)
    ax.set_xlabel("$V_\\pi$  (V)")
    ax.set_ylabel("single-channel PD swing  (mV)")
    ax.grid(True, which="major", color=GRID, lw=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4,
              labelspacing=0.35, borderpad=0.2)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        p = a.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print("wrote", p)
    print(f"B: {len(vpi_b)} ch, Vpi median {np.median(vpi_b):.2f}, "
          f"{int((vpi_b > vclamp).sum())} beyond clamp; "
          f"A: {len(vpi_a)} ch, Vpi median {np.median(vpi_a):.2f}, "
          f"{int((vpi_a > VMAX_A).sum())} beyond sweep")


if __name__ == "__main__":
    main()
