"""Section II figure: one heater's calibration, start to finish.

    PYTHONPATH=. python scripts/fig_calib.py

Raw sweep points for channel 0 read on PD8 (pic_data/census/sweep_b128_fringes.csv, 24
points over the 0-2 V band, repeats averaged), the fitted cosine-in-squared-voltage
response, and the operating points read off the fit: V0 (maximum transmission), Vnull
(dark), and the Vpi half-fringe spacing. Extrema beyond the swept band are drawn dashed:
that is what an extrapolated operating point is.

Replaces a paragraph of Section II prose; the fit numbers cross-check
pic_data/census/fringe_fits_b128.csv (ch0/pd8: swing 170.3 mV, Vpi 2.21, r2 0.998).
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ORANGE = "#eb6834"

CH, PD = "0", "pd8"


def model(v, A, B, p2, p0):
    return A + B * np.cos(p2 * v ** 2 + p0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = defaultdict(list)
    for r in csv.DictReader(open("pic_data/census/sweep_b128_fringes.csv")):
        if r["ch"] == CH and r["phase"] != "base":
            acc[float(r["drive_v"])].append(float(r[PD]))
    v = np.array(sorted(acc))
    y = np.array([np.mean(acc[k]) for k in v])
    sd = np.array([np.std(acc[k], ddof=1) for k in v])

    p, _ = curve_fit(model, v, y, p0=[y.mean(), (y.max() - y.min()) / 2, 0.65, 0.5],
                     maxfev=20000)
    A, B, p2, p0 = p
    if B < 0:
        B, p0 = -B, p0 + np.pi
    vpi = float(np.sqrt(np.pi / abs(p2)))
    rmse = float(np.sqrt(np.mean((model(v, *p) - y) ** 2)))

    # operating points: argmax / argmin of the fit. V0 sits at the 0 V boundary (this
    # channel starts bright); the null lands just past the swept band -- the "null past
    # the clamp, recovered from the fit" case the text describes.
    vv = np.linspace(0, 3.1, 3000)
    fv = model(vv, *p)
    v0 = float(vv[np.argmax(np.where(vv <= v.max(), fv, -np.inf))])
    vnull = float(vv[np.argmin(fv)])

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    vmax_swept = v.max()
    dd_in = np.linspace(0, vmax_swept, 300)
    dd_out = np.linspace(vmax_swept, 3.2, 160)
    ax.plot(dd_in, 1e3 * model(dd_in, *p), color=RAMP[2], lw=1.4, zorder=3,
            label="fit")
    ax.plot(dd_out, 1e3 * model(dd_out, *p), color=RAMP[2], lw=1.1, ls=(0, (4, 3)),
            zorder=3, label="extrapolation")
    ax.errorbar(v, 1e3 * y, yerr=1e3 * sd, fmt="o", color=INK, ms=3.6, mec="white",
                mew=0.6, elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=4,
                label="measured")

    for x, name, dx, dy in ((v0, "$V_0$", 0.09, -22), (vnull, "$V_{\\mathrm{null}}$", 0.09, 14)):
        yv = 1e3 * model(np.array([x]), *p)[0]
        ax.plot([x], [yv], "s", color=ORANGE, ms=5.5, mec="white", mew=0.7, zorder=5)
        ax.annotate(name, xy=(x, yv), xytext=(x + dx, yv + dy), color=ORANGE, fontsize=8.5)
    ax.text(0.03, 0.55, f"$V_\\pi = {vpi:.2f}$ V",
            transform=ax.transAxes, color=MUTED, fontsize=7, va="top")

    ax.axvspan(vmax_swept, 3.2, color="#f5f5f3", zorder=0)

    ax.set_xlim(-0.05, 3.2)
    ax.set_xlabel("drive voltage  $V$  (V)")
    ax.set_ylabel(f"PD{PD[2:]} response  (mV)")
    ax.grid(True, color=GRID, lw=0.4, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.5, borderpad=0.2)

    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"slides/images/calib.{ext}", dpi=400, bbox_inches="tight")
        print("wrote", f"slides/images/calib.{ext}")
    print(f"fit: Vpi={vpi:.3f}  V0={v0:.3f}  Vnull={vnull:.3f}  rmse={1e3*rmse:.2f} mV")


if __name__ == "__main__":
    main()
