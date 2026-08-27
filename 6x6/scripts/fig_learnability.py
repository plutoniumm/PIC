"""Section B figure: how well the digital-twin surrogate (the hardware-trained DPNN) predicts
each photodiode, and where that sits against the project's PIC-A forward models.

    PYTHONPATH=. python scripts/fig_learnability.py

Panel (a) per-PD held-out R^2, sorted descending, split by the PD signal tier. The per-PD
values are NOT stored anywhere -- `runs/dpnn_hw/resume*.log` only logs the aggregates -- so they
are recomputed here exactly as `scripts/train_hw_dpnn.py:train_round` does: reload
`runs/dpnn_hw/ckpt.pt` (weights + frozen norm), rebuild the features from
`runs/dpnn_hw/buffer.npz` (working-heaters^2 + dbm + laser telemetry), regenerate the same
15% validation split (`np.random.default_rng(seed=0)`), and score. The script asserts the
recomputed mean against the final `R2 mean` line parsed out of the resume logs, so the bars
are guaranteed to be the same model the logs report.

Panel (b) the same mean beside the two PIC-A forward-model results quoted in
`Anagha/README.md` ("Results worth citing"): without TEC and with TEC PID at 23 C.
Same project, different section, board and DAC map.

Reads: runs/dpnn_hw/{ckpt.pt,buffer.npz,meta.json,resume*.log}, pic_data/pic_b_config.json
(PD strong/weak tiers, drivable channel set), pic_data/pd_learnable.json (per-PD SNR).
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, validated
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# PIC-A forward models, verbatim from Anagha/README.md "Results worth citing".
PIC_A = [
    ("PIC A\nno TEC", 0.70, "PIC_ML.pdf (Jun 2025)"),
    ("PIC A\nTEC 23$^\\circ$C", 0.858, "pic_128_report.pdf (Jan 2026)"),
]


def per_pd_r2(ckpt_dir, config, seed=0):
    """Recompute per-PD held-out R^2 from the checkpoint + replay buffer (see module doc)."""
    import torch
    from src.prune import DynamicPrunedMLP

    cfg = json.load(open(config))
    feat = sorted(c["dac"] for c in cfg["channels"] if c.get("elec") == "ok")
    ck = torch.load(f"{ckpt_dir}/ckpt.pt", weights_only=False)
    model = DynamicPrunedMLP(ck["din"], ck["dout"], hidden=tuple(ck["widths"]),
                             activation=ck["meta"].get("act", "relu"), min_neurons=8)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    fm, fs, ym, ys = [np.asarray(v, float) for v in ck["norm"]]

    b = np.load(f"{ckpt_dir}/buffer.npz")
    F = np.hstack([b["H"][:, feat] ** 2, b["dbm"].reshape(-1, 1), b["tel"].reshape(-1, 3)])
    Y = b["Ypd"][:, :ck["dout"]]
    n = len(F)
    vi = np.random.default_rng(seed).choice(n, max(1, int(0.15 * n)), replace=False)
    val = np.zeros(n, bool)
    val[vi] = True
    with torch.no_grad():
        P = model(torch.tensor(((F - fm) / fs).astype(np.float32)[val])).numpy() * ys + ym
    yv = Y[val]
    r2 = 1 - ((yv - P) ** 2).sum(0) / (((yv - yv.mean(0)) ** 2).sum(0) + 1e-12)
    return r2, int(val.sum()), n, cfg


def logged_mean(ckpt_dir):
    """Final `R2 mean` printed by the training loop, for the self-check. None if absent."""
    vals = []
    for p in sorted(glob.glob(f"{ckpt_dir}/resume*.log")):
        vals += [float(m) for m in re.findall(r"R2 mean ([-+]?\d*\.\d+)", open(p).read())]
    return vals[-1] if vals else None


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/dpnn_hw")
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--out", default="slides/images/learnability.pdf")
    a = ap.parse_args(argv)

    r2, n_val, n_all, cfg = per_pd_r2(a.ckpt, a.config)
    meta = json.load(open(f"{a.ckpt}/meta.json"))
    lm = logged_mean(a.ckpt)
    if lm is not None and abs(lm - r2.mean()) > 5e-3:
        raise SystemExit(f"recomputed mean {r2.mean():.3f} != logged {lm:.3f}")

    strong = set(cfg["photodiodes"]["strong"])
    order = np.argsort(r2)                     # barh draws bottom-up -> ascending = top-down
    ypos = np.arange(len(order))

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                 gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---- (a) per-PD held-out R^2 -------------------------------------------
    cols = [BLUE if i in strong else ORANGE for i in order]
    ax.barh(ypos, r2[order], color=cols, height=0.72, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"PD{i}" for i in order])
    for y, i in zip(ypos, order):
        ax.text(r2[i] + 0.012, y, f"{r2[i]:.2f}", va="center", ha="left", fontsize=6.2,
                color=MUTED, zorder=4)
    ax.axvline(r2.mean(), color=INK, lw=0.9, ls="--", zorder=5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.2)
    ax.set_xlabel("held-out $R^2$")
    ax.set_title(f"(a)  DPNN surrogate, {len(order)} PDs "
                 f"({n_val} held out of {n_all})", loc="left", color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.4, zorder=1)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label=f"strong PD ({len(strong)})"),
                       Patch(facecolor=ORANGE, label=f"weak PD ({14 - len(strong)})"),
                       Line2D([0], [0], color=INK, lw=0.9, ls="--",
                              label=f"mean {r2.mean():.3f}")],
              frameon=False, loc="lower right", handlelength=1.4, handletextpad=0.5,
              borderpad=0.2)

    # ---- (b) beside the PIC-A forward models --------------------------------
    labels = [p[0] for p in PIC_A] + ["PIC B\nno TEC"]
    vals = [p[1] for p in PIC_A] + [float(r2.mean())]
    cols_b = [RAMP[0], RAMP[0], RAMP[2]]
    bx.bar(np.arange(3), vals, color=cols_b, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        bx.text(i, v + 0.018, f"{v:.3f}" if i == 2 else f"{v:.2f}", ha="center", va="bottom",
                fontsize=7, color=INK, zorder=4)
    bx.set_xticks(np.arange(3))
    bx.set_xticklabels(labels, fontsize=6.2, color=MUTED, linespacing=1.35)
    bx.tick_params(axis="x", length=0)
    bx.set_ylim(0, 1.05)
    bx.set_yticks(np.arange(0.0, 1.01, 0.2))
    bx.set_ylabel("mean forward-model $R^2$")
    bx.set_title("(b)  forward-model mean $R^2$, both sections", loc="left", color=INK)
    bx.grid(True, axis="y", color=GRID, lw=0.4, zorder=1)
    bx.set_axisbelow(True)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        p = a.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print("wrote", p)
    print(f"per-PD R2 {np.round(r2, 3).tolist()}")
    print(f"mean {r2.mean():.4f} (logged {lm}); n_params {meta['n_params']}, "
          f"widths {meta['widths']}, samples {meta['n_samples']}")


if __name__ == "__main__":
    main()
