"""Score the digital twin against every state in a real random-heater capture.

`pic.compute.ising.run_gate` already compares `theory.twin`'s monitor powers to the chip,
but only ever for the ONE configuration an Ising run programmes -- 6 numbers, from which
no honest correlation can be read. This runs the same comparison over the whole
`scripts/pd_learn_test.py` capture (150 random drivable-heater settings x 3 repeats, all
14 PDs, +15 dBm, TEC on), which is the largest real many-state dataset PIC B has.

The observable is deliberately the one the 4x4 uses (`4x4/twin_vs_hardware.py`), so the
two chips' numbers mean the same thing: the per-state monitor vector normalised to sum 1.
Normalisation is the measurement, not a convenience -- it divides out launch power, fibre
coupling and PD transimpedance, none of which the twin represents and all of which move
between states. What survives is the shape of the output distribution, which is exactly
what the twin claims to predict.

Two statistics, both over the flattened (n_state, n_pd) array:

    pearson   corrcoef(P, M)                 scale- and offset-free
    rel       ||P - M||_F / ||M||_F          the 4x4's `learn.surrogate.rel`

and three references that say what those numbers are worth: the mean measured
distribution (the null model), a state-shuffled twin (what zero predictive content scores)
and repeat-vs-repeat on the same configs (what the data's own noise permits).

    PYTHONPATH=. python scripts/twin_vs_hardware.py
    PYTHONPATH=. python scripts/twin_vs_hardware.py --pds all --dark 0.0098
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pic.homodyne import DEFAULT_PDMAP

CAPTURE = "pic_data/census/pdlearn_20260730_213948.csv"
MON_PDS = list(DEFAULT_PDMAP.monitor)                    # (2, 4, 6, 7, 9, 11)
ALL_PDS = MON_PDS + list(DEFAULT_PDMAP.homodyne) + \
    [DEFAULT_PDMAP.lo_tap_top, DEFAULT_PDMAP.lo_tap_bottom]


def pearson(P, M) -> float:
    return float(np.corrcoef(np.asarray(P).ravel(), np.asarray(M).ravel())[0, 1])


def rel(P, M) -> float:
    return float(np.linalg.norm(np.asarray(P) - M) / np.linalg.norm(M))


def load_capture(path=CAPTURE, dark: float = 0.0):
    """(volts, per-repeat PD volts) as (n_cfg, 128) and (n_cfg, n_rep, 14).

    `dark` is subtracted from every PD read. The capture has no laser-off row of its own;
    `runs/hwtests/dark_no_laser.csv` puts the true floor at one ADC LSB (~5-10 mV), which
    is why the default is 0 and the flag exists to show the answer does not turn on it.
    """
    by_cfg = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by_cfg[int(r["cfg"])].append(r)
    keys = sorted(by_cfg)
    V = np.array([[float(x) for x in by_cfg[k][0]["input"].split()] for k in keys])
    Y = np.array([[[float(r[f"pd{i}"]) for i in range(14)] for r in by_cfg[k]]
                  for k in keys])
    return V, np.clip(Y - dark, 0.0, None)


def twin_forward(V, config=None):
    """Twin monitor/homodyne/LO powers packed into a 14-PD vector, one row per state."""
    from pic.model import TwinModel
    tm = TwinModel(config) if config else TwinModel()
    return np.array([tm.predict(v) for v in V])


def normalise(A, pds):
    A = np.asarray(A, float)[:, pds]
    return A / np.maximum(A.sum(1, keepdims=True), 1e-12)


def score(P, M, shuffles: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    sh = np.array([[pearson(P[i], M), rel(P[i], M)]
                   for i in (rng.permutation(len(P)) for _ in range(shuffles))])
    return {"n": int(P.size), "pearson": pearson(P, M), "rel": rel(P, M),
            "shuffled_pearson": float(sh[:, 0].mean()), "shuffled_pearson_sd": float(sh[:, 0].std()),
            "shuffled_rel": float(sh[:, 1].mean()), "shuffled_rel_sd": float(sh[:, 1].std())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", nargs="?", default=CAPTURE)
    ap.add_argument("--pds", choices=("mon", "all"), default="mon",
                    help="mon: the 6 monitor taps the twin's mesh output feeds. "
                         "all: those plus the 6 homodyne PDs and 2 LO taps, whose twin "
                         "prediction also rides on the uncalibrated reference arms")
    ap.add_argument("--dark", type=float, default=0.0, help="PD floor subtracted, volts")
    ap.add_argument("--config", default=None, help="pic_b_config.json to calibrate from")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    pds = MON_PDS if a.pds == "mon" else ALL_PDS
    V, Y = load_capture(a.capture, a.dark)
    n_cfg, n_rep = Y.shape[0], Y.shape[1]
    P = normalise(twin_forward(V, a.config), pds)
    M = normalise(Y.mean(1), pds)

    s = score(P, M, seed=a.seed)
    mean = np.tile(M.mean(0), (len(M), 1))
    half = (normalise(Y[:, :n_rep // 2].mean(1), pds),
            normalise(Y[:, n_rep // 2:].mean(1), pds))

    print(f"{a.capture}\n  {n_cfg} configs x {n_rep} repeats, {len(pds)} PDs {pds}, "
          f"dark {a.dark * 1000:.1f} mV\n")
    print(f"  {'model':<26}{'N':>6}{'pearson':>10}{'rel':>9}")
    print(f"  {'twin (characterized)':<26}{s['n']:>6}{s['pearson']:>10.4f}{s['rel']:>9.4f}")
    print(f"  {'mean measured distrib':<26}{M.size:>6}{pearson(mean, M):>10.4f}{rel(mean, M):>9.4f}")
    print(f"  {'twin, states shuffled':<26}{s['n']:>6}{s['shuffled_pearson']:>10.4f}"
          f"{s['shuffled_rel']:>9.4f}"
          f"   +-{s['shuffled_pearson_sd']:.4f} / {s['shuffled_rel_sd']:.4f}")
    print(f"  {'repeats vs repeats':<26}{M.size:>6}{pearson(*half):>10.4f}{rel(*half):>9.4f}")
    print("\n  raw (un-normalised) pooled pearson, twin vs chip: "
          f"{pearson(twin_forward(V, a.config)[:, pds], Y.mean(1)[:, pds]):.4f}")
    return s


if __name__ == "__main__":
    main()
