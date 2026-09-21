"""Score the twin against every state in the four-port transfer capture.

The companion of `6x6/scripts/twin_vs_hardware.py`, and written to the same definitions so
the two chips' numbers can be put beside each other. The observable there is a per-state
monitor vector normalised to sum 1; here it is `learn.transfer_fit.load`'s column-normalised
|U|^2, which is the same object -- a conservation-normalised distribution of light over the
detectors -- measured once per input port instead of once per state.

Two statistics over the flattened array, both already defined in `learn.surrogate`:

    pearson   corrcoef(P, M)                 scale- and offset-free
    rel       ||P - M||_F / ||M||_F

Three models, because the twin's number only means something against them:

  * `--fit` off (default): the CHARACTERIZED twin, `pic_data/calib.json` straight out of
    `pic.characterize`, scored on the whole table. Nothing was fitted to this data, so
    in-sample and held-out are the same number and there is nothing to split.
  * `--fit`: `learn.transfer_fit.fit` refitted on the training states only and scored on
    the held-out ones. That is the 52-parameter instrument -- Vpi and the identifiable phi0,
    twelve coupler ratios, three detector gain ratios -- and it is a refiner warm-started
    from the characterized Vpi, not a search.
  * the mean measured transfer, the null model, and a state-shuffled twin, which is what a
    prediction with no content about which state produced which reading scores.

`--split random` holds out a scattered 25 percent (`Transfers.split`, the module default);
`--split region` holds out a slab of one channel's range at a time, the extrapolation split
`learn.surrogate` reports against. They are different questions and the gap between them is
the honest measure of how far the fit travels.

    python twin_vs_hardware.py                      # the characterized twin
    python twin_vs_hardware.py --fit                # + the refit, held out at random
    python twin_vs_hardware.py --fit --split region # + the refit, extrapolating
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from learn.transfer_fit import ROLES_LAYOUT, load, predict
from theory.calib import Calibration

CAPTURE = "pic_data/sessions/2026-08-27/raw_transfers_fresh.json"
CALIB = "pic_data/calib.json"


def pearson(P, M) -> float:
    return float(np.corrcoef(np.asarray(P).ravel(), np.asarray(M).ravel())[0, 1])


def rel(P, M) -> float:
    return float(np.linalg.norm(np.asarray(P) - M) / np.linalg.norm(M))


def shuffled(P, M, trials: int = 200, seed: int = 0):
    """What the same prediction scores once it no longer knows which state it belongs to."""
    rng = np.random.default_rng(seed)
    s = np.array([[pearson(P[i], M), rel(P[i], M)]
                  for i in (rng.permutation(len(P)) for _ in range(trials))])
    return s.mean(0), s.std(0)


def region_splits(tr, active, frac: float = 0.6):
    """Hold out the top of one channel's range at a time -- `learn.surrogate.splits`."""
    u = (tr.volts[:, active] / np.maximum(tr.volts[:, active].max(0), 1e-9)) ** 2
    for c in range(u.shape[1]):
        iva = np.flatnonzero(u[:, c] > frac)
        if len(iva):
            yield f"ch{active[c]}>{frac}", np.setdiff1d(np.arange(len(tr)), iva), iva


def refit(tr, split, steps: int, restarts: int, seed: int):
    """Fitted model evaluated through its own forward, not through `FitResult.calib`.

    `TransferModel.extract` stores phi0 mod 2 pi, and `theory.twin`'s MZI block carries a
    `exp(-i theta/2)` gauge factor -- a common phase per MZI, which is invisible on one
    block and NOT invisible once the blocks sit on different rail pairs of a mesh. The
    twin is therefore 4 pi periodic in theta, so a wrapped phi0 is a different chip: on
    the 2026-08-27 table the round trip moves two channels by 2 pi and the held-out
    residual from 0.054 to 0.103. The module has no such wrap.
    """
    from learn.transfer_fit import fit
    res = fit(tr, ROLES_LAYOUT, calib0=Calibration.load(CALIB), split=split,
              steps=steps, restarts=restarts, seed=seed)
    with torch.no_grad():
        return res.model(torch.as_tensor(tr.volts)).numpy()[0], res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", nargs="?", default=CAPTURE)
    ap.add_argument("--fit", action="store_true", help="also refit the instrument per split")
    ap.add_argument("--split", choices=("random", "region"), default="random")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--restarts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    tr = load(a.capture)
    M = tr.T
    active = np.flatnonzero(tr.swing > 0.2)
    P = predict(Calibration.load(CALIB), ROLES_LAYOUT, tr.volts)
    mean = np.tile(M.mean(0), (len(M), 1, 1))
    (sp, sr), (dp, dr) = shuffled(P, M, seed=a.seed)

    print(f"{a.capture}\n  {len(tr)} states x {M.shape[2]} input ports x {M.shape[1]} "
          f"detectors, {len(active)} moving channels (DAC {[int(c) for c in active]})\n")
    print(f"  {'model':<26}{'N':>6}{'pearson':>10}{'rel':>9}")
    print(f"  {'twin (characterized)':<26}{M.size:>6}{pearson(P, M):>10.4f}{rel(P, M):>9.4f}")
    print(f"  {'mean measured transfer':<26}{M.size:>6}{pearson(mean, M):>10.4f}{rel(mean, M):>9.4f}")
    print(f"  {'twin, states shuffled':<26}{M.size:>6}{sp:>10.4f}{sr:>9.4f}   +-{dp:.4f} / {dr:.4f}")

    if not a.fit:
        return
    if a.split == "random":
        splits = [("random", *tr.split(a.val_frac, a.seed))]
    else:
        splits = list(region_splits(tr, active))
    print(f"\n  refit ({a.split} holdout, {a.steps} steps x {a.restarts} restarts)")
    print(f"  {'split':<14}{'nval':>5}{'N':>7}{'pearson':>10}{'rel':>9}{'rel(train)':>12}"
          f"{'rel(mean)':>11}")
    rows = []
    for name, itr, iva in splits:
        Q, _ = refit(tr, (itr, iva), a.steps, a.restarts, a.seed)
        e, p = rel(Q[iva], M[iva]), pearson(Q[iva], M[iva])
        rows.append((p, e))
        print(f"  {name:<14}{len(iva):>5}{M[iva].size:>7}{p:>10.4f}{e:>9.4f}"
              f"{rel(Q[itr], M[itr]):>12.4f}"
              f"{rel(np.tile(M[itr].mean(0), (len(iva), 1, 1)), M[iva]):>11.4f}")
    if len(rows) > 1:
        r = np.array(rows)
        print(f"  {'MEDIAN':<14}{'':>5}{'':>7}{np.median(r[:, 0]):>10.4f}"
              f"{np.median(r[:, 1]):>9.4f}")


if __name__ == "__main__":
    main()
