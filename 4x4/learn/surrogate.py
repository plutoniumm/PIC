"""Can a learned volts -> |U|^2 surrogate replace the measured lookup table? No.

`pic.matvec.plan_from_table` picks the nearest of N measured heater states because the
physics twin is not predictive -- over a measured table its |U|^2 misses the chip by a
relative 1.14, worse than predicting the mean measured transfer (0.48). That lookup is now
the binding error term and it is saturated in table size, so the obvious next move is to
learn `volts -> T` and optimise volts through the model the way `theory.intensity_matvec.
plan_matvec` optimises through the twin. This module measures whether the data supports
that. It does not, and it fails in the twin's own way.

Everything is scored in the unit `talk/progress.md` quotes, ``rel = ||P - T||_F / ||T||_F``
over column-normalised transfers, so the numbers sit beside the twin's 1.14 and the mean's
0.48. On the 155-state 25 C table (`raw_transfers_fresh.json`), held out by construction:

    family                          region   corner   random     note
    mean measured transfer          0.46     0.49     0.44       the null model
    twin, pic_data/calib.json       1.12     1.12     1.12       whole table, not a split
    ridge on V^2                    0.317    0.325    0.282
    ridge on V                      0.293    0.308    0.260      V^2-only is NOT better here
    ridge on V^2 + cross terms      0.284    0.230    0.181
    ridge on pairwise cos/sin       0.266    0.231    0.182      the physical basis
    MLP (64,32) on V^2              0.221    0.190    0.153      best of everything tried
    twin + learned residual         0.392    0.356    0.314      WORSE than ignoring the twin
    transfer_fit refit per split    0.314                        `--physics`
    that refit + learned residual   0.216                        ties the plain MLP

A surrogate is therefore a real model of this chip -- 7x better than the twin and 3x better
than the null -- and still nowhere near good enough to plan through. Four findings, in the
order they decide it.

**Planning through the surrogate reproduces the twin's failure exactly** (`--plan`). Two
nets fitted on disjoint halves of the states are both legitimate models of the same chip.
Optimise volts continuously through A and A reports a hosting residual of **0.0001**; B,
just as entitled to an opinion, says that plan hosts at **0.216**, against **0.040** for
picking a measured state off the same 155. The optimiser walks to wherever the model is most
wrong -- 0.216 at the optimum against 0.158 disagreement at random states -- which is why a
residual a planner reports about itself is not evidence about the chip. Same shape as
`plan_matvec` through the twin reporting 0.0001 and delivering vectors 1.1 out.

**The chip does not reproduce itself better than the model would have to be.** The same 100
heater states measured in two captures disagree at rel 0.078-0.127 (`repeatability`). 0.08
is the floor any surrogate is aiming at, and the lookup's 0.0669 is *below* it -- because a
lookup residual is an in-sample fit to the very numbers it selects from, not a prediction.

**More states cannot close it** (`--curve`). Held-out rel falls as N^-0.43: 327 states for
0.10, 1612 for 0.05. 327 is already past the reproducibility ceiling and 1612 is about two
hours of capture, over which `talk/progress.md` measures the table moving further than the
model would gain.

**What the surrogate IS good for is ordering, not valuing** (`--rank`). Its residual ranks
candidate states it has never seen at spearman 0.916 against the truth. Its top-1 pick hosts
at 0.139, but measuring its top 3 gets 0.094 and its top 5 gets 0.089 -- which is the oracle
over all 38 candidates, for five reads instead of a capture. That is the deployable use:
shortlist, then measure. The continuous version of the same idea is weaker and still loses
to the table (B says 0.081 for the best of 8 optimised proposals against 0.040).

One side result worth keeping: the 6x6's feature finding does **not** carry over. Dropping
the linear V term is worth 0.00-0.01 rel here (`--basis`), not the +0.10 R^2 it bought
there, which is what `learn.dpnn`'s docstring already predicts for a chip where no heater
wraps.

    python -m learn.surrogate            # families x splits, on the newest 25 C table
    python -m learn.surrogate --plan     # plan through A, score under B  <- the answer
    python -m learn.surrogate --hosting  # what a model-picked state actually hosts
    python -m learn.surrogate --rank     # the one thing it does well
    python -m learn.surrogate --curve    # would more states fix it
    python -m learn.surrogate --physics  # transfer_fit as model and as base
    python -m learn.surrogate --basis    # V^2 against V, the 6x6's finding retested
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from theory.calib import Calibration
from theory.intensity_matvec import BEST_RAILS, sinkhorn

from .transfer_fit import ROLES_LAYOUT, load, predict

FRESH = "pic_data/sessions/2026-08-27/raw_transfers_fresh.json"
ALPHAS = 10.0 ** np.arange(-4, 5)


def dataset(path=FRESH):
    """Transfers plus the phase fraction ``u = (V/Vmax)^2``, the coordinate the physics is
    linear in. Only the channels that actually moved get a column: a static input cannot
    explain a moving target and is capacity to overfit with."""
    tr = load(path)
    meta = json.loads(Path(path).read_text())
    vmax = np.asarray(meta["vmax"], float)
    act = np.asarray(meta["trainable"], int)
    act = act[vmax[act] > 0]
    return tr, (tr.volts[:, act] / vmax[act]) ** 2, act


def rel(P, T) -> float:
    return float(np.linalg.norm(np.asarray(P) - T) / np.linalg.norm(T))


def r2(P, T) -> float:
    return float(1 - ((T - P) ** 2).sum() / ((T - T.mean()) ** 2).sum())


def norm_cols(P):
    """A prediction is only a transfer once it obeys the conservation law the data was
    normalised by; clipping first because a regression will happily predict negative light."""
    P = np.clip(np.asarray(P, float).reshape(-1, 4, 4), 0.0, None)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12)


def features(u, kind: str) -> np.ndarray:
    """Feature families. `u` is V^2 and `sqrt(u)` is V, so "V^2 only" versus "V only" versus
    both is a one-word change here -- the 6x6 found dropping V worth +0.10 R^2 and this chip
    does not reproduce it (see `_report`)."""
    n, m = u.shape
    one = np.ones((n, 1))
    if kind == "u":
        return np.hstack([one, u])
    if kind == "v":
        return np.hstack([one, np.sqrt(u)])
    if kind == "uv":
        return np.hstack([one, u, np.sqrt(u)])
    if kind == "u2":
        cross = np.array([u[:, i] * u[:, j]
                          for i, j in itertools.combinations_with_replacement(range(m), 2)]).T
        return np.hstack([one, u, cross])
    # |U|^2 of a mesh is a trigonometric polynomial in the phases, so cosines of single
    # phases and of pairwise sums and differences are the exact basis truncated at two modes.
    a = np.pi * u
    if kind == "trig1":
        return np.hstack([one, np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])
    if kind == "trig2":
        cols = [one, np.cos(a), np.sin(a)]
        for i, j in itertools.combinations(range(m), 2):
            for s in (1, -1):
                cols += [np.cos(a[:, [i]] + s * a[:, [j]]), np.sin(a[:, [i]] + s * a[:, [j]])]
        return np.hstack(cols)
    raise ValueError(kind)


def ridge(F, Y, alpha: float):
    """Standardised ridge with an unpenalised intercept, returned as a predictor."""
    mu, sd = F.mean(0), F.std(0) + 1e-9
    sd[0], mu[0] = 1.0, 0.0
    Z = (F - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    A[0, 0] -= alpha
    W = np.linalg.solve(A, Z.T @ Y)
    return lambda G: ((G - mu) / sd) @ W


def fit_ridge(u, T, itr, iva, kind: str, base=None, seed: int = 0):
    """One family, one split. Alpha is chosen on an inner split of the TRAINING states only,
    so the held-out number is never used to choose anything.

    `base` is a prediction to learn the residual of -- the twin, or a physics refit."""
    F = features(u, kind)
    Y = T.reshape(len(T), 16).copy()
    if base is not None:
        Y = Y - np.asarray(base, float).reshape(len(T), 16)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(itr)
    cut = max(4, int(0.75 * len(perm)))
    i1, i2 = perm[:cut], perm[cut:]
    add = (lambda P, idx: P) if base is None else \
        (lambda P, idx: P + np.asarray(base, float).reshape(len(T), 16)[idx])
    err = [rel(norm_cols(add(ridge(F[i1], Y[i1], a)(F[i2]), i2)), T[i2]) for a in ALPHAS]
    a = float(ALPHAS[int(np.argmin(err))])
    P = norm_cols(add(ridge(F[itr], Y[itr], a)(F[iva]), iva))
    return P, a


def fit_mlp(u, T, itr, iva, hidden=(64, 32), epochs: int = 3000, seed: int = 0,
            kind: str = "u", base=None):
    """The `learn.dpnn` architecture on this data: sixteen outputs, early-stopped on an
    inner split of the training states.

    `kind` is what tests the 6x6's feature finding -- V^2 alone, V alone, or both. That rig
    gained 0.10 R^2 by dropping V; this one does not (`_report`), which is what the physics
    predicts once no heater reaches even 1.5 pi: over a monotone segment a net learns the
    squaring either way and the basis stops mattering."""
    import torch
    torch.manual_seed(seed)
    X = torch.as_tensor({"u": u, "v": np.sqrt(u),
                         "uv": np.hstack([u, np.sqrt(u)])}[kind], dtype=torch.float32)
    R = T.reshape(len(T), 16)
    off = None if base is None else np.asarray(base, float).reshape(len(T), 16)
    Y = torch.as_tensor(R if off is None else R - off, dtype=torch.float32)
    add = (lambda P, idx: P) if off is None else (lambda P, idx: P + off[idx])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(itr)
    cut = max(4, int(0.75 * len(perm)))
    i1, i2 = perm[:cut], perm[cut:]
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 16))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state = np.inf, None
    for ep in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(X[i1]), Y[i1]).backward()
        opt.step()
        if ep % 25 == 0:
            with torch.no_grad():
                e = rel(norm_cols(add(net(X[i2]).numpy(), i2)), T[i2])
            if e < best:
                best, best_state = e, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        return norm_cols(add(net(X[iva]).numpy(), iva))


def splits(u, n, kind: str = "region", seed: int = 0):
    """Held out by CONSTRUCTION. A scattered split measures interpolation between states
    that were measured; the surrogate's whole value is states nobody measured, so the
    default holds out a slab of phase space per channel."""
    if kind == "random":
        rng = np.random.default_rng(seed)
        for s in range(5):
            rng = np.random.default_rng(seed + s)
            iva = rng.choice(n, max(1, n // 4), replace=False)
            yield f"random s{s}", np.setdiff1d(np.arange(n), iva), iva
    elif kind == "region":
        for c in range(u.shape[1]):
            iva = np.flatnonzero(u[:, c] > 0.6)
            yield f"ch{c}>0.6", np.setdiff1d(np.arange(n), iva), iva
    elif kind == "corner":
        for s in range(5):
            w = np.random.default_rng(seed + 100 + s).normal(size=u.shape[1])
            p = u @ w
            iva = np.flatnonzero(p > np.quantile(p, 0.75))
            yield f"corner s{s}", np.setdiff1d(np.arange(n), iva), iva
    else:
        raise ValueError(kind)


def _resid(blocks, H, r, s, A):
    """`pic.matvec._pick_state`'s score, vectorised over candidate blocks: fit the
    brightness, undo the Sinkhorn diagonals, measure against the nonnegative target."""
    g = (blocks * A).sum((-2, -1)) / max(float((A * A).sum()), 1e-18)
    rec = np.einsum("i,nij,j->nij", 1 / r, blocks,
                    1 / s) / np.where(g > 0, g, 1.0)[:, None, None]
    return np.linalg.norm(rec - H, axis=(-2, -1)) / max(np.linalg.norm(H), 1e-18)


def hosting(u, T, itr, iva, kind: str = "trig2", trials: int = 200, rails=BEST_RAILS,
            seed: int = 0, P=None):
    """The number the whole question turns on: if the model picks the state, what does the
    chip actually host?

    A continuous optimisation through the surrogate cannot be checked without hardware, so
    this is its honest stand-in -- the model ranks states it has never seen and the residual
    is then read off the MEASURED transfer of whatever it picked. Optimising volts freely can
    only be worse conditioned than ranking a finite set, because a free optimiser is at
    liberty to walk to wherever the model is most wrong. If the model cannot rank measured
    states it has not seen, planning through it hosts a matrix nobody has seen -- exactly the
    failure `plan_from_table` exists to escape."""
    if P is None:
        P = fit_mlp(u, T, itr, iva) if kind == "mlp" else fit_ridge(u, T, itr, iva, kind)[0]
    out, inp = rails
    Bm = T[:, list(out), :][:, :, list(inp)]
    Pm = P[:, list(out), :][:, :, list(inp)]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        B = rng.normal(size=(2, 2))
        H = B + max(0.0, -float(B.min())) + 0.25 * float(np.abs(B).mean())
        A, r, s = sinkhorn(H)
        r_all, r_va, r_pred = (_resid(Bm, H, r, s, A), _resid(Bm[iva], H, r, s, A),
                               _resid(Pm, H, r, s, A))
        k = int(np.argmin(r_pred))
        rows.append((r_all.min(), r_va.min(), r_va[k], r_pred[k]))
    a = np.array(rows, float)
    return {"table_all": float(np.median(a[:, 0])),
            "oracle_heldout": float(np.median(a[:, 1])),
            "model_pick": float(np.median(a[:, 2])),
            "model_believes": float(np.median(a[:, 3])),
            "n_heldout": int(len(iva))}


def repeatability(paths=None) -> list[tuple]:
    """The same heater states in two captures. This is the ceiling on any surrogate: a
    model fitted on one capture is being asked to predict a chip that does not reproduce
    itself better than this."""
    paths = paths or [FRESH, "pic_data/quarantine_not25C/raw_transfers_fresh.json",
                      "pic_data/quarantine_not25C/raw_transfers_big.json",
                      "pic_data/quarantine_not25C/raw_transfers_20C.json"]
    trs = [("/".join(Path(p).parts[-2:]), load(p)) for p in paths if Path(p).exists()]
    out = []
    for (na, a), (nb, b) in itertools.combinations(trs, 2):
        i, j = [], []
        for k, v in enumerate(a.volts):
            d = np.abs(b.volts - v).max(1)
            m = int(np.argmin(d))
            if d[m] < 1e-6:
                i.append(k)
                j.append(m)
        if len(i) >= 16:
            out.append((na, nb, len(i), rel(a.T[i], b.T[j])))
    return out


FAMILIES = ("u", "v", "uv", "u2", "trig1", "trig2")


def torch_mlp(u, T, idx, hidden=(64, 32), epochs: int = 3000, seed: int = 0):
    """The same net as `fit_mlp`, returned as a callable on arbitrary new `u` -- which is
    what a planner needs and a per-split predictor is not."""
    import torch
    n = len(T)
    X = torch.as_tensor(u, dtype=torch.float32)
    Y = torch.as_tensor(T.reshape(n, 16), dtype=torch.float32)
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(idx)
    cut = max(4, int(0.8 * len(perm)))
    i1, i2 = perm[:cut], perm[cut:]
    layers, d = [], u.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 16))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, state = np.inf, None
    for ep in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(X[i1]), Y[i1]).backward()
        opt.step()
        if ep % 25 == 0:
            with torch.no_grad():
                e = float(torch.nn.functional.mse_loss(net(X[i2]), Y[i2]))
            if e < best:
                best, state = e, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)

    def predict_u(uu):
        with torch.no_grad():
            P = net(torch.as_tensor(np.atleast_2d(uu), dtype=torch.float32)).reshape(-1, 4, 4)
            P = torch.clamp(P, min=0.0)
            return (P / P.sum(1, keepdim=True).clamp(min=1e-9)).numpy()
    return predict_u


def plan_ab(path=FRESH, trials: int = 60, rails=BEST_RAILS, seed: int = 0,
            starts: int = 96, keep: int = 8):
    """Plan volts CONTINUOUSLY through one surrogate, score the plan under an independent
    one. This is the experiment that answers the question, and it needs no hardware.

    Two nets fitted on disjoint halves of the states are both legitimate models of the same
    chip, so their disagreement at a point neither was told about is an honest estimate of
    what either one is worth there. Planning is free to walk to wherever the model is most
    wrong, which is why the disagreement at the OPTIMISED point is larger than the
    disagreement at random states -- and why a residual the planner reports about itself is
    not evidence about the chip. `plan_matvec` through the twin reported 0.0001 and hosted a
    matrix 1.1 out; this reproduces that failure with a fitted model instead of a nominal
    one.

    `keep` local optima are carried, not one, because a shortlist is the honest version of
    the idea: if the plan cannot be trusted, measuring a few of its proposals still might be
    cheaper than a capture. B judges each of them. Both halves are half-size models, so B's
    verdict is pessimistic by roughly the difference between a 78-state and a 155-state fit.
    """
    from scipy.optimize import minimize
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    out, inp = rails
    perm = np.random.default_rng(seed).permutation(n)
    A = torch_mlp(u, T, perm[:n // 2], seed=seed + 1)
    B = torch_mlp(u, T, perm[n // 2:], seed=seed + 2)
    meas = T[:, list(out), :][:, :, list(inp)]
    blk = lambda f, z: f(np.clip(np.atleast_2d(z), 0, 1))[:, list(out), :][:, :, list(inp)]
    rng = np.random.default_rng(seed + 7)
    rows = []
    for _ in range(trials):
        Bt = rng.normal(size=(2, 2))
        H = Bt + max(0.0, -float(Bt.min())) + 0.25 * float(np.abs(Bt).mean())
        Aq, r, s = sinkhorn(H)
        cost = lambda f, z: _resid(blk(f, z), H, r, s, Aq)
        z0 = rng.random((starts, u.shape[1]))
        cands = []
        for z in z0[np.argsort(cost(A, z0))[:keep]]:
            res = minimize(lambda q: float(cost(A, q)[0]), z, method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-5})
            cands.append((float(res.fun), np.clip(res.x, 0, 1)))
        cands.sort(key=lambda c: c[0])
        b = cost(B, np.array([c[1] for c in cands]))
        rows.append([float(_resid(meas, H, r, s, Aq).min()), cands[0][0], float(b[0]),
                     float(b[:3].min()), float(b[:5].min()), float(b.min())])
    m = np.median(np.array(rows, float), 0)
    return {"table": m[0], "believes": m[1], "independent": m[2], "best_of_3": m[3],
            "best_of_5": m[4], f"best_of_{keep}": m[5], "cross_rel": rel(A(u), B(u)),
            "n_states": n}


def learning_curve(path=FRESH, sizes=(20, 40, 60, 80, 100, 125), seeds: int = 4,
                   n_val: int = 30):
    """Held-out rel against training-set size, and the power law through it. What this is
    for is the question "would more states fix it" -- answered by extrapolating the fit
    against the reproducibility ceiling, not by opinion."""
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    acc = {N: [] for N in sizes}
    for s in range(seeds):
        perm = np.random.default_rng(s).permutation(n)
        iva, pool = perm[:n_val], perm[n_val:]
        for N in sizes:
            if N <= len(pool):
                acc[N].append(rel(fit_mlp(u, T, pool[:N], iva, seed=s), T[iva]))
    N = np.array([k for k in sizes if acc[k]])
    y = np.array([np.median(acc[k]) for k in N])
    slope, icept = np.polyfit(np.log(N), np.log(y), 1)
    need = lambda target: float(np.exp((np.log(target) - icept) / slope))
    return {"sizes": N.tolist(), "rel": y.tolist(), "slope": float(slope),
            "n_for_0.10": need(0.10), "n_for_0.05": need(0.05)}


def physics_report(path=FRESH, split_kind="region", steps: int = 1200, restarts: int = 16):
    """`learn.transfer_fit` refitted per split, alone and as the base an MLP corrects.

    The nominal twin is a bad base -- learning its residual scores WORSE than ignoring it
    (0.39 against 0.22) because a 1.12-relative offset is most of what there is to learn. A
    twin refitted on the training states is a good base and still not a better model than the
    net on its own: the physics carries the extrapolation and the net carries everything the
    mesh model has no term for, and on this data those add up to the same number."""
    import torch
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    calib0 = Calibration.load("pic_data/calib.json")
    V = torch.as_tensor(tr.volts, dtype=torch.float64)
    acc = {"physics": [], "physics+mlp": [], "mlp": []}
    print(f"  {'split':<12}{'nval':>5}{'physics':>10}{'physics+mlp':>13}{'mlp':>8}")
    for name, itr, iva in splits(u, n, split_kind):
        r = _fit_physics(tr, calib0, (itr, iva), steps, restarts)
        with torch.no_grad():
            base = r.model(V).numpy()[0]
        row = (rel(base[iva], T[iva]),
               rel(fit_mlp(u, T, itr, iva, base=base), T[iva]),
               rel(fit_mlp(u, T, itr, iva), T[iva]))
        for k, v in zip(acc, row):
            acc[k].append(v)
        print(f"  {name:<12}{len(iva):>5}{row[0]:>10.4f}{row[1]:>13.4f}{row[2]:>8.4f}")
    print(f"  {'MEDIAN':<12}{'':>5}" +
          "".join(f"{np.median(v):>{w}.4f}" for v, w in zip(acc.values(), (10, 13, 8))))
    return acc


def _fit_physics(tr, calib0, split, steps, restarts):
    from .transfer_fit import fit
    return fit(tr, ROLES_LAYOUT, calib0=calib0, split=split, steps=steps, restarts=restarts)


def basis_report(path=FRESH):
    """The 6x6's finding, retested on the family that could use it. That rig gained 0.10
    held-out R^2 by dropping the linear V term; here V^2 alone, V alone and both land within
    0.01 rel of each other on every split, which is smaller than the split-to-split spread."""
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    print(f"  {'split':<8}" + "".join(f"{k:>9}" for k in ("V^2", "V", "both")))
    for sk in ("region", "corner", "random"):
        acc = {k: [] for k in ("u", "v", "uv")}
        for _, itr, iva in splits(u, n, sk):
            for k in acc:
                acc[k].append(rel(fit_mlp(u, T, itr, iva, kind=k), T[iva]))
        print(f"  {sk:<8}" + "".join(f"{np.median(v):>9.4f}" for v in acc.values()))


def rank_report(path=FRESH, trials: int = 200, rails=BEST_RAILS, split_kind="random"):
    """How well the model ORDERS states it has never seen, which is a different and much
    easier question than how well it values them.

    It is the one thing the surrogate is good at, and the only use this data supports: the
    ranking is worth a shortlist, not a plan. Scoring by the model and then MEASURING its
    top few recovers most of what measuring the whole candidate set would have found, for a
    handful of reads."""
    from scipy.stats import spearmanr
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    out, inp = rails
    rho, top = [], {1: [], 3: [], 5: [], 10: []}
    oracle = []
    for _, itr, iva in splits(u, n, split_kind):
        P = fit_mlp(u, T, itr, iva)
        meas = T[:, list(out), :][:, :, list(inp)][iva]
        pred = P[:, list(out), :][:, :, list(inp)]
        rng = np.random.default_rng(0)
        for _ in range(trials):
            B = rng.normal(size=(2, 2))
            H = B + max(0.0, -float(B.min())) + 0.25 * float(np.abs(B).mean())
            A, r, s = sinkhorn(H)
            rt, rp = _resid(meas, H, r, s, A), _resid(pred, H, r, s, A)
            rho.append(spearmanr(rt, rp).statistic)
            order = np.argsort(rp)
            oracle.append(rt.min())
            for k in top:
                top[k].append(rt[order[:k]].min())
    print(f"  spearman(model, truth) over candidates: median {np.median(rho):.3f}")
    print(f"  {'measure the models top k':<26}{'hosted':>9}")
    for k, v in top.items():
        print(f"  {'k = ' + str(k):<26}{np.median(v):>9.4f}")
    print(f"  {'oracle over all candidates':<26}{np.median(oracle):>9.4f}")


def _report(path=FRESH, split_kind="region", with_mlp=True, with_twin_residual=True):
    tr, u, act = dataset(path)
    T, n = tr.T, len(tr)
    twin = predict(Calibration.load("pic_data/calib.json"), ROLES_LAYOUT, tr.volts)
    print(f"{path}  {n} states, {u.shape[1]} moving channels (DAC {[int(a) for a in act]})")
    print(f"  whole-table baselines:  mean transfer {rel(T.mean(0)[None], T):.4f}   "
          f"twin {rel(twin, T):.4f}")
    cols = list(FAMILIES) + (["mlp"] if with_mlp else []) + \
           (["twin+u2"] if with_twin_residual else [])
    print(f"\n  {'split':<12}{'nval':>5}{'mean':>8}" + "".join(f"{c:>9}" for c in cols))
    acc = {c: [] for c in cols}
    for name, itr, iva in splits(u, n, split_kind):
        row = f"  {name:<12}{len(iva):>5}{rel(T[itr].mean(0)[None], T[iva]):>8.4f}"
        for c in cols:
            if c == "mlp":
                P = fit_mlp(u, T, itr, iva)
            elif c == "twin+u2":
                P, _ = fit_ridge(u, T, itr, iva, "u2", base=twin)
            else:
                P, _ = fit_ridge(u, T, itr, iva, c)
            e = rel(P, T[iva])
            acc[c].append(e)
            row += f"{e:>9.4f}"
        print(row)
    print(f"  {'MEDIAN':<12}{'':>5}{'':>8}" +
          "".join(f"{np.median(acc[c]):>9.4f}" for c in cols))
    return acc


def _hosting_report(path=FRESH, split_kind="region", trials: int = 200):
    tr, u, _ = dataset(path)
    T, n = tr.T, len(tr)
    print(f"  {'split':<12}{'nval':>5}{'rel':>8}{'table(all)':>12}{'oracle(val)':>13}"
          f"{'model pick':>12}{'model thinks':>14}")
    rows = []
    for name, itr, iva in splits(u, n, split_kind):
        P = fit_mlp(u, T, itr, iva)
        h = hosting(u, T, itr, iva, trials=trials, P=P)
        e = rel(P, T[iva])
        rows.append([e, h["table_all"], h["oracle_heldout"], h["model_pick"],
                     h["model_believes"]])
        print(f"  {name:<12}{len(iva):>5}{e:>8.4f}{h['table_all']:>12.4f}"
              f"{h['oracle_heldout']:>13.4f}{h['model_pick']:>12.4f}"
              f"{h['model_believes']:>14.4f}")
    m = np.median(np.array(rows), 0)
    print(f"  {'MEDIAN':<12}{'':>5}" +
          "".join(f"{v:>{w}.4f}" for v, w in zip(m, (8, 12, 13, 12, 14))))


if __name__ == "__main__":
    import sys
    kinds = ("region", "corner", "random")
    if "--hosting" in sys.argv:
        for k in kinds:
            print(f"\n=== hosting, split by {k} ===")
            _hosting_report(split_kind=k)
    elif "--plan" in sys.argv:
        r = plan_ab()
        print(f"continuous planning, {r['n_states']} states, two nets on disjoint halves")
        print(f"  the two nets disagree by rel {r['cross_rel']:.4f} at the measured states")
        for k, v in r.items():
            if k not in ("cross_rel", "n_states"):
                print(f"  {k:<16}{v:.4f}")
    elif "--rank" in sys.argv:
        rank_report()
    elif "--basis" in sys.argv:
        basis_report()
    elif "--physics" in sys.argv:
        physics_report()
    elif "--curve" in sys.argv:
        c = learning_curve()
        for N, e in zip(c["sizes"], c["rel"]):
            print(f"  {N:>4} states  rel {e:.4f}")
        print(f"  rel ~ N^{c['slope']:.2f}:  {c['n_for_0.10']:.0f} states for 0.10, "
              f"{c['n_for_0.05']:.0f} for 0.05")
    else:
        for na, nb, k, e in repeatability():
            print(f"  repeatability  {na} vs {nb}  ({k} shared)  rel {e:.4f}")
        for k in kinds:
            print(f"\n=== split by {k} ===")
            _report(split_kind=k)
