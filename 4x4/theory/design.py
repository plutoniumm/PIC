"""How many transfer measurements a table needs, which ones, and how to keep them fresh.

`pic.matvec.plan_from_table` hosts a target by picking the closest of ~100 MEASURED heater
states, so the table is the whole instrument: its size is the resolution, and its age is
the error. Capturing 100 states costs 12-15 minutes, which is the same order as the drift
time constant, so the table is stale before it is finished. This module measures the three
ways out of that -- fewer states, better-chosen states, and a correction that transports an
old table onto today's chip -- against the session dumps in `pic_data/sessions/`.

Three results, and the third is the one that matters.

**Size buys resolution slowly.** Picking one of n measured blocks is nearest-neighbour
quantisation on a 3-dimensional manifold (a 2x2 block has four entries and the hosting
metric fits its scale), so the residual falls like n^(-1/3) and nothing flattens it. It
stops falling only when it reaches the repeatability floor, which on this bench happens
near n = 30.

That quantisation residual is real but it is NOT what limits the bench, and the way out is
not to leave the codebook. Hosting a target as a weighted sum of K measured states spans the
block's four dimensions at K = 4 and drives the quantisation residual to zero, and on the
chip it bought nothing: measured matrix 0.1875 at K = 1 against 0.1876 at K = 4, with vectors
1.7x *worse* (0.0419 -> 0.0701) because `sum |c|` runs to about 6 and multiplies the read
noise by it. Simulation had predicted 0.073 against 0.657 -- a case that rested entirely on
single-state hosting being catastrophic, which on hardware it is not. So the measured error
is not codebook residual, and the levers that work are the ones below: more states, better
chosen, and above all fresher.

**Choice buys a factor of two to two and a half in n, and it has to be somebody else's
choice.** Greedy max-min distance between the measured blocks, taken as directions, lands
reliably where the best of forty random draws of the same size lands -- n = 15 designed
matches n = 35 random and n = 30 matches n = 60. The design does not need today's
measurements and must not be taken from them: computed off a five-hour-old table it beats
random at every size, and computed off the very table it will then plan with it is *worse
than random* below n = 20 (0.201 against 0.193 at n = 10). Max-min seeks extremes, and a
table's extremes are its noisiest states; a design taken from a different capture cannot
chase the plan table's own noise. The alternatives are measured off the same prior and
lose: k-means owns n = 5 (0.186), max-min everything from n = 10 up, and D-optimality and
greedy SVD coverage never lead anywhere. Volume is the wrong objective -- the picked state
is a nearest neighbour, so what matters is the largest gap, not the spanned determinant.

**Drift is 15 parameters, and five fresh states measure them.** State for state, an old
table differs from a new one by a fixed 4x4 mixing on the OUTPUT side -- `T_now =
colnorm(M T_then)`. Fitting M on five re-measured states takes the disagreement from 0.127
to 0.016 per entry, under the 0.018 repeat floor, and takes hosting from 0.174 to 0.073
against 0.079 for capturing all hundred states again. Twenty seconds replaces twelve
minutes. Three states reach nearly the same median with a much longer tail (p90 0.115 against
0.079 at five; at two the worst of sixty draws is 0.52, because four columns cannot
determine fifteen parameters at all), so five is the number to run and the marginal state
costs four seconds.

Why the output side and not the input: `Calibration.to_transfer` divides each column by its
own sum, so launch power, fibre coupling and switch loss are already gone and an input-side
model has nothing left to fit (0.127 -> 0.089, against 0.014 for the output side). What
survives normalisation is per-detector response and detector-to-detector leakage, which is
exactly a left multiplication.

Two things this module refuses.

**Interpolation does not work here** and the number is not close. The best local
predictor of a held-out block -- affine regression on its forty nearest states -- lands at
0.072 per entry, four times the 0.018 repeat floor and only 40 percent better than
predicting the table's overall average (0.120) with no neighbours at all. A hundred states
scattered in an 8-dimensional heater cube sit a median 0.52 apart on a side of 1, half a
fringe on the wide channels: the map is smooth, the sampling is nowhere near dense enough
to exploit it. A coarse grid plus interpolation is not a scaling law available on this
chip, and it would need roughly 2^8 states before it became one.

**Do not correct a table that is not stale, and never with stale tie-points.** Fitting 15
parameters on a table already at the floor makes it worse, because the fit has nothing to
find and copies the tie-points' noise instead. `should_correct` is that gate and it is
derived, not tuned: a mixing fitted on m states and applied to a state that was not one of
them is out-of-sample least-squares prediction, whose variance inflates by `p/(n - p - 1)`
with `p = 15` free parameters (colnorm eats M's overall scale) and `n = 12m` observations
(four columns per state, each normalised to sum to 1). So the corrected table floors at
`sqrt(1 + 15/(12m - 16))` times the repeatability -- 1.70 at m = 2, 1.16 at m = 5, 1.03 at
m = 20, and *infinite* below m = 2, where 12 observations cannot determine 15 parameters at
all. `measured_floor` puts that against a synthetic table with no drift planted and gets
1.81/1.30/1.18/1.07 at m = 2/3/5/10; `_selftest` asserts the agreement, so the constant
cannot drift away from the derivation unnoticed.

The sharper failure is tie-points that have aged: the correction transports the table onto
the state the TIE-POINTS were in, so ties taken 74 minutes before the run left the table
worse than not correcting at all (0.131 -> 0.195).
Measure the ties immediately before use -- which costs twenty seconds, which is the point.

    python -m theory.design             # the whole report against the bench sessions
    python -m theory.design protocol    # and the n x m cost grid the recommendation is from
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calib import Calibration
from .intensity_matvec import BEST_RAILS, sinkhorn

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "pic_data/sessions"

# `pic.matvec`'s decomposition constants. Reproduced rather than imported: `theory` must not
# import `pic` (that is the circular import the layering exists to prevent), and a design
# study that scored a different decomposition from the planner would not be about the
# planner.
# 0.25 is a measured optimum, not a round number. Swept against matrix error on 120 random
# targets with the table stale by its measured 0.0157, as margin: host / matrix / relative
# amplification -- 0.05: .073/.110/1.89, 0.10: .059/.090/1.45, 0.25: .051/.085/1.11,
# 0.60: .069/.127/1.00, 2.00: .080/.241/0.99. A larger shift makes the hosted matrix more
# uniform, so Sinkhorn's diagonals stop being extreme and a stale table is amplified less; but
# the target then sits under a constant the recovery subtracts back off, and past 0.25 that
# costs more signal than the conditioning buys. Clean interior minimum -- do not tune by eye.
SHIFT_MARGIN, SINK_FLOOR = 0.25, 1e-3

# One table entry costs a heater write, the thermal settle, and four switch moves with a
# read each. The settle dominates and is paid once per state, not once per port.
STATE_COST_S = 4.0


@dataclass
class Table:
    """A session's measured transfer table: `T[n]` is |U|^2 at heater state `volts[n]`."""

    volts: np.ndarray  # (n, N_HEATERS)
    T: np.ndarray  # (n, 4, 4), columns summing to 1
    vmax: np.ndarray  # (N_HEATERS,) the ceiling each channel was swept to
    trainable: np.ndarray  # which channels moved
    label: str = ""

    def __len__(self):
        return len(self.T)

    def block(self, rails=BEST_RAILS) -> np.ndarray:
        return self.T[:, list(rails[0]), :][:, :, list(rails[1])]

    @property
    def phase(self) -> np.ndarray:
        """Heater state in the coordinate the physics is linear in: phase goes as V^2, so a
        fraction of full phase is (V/vmax)^2 and the cube is what the sampling lives in."""
        return (self.volts[:, self.trainable] / self.vmax[self.trainable]) ** 2

    def subset(self, idx) -> "Table":
        idx = np.asarray(idx, int)
        return Table(self.volts[idx], self.T[idx], self.vmax, self.trainable, self.label)


def load_table(path, calib: Calibration | None = None) -> Table:
    """A `raw_transfers*.json` dump -> the normalised table.

    The JSON is re-read here rather than through `pic.normalise.load_session` for the
    layering reason in the module note; the two must stay identical, and the shared piece --
    the port-major transpose and `Calibration.to_transfer` -- is the part that matters."""
    path = Path(path)
    d = json.loads(path.read_text())
    calib = calib or Calibration.load_or_nominal(ROOT / "pic_data/calib.json")
    dark = np.asarray(d["dark_switch_off"], float)
    return Table(
        np.asarray([s["volts"] for s in d["states"]], float),
        np.stack([calib.to_transfer(np.asarray(s["raw"], float).T, dark) for s in d["states"]]),
        np.asarray(d["vmax"], float),
        np.asarray(d["trainable"], int),
        path.stem,
    )


def align(a: Table, b: Table) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs where two tables hold the SAME heater state. Every drift number here is
    state for state, because a table difference taken any other way is mostly the states."""
    m = {tuple(np.round(v, 6)): i for i, v in enumerate(a.volts)}
    ij = [(m[k], j) for j, v in enumerate(b.volts) if (k := tuple(np.round(v, 6))) in m]
    return np.array([i for i, _ in ij], int), np.array([j for _, j in ij], int)


def colnorm(T) -> np.ndarray:
    """The one normalisation the mesh's unitarity licenses -- see `Calibration.to_transfer`,
    which this reproduces for a stack so a corrected table passes through the same map the
    measured one did."""
    T = np.asarray(T, float)
    s = T.sum(-2, keepdims=True)
    return np.divide(T, s, out=np.zeros_like(T), where=s > 1e-12)


def targets(n: int, k: int = 2, seed: int = 0) -> list:
    return list(np.random.default_rng(seed).normal(size=(n, k, k)))


def shift(B):
    """`pic.matvec._shift`: B -> (c, sinkhorn(B + c)), the plan's own decomposition."""
    B = np.asarray(B, float)
    c = max(0.0, -float(B.min())) + SHIFT_MARGIN * float(np.abs(B).mean())
    return c, sinkhorn(B + c, floor=SINK_FLOOR)


def prepare(tgts) -> list:
    return [shift(B) for B in tgts]


def host(H, sink, blocks_plan, blocks_truth=None) -> tuple[int, float, float]:
    """`pic.matvec._pick_state`, plus what the chip then does.

    The planner scores every table block on the RECOVERED matrix -- Sinkhorn diagonals and
    hosted scale undone -- and keeps the best. `blocks_truth` is the same states re-measured
    (a fresher table, or a different one); recovering through the plan's own `scale`, `r`
    and `s` and comparing to H is exactly `pic.matvec.measured_matrix` against the target,
    which is the only number a bench run can check."""
    A, r, s = sink
    scale = (blocks_plan * A).sum((-2, -1)) / max(float((A * A).sum()), 1e-18)
    rec = (
        np.einsum("i,nij,j->nij", 1 / r, blocks_plan, 1 / s)
        / np.where(scale > 0, scale, 1.0)[:, None, None]
    )
    err = np.linalg.norm(rec - H, axis=(-2, -1)) / max(np.linalg.norm(H), 1e-18)
    k = int(np.argmin(np.where(scale > 0, err, np.inf)))
    if blocks_truth is None:
        return k, float(err[k]), float(err[k])
    got = np.diag(1 / r) @ (blocks_truth[k] / (scale[k] if scale[k] > 0 else 1.0)) @ np.diag(1 / s)
    return k, float(err[k]), float(np.linalg.norm(got - H) / max(np.linalg.norm(H), 1e-18))


def score(prepped, tgts, blocks_plan, blocks_truth=None) -> float:
    """Median realised hosting error over a batch of random signed targets."""
    return float(
        np.median(
            [host(B + c, sk, blocks_plan, blocks_truth)[2] for (c, sk), B in zip(prepped, tgts)]
        )
    )


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _farthest(Y, n: int) -> np.ndarray:
    """Greedy max-min: open at the point furthest from the mean, then always take the point
    furthest from everything already taken."""
    chosen = [int(np.argmax(np.linalg.norm(Y - Y.mean(0), axis=1)))]
    d = np.linalg.norm(Y - Y[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(Y - Y[k], axis=1))
        d[chosen] = -np.inf
    return np.array(chosen)


def select(kind: str, X, n: int, rng=None) -> np.ndarray:
    """n row indices of X to keep.

    X is the candidate set in whatever space the design lives in -- measured blocks
    (`Table.block(...).reshape(len, -1)`) when a previous table exists, `Table.phase` when
    none does. Block designs work on DIRECTIONS: the hosting metric fits each block's scale,
    so two states differing only in brightness are one state as far as coverage goes."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    if kind == "random":
        return rng.choice(len(X), n, replace=False)
    if kind in ("maxmin", "spread"):  # `spread` is the same rule without the
        return _farthest(_unit(X) if kind == "maxmin" else X, n)  # direction normalisation
    if kind == "svd":  # cover the singular directions, not the ball
        Y, chosen = _unit(X), []
        R = Y.copy()
        for _ in range(n):
            nr = np.linalg.norm(R, axis=1)
            nr[chosen] = -np.inf
            k = int(np.argmax(nr))
            chosen.append(k)
            v = R[k] / max(np.linalg.norm(R[k]), 1e-12)
            R = R - np.outer(R @ v, v)
            if len(chosen) >= Y.shape[1]:  # the basis is full; restart on what it misses
                R = Y - Y @ np.linalg.pinv(Y[chosen]) @ Y[chosen]
                R[chosen] = 0.0
        return np.array(chosen)
    if kind == "kmeans":
        Y = _unit(X)
        c = Y[rng.choice(len(Y), n, replace=False)].copy()
        for _ in range(50):
            lab = np.argmin(((Y[:, None] - c[None]) ** 2).sum(-1), axis=1)
            for j in range(n):
                if (lab == j).any():
                    c[j] = Y[lab == j].mean(0)
        out, used = [], set()
        for j in range(n):
            for k in np.argsort(((Y - c[j]) ** 2).sum(1)):
                if int(k) not in used:
                    used.add(int(k))
                    out.append(int(k))
                    break
        return np.array(out)
    if kind == "dopt":
        Y = _unit(X)
        G, chosen = 1e-6 * np.eye(Y.shape[1]), []
        for _ in range(n):
            gain = np.einsum("ij,jk,ik->i", Y, np.linalg.inv(G), Y)
            gain[chosen] = -np.inf
            k = int(np.argmax(gain))
            chosen.append(k)
            G = G + np.outer(Y[k], Y[k])
        return np.array(chosen)
    raise ValueError(f"kind={kind!r}")


KINDS = ("random", "maxmin", "kmeans", "dopt", "svd")


def size_curve(
    design: Table,
    plan: Table,
    truth: Table,
    ns,
    tgts,
    prepped,
    kinds=KINDS,
    reps: int = 12,
    rails=BEST_RAILS,
    seed: int = 0,
) -> dict:
    """Hosting error against table size, for each selection rule.

    `design` is the table the states are CHOSEN from -- normally an old one, since choosing
    from today's requires having measured it. `plan` is what the planner reads and `truth`
    is what the chip then does; passing two different sessions is what makes the number
    honest, since a plan scored against its own measurement shares that measurement's
    noise."""
    Xd = design.block(rails).reshape(len(design), -1)
    Bp, Bt = plan.block(rails), truth.block(rails)
    out = {}
    for kind in kinds:
        row = []
        for n in ns:
            e = []
            for rep in range(reps if kind in ("random", "kmeans") else 1):
                idx = select(kind, Xd, n, np.random.default_rng(1000 * seed + rep))
                e.append(score(prepped, tgts, Bp[idx], Bt[idx]))
            row.append(float(np.median(e)))
        out[kind] = row
    return out


def apply_mixing(M, T) -> np.ndarray:
    """T -> colnorm(M T): the drift model, applied through the measurement's own map."""
    return colnorm(
        np.clip(
            np.einsum("ab,nbk->nak", np.asarray(M, float).reshape(4, 4), np.asarray(T, float)),
            0.0,
            None,
        )
    )


def fit_mixing(Ts, Tf, both: bool = False):
    """The 4x4 output mixing that carries a stale table onto a fresh one.

    Least squares from the identity, not a null-space solve. Writing `(I - f 1^T) M s = 0`
    makes the model linear in M and says exactly how much data it needs -- each measured
    column is 3 independent constraints on 15 free parameters, so five columns, and a state
    is four of them -- but the algebraic residual it minimises is the true one multiplied by
    the predicted column sum, and reweighting that back was worse than the direct fit at
    every tie-point count tried (0.043 against 0.020 at three states). The identifiability
    argument is worth keeping; the estimator is not.

    `both` adds an input-side factor. It is measured and it is not worth it -- the column
    normalisation has already removed what it would model -- so it exists to be refused."""
    from scipy.optimize import least_squares

    Ts, Tf = np.asarray(Ts, float), np.asarray(Tf, float)
    if not both:
        r = least_squares(
            lambda p: (apply_mixing(p, Ts) - Tf).ravel(),
            np.eye(4).ravel(),
            method="lm",
            max_nfev=8000,
        )
        return r.x.reshape(4, 4)

    def resid(p):
        MT = np.einsum("ab,nbc,ck->nak", p[:16].reshape(4, 4), Ts, p[16:].reshape(4, 4))
        return (colnorm(np.clip(MT, 0.0, None)) - Tf).ravel()

    r = least_squares(
        resid,
        np.concatenate([fit_mixing(Ts, Tf).ravel(), np.eye(4).ravel()]),
        method="lm",
        max_nfev=20000,
    )
    return r.x[:16].reshape(4, 4), r.x[16:].reshape(4, 4)


def apply_input(N, T) -> np.ndarray:
    return colnorm(
        np.clip(
            np.einsum("nab,bk->nak", np.asarray(T, float), np.asarray(N, float).reshape(4, 4)),
            0.0,
            None,
        )
    )


def fit_input(Ts, Tf) -> np.ndarray:
    """The input-side mirror of `fit_mixing`, kept so the comparison is like for like.

    It has almost nothing to fit and that is the point: `to_transfer` divides each column by
    its own sum, so per-port launch and coupling are gone before this sees the data and only
    port-to-port leakage is left."""
    from scipy.optimize import least_squares

    r = least_squares(
        lambda p: (apply_input(p, Ts) - Tf).ravel(), np.eye(4).ravel(), method="lm", max_nfev=8000
    )
    return r.x.reshape(4, 4)


def fit_pd_gain(Ts, Tf) -> np.ndarray:
    """The diagonal special case: per-detector response only, no leakage. 4 parameters."""
    from scipy.optimize import least_squares

    r = least_squares(
        lambda g: (
            colnorm(np.clip(g, 1e-6, None)[None, :, None] * np.asarray(Ts, float)) - Tf
        ).ravel(),
        np.ones(4),
        method="lm",
        max_nfev=4000,
    )
    return r.x


DRIFT_MODELS = {
    "none": (lambda Ts, Tf: None, lambda p, T: colnorm(T)),
    "pd gain": (fit_pd_gain, lambda g, T: colnorm(np.clip(g, 1e-6, None)[None, :, None] * T)),
    "out mix": (fit_mixing, apply_mixing),
    "in mix": (lambda Ts, Tf: fit_input(Ts, Tf), apply_input),
    "both": (
        lambda Ts, Tf: fit_mixing(Ts, Tf, both=True),
        lambda p, T: colnorm(np.clip(np.einsum("ab,nbc,ck->nak", p[0], T, p[1]), 0, None)),
    ),
}


def correct(stale: Table, tie_idx, fresh_T, model: str = "out mix") -> Table:
    """Transport a whole stale table onto today's chip from a handful of re-measured states.

    `fresh_T` are the tie states as measured NOW; only rows `tie_idx` of `stale` are used to
    fit, and the correction is then applied to all of them. The tie states themselves are
    returned corrected rather than replaced, so the caller can hold out whichever states it
    means to score on."""
    fit, ap = DRIFT_MODELS[model]
    return Table(
        stale.volts,
        ap(fit(stale.T[np.asarray(tie_idx, int)], np.asarray(fresh_T, float)), stale.T),
        stale.vmax,
        stale.trainable,
        stale.label + "+corr",
    )


# Applying a mixing fitted on m states to a state that was NOT one of them is out-of-sample
# prediction from a least-squares fit, so what it costs is set by the parameter count against
# the observation count and by nothing else. `apply_mixing` is colnorm(M T), where M's overall
# scale cancels -- 15 free parameters, not 16 -- and each measured state contributes four
# columns whose four entries are normalised to sum to 1, so a state is 12 independent
# observations and not 16. The expected prediction variance of a linear fit at a fresh design
# point is sigma^2 (1 + p/(n - p - 1)), which gives the floor below and, as a bonus, the right
# behaviour at small m: it diverges exactly where 12m stops being able to determine 15
# parameters.
#
# `measured_floor` checks that against a synthetic table with NO drift at all -- the case the
# gate has to get right -- and `_selftest` asserts the agreement: 1.81/1.30/1.18/1.07 at
# m = 2/3/5/10 against a predicted 1.70/1.32/1.16/1.07. The constant here used to be an
# empirical sqrt(1 + 4.6/m) fitted to a single run; it happens to land on the truth at m = 2
# and is 20-35 percent too conservative from m = 5 up, which is the whole range the bench
# uses -- so it refused corrections that would have helped.
MIX_PARAMS = 15  # a 4x4 output mixing, less the overall scale colnorm removes
OBS_PER_STATE = 12  # four columns of four entries, each normalised to sum to 1


def correction_floor(floor: float, m: int) -> float:
    """The best a 15-parameter correction fitted on m states can do, given `floor`.

    Infinite below m = 2, because 12 observations cannot determine 15 parameters: the fit is
    exact on the ties and carries the whole of their read noise into every other state.
    Measured at m = 1 that is 4.1x the floor, which is the documented "one tie-point makes
    the table WORSE" with a number on it."""
    dof = OBS_PER_STATE * int(m) - MIX_PARAMS - 1
    return float("inf") if dof <= 0 else float(floor) * np.sqrt(1.0 + MIX_PARAMS / dof)


def should_correct(stale_err: float, floor: float, m: int) -> bool:
    """Is a stale table far enough out to be worth a 15-parameter correction on m states?"""
    return stale_err > correction_floor(floor, m)


def measured_floor(
    ms=(2, 3, 5, 10), n: int = 60, sigma: float = 0.02, reps: int = 24, seed: int = 0
) -> dict:
    """`correction_floor` against measurement, on a table that has not drifted at all.

    No drift is planted, so there is nothing for the mixing to find and everything it does
    to a held-out state is noise it imported from the tie-points -- which is exactly the
    regime `should_correct` has to decide, and the one where a gate set too low does damage.
    The floor is measured the way the bench measures its own, as the disagreement between two
    independent reads of the same true table, so the returned ratios are dimensionless and do
    not depend on `sigma`."""
    rng = np.random.default_rng(seed)
    U = [np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0] for _ in range(n)]
    T = colnorm(np.stack([np.abs(u) ** 2 for u in U]))
    read = lambda: colnorm(np.clip(T + rng.normal(0, sigma, T.shape), 1e-9, None))
    floor = float(np.mean([np.abs(read() - read()).mean() for _ in range(8)]))
    out = {}
    for m in ms:
        e = []
        for _ in range(reps):
            stale, tie, truth = read(), read(), read()
            idx = rng.choice(n, m, replace=False)
            hold = np.setdiff1d(np.arange(n), idx)
            C = apply_mixing(fit_mixing(stale[idx], tie[idx]), stale)
            e.append(float(np.abs(C[hold] - truth[hold]).mean()))
        out[m] = float(np.mean(e)) / floor
    return {"floor": floor, "ratio": out, "reps": reps, "n": n}


def tie_curve(
    stale: Table,
    tie: Table,
    truth: Table,
    ms,
    tgts,
    prepped,
    model: str = "out mix",
    reps: int = 9,
    rails=BEST_RAILS,
    design: Table | None = None,
) -> dict:
    """Hosting error against the number of freshly re-measured states.

    Three tables and not two, deliberately. `tie` is where the fresh measurements come from
    and `truth` is what the chip does when the plan is run; if they were the same capture
    the correction would be scored against the noise it was fitted on. Tie states are
    excluded from the candidate pool for the same reason."""
    ia, ib = align(stale, tie)
    ja, jb = align(stale, truth)
    keep_all = np.intersect1d(ia, ja)
    Bt = truth.block(rails)
    truth_of = {a: b for a, b in zip(ja, jb)}
    rows = {}
    for m in ms:
        e, d = [], []
        for rep in range(reps):
            rng = np.random.default_rng(rep)
            pick = rng.choice(len(ia), m, replace=False)
            tie_s, tie_f = ia[pick], ib[pick]
            keep = np.setdiff1d(keep_all, tie_s)
            C = correct(stale, tie_s, tie.T[tie_f], model)
            if design is not None:  # only score states we would have kept
                Xd = design.block(rails).reshape(len(design), -1)
                keep = np.intersect1d(keep, select("maxmin", Xd, len(keep), rng))
            bt = np.stack([Bt[truth_of[i]] for i in keep])
            tt = np.stack([truth.T[truth_of[i]] for i in keep])
            e.append(score(prepped, tgts, C.block(rails)[keep], bt))
            d.append(float(np.abs(C.T[keep] - tt).mean()))
        # p90 and not just the median: at two tie-points the model is under-determined and
        # the median hides it, so the tail is the number that chooses m
        rows[m] = (float(np.median(e)), float(np.percentile(e, 90)), float(np.median(d)))
    return rows


def neighbour_stats(P) -> dict:
    D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e9
    return {
        "median_nn": float(np.median(D.min(1))),
        "min_nn": float(D.min()),
        "side": float(np.sqrt(P.shape[1])),
        "D": D,
    }


def interp_curve(table: Table, ks=(3, 5, 8, 12, 20, 40), orders=(0, 1)) -> dict:
    """Leave-one-out prediction of a held-out block from its k nearest states in V^2.

    Order 0 averages the neighbours, order 1 fits an affine model in phase and reads off the
    constant. If the map were smooth at this sampling density order 1 would beat order 0 and
    both would approach the repeatability floor. Neither happens, which is the finding."""
    P, Y = table.phase, table.T.reshape(len(table), -1)
    D = neighbour_stats(P)["D"]
    out = {}
    for k in ks:
        row = []
        for order in orders:
            e = []
            for i in range(len(P)):
                nb = np.argsort(D[i])[:k]
                X = P[nb] - P[i]
                A = np.ones((k, 1)) if order == 0 else np.hstack([np.ones((k, 1)), X])
                if A.shape[1] > k - 1:
                    continue
                w, *_ = np.linalg.lstsq(A, Y[nb], rcond=None)
                e.append(np.abs(w[0] - Y[i]).mean())
            row.append(float(np.mean(e)) if e else float("nan"))
        out[k] = row
    # the number every row above has to beat: predict the whole table's average and never
    # measure a neighbour at all
    out["mean"] = [
        float(
            np.mean(
                [np.abs(Y[np.arange(len(Y)) != i].mean(0) - Y[i]).mean() for i in range(len(Y))]
            )
        )
    ] * len(orders)
    return out


def cost_s(n_states: int, n_ties: int = 0) -> float:
    return STATE_COST_S * (n_states + n_ties)


BENCH = {
    "big": "2026-08-27/raw_transfers_big.json",
    "20C": "2026-08-27/raw_transfers_20C.json",
    "25C": "2026-08-27/raw_transfers_25C.json",
    "fresh": "2026-08-27/raw_transfers_fresh.json",
}


def load_bench(root=SESSIONS) -> dict:
    """The session dumps that are present. They are untracked, so absence is normal."""
    out = {}
    for k, rel in BENCH.items():
        p = Path(root) / rel
        if p.exists():
            out[k] = load_table(p)
    return out


def report(tabs=None, n_targets: int = 200, draws_n: int = 40, tie_reps: int = 24) -> dict:
    tabs = tabs or load_bench()
    tgts = targets(n_targets)
    prep = prepare(tgts)
    names = [k for k in ("big", "20C", "25C", "fresh") if k in tabs]
    res = {"names": names}

    res["dT"] = {
        (a, b): float(
            np.abs(
                tabs[a].T[align(tabs[a], tabs[b])[0]] - tabs[b].T[align(tabs[a], tabs[b])[1]]
            ).mean()
        )
        for a in names
        for b in names
    }
    res["host"] = {
        (a, b): score(prep, tgts, tabs[a].block(), tabs[b].block()) for a in names for b in names
    }
    if {"big", "20C"} <= set(names):
        res["floor_dT"] = res["dT"][("big", "20C")]
        res["floor_host"] = res["host"][("big", "20C")]
        ns = (5, 10, 15, 20, 30, 40, 60, 100)
        res["ns"] = ns
        Bp, Bt = tabs["big"].block(), tabs["20C"].block()

        def at(idx):
            return score(prep, tgts, Bp[idx], Bt[idx])

        # random is a DISTRIBUTION and a designed pick has to be read against its spread,
        # not against its median: a rule that lands where the best of forty draws lands is
        # worth having, and one that lands at the median is not.
        draws = [
            [
                at(select("random", Bp.reshape(len(Bp), -1), n, np.random.default_rng(r)))
                for r in range(draws_n)
            ]
            for n in ns
        ]
        res["size"] = {
            "rand p25": [float(np.percentile(d, 25)) for d in draws],
            "rand med": [float(np.median(d)) for d in draws],
            "rand best": [float(np.min(d)) for d in draws],
        }
        # every rule scored off the SAME prior -- a five-hour-old table -- which is the only
        # information a design is allowed to use before the states are measured
        if "25C" in names:
            res["rules"] = size_curve(
                tabs["25C"],
                tabs["big"],
                tabs["20C"],
                ns,
                tgts,
                prep,
                kinds=[k for k in KINDS if k != "random"],
            )
            res["size"]["mm/stale"] = res["rules"]["maxmin"]
        # the two ways of designing that are available with no prior table at all, and the
        # one that looks free and is not
        res["size"]["mm/own"] = [at(select("maxmin", Bp.reshape(len(Bp), -1), n)) for n in ns]
        res["size"]["phase"] = [at(select("spread", tabs["big"].phase, n)) for n in ns]
    if "big" in names:
        res["neighbours"] = {
            k: v for k, v in neighbour_stats(tabs["big"].phase).items() if k != "D"
        }
        res["interp"] = interp_curve(tabs["big"])
    if {"25C", "big", "20C"} <= set(names):
        S, F = tabs["25C"], tabs["20C"]
        ia, ib = align(S, F)
        res["models"] = {}
        out, inp = list(BEST_RAILS[0]), list(BEST_RAILS[1])
        for name, (fit, ap) in DRIFT_MODELS.items():
            C = ap(fit(S.T[ia], F.T[ib]), S.T)
            res["models"][name] = (
                float(np.abs(C[ia] - F.T[ib]).mean()),
                score(prep, tgts, C[ia][:, out][:, :, inp], F.block()[ib]),
            )
        res["ties"] = tie_curve(
            tabs["25C"], tabs["big"], tabs["20C"], (1, 2, 3, 5, 10, 20), tgts, prep, reps=tie_reps
        )

    return res


def protocol(
    tabs=None, ns=(15, 20, 30, 50, 100), ms=(0, 2, 3, 5), n_targets: int = 200, reps: int = 7
) -> dict:
    """The two numbers the recommendation is made of, measured rather than extrapolated.

    `grid` is hosting error for a table of n designed states corrected from m tie-points,
    with its wall-clock cost. `aging` is the same correction with tie-points that were
    themselves taken in an earlier session, which is where it stops working."""
    tabs = tabs or load_bench()
    tgts = targets(n_targets)
    prep = prepare(tgts)
    Xd = tabs["25C"].block().reshape(len(tabs["25C"]), -1)
    grid = {}
    for n in ns:
        sub = tabs["25C"].subset(select("maxmin", Xd, n, np.random.default_rng(0)))
        row = []
        for m in ms:
            if m == 0:
                ia, ib = align(sub, tabs["20C"])
                row.append(score(prep, tgts, sub.block()[ia], tabs["20C"].block()[ib]))
            else:
                row.append(
                    tie_curve(sub, tabs["big"], tabs["20C"], (m,), tgts, prep, reps=reps)[m][0]
                )
        grid[n] = row
    aging = {}
    for stale, tie, truth in [
        ("25C", "big", "20C"),
        ("fresh", "big", "20C"),
        ("big", "25C", "fresh"),
        ("20C", "25C", "fresh"),
    ]:
        if not {stale, tie, truth} <= set(tabs):
            continue
        r = tie_curve(tabs[stale], tabs[tie], tabs[truth], (3, 5), tgts, prep, reps=reps)
        aging[(stale, tie, truth)] = (
            score(prep, tgts, tabs[stale].block(), tabs[truth].block()),
            score(prep, tgts, tabs[tie].block(), tabs[truth].block()),
            r[3][0],
            r[5][0],
        )
    return {"ns": ns, "ms": ms, "grid": grid, "aging": aging}


def protocol_digest(pr: dict) -> str:
    L = [
        "a table of n designed states, corrected from m tie-points measured now",
        "  n   cost s" + "".join(f"{'m=' + str(m):>9s}" for m in pr["ms"]),
    ]
    for n, row in pr["grid"].items():
        L.append(
            f"  {n:<4d}{cost_s(n):>7.0f}"
            + "".join(f"{v:9.4f}" for v in row)
            + ("   <- keep this table; only the m column is paid per session" if n == 100 else "")
        )
    L += [
        "",
        "tie-points age too: the correction lands the table where the TIES were",
        f"  {'stale':>6}{'ties':>6}{'truth':>6}{'uncorr':>9}{'tie tbl':>9}{'m=3':>9}{'m=5':>9}",
    ]
    for (a, b, c), (u, t, h3, h5) in pr["aging"].items():
        L.append(f"  {a:>6}{b:>6}{c:>6}{u:9.4f}{t:9.4f}{h3:9.4f}{h5:9.4f}")
    return "\n".join(L)


def digest(res: dict) -> str:
    n = res["names"]
    L = [
        f"tables: {', '.join(n)}",
        "",
        "mean |dT| per entry, same heater state (row vs col)",
        "        " + "".join(f"{b:>9s}" for b in n),
    ]
    for a in n:
        L.append(f"{a:>8s}" + "".join(f"{res['dT'][(a, b)]:9.4f}" for b in n))
    L += [
        "",
        "median hosting error, 2x2 on out(1,2) in(1,2): planned on row, realised on col",
        "        " + "".join(f"{b:>9s}" for b in n),
    ]
    for a in n:
        L.append(f"{a:>8s}" + "".join(f"{res['host'][(a, b)]:9.4f}" for b in n))
    if "size" in res:
        ns, keys = res["ns"], list(res["size"])
        L += [
            "",
            "Q1/Q2 size and selection: plan on big (20:29), realised on 20C (21:36)",
            "  n   " + "".join(f"{k:>11s}" for k in keys),
        ]
        for i, nn in enumerate(ns):
            L.append(f"  {nn:<4d}" + "".join(f"{res['size'][k][i]:11.4f}" for k in keys))
    if "rules" in res:
        ns, keys = res["ns"], list(res["rules"])
        L += [
            "",
            "Q2 selection rules, all designed off the 25C table, planned on big, "
            "realised on 20C",
            "  n   " + "".join(f"{k:>11s}" for k in keys),
        ]
        for i, nn in enumerate(ns):
            L.append(f"  {nn:<4d}" + "".join(f"{res['rules'][k][i]:11.4f}" for k in keys))
    if "interp" in res:
        nb = res["neighbours"]
        L += [
            "",
            f"Q3 interpolation -- median nearest-neighbour distance {nb['median_nn']:.3f} "
            f"of a {nb['side']:.2f} cube diagonal",
            "  k        mean      affine",
        ]
        for k, v in res["interp"].items():
            L.append(f"  {str(k):<4}" + "".join(f"{x:12.4f}" for x in v))
    if "models" in res:
        L += [
            "",
            "Q4 drift model, stale 25C -> truth 20C, fitted on all 100 states",
            f"  {'model':<10}{'|dT|':>9}{'host':>9}",
        ]
        for k, (d, h) in res["models"].items():
            L.append(f"  {k:<10}{d:9.4f}{h:9.4f}")
        L += [
            "",
            "Q4/Q5 tie-points: stale 25C corrected from m states measured in the 20:29 "
            "session, realised at 21:36",
            f"  {'m':>4}{'host':>9}{'host p90':>10}{'|dT|':>9}{'cost s':>9}",
        ]
        for m, (h, h9, d) in res["ties"].items():
            L.append(f"  {m:>4}{h:9.4f}{h9:10.4f}{d:9.4f}{cost_s(0, m):9.0f}")
        L.append(
            f"  {'100':>4}{res['floor_host']:9.4f}{res['floor_host']:10.4f}"
            f"{res['floor_dT']:9.4f}{cost_s(100):9.0f}   (recapture the whole table)"
        )
    return "\n".join(L)


def _selftest(seed: int = 0) -> dict:
    """Synthetic first, because the bench files are untracked and absence must not pass
    vacuously; the bench half is asserted separately and skipped when it is missing."""
    rng = np.random.default_rng(seed)

    def rand_table(n, r=None):
        r = r or rng
        U = [np.linalg.qr(r.normal(size=(4, 4)))[0] for _ in range(n)]
        T = np.stack([np.abs(u) ** 2 for u in U])
        V = r.uniform(0, 3, (n, 16))
        return Table(V, T, np.full(16, 3.0), np.arange(8))

    t = rand_table(120)
    tg = targets(24, seed=1)
    prep = prepare(tg)

    # the metric IS the planner's: a table holding the target's own block, at any
    # brightness, hosts it exactly -- the hosted scale is fitted, so only direction counts
    B = tg[0]
    c, (A, r, s) = prep[0]
    exact = np.diag(r) @ (B + c) @ np.diag(s)
    _, e_plan, _ = host(B + c, (A, r, s), exact[None] * 0.37)
    assert e_plan < 1e-9, e_plan

    # a known mixing must come back out of THREE states -- 12 columns against 15 parameters
    M = np.abs(np.eye(4) + 0.25 * rng.normal(size=(4, 4)))
    fresh = apply_mixing(M, t.T)
    rec = apply_mixing(fit_mixing(t.T[:3], fresh[:3]), t.T)
    assert np.abs(rec - fresh).mean() < 1e-6, np.abs(rec - fresh).mean()
    # a real fit, not the identity smuggled through
    assert np.abs(colnorm(t.T) - fresh).mean() > 1e-3

    # every rule returns n distinct states and is reproducible. Whether a rule BEATS random
    # is a property of the measured block distribution, not of the code -- on Haar-random
    # blocks it does not, and only the bench half of this test is entitled to claim it.
    Xb = t.block().reshape(len(t), -1)
    for kind in KINDS:
        idx = select(kind, Xb, 12, np.random.default_rng(0))
        assert len(set(idx.tolist())) == 12, (kind, idx)
        assert np.array_equal(idx, select(kind, Xb, 12, np.random.default_rng(0))), kind
    e_rand = float(
        np.median(
            [
                score(prep, tg, t.block()[select("random", Xb, 24, np.random.default_rng(i))])
                for i in range(8)
            ]
        )
    )
    e_mm = score(prep, tg, t.block()[select("maxmin", Xb, 24)])

    # more states never host worse, at a fixed selection rule
    curve = [score(prep, tg, t.block()[select("maxmin", Xb, n)]) for n in (8, 16, 32, 64)]
    assert curve[-1] < curve[0], curve

    assert should_correct(0.13, 0.018, 3) and not should_correct(0.02, 0.018, 3)
    # the gate's m-dependence is DERIVED, so it is checkable: fit the mixing on m states of a
    # table that has not drifted and the corrected error must land on `correction_floor`
    mf = measured_floor()
    for m, got in mf["ratio"].items():
        want = correction_floor(1.0, m)
        assert abs(got - want) < 0.12 * want, (m, got, want)
    assert not should_correct(1e9, 0.018, 1)  # 12 observations cannot fit 15 parameters

    return {
        "synthetic": dict(
            e_plan=e_plan,
            e_mm=float(e_mm),
            e_rand=float(e_rand),
            curve=curve,
            mixing=float(np.abs(rec - fresh).mean()),
        ),
        "floor": mf,
        "bench": _selftest_bench(),
    }


def _selftest_bench() -> dict | None:
    """The claims in the module note, against the 2026-08-27 session. Thresholds are what
    those files measure; they are here so that a change to the readout chain or to
    `_pick_state`'s scoring shows up as a failure rather than as a quietly different
    recommendation."""
    tabs = load_bench()
    if not {"big", "20C", "25C"} <= set(tabs):
        return None
    res = report(tabs, n_targets=120)

    # two captures of the same chip state 67 min apart -- the repeatability floor everything
    # else is measured against
    assert res["floor_dT"] < 0.025, res["floor_dT"]
    # and one across a session boundary, which is what "stale" costs
    assert res["dT"][("25C", "20C")] > 5 * res["floor_dT"], res["dT"][("25C", "20C")]

    # size: n = 20 random is still well short of n = 100, so the curve has NOT flattened
    sz = res["size"]
    i10, i20, i100 = (res["ns"].index(k) for k in (10, 20, 100))
    assert sz["rand med"][i20] > 1.5 * sz["rand med"][i100], sz["rand med"]
    # a design taken off a DIFFERENT capture transfers and lands near the best of 40 draws
    assert sz["mm/stale"][i20] < sz["rand p25"][i20], (sz["mm/stale"][i20], sz["rand p25"][i20])
    # the same rule on the plan table's own blocks does not, because it chases its noise
    assert sz["mm/own"][i10] > sz["rand med"][i10], (sz["mm/own"][i10], sz["rand med"][i10])

    # nothing local reaches the floor, and the best of it barely beats predicting the table
    # average -- which is what "not dense enough to interpolate" means quantitatively
    best = min(min(v) for k, v in res["interp"].items() if k != "mean")
    assert best > 3 * res["floor_dT"], (best, res["floor_dT"])
    assert best > 0.5 * res["interp"]["mean"][0], (best, res["interp"]["mean"])

    # the drift is output-side mixing: the diagonal special case and the input side both fail
    m = res["models"]
    assert m["out mix"][0] < 0.35 * m["none"][0], m
    assert m["out mix"][0] < 0.5 * m["pd gain"][0], m
    assert m["out mix"][0] < 0.5 * m["in mix"][0], m
    assert m["both"][0] > 0.9 * m["out mix"][0], m  # the input factor adds nothing

    # one tie-point is under-determined and makes the table WORSE: 15 free parameters, and a
    # state is four columns of three constraints
    assert res["ties"][1][0] > res["host"][("25C", "20C")], res["ties"][1]
    # five fresh states buy a fresh table, tail included
    med, p90, dT = res["ties"][5]
    assert med < 0.6 * res["host"][("25C", "20C")], (med, res["host"][("25C", "20C")])
    assert med < 1.05 * res["floor_host"], (med, res["floor_host"])
    assert p90 < 1.15 * res["floor_host"], (p90, res["floor_host"])
    assert dT < res["floor_dT"], (dT, res["floor_dT"])
    # and two do not -- the tail is where an under-determined fit shows up
    assert res["ties"][2][1] > 1.3 * res["ties"][5][1], (res["ties"][2], res["ties"][5])
    return res


if __name__ == "__main__":
    import sys

    r = _selftest()
    b = r["bench"]
    if b is None:
        print(f"no bench sessions under {SESSIONS} -- session dumps are untracked")
    else:
        print(digest(b))
        if "protocol" in sys.argv[1:]:
            print("\nQ5 " + protocol_digest(protocol()))
    print(
        f"\nsynthetic: exact-block residual {r['synthetic']['e_plan']:.2e}, "
        f"mixing recovered from 3 states to {r['synthetic']['mixing']:.2e}, "
        f"maxmin {r['synthetic']['e_mm']:.4f} vs random {r['synthetic']['e_rand']:.4f} at n=24"
    )
    mf = r["floor"]
    print(
        f"correction noise floor, no drift planted ({mf['n']} states, {mf['reps']} draws, "
        f"floor {mf['floor']:.4f}/entry): "
        + "  ".join(
            f"m={m} {v:.2f}x (predicted {correction_floor(1.0, m):.2f})"
            for m, v in mf["ratio"].items()
        )
    )
