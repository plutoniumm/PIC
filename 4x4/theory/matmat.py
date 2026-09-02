"""Y = B X on the mesh, and a matrix bigger than the mesh, tiled onto it.

`intensity_matvec` gets one signed y = B x out of the photodiodes. These are the two rungs
above it, and neither needs a new decomposition -- both are scheduling on top of the same
`MatvecPlan`.

**A column of X is nearly free; the program is not.** A plan is a set of heater volts plus
the arithmetic that undoes the Sinkhorn diagonals, and finding it costs a multi-start fit
through the twin -- about a second per 2x2 tile -- and, on the bench, one thermal settle
when the volts are written. None of that depends on X, so `matmat` writes the program once
and walks the columns of X underneath it. Ordering the passes port-major (hold one switch
position, serve every column that needs it, then move) takes the switch out of the marginal
cost as well: measured on the twin, a column of a 2x2 costs 2 reads and 0.3 switch moves
port-major against 2 reads and 2.0 moves column-major, and the two orderings return the
same numbers to 1e-16. What is left in the marginal cost is photons.

**A tiled product is honest about where the adds happen.** `BlockPlan` cuts an n x n target
into k x k tiles, hosts each one in turn, and passes the matching k-row stripe of X through
it. The optics do every multiply; the HOST does the sum over tiles. That is not a shortcut
being hidden -- this bench has no optical path that can hold two tiles at once or add their
outputs, so a k x k mesh computing an n x n product must return (n/k)^2 partial products for
software to accumulate, and the interesting quantity is what fraction of the arithmetic
stays in light. It is k/2 multiply-accumulates per photodiode read: a lit rail multiplies
one input weight into k outputs at once, and the signed encoding costs a factor of two. On
this die k is 2, so the chip does about one MAC per read; the whole optical advantage is
linear in the block size, which is why the reachability ceiling below is the number that
matters and not the residual.

**Tile size, and why 3 is not on the table.** A 16x16 in 3x3 tiles is 36 programs against
64, so the question is worth asking properly. It fails twice over, independently, and both
halves are measured.

`intensity_matvec.reachability` on the calibration in force -- a best-case FIT, no table:
2x2 hosts to 0.0007, 3x3 to 0.112, 4x4 to 0.258. On the mesh as DESIGNED the same fits reach
0.0009 and 0.0001, so the k = 3 gap is heater span and nothing else. It is span and not
geometry: scaling the current limit by 1.25 takes the widest heater past 2 pi and the k = 3
fit from 0.138 to 0.020, by 1.5 to 0.007. (The "heater span does not bind" result recorded
in `talk/progress.md` is a k = 2 result and does not carry over -- at k = 2 three free
numbers are chased with eight heaters, at k = 3 it is eight with eight.)

Then the table. Picking off the 115-state bench table rather than fitting, over 48 random
signed targets on every rail triple, k = 3's best pair hosts at a median 0.83 and a worst
1.84, against 0.14 and 0.32 for the best 2x2 pair. A relative error of 0.83 is barely better
than answering zero. And it does not improve with table size the way k = 2 does: from 15 to
115 states k = 2 falls 0.53 -> 0.16 and k = 3 only 1.14 -> 0.72, roughly N^-0.6 against
N^-0.22, so reaching k = 2's accuracy would take on the order of 10^5 states -- days of
capture for a table that is stale in an hour.

So k = 3 would buy 44 percent of the wall clock for a 5.8x worse block, which is not a
trade. k = 4 is worse again and has a second limit that is nobody's fault: |U|^2 of a unitary
is not merely doubly stochastic, it is UNISTOCHASTIC, and for n >= 3 that is a proper subset
of the Birkhoff polytope.

That k = 4 limit again, measured: Sinkhorn lands a target with a wide dynamic range near the
polytope boundary,
where the subset has holes. Measured on the ideal box over twelve random 4x4 targets: eleven
fit under 0.010 and one plateaus at 0.033, unmoved to four decimal places from 12 restarts
to 64 -- which is the plateau signature, since everything else in that table moves when the
budget does. Against it, |V|^2 of a random unitary -- unistochastic by construction -- fits
to 0.017 or better on every draw at the *low* budget (`_selftest` checks that control). So a
k = 4 tile fails on this die by heater span, and a minority of targets would still fail with
perfect heaters. One in twelve is the rate this many draws supports and no more.

**How the error accumulates, measured.** Each tile carries its own hosting residual and its
own read noise, and the output block is a sum of t = n/k of them. If those errors are
independent the sum grows as sqrt(t); if they are one systematic error repeated it grows as
t. `accumulation` computes both bounds from the *same* measured per-tile errors, so the
comparison is about correlation and not about how large the residuals happened to be. Over
12 trials at seed 0 the observed sum sits ON the independent bound at every tile count and
both noise levels: 0.96, 0.97, 0.98 times it at t = 2, 4, 8 noiseless, and 1.00, 1.05, 1.08
at 1.2 percent read noise. Six cells, all within 8 percent of 1. The errors add incoherently
-- sqrt(t), not t -- and this is the strongest form of the claim available, because the
denominator is computed from the very same per-tile errors as the numerator.

The SPREAD is as much of the result as the ratio. The per-trial ratio has sd 0.08, 0.04,
0.17 noiseless and 0.15, 0.17, 0.25 at bench noise, rising with t because summing t vectors
in k*cols = 8 dimensions carries that much sampling scatter on its own. So a two- or
three-trial run of this same function returns anything from 0.5 to 1.3, and no single triple
taken out of one is a measurement of anything -- which is why `trials` defaults to 12 and why
`vs_indep_sd` and `trials` come back beside the ratio. Quote the mean with its spread or do
not quote it.

`vs_corr` is not the mirror image of `vs_indep` and is not evidence about correlation. It
factors exactly as `vs_indep * (indep/corr)`, and `indep/corr` says only how UNEQUAL the
per-tile residuals are: 1/sqrt(t) if they are all the same size, 1 if one tile dominates.
Measured it runs 0.88, 0.90, 0.75 noiseless against a 1/sqrt(t) of 0.71, 0.50, 0.35, because
the per-tile residual is heavy tailed (nominally identical targets land anywhere from 1e-4 to
2e-2); read noise evens the tiles out and pulls it back to 0.78, 0.56, 0.41. So the two
bounds sit only 1.3x apart noiseless at t = 8, and the correlated hypothesis is excluded by
`vs_indep` sitting on 1, not by `vs_corr` being small.

The signal accumulates incoherently too, so the RELATIVE error of a block product does not
grow with the tile count: at bench noise it is flat, 0.080, 0.084, 0.073 at t = 2, 4, 8.
Noiseless it is two orders of magnitude smaller and set entirely by which tiles happened to
fit badly -- 0.0007, 0.0077, 0.0048, no trend in t and none to be had from three points
against that tail. Tiling costs time, not accuracy.

**Unitarity as a prior, and what it is honestly a prior over.** The photodiodes measure
|U|^2, which is moduli; the differential passes recover the SIGN of each entry; no phase is
measured anywhere on this bench. So a measured product is a real matrix, and the manifold to
project onto is O(k), the real orthogonal group -- not U(k). Projecting onto U(k) would be
claiming the imaginary part is small when in fact it is absent. (The mesh's own transfer is
complex and unitary; the hosted block is |U|^2 after Sinkhorn and the shift, and is real by
construction.) `compose` uses that: host two orthogonal matrices, read the first back with
`matmat(plan_a, I)`, feed the measured result through program b, and the group property says
the answer must still be orthogonal.

`manifold_dist` measures how far it is from being so and needs no ground truth, which makes
it the only one of these numbers a real bench can compute. It is also a lower bound on the
true error and not an estimate of it -- the nearest orthogonal matrix is at least as close
as the right one -- so what it misses is exactly the error along the manifold; it read 0.8
to 0.9 of the truth here. A large distance says the measurement is inconsistent with *any*
orthogonal matrix, which is a stronger statement than being far from one particular target.

Projection then removes the normal part and cannot touch the rest: over 24 draws, 0.0099 ->
0.0046 noiseless and 0.096 -> 0.035 at 1.2 percent read noise, and polar improves on the raw
product in every single draw at both levels. Twenty-four draws and not five, because a single
draw ranges over 7x: the same quantity over the first five reads 0.0056 -> 0.0039, which is a
different-looking answer to the same question. An isotropic error at k = 2 would halve exactly
-- one of the four dimensions is tangent to O(2) -- and the 2.2x and 2.7x observed say this
error is at least as normal-to-manifold as isotropic, which is what a gain-like error looks
like (the hosted scale and the Sinkhorn diagonals are gains, and a gain is entirely normal).
The prior is doing real work rather than dressing the answer up: agreement with the TRUE
product improves, and that is a claim only the twin can check, which is why the check lives
here.

**Polar, not QR.** `polar` is the nearest orthogonal matrix in Frobenius norm and has no
preferred column order. `qr_orth` returns *an* orthogonal matrix: Gram-Schmidt keeps the
first column exactly and spends every correction on the last, so it is by construction no
closer *to the measurement* -- `polar_step <= qr_step` identically, which `_selftest` checks
-- and permuting the columns moves its answer while polar's does not.

Closer to the TRUTH is a separate question and the answer depends on how large the error is.
At 1.2 percent read noise polar wins on the mean at every draw count tried (0.035 against
0.047 over 24, 1.4x) and in 58 percent of individual draws, so that is what `_selftest`
asserts. Noiseless the composed error is small enough that the two sit inside the
draw-to-draw spread -- 0.0046 against 0.0052 over 24 draws, with QR ahead over the first 5,
8 and 12 -- and asserting an ordering there would be asserting the sampling noise. What is
not a coin flip is the arbitrariness: `qr_order_spread` says QR's answer shifts under a
column permutation by roughly the size of the correction it just applied, and polar's does
not move at all.

    python -m theory.matmat        # the ladder, both boxes, and the group-property check
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .calib import Calibration
from .intensity_matvec import (HeaterBox, MatvecPlan, block_rails, fit_nonneg, plan_matvec,
                               score, twin_probe)
from .twin import Twin

TILE_K = 2   # see the docstring: fitted, 3 misses by 11 percent on this die and 4 by 26;
             # picked off the measured table, 3 misses by 83 percent and is unusable


class CountedProbe:
    """A probe that reports what it cost the bench, charged the way the rig charges it.

    `pic.matvec.rig_probe` moves the switch only when the port changes and writes the
    heaters -- paying the thermal settle -- only when the volts change, so counting
    transitions in the request stream is the hardware cost rather than an estimate of it.
    Writes are the expensive ones: a settle is seconds and a read is milliseconds."""

    def __init__(self, probe):
        self.probe = probe
        self.reads = self.switch_moves = self.writes = 0
        self._port, self._volts = None, None

    def __call__(self, volts, port, power: float = 1.0):
        v = np.asarray(volts, float)
        if self._volts is None or not np.array_equal(v, self._volts):
            self.writes += 1
            self._volts = v.copy()
        if self._port != int(port):
            self.switch_moves += 1
            self._port = int(port)
        self.reads += 1
        return self.probe(volts, port, power)

    def per_column(self, n_col: int) -> dict:
        return {"reads": self.reads / n_col, "switch_moves": self.switch_moves / n_col,
                "writes": self.writes}

    def __str__(self):
        return (f"{self.writes} heater writes, {self.switch_moves} switch moves, "
                f"{self.reads} reads")


def matmat(plan: MatvecPlan, probe, X, order: str = "port") -> np.ndarray:
    """Y = B X under ONE heater program, the columns of X applied in turn.

    The plan does not depend on X, so nothing here refits or rewrites anything and the whole
    marginal cost of a column is its photodiode reads.

    `order="port"` walks (term, input rail, column), so the switch visits each rail once per
    program however many columns follow; `order="column"` is the literal
    `[plan.matvec(probe, x) for x in X.T]` and pays those moves again for every column. They
    compute the same numbers -- `_selftest` checks they agree to 1e-12 -- so the ordering is
    purely what the bench is charged for. Port-major is not free on a rig that sets the
    input weight with the laser (`power="laser"`): the retune moves per read either way.
    """
    X = np.atleast_2d(np.asarray(X, float))
    out, inp = plan.rails
    if X.shape[0] != len(inp):
        raise ValueError(f"X has {X.shape[0]} rows; the plan takes {len(inp)}")
    if order == "column":
        return np.stack([plan.matvec(probe, X[:, c]) for c in range(X.shape[1])], axis=1)
    if order != "port":
        raise ValueError(f"order={order!r}; use 'port' or 'column'")

    Y = np.zeros((len(out), X.shape[1]))
    for t in plan.terms:
        acc = np.zeros_like(Y)
        P = X / t.s[:, None]                      # Sinkhorn input scaling, then the signs
        for j, port in enumerate(inp):
            for c in range(X.shape[1]):
                for sgn in (1.0, -1.0):
                    p = sgn * P[j, c]
                    if p > 0:                     # an unlit rail is not a read
                        acc[:, c] += sgn * np.asarray(probe(t.prog.volts, port, p))[list(out)]
        Y = Y + t.coeff * acc / t.prog.scale / t.r[:, None]
    return Y - plan.offset * X.sum(0)


@dataclass
class BlockPlan:
    """A matrix too big for the mesh, cut into k x k tiles it can host one at a time.

    `tiles` is sparse: a tile that is entirely zero -- which is what a block-structured
    target and enough padding produce -- is absent, because an all-zero block needs no light
    and no program. A zero-padded row *inside* a live tile is hosted like any other row and
    discarded on read; it costs fit accuracy like any other row, which is the price of a
    size that is not a multiple of k.
    """

    B: np.ndarray
    k: int
    rails: tuple
    tiles: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.B.shape

    @property
    def programs(self) -> int:
        """Heater writes for the whole product, and thermal settles with it. This is the
        cost that scales as (n/k)^2 and it is paid once, not per column."""
        return len(self.tiles)

    @property
    def err(self) -> float:
        """Worst hosted residual over the tiles -- the honest bound, as for one plan."""
        return max((p.err for p in self.tiles.values()), default=0.0)

    @property
    def passes(self) -> int:
        """Photodiode reads per column of X at full input support. Real dense inputs cost
        half of this: each rail is either positive or negative, never both."""
        return sum(p.passes for p in self.tiles.values())

    def predict(self) -> np.ndarray:
        """The signed matrix the tiling actually realises, noise aside."""
        M, N = self.B.shape
        Bh = np.zeros(((M + self.k - 1) // self.k * self.k,
                       (N + self.k - 1) // self.k * self.k))
        for (i, j), plan in self.tiles.items():
            Bh[i * self.k:(i + 1) * self.k, j * self.k:(j + 1) * self.k] = plan.predict()
        return Bh[:M, :N]

    def matmat(self, probe, X) -> np.ndarray:
        """Y = B X, tile by tile, program-major.

        The tile loop is OUTSIDE the column loop and that ordering is the whole point: the
        heater program is the expensive state, so every column that a tile can serve is
        served while it is hosted. Inverted -- columns outside, tiles inside -- an n x n
        product pays (n/k)^2 thermal settles per column instead of once.

        The `+=` is the software accumulation. It is the only arithmetic here the optics did
        not do."""
        X = np.atleast_2d(np.asarray(X, float))
        M, N = self.B.shape
        k = self.k
        Xp = np.zeros(((N + k - 1) // k * k, X.shape[1]))
        Xp[:N] = X
        Y = np.zeros(((M + k - 1) // k * k, X.shape[1]))
        for (i, j), plan in self.tiles.items():
            Y[i * k:(i + 1) * k] += matmat(plan, probe, Xp[j * k:(j + 1) * k])
        return Y[:M]


def plan_block(B, box: HeaterBox, twin: Twin | None = None, k: int = TILE_K, rails=None,
               seed: int = 0, **fit_kw) -> BlockPlan:
    """Tile B into k x k blocks and fit a mesh program for each.

    Every tile is an independent `plan_matvec`, which is what makes the residuals
    independent (`accumulation` measures that rather than assuming it) and what makes this
    embarrassingly parallel if a second chip ever exists. All tiles share one rail pair by
    default; `intensity_matvec.scan_rails` per tile would buy a little more brightness for
    36x the fitting time and is left to the caller."""
    B = np.atleast_2d(np.asarray(B, float))
    M, N = B.shape
    rails = block_rails(k) if rails is None else rails
    Bp = np.zeros(((M + k - 1) // k * k, (N + k - 1) // k * k))
    Bp[:M, :N] = B
    tiles = {}
    for i in range(Bp.shape[0] // k):
        for j in range(Bp.shape[1] // k):
            tile = Bp[i * k:(i + 1) * k, j * k:(j + 1) * k]
            if not tile.any():
                continue
            tiles[(i, j)] = plan_matvec(tile, box, twin, rails=rails,
                                        seed=seed + 101 * i + 7 * j, **fit_kw)
    return BlockPlan(B, k, rails, tiles)


def polar(M) -> np.ndarray:
    """Nearest orthogonal matrix in Frobenius norm: M = W S Vh -> W Vh.

    This is *the* projection. It minimises |M - Q| over all orthogonal Q, and it treats
    every column alike, so it has nothing to depend on but M itself."""
    W, _s, Vh = np.linalg.svd(np.asarray(M, float))
    return W @ Vh


def qr_orth(M) -> np.ndarray:
    """Gram-Schmidt orthogonalisation, sign-fixed so R has a positive diagonal.

    Without that convention LAPACK's Q comes back with arbitrary column signs and any
    comparison against a target is meaningless. Even with it this is *an* orthogonal matrix
    and not the nearest one: column 0 survives exactly and the last column absorbs every
    correction, so the answer depends on the column order (`qr_order_spread`)."""
    Q, R = np.linalg.qr(np.asarray(M, float))
    return Q * np.sign(np.diag(R))


def qr_order_spread(M) -> float:
    """How much `qr_orth` changes when the columns are permuted and the result permuted
    back. Zero for `polar` by construction; this is the number that says how arbitrary the
    QR answer is."""
    M = np.asarray(M, float)
    base = qr_orth(M)
    worst = 0.0
    for perm in itertools.permutations(range(M.shape[1])):
        p = np.asarray(perm)
        got = qr_orth(M[:, p])[:, np.argsort(p)]
        worst = max(worst, float(np.linalg.norm(got - base) / np.sqrt(M.shape[1])))
    return worst


def manifold_dist(M) -> float:
    """Frobenius distance from M to the nearest orthogonal matrix, per entry.

    It equals |sigma - 1| over the singular values, and dividing by sqrt(k) puts it on the
    same scale as a relative error against a target of unit-norm columns. Worth reporting on
    its own: it needs no ground truth, so on a real measurement it is the only one of these
    numbers available, and it says the measured product is inconsistent with *every*
    orthogonal matrix rather than merely far from the intended one."""
    s = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    return float(np.linalg.norm(s - 1) / np.sqrt(len(s)))


def random_orthogonal(rng, k: int) -> np.ndarray:
    return qr_orth(rng.normal(size=(k, k)))


def _rel(A, B) -> float:
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B)) / np.linalg.norm(B))


def compose(box: HeaterBox, twin: Twin | None = None, k: int = TILE_K, noise: float = 0.0,
            repeats: int = 1, trials: int = 4, seed: int = 0, rails=None, probe=None,
            plan_fn=None, **fit_kw) -> dict:
    """The group property, measured: host two orthogonal matrices and run one through the
    other.

    There is no optical loopback on this bench, so the composition is two programs and a
    handoff in software: read A back with `matmat(plan_a, I)` -- the identity is just the k
    basis vectors, which is the cheapest possible use of the matmat rung -- and then feed
    that MEASURED matrix in as the input vectors of program b. Both stages' errors are
    therefore in the answer, which is the point: the product of two orthogonal matrices is
    orthogonal, so `manifold_dist` of the result is a consistency test that never sees the
    truth -- pass `probe=pic.matvec.rig_probe(rig, calib)` and it becomes a bench diagnostic,
    since `manifold_dist` is the one number here that does not need to know the answer.

    Returns the distance to the manifold before projection, and the error against the true
    product raw, polar-projected and QR-projected. The comparison that matters is whether
    projection IMPROVES agreement with the true product or merely makes the answer look
    well-formed; on the twin it improves it, by about the factor dimension counting predicts
    when the error is isotropic and by much more when it is gain-like."""
    twin = twin or Twin()
    rows = []
    for t in range(trials):
        rng = np.random.default_rng(seed + 17 * t)
        Oa, Ob = random_orthogonal(rng, k), random_orthogonal(rng, k)
        # `plan_fn` is how the bench gets its own planner in. Planning through the twin is
        # right on the twin and wrong on the chip: `matrix_err` there is 1.14 relative, worse
        # than predicting a uniform matrix, so a twin-planned pair hosts two matrices nobody
        # has seen and the composed product comes back a near-orthogonal matrix at 13x the
        # wrong scale -- which reads as manifold_dist 0.925 and gets blamed on the projection.
        plan = plan_fn or (lambda B, s: plan_matvec(B, box, twin, rails=rails, seed=s,
                                                    **fit_kw))
        pa, pb = plan(Oa, seed + t), plan(Ob, seed + t + 50)
        pr = twin_probe(twin, box, noise=noise, repeats=repeats, rng=rng) if probe is None \
            else probe
        Ah = matmat(pa, pr, np.eye(k))
        Ch = matmat(pb, pr, Ah)
        Ct = Ob @ Oa
        rows.append({
            "dist_a": manifold_dist(Ah), "dist_c": manifold_dist(Ch),
            "raw": _rel(Ch, Ct), "polar": _rel(polar(Ch), Ct), "qr": _rel(qr_orth(Ch), Ct),
            "polar_step": float(np.linalg.norm(Ch - polar(Ch))),
            "qr_step": float(np.linalg.norm(Ch - qr_orth(Ch))),
            "qr_spread": qr_order_spread(Ch),
            "sign": score(polar(Ch).ravel(), Ct.ravel())[1],
        })
    return {"k": k, "noise": noise, "trials": trials,
            **{n: float(np.mean([r[n] for r in rows])) for n in rows[0]}}


def accumulation(box: HeaterBox, twin: Twin | None = None, k: int = TILE_K,
                 tiles=(2, 4, 8), cols: int = 4, trials: int = 12, noise: float = 0.0,
                 seed: int = 0, rails=None, **fit_kw) -> list[dict]:
    """Does tile error add as sqrt(t) or as t? Measured, from the tile errors themselves.

    One output block fed by t tiles: the error is sum_j d_j with d_j = (B_hat_j - B_j) X_j,
    and both bounds are computed from the same measured d_j -- sqrt(sum |d_j|^2) if they are
    independent, sum |d_j| if they are one systematic error repeated. `vs_indep` is the
    answer; see the module note for what it measures and for why `vs_corr` does not measure
    the mirror image of it.

    `vs_indep` is the MEAN OF THE PER-TRIAL RATIOS and `vs_indep_sd` is their spread, not a
    ratio of two averaged norms -- the residual is heavy tailed enough that averaging the
    norms first hides which trials the ratio came from. The spread is the reason `trials`
    defaults to 12: it reaches 0.24 at t = 8 under bench noise, so a 2- or 3-trial call
    returns a number with no significant figures.

    Deterministic in `seed`. Every draw is keyed on (seed, t, trial) and every tile's
    optimiser on (seed, t, trial, tile), so no two cells of the table share a stream, no
    arithmetic on the seed can collide two of them, and a rerun returns the same numbers.
    The keys deliberately include t: a tile fitted for the t = 2 row is not the same
    measurement as the same-indexed tile of the t = 8 row and must not reuse its start."""
    twin = twin or Twin()
    rows = []
    for t in tiles:
        acc, ratio = [], []
        for s in range(trials):
            rng = np.random.default_rng([seed, t, s])
            B = rng.normal(size=(k, k * t))
            X = rng.normal(size=(k * t, cols))
            probe = twin_probe(twin, box, noise=noise, rng=rng)
            d = []
            for j in range(t):
                Bj, Xj = B[:, j * k:(j + 1) * k], X[j * k:(j + 1) * k]
                fit_seed = int(np.random.default_rng([seed, t, s, j]).integers(1 << 30))
                plan = plan_matvec(Bj, box, twin, rails=rails, seed=fit_seed, **fit_kw)
                d.append(matmat(plan, probe, Xj) - Bj @ Xj)
            d = np.asarray(d)
            per = np.linalg.norm(d, axis=(1, 2))
            o, i_, c = (float(np.linalg.norm(d.sum(0))), float(np.linalg.norm(per)),
                        float(per.sum()))
            acc.append((o, i_, c, float(np.linalg.norm(B @ X))))
            ratio.append((o / max(i_, 1e-18), o / max(c, 1e-18), i_ / max(c, 1e-18)))
        obs, ind, cor, nrm = np.mean(acc, axis=0)
        vi, vc, ic = np.mean(ratio, axis=0)
        rows.append({"tiles": t, "trials": trials, "observed": obs, "indep": ind, "corr": cor,
                     "vs_indep": float(vi), "vs_corr": float(vc), "spread": float(ic),
                     "vs_indep_sd": float(np.std([r[0] for r in ratio])),
                     "sqrt_bound": float(1 / np.sqrt(t)),
                     "rel": obs / max(nrm, 1e-18), "noise": noise})
    return rows


def validate_block(box: HeaterBox, twin: Twin | None = None, k: int = TILE_K, n=(4, 4),
                   cols: int = 3, noise: float = 0.0, repeats: int = 1, seed: int = 0,
                   rails=None, label: str = "", **fit_kw) -> dict:
    """One row of the ladder table: plan an n[0] x n[1] target in k x k tiles, then measure
    Y = B X and score it against the truth."""
    twin = twin or Twin()
    rng = np.random.default_rng(seed)
    B = rng.normal(size=tuple(n))
    X = rng.normal(size=(n[1], cols))
    plan = plan_block(B, box, twin, k=k, rails=rails, seed=seed, **fit_kw)
    probe = CountedProbe(twin_probe(twin, box, noise=noise, repeats=repeats, rng=rng))
    Y = plan.matmat(probe, X)
    err, sign = score(Y.ravel(), (B @ X).ravel())
    m_err, m_sign = score(plan.predict().ravel(), B.ravel())
    return {"label": label, "k": k, "shape": tuple(n), "cols": cols, "noise": noise,
            "tiles": plan.programs, "fit_err": plan.err, "matrix_err": m_err,
            "matrix_sign": m_sign, "vec_err": err, "sign_acc": sign,
            **probe.per_column(cols), "plan": plan}


def digest(rows) -> str:
    head = (f"{'box':<10}{'k':>2}{'shape':>9}{'tiles':>7}{'hosted':>9}{'matrix':>9}"
            f"{'Y err':>9}{'sign':>7}{'reads/col':>11}{'writes':>8}")
    out = [head]
    for r in rows:
        shape = f"{r['shape'][0]}x{r['shape'][1]}"
        out.append(f"{r['label']:<10}{r['k']:>2}{shape:>9}"
                   f"{r['tiles']:>7}{r['fit_err']:>9.4f}{r['matrix_err']:>9.4f}"
                   f"{r['vec_err']:>9.4f}{r['sign_acc']:>7.0%}{r['reads']:>11.1f}"
                   f"{r['writes']:>8}")
    return "\n".join(out)


def _selftest(k: int = TILE_K, cols: int = 6, noise: float = 0.012, restarts: int = 12,
              steps: int = 300, sizes=((2, 2), (4, 4), (8, 8), (5, 5)), seed: int = 0):
    """Every claim above, against the twin, on the calibration in force and on the ideal
    box, so "what this die can do" and "what the method can do" stay separate."""
    twin = Twin()
    box = HeaterBox.from_calibration(Calibration.load_or_nominal())
    ideal = HeaterBox.ideal()
    kw = dict(restarts=restarts, steps=steps)
    rng = np.random.default_rng(seed)

    # one program, many columns: the two orderings must agree, and port-major must take the
    # switch out of the marginal cost
    B = rng.normal(size=(k, k))
    X = rng.normal(size=(k, cols))
    plan = plan_matvec(B, box, twin, seed=seed, **kw)
    port = CountedProbe(twin_probe(twin, box))
    col = CountedProbe(twin_probe(twin, box))
    Yp = matmat(plan, port, X)
    Yc = matmat(plan, col, X, order="column")
    order_gap = float(np.abs(Yp - Yc).max())
    one, many = (CountedProbe(twin_probe(twin, box)), CountedProbe(twin_probe(twin, box)))
    matmat(plan, one, X[:, :1])
    matmat(plan, many, X)
    marginal = {"port": port.per_column(cols), "column": col.per_column(cols),
                "first_column": one.per_column(1),
                "extra_per_column": {n: (getattr(many, n) - getattr(one, n)) / (cols - 1)
                                     for n in ("reads", "switch_moves", "writes")}}

    rows = [validate_block(box, twin, k=k, n=n, cols=3, seed=seed, label="measured", **kw)
            for n in sizes]
    rows += [validate_block(box, twin, k=3, n=(6, 6), cols=3, seed=seed, label="measured",
                            **kw)]
    rows += [validate_block(ideal, twin, k=kk, n=(2 * kk, 2 * kk), cols=3, seed=seed,
                            label="ideal", **kw) for kk in (2, 3, 4)]
    noisy = validate_block(box, twin, k=k, n=(8, 8), cols=3, noise=noise, seed=seed,
                           label=f"measured {noise:.1%}", **kw)

    acc = accumulation(box, twin, k=k, tiles=(2, 4, 8), trials=4, seed=seed, **kw)
    # 24 draws, because a single one is not a measurement: the projected error moves by 7x
    # across seeds, and at five draws the polar-vs-QR mean flips sign in the noiseless case
    grp = {nz: compose(box, twin, k=k, noise=nz, trials=24, seed=seed, **kw)
           for nz in (0.0, noise)}

    # the control behind the unistochastic claim: |V|^2 of a random unitary IS reachable by
    # construction, so if those fit and Sinkhorn-scaled Gaussians sometimes do not, the
    # 4x4 tail is the constraint and not the optimiser
    from .clements import random_unitary
    uni = max(fit_nonneg(np.abs(random_unitary(rng)) ** 2, ideal, twin, block_rails(4),
                         seed=s, **kw).err for s in range(3))

    big = next(r for r in rows if r["shape"] == (8, 8) and r["label"] == "measured")
    assert order_gap < 1e-12, order_gap
    assert port.reads == col.reads, (port.reads, col.reads)
    assert port.switch_moves < col.switch_moves / 2, (port.switch_moves, col.switch_moves)
    assert marginal["extra_per_column"]["writes"] == 0, marginal
    assert big["tiles"] == (8 // k) ** 2, big["tiles"]
    assert big["writes"] == big["tiles"], (big["writes"], big["tiles"])
    assert big["vec_err"] < 0.05, big["vec_err"]
    assert big["sign_acc"] == 1.0, big["sign_acc"]
    # the ideal box separates method from die -- but only up to k = 3. At k = 4 the target
    # has to be UNISTOCHASTIC, not merely doubly stochastic, and a minority of random tiles
    # are not: one of twelve plateaus at 0.033, unmoved from restarts=12 to restarts=64,
    # while `uni` shows a reachable-by-construction target still fits at the LOW budget.
    # The bar is 0.03 and not 0.01 because this is the low budget: the worst of these three
    # draws is 0.017 and the worst of six in a wider check was 0.011, against a plateau at
    # 0.033. Tightening it to 0.01 is asserting that 12 restarts always converge, which is
    # a claim about the optimiser and not about the set.
    assert uni < 0.03, uni
    for r in rows:
        if r["label"] == "ideal":
            lim = 0.05 if r["k"] < 4 else 0.10
            assert r["vec_err"] < lim, (r["label"], r["k"], r["shape"], r["vec_err"])
    # errors add incoherently, sqrt(t) not t. The band is 0.65-1.35 because the per-trial
    # ratio has sd up to 0.24 and this runs four trials; a tighter bar would be testing the
    # sampling noise. `vs_corr` is NOT asserted -- it is `vs_indep` times the residual
    # inequality, so a bar on it is a bar on how heavy the tail happened to be.
    for a in acc:
        assert 0.65 < a["vs_indep"] < 1.35, a
    # and the observed sum is on the independent SIDE of the two bounds, not merely near
    # one of them: below their geometric midpoint, which self-scales with the tail
    assert acc[-1]["observed"] < np.sqrt(acc[-1]["indep"] * acc[-1]["corr"]), acc[-1]
    for nz, g in grp.items():
        assert g["polar_step"] <= g["qr_step"] + 1e-12, g   # polar IS the nearest, exactly
        assert g["polar"] < g["raw"], (nz, g)              # and projecting improves the answer
    # closer to the TRUTH than QR is only asserted where it reproduces. Noiseless the two are
    # inside the draw-to-draw spread (QR ahead over the first 5, 8 and 12 of the same 24
    # draws, polar ahead over all 24), so a bar there would be a bar on the sampling noise.
    assert grp[noise]["polar"] < grp[noise]["qr"], grp[noise]
    return {"rows": rows, "noisy": noisy, "marginal": marginal, "order_gap": order_gap,
            "accumulation": acc, "group": grp, "box": box, "unistochastic_control": uni,
            "calibrated": bool(Calibration.load_or_nominal().meta)}


if __name__ == "__main__":
    r = _selftest()
    src = "measured calibration" if r["calibrated"] else "NOMINAL (uncharacterized)"
    print(f"heater box: {src}; {int(r['box'].trainable.sum())} steerable channels, widest "
          f"span {r['box'].span_pi[r['box'].trainable].max(initial=0.0):.2f} pi\n")

    m = r["marginal"]
    print(f"one program, many columns (2x2, {m['port']['writes']} write): the port-major and "
          f"column-major orderings agree to {r['order_gap']:.1e}")
    print(f"  {'':<14}{'reads/col':>11}{'switch/col':>12}{'writes':>8}")
    for name in ("port", "column"):
        print(f"  {name + '-major':<14}{m[name]['reads']:>11.2f}"
              f"{m[name]['switch_moves']:>12.2f}{m[name]['writes']:>8}")
    print(f"  {'marginal':<14}{m['extra_per_column']['reads']:>11.2f}"
          f"{m['extra_per_column']['switch_moves']:>12.2f}"
          f"{m['extra_per_column']['writes']:>8.0f}   (column 2 onwards; the first column "
          f"pays the write)")

    print("\nblock matmat: the optics multiply, the host adds the partial products")
    print(digest(r["rows"]))
    n = r["noisy"]
    print(f"{'at ' + format(n['noise'], '.1%') + ' read noise':<10}{n['k']:>2}"
          f"{'8x8':>9}{n['tiles']:>7}{n['fit_err']:>9.4f}{n['matrix_err']:>9.4f}"
          f"{n['vec_err']:>9.4f}{n['sign_acc']:>7.0%}{n['reads']:>11.1f}{n['writes']:>8}")

    a0 = r["accumulation"][0]
    print(f"\nhow tile errors add, {a0['trials']} trials (both bounds from the same measured "
          f"per-tile errors)")
    print(f"{'tiles':>6}{'observed':>10}{'if independent':>16}{'if correlated':>15}"
          f"{'obs/indep':>11}{'sd':>7}{'ind/corr':>10}{'1/sqrt(t)':>11}{'rel err':>9}")
    for a in r["accumulation"]:
        print(f"{a['tiles']:>6}{a['observed']:>10.4f}{a['indep']:>16.4f}{a['corr']:>15.4f}"
              f"{a['vs_indep']:>11.2f}{a['vs_indep_sd']:>7.2f}{a['spread']:>10.2f}"
              f"{a['sqrt_bound']:>11.2f}{a['rel']:>9.4f}")
    print("  obs/indep is the result; ind/corr against 1/sqrt(t) says how unequal the "
          "per-tile residuals were, which is all obs/corr would have added")

    print("\ngroup property: host O_a and O_b, run the measured A through program b")
    print(f"{'noise':>7}{'dist(A)':>9}{'dist(C)':>9}{'C raw':>8}{'C polar':>9}{'C qr':>8}"
          f"{'qr order spread':>17}{'sign':>7}")
    for nz, g in r["group"].items():
        print(f"{nz:>7.3f}{g['dist_a']:>9.4f}{g['dist_c']:>9.4f}{g['raw']:>8.4f}"
              f"{g['polar']:>9.4f}{g['qr']:>8.4f}{g['qr_spread']:>17.4f}{g['sign']:>7.0%}")
    print("  dist is to O(k) and needs no truth; polar is the nearest orthogonal matrix and "
          "QR is not")
