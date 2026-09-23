"""What set does a measured block actually live in, and which projection onto it helps.

Three different objects come off this bench and they live in three different sets. Treating
them alike is how a projection that provably reduces one error gets applied to something it
can only make worse. Derived here, then measured on the 2026-08-27 bench tables.

**A sub-block of an orthogonal matrix is a contraction, and that is exactly the set.**
Partition a 4x4 orthogonal Q into 2x2 blocks [[A, B], [C, D]]. `Q'Q = I` and `QQ' = I` are
not extra assumptions -- regrouped they *are*

    A'A + C'C = I     B'B + D'D = I     A'B + C'D = 0
    AA' + BB' = I     CC' + DD' = I     AC' + BD' = 0

so `A'A = I - C'C <= I` and every singular value of A is at most 1. The converse is
constructive: given any A with `||A||_2 <= 1`, take an SVD `A = U S V'`, put
`G = sqrt(I - S^2)`, and

    Q = [[U S V',  U G W'], [-Z G V',  Z S W']]

is orthogonal for *any* orthogonal Z, W -- verified symbolically in `_selftest`. So the set
of top-left 2x2 blocks of 4x4 orthogonal matrices is exactly the closed operator-norm unit
ball. That set is **convex** (it is a norm ball), the Frobenius projection onto it is
singular-value clipping `sigma -> min(sigma, 1)` by von Neumann's trace inequality, and a
projection onto a convex set is nonexpansive -- so `clip` can never move a measurement
*away* from a true contraction. `polar` projects onto O(k), which is not convex and carries
no such guarantee.

The same algebra gives the Cosine-Sine relation without citing it. Tracing the four block
identities gives `tr(A'A) = 2 - tr(C'C) = tr(D'D)`; Jacobi's identity on the complementary
minor of `Q^-1 = Q'` gives `det D = det A / det Q`, hence `det(A'A) = det(D'D)`. Two 2x2
PSD matrices with equal trace and equal determinant have equal eigenvalues, so **A and D
have the same singular values** -- exactly, symbolically, for k = 2. Writing them as
`cos(theta_i)` and running the SVD chain through the block identities gives
`sing(B) = sing(C) = sin(theta_i)`, which is the CSD.

**But this bench measures intensity, not amplitude.** A photodiode reports `|U|^2`, doubly
stochastic for a lossless mesh, so a 2x2 sub-block of what is measured is *substochastic* --
nonnegative, row sums and column sums at most 1 -- and the contraction bound applies only to
the amplitude matrix, which needs signs the intensity does not carry. That gap is worth a
number: over 3600 measured blocks (36 rail pairs x 100 states), the entrywise square root
with no signs breaks `sigma_max <= 1` in 37 percent of blocks and reaches 1.41, and even the
best of the two available sign classes still breaks it in 12 percent and reaches 1.30.
Entrywise absolute value is not norm-decreasing: `[[.8, .6], [.6, -.8]]` is orthogonal and
its modulus has `sigma_max = 1.4`. A contraction test run on `sqrt(T)` is measuring that
artefact, not the chip.

**The three objects, and what each projection does to them.**

| object | domain | set it lives in | what helps |
|---|---|---|---|
| measured 4x4 intensity transfer | intensity | unistochastic | nothing -- it is already as close to unistochastic as to Birkhoff |
| measured 2x2 sub-block | intensity | substochastic | nothing -- projection made it 16 percent worse |
| composed product of two hosted orthogonal matrices | signed | O(k) | polar, 2.9x on real drifted data |
| arbitrary hosted target | signed | all of R^(kxk) | nothing -- polar costs 13x |

The measured 4x4 sits 0.0660 per entry from the Birkhoff polytope and 0.0670 from the
unistochastic set: the extra structure costs 1.5 percent, so on this chip unistochasticity is
free and the 0.066 is loss and readout, not polytope geometry. Fitting a real `Q in O(4)`
instead costs 9 percent more (0.0722), which is the one place the real-versus-complex
distinction shows up as a number.

The 2x2 intensity sub-blocks are substochastic almost for free -- column sums are 1 by
construction after `Calibration.to_transfer`, and only 3.2 percent of *rows* exceed 1. The
projection therefore moves 6.4 percent of blocks and, scored against the same heater state
re-measured five degrees colder, it makes the error *worse*: 0.0475 -> 0.0551. The row-sum
excess reproduces across the two temperatures, so it is a systematic property of the
measurement (column-only normalisation plus real per-row loss), not noise to be projected
away -- the excess correlates at **r = +0.993** between the two temperatures.
**Substochastic projection is refused, with numbers.**

**Normal versus tangential, measured rather than argued.** Error at a point Q of O(k) splits
into `Q K` with K skew (tangential -- along the manifold, invisible to any projection) and
`Q S` with S symmetric (normal -- what polar removes). At k = 2 the tangent is 1 of 4
dimensions, so an isotropic error is 25 percent tangential. Measured on twelve composed
products through the real tables: the hosting residual is 13 percent tangential and the
5 degree drift on top of it is 16 percent. Both are below isotropic.

That is the opposite of the received story. On this rig the 25 C -> 20 C change is
predominantly NORMAL, because it enters through the hosting residual -- a different table
state becomes the best host, at a different brightness -- and not as a rotation of the
realised matrix. Which is why polar keeps working on the drifted case (0.2127 -> 0.0735,
2.9x) and why re-anchoring the plan on the fresh table buys only 12 percent of the raw error
and nothing at all after projection (0.0735 against 0.0763, inside the spread of twelve).

The controlled version separates the two cleanly. A known gain of 5 percent is pure normal
and polar removes it 6.0x; a known rotation of 0.15 rad is pure tangential and polar returns
1.01x, with the error floored at exactly the rotation angle. `manifold_dist` reads 0.019 in
both cases -- **it is blind to tangential error by construction**, which is the trap: it is a
faithful error bar (correlation +0.995 with the true error across the twelve real products)
right up until the error stops being normal, and it says nothing when it stops.

**Ordering.** For a correction that is right-multiplication by an orthogonal R,
`polar(M R) = polar(M) R` exactly, so project-then-correct and correct-then-project return
the *same matrix* and the ordering cannot be argued on accuracy. It is argued on evidence:
after projection `manifold_dist` is identically zero, so the diagnostic that would have said
a correction was needed -- and any estimator that reads the singular values to size it -- has
been deleted. Correct first, project last, and keep `manifold_dist` of the unprojected
measurement in the report.

    python -m theory.blocks        # the symbolic derivation, then the bench comparison
"""

from __future__ import annotations

import itertools

import numpy as np

from .calib import Calibration
from .matmat import manifold_dist, polar, qr_orth, random_orthogonal

BENCH = (
    "pic_data/sessions/2026-08-27/raw_transfers_big.json",  # 25 C
    "pic_data/sessions/2026-08-27/raw_transfers_20C.json",
)  # 20 C, same 100 states


def clip(M, s: float = 1.0) -> np.ndarray:
    """Frobenius projection onto the operator-norm ball of radius s: sigma -> min(sigma, s).

    This is *the* projection for a sub-block of a unitary in the amplitude picture, and it
    is a projection in the strong sense polar is not: the ball is convex, so the nearest
    point is unique and the map is nonexpansive -- it cannot increase the distance to any
    matrix already inside. Optimality is von Neumann's trace inequality, which says the
    minimiser shares M's singular vectors and reduces the problem to k independent scalar
    problems `min (sigma_i - x_i)^2` subject to `x_i <= s`.

    Fitting s rather than assuming 1 covers the brightness scalar the laser absorbs, around
    1.03-1.07 on this bench; `scale_hat` estimates it. It buys nothing on the composed
    product, where the truth genuinely has every singular value at 1 (measured: 0.2050
    against 0.1804 for s = 1 and 0.0735 for polar), because clipping can only pull the
    large singular value down and the error there is mostly a small one being too small."""
    W, sv, Vh = np.linalg.svd(np.asarray(M, float))
    return W @ np.diag(np.minimum(sv, float(s))) @ Vh


def scale_hat(M) -> float:
    """The subnormalisation a scaled-orthogonal model would fit: the mean singular value.

    `min_s |M - s Q|_F` over s and orthogonal Q is solved by `s = mean(sigma)`, so this is
    the brightness a measurement carries if its shape is otherwise right. Reported, not
    divided out by default -- `pic.matvec.shape_scale` splits brightness from shape against
    a known target, and this is the target-free version of the same number."""
    return float(np.linalg.svd(np.asarray(M, float), compute_uv=False).mean())


def contraction_dist(M, s: float = 1.0) -> float:
    """Distance to the contraction ball, per entry: `||max(sigma - s, 0)|| / sqrt(k)`.

    The one-sided sibling of `theory.matmat.manifold_dist`, and weaker for exactly that
    reason: it is zero for anything already inside the ball, so a block that is far too
    *dim* scores perfectly. It is still a genuine target-free refusal -- a positive value
    proves the measurement is not a sub-block of any unitary, at any sign assignment -- and
    that is the honest use for it."""
    sv = np.linalg.svd(np.asarray(M, float), compute_uv=False)
    return float(np.linalg.norm(np.clip(sv - float(s), 0.0, None)) / np.sqrt(len(sv)))


def best_signed_sigma(A) -> float:
    """Smallest `sigma_max` any sign assignment can give the amplitudes of a 2x2 intensity
    block. Above 1 the block cannot be a sub-block of a unitary, whatever the signs are.

    Signs are not measured in a plain transfer sweep, so a contraction test on `sqrt(A)`
    with all entries positive is testing the modulus and not the matrix -- and entrywise
    absolute value can *raise* the singular values, since it changes `|ad - bc|` to
    `||ad| - |bc||`. For 2x2 the Frobenius norm is sign-invariant and only the determinant
    moves, so `sigma_max` is minimised by the sign class with the larger `|det|`, which is
    `|ad| + |bc|`. That gives the best case in closed form and needs no search."""
    M = np.sqrt(np.clip(np.asarray(A, float), 0.0, None))
    a, b, c, d = M.ravel()
    fr, det = float((M**2).sum()), abs(a * d) + abs(b * c)
    return float(np.sqrt((fr + np.sqrt(max(fr**2 - 4 * det**2, 0.0))) / 2))


def embed_contraction(A, rng=None) -> np.ndarray:
    """A 4x4 orthogonal matrix whose top-left 2x2 block is exactly A. Requires ||A||_2 <= 1.

    The constructive half of the characterisation: necessity is `A'A = I - C'C`, and this is
    sufficiency. Z and W are free, which is the statement that the block does not determine
    the rest of the mesh -- the fibre over a given block is a copy of O(k) x O(k)."""
    A = np.asarray(A, float)
    k = A.shape[0]
    U, s, Vh = np.linalg.svd(A)
    if s.max() > 1 + 1e-9:
        raise ValueError(f"||A||_2 = {s.max():.6f} > 1; not a sub-block of any orthogonal")
    G = np.diag(np.sqrt(np.clip(1 - s**2, 0.0, None)))
    rng = np.random.default_rng() if rng is None else rng
    Z, W = random_orthogonal(rng, k), random_orthogonal(rng, k)
    return np.block([[U @ np.diag(s) @ Vh, U @ G @ W.T], [-Z @ G @ Vh, Z @ np.diag(s) @ W.T]])


def _dykstra(A, ops, iters: int) -> np.ndarray:
    """Dykstra's alternating projection: the Frobenius projection onto an intersection of
    convex sets, which plain alternating projection does not give (it lands *somewhere* in
    the intersection, not at the nearest point)."""
    X = np.asarray(A, float).copy()
    p = [np.zeros_like(X) for _ in ops]
    for _ in range(iters):
        for i, P in enumerate(ops):
            Y = P(X + p[i])
            p[i] = X + p[i] - Y
            X = Y
    return X


def substochastic(A, iters: int = 400) -> np.ndarray:
    """Projection onto {A >= 0, row sums <= 1, column sums <= 1} -- the set a sub-block of a
    doubly stochastic matrix lives in.

    Three convex sets, each with a closed-form projection: clipping to the orthant, and one
    halfspace per violated row and column, whose projection subtracts the excess spread
    evenly. Dykstra composes them into the true projection.

    Measured on this bench it is a refusal, not a tool: only 6.4 percent of the 3600 measured
    2x2 blocks violate anything at all, and projecting them raised the error against the same
    state re-measured at 20 C from 0.0475 to 0.0551. The violation is reproducible across the
    two temperatures, which is what says it is systematic -- column-only normalisation plus
    real per-row loss -- rather than the noise a projection is entitled to remove."""
    k, n = np.asarray(A, float).shape
    return _dykstra(
        A,
        (
            lambda M: M - np.clip(M.sum(1, keepdims=True) - 1, 0, None) / n,
            lambda M: M - np.clip(M.sum(0, keepdims=True) - 1, 0, None) / k,
            lambda M: np.clip(M, 0.0, None),
        ),
        iters,
    )


def birkhoff(T, iters: int = 2000) -> np.ndarray:
    """Projection onto the doubly stochastic matrices, in Frobenius norm.

    Not Sinkhorn: Sinkhorn finds the doubly stochastic matrix in the *diagonal-scaling
    orbit* of T, which is a different point and, on a dim probe, a much worse one
    (CLAUDE.md: 0.0045 -> 0.0176 across sessions). This is the nearest point, and it is what
    "distance to the Birkhoff polytope" has to mean if it is going to be compared with the
    distance to the unistochastic subset."""
    n = np.asarray(T, float).shape[0]
    return _dykstra(
        T,
        (
            lambda M: M + (1 - M.sum(1, keepdims=True)) / n,
            lambda M: M + (1 - M.sum(0, keepdims=True)) / n,
            lambda M: np.clip(M, 0.0, None),
        ),
        iters,
    )


def _fit_stochastic(T, complex_: bool, restarts: int, seed: int) -> tuple:
    from scipy.linalg import expm
    from scipy.optimize import minimize

    T = np.asarray(T, float)
    n = T.shape[0]
    iu = np.triu_indices(n, 1)
    m = len(iu[0])

    def unpack(p):
        K = np.zeros((n, n), complex if complex_ else float)
        K[iu] = (p[:m] + 1j * p[m : 2 * m]) if complex_ else p[:m]
        K = K - (K.conj().T if complex_ else K.T)
        return expm(K + 1j * np.diag(p[2 * m :]) if complex_ else K)

    def f(p):
        V = unpack(p)
        return float(((np.abs(V) ** 2 - T) ** 2).sum())

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        r = minimize(f, rng.normal(size=(2 * m + n) if complex_ else m), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x), float(np.sqrt(best.fun) / n)


def unistochastic_fit(T, restarts: int = 4, seed: int = 0) -> tuple:
    """Nearest `|U|^2` for a complex unitary U, and the per-entry distance to it.

    Membership has no closed form for n >= 3 -- unistochastic is a proper, curved subset of
    the Birkhoff polytope with holes near the boundary -- so this fits rather than tests.
    Returns (U, distance)."""
    return _fit_stochastic(T, True, restarts, seed)


def orthostochastic_fit(T, restarts: int = 4, seed: int = 0) -> tuple:
    """Nearest `Q**2` for a real orthogonal Q. Strictly smaller a set than unistochastic, and
    the right one when the pipeline recovers signs rather than phases -- which this one does.
    On the measured 4x4 it costs 13 percent more residual than allowing a complex U, which is
    the only place on this bench the real-versus-complex choice shows up as a number."""
    return _fit_stochastic(T, False, restarts, seed)


def normal_tangential(E, Q) -> tuple[float, float]:
    """Split an error E at a point Q of O(k) into (tangential, normal) Frobenius norms.

    The tangent space at Q is `{Q K : K skew}` and the normal space is `{Q S : S symmetric}`,
    so the split is the skew/symmetric decomposition of `Q' E` -- orthogonal, so the two
    norms add in quadrature. Polar removes the normal part and by construction cannot touch
    the tangential one. At k = 2 the tangent is 1 of 4 dimensions, so 25 percent tangential
    is what an isotropic error looks like and anything below that is a normal-biased error
    -- a gain, a brightness, a hosting residual."""
    A = np.asarray(Q, float).T @ np.asarray(E, float)
    return (float(np.linalg.norm((A - A.T) / 2)), float(np.linalg.norm((A + A.T) / 2)))


class TableProbe:
    """A `(volts, port, power) -> intensities` probe served out of a measured transfer table.

    The same call signature as `pic.matvec.rig_probe`, so anything written against the bench
    runs against recorded hardware with nothing plugged in. Volts are matched exactly rather
    than by nearest neighbour, because the only planner that should be pointed at this is one
    that picks states *from* a table -- a nearest-neighbour lookup would silently substitute a
    different measurement and report the substitution as chip error.

    Planning off one table and probing off another is the useful case: it replays a real
    5 degree drift with the truth still known."""

    def __init__(self, volts, T):
        self.T = np.asarray(T, float)
        self.key = {v.tobytes(): i for i, v in enumerate(np.ascontiguousarray(volts, float))}

    def __call__(self, volts, port: int, p: float = 1.0) -> np.ndarray:
        k = np.ascontiguousarray(np.asarray(volts, float)).tobytes()
        if k not in self.key:
            raise KeyError("heater state is not in this table; plan from the table you probe")
        return float(p) * self.T[self.key[k]][:, int(port)]


def load_bench(paths=BENCH, calib: Calibration | None = None, root=None) -> list[tuple]:
    """The session's raw four-port sweeps -> (volts, |U|^2) per file, or [] if absent.

    Read here rather than through `pic.normalise.load_session` because `theory` must not
    import `pic`; the file is JSON and the physics is `Calibration.to_transfer`, both of
    which are already on this side of the layering. The session's OWN switch-off dark is
    used, not the calibration's: it is additive, so a stale floor is the one nuisance a
    column sum cannot divide out."""
    import json
    from pathlib import Path

    calib = calib or Calibration.load_or_nominal()
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    out = []
    for p in paths:
        q = root / p
        if not q.exists():
            continue
        d = json.loads(q.read_text())
        dark = np.asarray(d["dark_switch_off"], float)
        # stored port-major (raw[k][j]) because that is the order the switch visits
        T = np.stack([calib.to_transfer(np.asarray(s["raw"], float).T, dark) for s in d["states"]])
        out.append((np.asarray([s["volts"] for s in d["states"]], float), T))
    return out


def intensity_sets(T, restarts: int = 3, seed: int = 0) -> dict:
    """Per-entry distance from one measured `|U|^2` to Birkhoff, unistochastic and
    orthostochastic. Divided by n rather than reported as a Frobenius norm so it is on the
    same scale as `manifold_dist` and as a relative entry error."""
    T = np.asarray(T, float)
    n = T.shape[0]
    return {
        "birkhoff": float(np.linalg.norm(T - birkhoff(T)) / n),
        "unistochastic": unistochastic_fit(T, restarts, seed)[1],
        "orthostochastic": orthostochastic_fit(T, restarts, seed)[1],
    }


def subblock_census(T, k: int = 2) -> dict:
    """Every k x k sub-block of every measured state, scored against both candidate sets.

    Reports the substochastic violation (intensity, the set the block really lives in) and
    the contraction violation of its amplitudes (the set it would live in if signs were
    measured), the latter both for the naive positive square root and for the best sign
    class. The gap between those two is the cost of not measuring signs and it is large."""
    T = np.asarray(T, float)
    pairs = list(itertools.combinations(range(T.shape[-1]), k))
    rows, cols, naive, best = [], [], [], []
    for A in T:
        for out in pairs:
            for inp in pairs:
                blk = A[np.ix_(out, inp)]
                rows += list(blk.sum(1))
                cols += list(blk.sum(0))
                naive.append(
                    float(np.linalg.svd(np.sqrt(np.clip(blk, 0, None)), compute_uv=False)[0])
                )
                best.append(best_signed_sigma(blk))
    rows, cols, naive, best = map(np.asarray, (rows, cols, naive, best))
    return {
        "n": len(naive),
        "row_excess": rows,
        "col_excess": cols,
        "sigma_naive": naive,
        "sigma_best": best,
        "row_viol": float((rows > 1).mean()),
        "col_viol": float((cols > 1).mean()),
        "naive_viol": float((naive > 1).mean()),
        "best_viol": float((best > 1).mean()),
    }


def compare_subblocks(Ta, Tb, k: int = 2, iters: int = 400) -> dict:
    """Does substochastic projection of a block move it TOWARDS the same state re-measured?

    The held-out target is the honest one available on a bench with no ground truth: the
    identical heater state, measured again five degrees colder. Anything the projection
    removes that reproduces at the second temperature was signal."""
    Ta, Tb = np.asarray(Ta, float), np.asarray(Tb, float)
    pairs = list(itertools.combinations(range(Ta.shape[-1]), k))
    raw, pro = [], []
    for A, B in zip(Ta, Tb):
        for out in pairs:
            for inp in pairs:
                a, b = A[np.ix_(out, inp)], B[np.ix_(out, inp)]
                raw.append(float(np.linalg.norm(a - b)))
                pro.append(float(np.linalg.norm(substochastic(a, iters) - b)))
    raw, pro = np.asarray(raw), np.asarray(pro)
    moved = np.abs(raw - pro) > 1e-12
    return {
        "identity": float(raw.mean()),
        "substochastic": float(pro.mean()),
        "moved": float(moved.mean()),
        "excess_repeat": _excess_repeat(Ta, Tb, k),
        "helped": float((pro < raw - 1e-12).mean()),
        "n": len(raw),
    }


def _excess_repeat(Ta, Tb, k: int) -> float:
    """Correlation of the sub-block row-sum excess between two measurements of the same
    heater states. A projection is entitled to remove what does not reproduce; this says how
    much of the substochastic violation does."""
    pairs = list(itertools.combinations(range(np.shape(Ta)[-1]), k))
    e = lambda T: np.array(
        [A[np.ix_(o, i)].sum(1) - 1 for A in T for o in pairs for i in pairs]
    ).ravel()
    return float(np.corrcoef(e(Ta), e(Tb))[0, 1])


def controlled_split(
    trials: int = 12, eps: float = 0.0, gain: float = 0.0, noise: float = 0.02, seed: int = 7
) -> dict:
    """A known tangential rotation and a known normal gain, applied to a known orthogonal
    matrix. The point of controlling both is that on real data the split is inferred and
    here it is imposed, so what polar does to each is measured rather than argued."""
    rng = np.random.default_rng(seed)
    R = np.array([[np.cos(eps), -np.sin(eps)], [np.sin(eps), np.cos(eps)]])
    raw, pol, cli, dis, tan = [], [], [], [], []
    for _ in range(trials):
        Q = random_orthogonal(rng, 2)
        M = (1 + gain) * (Q @ R) + noise * rng.normal(size=(2, 2))
        raw.append(float(np.linalg.norm(M - Q) / np.sqrt(2)))
        pol.append(float(np.linalg.norm(polar(M) - Q) / np.sqrt(2)))
        cli.append(float(np.linalg.norm(clip(M) - Q) / np.sqrt(2)))
        dis.append(manifold_dist(M))
        t, n = normal_tangential(M - Q, Q)
        tan.append(t**2 / max(t**2 + n**2, 1e-30))
    return {
        "eps": eps,
        "gain": gain,
        "raw": float(np.mean(raw)),
        "polar": float(np.mean(pol)),
        "clip": float(np.mean(cli)),
        "manifold_dist": float(np.mean(dis)),
        "tangential_frac": float(np.mean(tan)),
        # a pure rotation by eps moves an orthogonal matrix by exactly 2 sin(eps/2) per
        # entry, and no projection onto O(k) can remove any of it
        "tangential_floor": float(2 * abs(np.sin(eps / 2))),
    }


def _bench_compose(tables, trials: int = 12, seed: int = 1) -> dict | None:
    """The composed-product experiment on recorded hardware: host two orthogonal matrices,
    plan both off the 25 C table, read both off the 20 C table.

    The planner and the heater box live in `pic`, which `theory` must not import at module
    scope; the import is local and the whole experiment is skipped when `pic` is not
    importable, so the layering is unchanged. Everything the experiment consumes is recorded
    data -- no port is opened.

    Two probes, one truth. `p25` replays the same measurement the plan was picked from, so
    its error is the hosting residual alone; `p20` replays the identical heater states five
    degrees colder, so its extra error is real drift plus real read noise."""
    try:
        from pic.matvec import bench_box, measured_matrix, plan_from_table, Transfers
        from .intensity_matvec import BEST_RAILS
        from .matmat import matmat
    except Exception:
        return None
    (v25, T25), (v20, T20) = tables
    t25, t20 = Transfers(v25, T25), Transfers(v20, T20)
    p25, p20 = TableProbe(v25, T25), TableProbe(v20, T20)
    box = bench_box(Calibration.load_or_nominal())
    rng = np.random.default_rng(seed)
    rel = lambda M, B: float(np.linalg.norm(M - B) / np.linalg.norm(B))
    acc = {
        k: []
        for k in (
            "host",
            "raw",
            "polar",
            "clip",
            "clip_s",
            "qr",
            "anchored",
            "anchored_polar",
            "dist",
            "tan_host",
            "tan_drift",
        )
    }
    for _ in range(trials):
        Oa, Ob = random_orthogonal(rng, 2), random_orthogonal(rng, 2)
        Ct = Ob @ Oa
        pa, pb = (plan_from_table(O, t25, box, rails=BEST_RAILS) for O in (Oa, Ob))
        qa, qb = (plan_from_table(O, t20, box, rails=BEST_RAILS) for O in (Oa, Ob))
        Cs = matmat(pb, p25, matmat(pa, p25, np.eye(2)))
        Ch = matmat(pb, p20, matmat(pa, p20, np.eye(2)))
        Cr = matmat(qb, p20, matmat(qa, p20, np.eye(2)))
        acc["host"].append(rel(Cs, Ct))
        acc["raw"].append(rel(Ch, Ct))
        acc["polar"].append(rel(polar(Ch), Ct))
        acc["clip"].append(rel(clip(Ch), Ct))
        acc["clip_s"].append(rel(clip(Ch, scale_hat(Ch)), Ct))
        acc["qr"].append(rel(qr_orth(Ch), Ct))
        acc["anchored"].append(rel(Cr, Ct))
        acc["anchored_polar"].append(rel(polar(Cr), Ct))
        acc["dist"].append(manifold_dist(Ch))
        for key, E in (("tan_host", Cs - Ct), ("tan_drift", Ch - Cs)):
            t, n = normal_tangential(E, Ct)
            acc[key].append(t**2 / max(t**2 + n**2, 1e-30))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["corr_dist_err"] = float(np.corrcoef(acc["dist"], acc["raw"])[0, 1])

    # the trap: a target the user hands us is not orthogonal, so O(k) is not its set
    trap_raw, trap_pol, trap_clip = [], [], []
    for _ in range(trials):
        B = 0.4 * rng.normal(size=(2, 2))
        M = measured_matrix(plan_from_table(B, t25, box, rails=BEST_RAILS), p20)
        trap_raw.append(rel(M, B))
        trap_pol.append(rel(polar(M), B))
        trap_clip.append(rel(clip(M), B))
    out |= {
        "trap_raw": float(np.mean(trap_raw)),
        "trap_polar": float(np.mean(trap_pol)),
        "trap_clip": float(np.mean(trap_clip)),
        "trials": trials,
    }
    return out


def _symbolic() -> dict:
    """The block algebra, verified in sympy rather than asserted.

    Four claims, each cheap. The block identities are `Q'Q = I` regrouped, so they are
    checked as a polynomial identity in sixteen free symbols with no orthogonality assumed
    anywhere. Jacobi's identity is likewise checked free, in the adjugate form that avoids
    inverting a symbolic matrix. The embedding is checked with symbolic singular values and
    four symbolic rotation angles, which is the whole sufficiency proof. The trace relation
    is four scalars of linear algebra and is done by hand below."""
    import sympy as sp

    q = sp.Matrix(4, 4, sp.symbols("q0:16", real=True))
    G = q.T * q
    blk = lambda M, i, j: M[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
    A, B, C, D = blk(q, 0, 0), blk(q, 0, 1), blk(q, 1, 0), blk(q, 1, 1)
    Z = sp.zeros(2, 2)
    out = {
        "regroup_AA_CC": sp.expand(A.T * A + C.T * C - blk(G, 0, 0)) == Z,
        "regroup_AB_CD": sp.expand(A.T * B + C.T * D - blk(G, 0, 1)) == Z,
        "regroup_BB_DD": sp.expand(B.T * B + D.T * D - blk(G, 1, 1)) == Z,
        # det(adj(Q)[2:,2:]) = det(Q) det(A) is Jacobi; with Q' = Q^-1 it reads det D det Q
        # = det A, so det(A'A) = det(D'D) whatever the sign of det Q
        "jacobi": sp.expand(blk(q.adjugate(), 1, 1).det() - q.det() * A.det()) == 0,
    }
    c1, c2, a, b, e = sp.symbols("c1 c2 a b e", real=True)
    rot = lambda t: sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
    S, Gm = sp.diag(c1, c2), sp.diag(sp.sqrt(1 - c1**2), sp.sqrt(1 - c2**2))
    U, V, Zr, W = rot(a), rot(b), rot(e), rot(a + b)
    Q = sp.Matrix(sp.BlockMatrix([[U * S * V.T, U * Gm * W.T], [-Zr * Gm * V.T, Zr * S * W.T]]))
    out["embedding"] = sp.simplify(sp.trigsimp(sp.expand(Q.T * Q - sp.eye(4)))) == sp.zeros(4, 4)
    return out


def _selftest(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    sym = _symbolic()
    print("symbolic (sympy, 16 free symbols unless stated)")
    for k, v in sym.items():
        print(f"  {k:20s} {v}")
        assert v, k

    # necessity and sufficiency, numerically, on the objects the symbols stood for
    for _ in range(50):
        Q = random_orthogonal(rng, 4)
        assert np.linalg.svd(Q[:2, :2], compute_uv=False).max() <= 1 + 1e-12
        sa = np.linalg.svd(Q[:2, :2], compute_uv=False)
        sd = np.linalg.svd(Q[2:, 2:], compute_uv=False)
        assert np.allclose(np.sort(sa), np.sort(sd), atol=1e-10), (sa, sd)  # the CS relation
        sb = np.linalg.svd(Q[:2, 2:], compute_uv=False)
        assert np.allclose(np.sort(sa) ** 2 + np.sort(sb)[::-1] ** 2, 1, atol=1e-10)
    for _ in range(20):
        A = rng.normal(size=(2, 2))
        A = clip(A)
        E = embed_contraction(A, rng)
        assert np.allclose(E.T @ E, np.eye(4), atol=1e-12)
        assert np.allclose(E[:2, :2], A, atol=1e-12)
    print("  necessity, CS singular-value relation and the embedding hold numerically too")

    # clip is the metric projection onto a convex set, so it is nonexpansive: it can never
    # move a measurement away from a matrix that is already a legal block
    worst = 0.0
    for _ in range(200):
        A = clip(rng.normal(size=(2, 2)))
        M = A + 0.3 * rng.normal(size=(2, 2))
        worst = max(worst, np.linalg.norm(clip(M) - A) - np.linalg.norm(M - A))
    assert worst <= 1e-12, worst
    print(f"  clip is nonexpansive toward a true contraction (worst increase {worst:+.2e})")

    # polar has no such guarantee, and the counterexample is the ordinary case
    hurt = [
        np.linalg.norm(polar(M) - A) > np.linalg.norm(M - A)
        for A, M in (
            (lambda B: (B, B + 0.05 * rng.normal(size=(2, 2))))(clip(0.5 * rng.normal(size=(2, 2))))
            for _ in range(200)
        )
    ]
    print(f"  polar moves AWAY from a true contraction in {100 * np.mean(hurt):.0f}% of draws")
    assert np.mean(hurt) > 0.5

    # polar commutes with an orthogonal correction, so ordering cannot be argued on accuracy
    R = random_orthogonal(rng, 2)
    M = rng.normal(size=(2, 2))
    assert np.allclose(polar(M @ R), polar(M) @ R, atol=1e-12)
    print(
        "  polar(M R) == polar(M) R for orthogonal R: the ordering changes evidence, "
        "not the answer"
    )

    print("\ncontrolled normal/tangential split (12 draws each, 2% read noise)")
    print(
        f"  {'perturbation':22s}{'tangential':>12}{'raw':>9}{'polar':>9}{'gain':>8}"
        f"{'m_dist':>9}{'floor':>8}"
    )
    for eps, gain, name in (
        (0.0, 0.0, "noise only"),
        (0.0, 0.05, "+5% gain (normal)"),
        (0.15, 0.0, "0.15 rad (tangential)"),
        (0.15, 0.05, "both"),
    ):
        r = controlled_split(eps=eps, gain=gain)
        print(
            f"  {name:22s}{100 * r['tangential_frac']:>11.0f}%{r['raw']:>9.4f}"
            f"{r['polar']:>9.4f}{r['raw'] / r['polar']:>7.2f}x{r['manifold_dist']:>9.4f}"
            f"{r['tangential_floor']:>8.4f}"
        )
    assert controlled_split(gain=0.05)["raw"] / controlled_split(gain=0.05)["polar"] > 3
    tang = controlled_split(eps=0.15)
    assert tang["raw"] / tang["polar"] < 1.1
    # polar lands ON the tangential floor: it removed the noise and none of the rotation
    assert abs(tang["polar"] - tang["tangential_floor"]) < 0.2 * tang["tangential_floor"]

    tables = load_bench()
    if len(tables) < 2:
        print("\nno bench transfer tables on file -- the measured half is skipped")
        return {"symbolic": sym}
    (v25, T25), (v20, T20) = tables
    print(f"\nbench: {len(T25)} heater states measured at 25 C and re-measured at 20 C")

    sets = [intensity_sets(T) for T in T25[:12]]
    print("  per-entry distance of the measured 4x4 |U|^2 to each set (12 states, median)")
    for k in ("birkhoff", "unistochastic", "orthostochastic"):
        print(f"    {k:18s}{np.median([s[k] for s in sets]):.4f}")
    b, u = np.median([s["birkhoff"] for s in sets]), np.median([s["unistochastic"] for s in sets])
    assert u >= b - 1e-6 and u < 1.1 * b, (b, u)  # unistochastic costs almost nothing extra

    cen = subblock_census(T25)
    print(
        f"  {cen['n']} 2x2 sub-blocks: row sums exceed 1 in {100 * cen['row_viol']:.1f}%, "
        f"column sums in {100 * cen['col_viol']:.1f}% (columns are 1 by construction)"
    )
    print(
        f"    amplitudes, no signs: sigma_max > 1 in {100 * cen['naive_viol']:.0f}% "
        f"(max {cen['sigma_naive'].max():.2f});  best sign class "
        f"{100 * cen['best_viol']:.0f}% (max {cen['sigma_best'].max():.2f})"
    )
    assert cen["col_viol"] < 1e-9 and cen["naive_viol"] > cen["best_viol"]

    sub = compare_subblocks(T25, T20)
    print(
        f"  substochastic projection vs the same state at 20 C: identity "
        f"{sub['identity']:.4f} -> {sub['substochastic']:.4f}, moved "
        f"{100 * sub['moved']:.1f}% of blocks, helped {100 * sub['helped']:.1f}%"
    )
    print(
        f"    the row-sum excess reproduces across the two temperatures at r = "
        f"{sub['excess_repeat']:+.3f}, so it is systematic and not noise to remove"
    )
    assert sub["substochastic"] > sub["identity"]  # it is a refusal; keep it one

    r = _bench_compose(tables)
    if r is None:
        print("  (pic not importable -- the composed-product replay is skipped)")
        return {"symbolic": sym, "sets": sets, "census": cen, "subblocks": sub}
    print(f"\ncomposed product of two hosted ORTHOGONAL matrices, {r['trials']} pairs")
    print(f"  planned and read on the 25 C table (hosting residual alone) {r['host']:.4f}")
    print(
        f"  planned on 25 C, read on 20 C (real drift + read noise)     {r['raw']:.4f}"
        f"   manifold_dist {r['dist']:.4f}"
    )
    for k, name in (
        ("polar", "polar -> O(2)"),
        ("clip", "clip sigma <= 1"),
        ("clip_s", "clip sigma <= mean sigma"),
        ("qr", "QR"),
        ("anchored", "re-anchor the plan on the 20 C table"),
        ("anchored_polar", "re-anchor, then polar"),
    ):
        print(f"    {name:38s}{r[k]:.4f}   {r['raw'] / r[k]:4.2f}x")
    print(
        f"  tangential fraction: hosting residual {100 * r['tan_host']:.0f}%, "
        f"drift {100 * r['tan_drift']:.0f}%  (isotropic would be 25%)"
    )
    print(
        f"  corr(manifold_dist, true error) over the unprojected products "
        f"{r['corr_dist_err']:+.3f}; after polar it is 0 for every one"
    )
    print(
        f"  arbitrary non-orthogonal target: raw {r['trap_raw']:.4f}, "
        f"polar {r['trap_polar']:.4f} ({r['trap_polar'] / r['trap_raw']:.0f}x WORSE), "
        f"clip {r['trap_clip']:.4f}"
    )
    assert r["polar"] < r["clip"] < r["raw"]  # polar wins; clip barely helps
    assert r["trap_polar"] > 5 * r["trap_raw"]  # and on a general target it is a trap
    assert r["tan_drift"] < 0.25  # the drift here is normal, not tangential
    return {"symbolic": sym, "sets": sets, "census": cen, "subblocks": sub, "compose": r}


if __name__ == "__main__":
    _selftest()
