"""Ising energies on the 4x4 unitary mesh.

What this chip can compute is decided by how it is read. Light enters one input port at a
time through the 1x4 switch and leaves into four photodiodes, so one shot is one column of

    T[j, k] = |U[j, k]|^2

and four switch positions return the whole intensity transfer matrix. T is doubly
stochastic because U is unitary, and only its symmetric part S = (T + T^T)/2 can enter a
quadratic form. The n(n-1)/2 off-diagonal entries of S are exactly the degrees of freedom
of an n-spin coupling matrix, so the mesh *hosts* J and the spin contraction is arithmetic
on the four measured columns:

    s^T J s  =  [ 2 sum_{i<j} S_ij s_i s_j  -  b ((1.s)^2 - n) ] / a

where (a, b) is the affine map fitting S's off-diagonal onto J's. Neither half of that
gauge costs anything. J's diagonal is a constant because s_i^2 = 1, so double stochasticity
pinning S's diagonal is harmless; and b only adds a uniform ferromagnetic term in (1.s)^2,
which is known and subtracted. b is also the only reason a non-negative S can host
antiferromagnetic couplings at all.

**The 6x6's power route does not port.** That chip launched the spin vector as 0/pi input
phases through a splitting tree and read the energy off the total monitor power. This one
has a switch, not a splitter: no coherent superposition of input ports exists, so the spins
cannot ride in on the light. What survives is the better-conditioned half of the same idea
-- four shots score the entire 2^n landscape, because once T is measured every
configuration is a dot product against the same sixteen numbers.

**The reachable set is small, and that is the whole difficulty.** A heater only *adds*
phase, phi = pi (V/Vpi)^2 + phi0, and at the 40 mA ceiling the measured spans run 0.18 to
1.51 pi where a free phase needs 2 pi. So a target cannot be inverted -- `theory.program`
hands back phases the chip refuses -- and `encode` searches the reachable set by gradient
descent on the heater voltages through the differentiable twin. That is what the 6x6 did
through its surrogate, for the same reason and a different cause.

**Contrast is not a side condition, it is half the objective.** Whatever survives the
hosting fit still has to be read off a photodiode, so the error in the recovered couplings
is the unhosted residual *and* the readout noise, both divided by the hosted swing:

    eps = sqrt( rel_err^2 + (sigma / contrast)^2 )

`noise_curve` measures the ground-state rate against that one number and finds it very
nearly independent of the split, which is what makes it the thing to minimise; `encode`
minimises it directly. Most of the contrast that buys turns out to cost no fidelity at all
-- the old objective simply never asked for it. Trading fidelity for contrast *beyond* that
is worth two or three points of ground-state rate at the noisiest readout measured and costs
fifteen at the quietest, which is why `SIGMA_DESIGN` is a hedge and not the bench's number.

The one way to raise contrast without paying for it in fidelity is to spend shots: `passes`
designs several held states and hosts J in their signed difference, which is unbounded below
zero where a single doubly stochastic matrix is not. See `hosted`.

    enc = encode(J, twin, calib)              # design: volts the chip can actually hold
    T   = measure_the_four_ports(enc.volts)   # or twin_transfer(twin, enc.phases)
    res = decode(T, J, modes=enc.modes)       # res["spins"], res["energies"], res["fit"]

`enc.modes` is not decoration. Which optical pair hosts which coupling is free, `encode`
searches it per instance, and the decode is wrong without it.

n is at most NMODE = 4. There are four optical modes and no way to embed a fifth spin.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .calib import VOLTAGE_MAX, VPI_NOMINAL, Calibration
from .clements import NMODE
from .layout import N_HEATERS

NSPIN_MAX = NMODE

# The uniform doubly stochastic matrix is the image of J = 0: every coupling equal is no
# coupling at all, once the (1.s)^2 term it produces has been subtracted. Carrying it as one
# extra pseudo-observation makes the (a, b) fit determined at n = 2, where a single measured
# coupling cannot otherwise separate scale from offset.
ANCHOR = 1.0 / NMODE
ANCHOR_WEIGHT = 1.0

# Per-entry noise on a normalised T. The bench reads ~0.3 mV RMS into a ~0.33 V full scale
# (pic.sim, MockPIC), and the four-port probe averages three port cycles, so ~1e-3 of full
# scale per entry is the honest photodiode number.
SIGMA_READ = 1e-3

# ...and it is not the number a design has to survive. On the 2026-08-27 bench the residual
# left after hosting was 0.075 per entry at n = 3 and 0.118 at n = 4, in doubly stochastic
# units -- seventy to a hundred times SIGMA_READ -- against designs whose *predicted* error
# was 0.001. That gap is the twin disagreeing with the chip, and it enters the decode at
# exactly the point read noise does: divided into the hosted swing. So it is what `encode`
# budgets against, and SIGMA_READ is not.
#
# 0.03 rather than the measured 0.08 because the budget is a hedge, not an estimate. Over
# 160 instances per cell, designing at 0.03 was the best or within one point of the best at
# every readout noise from 1e-3 to 0.12, while budgeting 0.1 cost 7 points of ground-state
# rate at n = 4 on a quiet readout and bought nothing on a loud one -- the objective is flat
# in `sigma` across 0.01 to 0.05 and falls off outside it. Budget for the middle of the
# range the chip might be in, not for its worst corner, because over-budgeting spends
# hosting fidelity that cannot be recovered if the readout turns out to be quiet.
SIGMA_DESIGN = 0.03

# What the couplings were actually wrong by on the bench: the 2026-08-27 run's rel_err times
# its contrast, per entry. `noise_curve` divides an eps target by this to state a contrast
# target, so it is the number to update when the twin stops missing the chip.
SIGMA_BENCH = 0.08

# Below this the hosted swing is comparable to the readout noise and the energies are a
# noise ranking. It is a floor on the *design*, not a fitted threshold: 1e-2 in doubly
# stochastic units is ten times SIGMA_READ.
MIN_CONTRAST = 1e-2


def measured(calib) -> np.ndarray:
    """Channels whose heater law was actually fitted, as a boolean over the DAC channels.

    A characterization leaves the nominal Vpi and a zero phi0 on every heater it failed to
    fit, and on this die the nominal is off by a factor of three -- so an unfitted channel
    advertises a 4 pi swing it cannot make. Letting the optimiser plan through one is the
    difference between a design that works and a design that only looks like it does."""
    return ~(
        np.isclose(np.asarray(calib.vpi, float), VPI_NOMINAL)
        & (np.asarray(calib.phi0, float) == 0.0)
    )


def movers(twin, phases, trainable=None, tol: float = 1e-6) -> np.ndarray:
    """Channels whose phase actually changes |U|^2 at this operating point.

    Intensity is blind to the input and output phase screens, so some wired, characterized
    heaters move nothing a photodiode can see. Measured rather than read off the role
    labels, because `theory.layout` says itself that the phi/alpha split is provisional."""
    import torch

    def grad_at(offset):
        # sum |U|^2 is the constant NMODE for any unitary, so it has no gradient at all;
        # the sum of squares is the cheapest scalar that actually tracks where the power went
        ph = (
            torch.as_tensor(np.asarray(phases, float) + offset, dtype=torch.float32)
            .clone()
            .requires_grad_(True)
        )
        ((twin.matrix(ph).abs() ** 2) ** 2).sum().backward()
        return ph.grad.abs().numpy()

    # a zero gradient at one point can be a stationary point rather than a blind direction,
    # so probe a second, offset point and keep a channel that moves at either
    m = (grad_at(0.0) > tol) | (grad_at(0.7) > tol)
    return m if trainable is None else m & np.asarray(trainable, bool)


def random_ising(rng, n: int = NSPIN_MAX, kind: str = "sk") -> np.ndarray:
    """Symmetric zero-diagonal coupling matrix, spectrally normalised.

    'sk' is Sherrington-Kirkpatrick gaussian; anything else gives +-1 couplings."""
    if not 2 <= n <= NSPIN_MAX:
        raise ValueError(f"n must be 2..{NSPIN_MAX}; this mesh has {NMODE} modes")
    J = rng.normal(size=(n, n)) if kind == "sk" else rng.choice([-1.0, 1.0], (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    return J / np.linalg.norm(J, 2)


def energy(J, s) -> float:
    return -0.5 * float(np.asarray(s, float) @ np.asarray(J, float) @ np.asarray(s, float))


def configs(n: int) -> np.ndarray:
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n)))


def brute_force(J):
    """Every configuration with its energy, sorted ascending."""
    cfgs = configs(len(J))
    E = np.array([energy(J, s) for s in cfgs])
    order = np.argsort(E)
    return cfgs[order], E[order]


def offdiag(M, n: int | None = None) -> np.ndarray:
    """Upper-triangle of the symmetrised n x n block: the couplings a matrix carries."""
    M = np.asarray(M, float)
    n = M.shape[-1] if n is None else n
    S = (M[..., :n, :n] + np.swapaxes(M, -1, -2)[..., :n, :n]) / 2
    iu = np.triu_indices(n, 1)
    return S[..., iu[0], iu[1]]


def mode_maps(n: int, nmode: int = NMODE):
    """Every distinct way of laying n spins onto the mesh's modes, as gather indices.

    Which optical pair carries which coupling is free and it is not a symmetry. The heaters
    reach different swings in different entries of |U|^2 -- `intensity_matvec.scan_rails`
    asks the same question of a matrix block and finds brightness spanning 0.007 to 0.96
    across choices -- so the assignment changes both how well J is hosted and how brightly. At
    n < NMODE it also chooses which modes to leave out, and a left-out mode is a sink the
    others can push power into rather than a wasted port.

    Worth 16 percent of contrast at n = 3 and 15 at n = 4 over 96 instances, on top of
    whatever the objective finds -- and it costs one gather per assignment, not one twin
    evaluation, so `encode` scores all 24 at every step of every restart.

    Returns (rows, cols), each (P, m); row p is one assignment, and the pair
    (rows[p, s], cols[p, s]) is the entry of |U|^2 that hosts coupling s. Deduplicated,
    because S is symmetric: 6 assignments at n = 2 and 24 at n = 3 and n = 4."""
    iu = np.triu_indices(n, 1)
    seen, rows, cols = set(), [], []
    for p in itertools.permutations(range(nmode), n):
        r, c = np.asarray(p)[iu[0]], np.asarray(p)[iu[1]]
        key = tuple(zip(np.minimum(r, c).tolist(), np.maximum(r, c).tolist()))
        if key not in seen:
            seen.add(key)
            rows.append(r)
            cols.append(c)
    return np.array(rows, int), np.array(cols, int)


def couplings(T, modes=None, n: int | None = None) -> np.ndarray:
    """The entries of a transfer matrix that host J, under one mode assignment.

    `modes` is a (rows, cols) pair naming the entry per coupling -- one row of `mode_maps`.
    None is the natural leading block, which is what `offdiag` returns. Batched over any
    leading dimensions, like `offdiag`."""
    if modes is None:
        return offdiag(T, n)
    T = np.asarray(T, float)
    S = (T + np.swapaxes(T, -1, -2)) / 2
    return S[..., np.asarray(modes[0]), np.asarray(modes[1])]


def hosted(T, modes=None, n: int | None = None) -> np.ndarray:
    """What one measurement offers the fit: the signed combination of its passes.

    A single (NMODE, NMODE) matrix is `couplings` unchanged. A stack of `passes` matrices is
    contracted with alternating signs, so a two-pass design hosts J in S1 - S2. Every entry
    of a doubly stochastic matrix is non-negative, which is why a single pass has to buy its
    antiferromagnetic couplings out of the offset b; a difference is signed to begin with,
    ranges over [-1, 1] instead of [0, 1], and its reachable set is the difference of the
    reachable set with itself -- a far larger object than the set. Measured on the twin at
    n = 4: two passes reach 2.1x the contrast at a third of the hosting error, three reach
    3.2x. Readout noise grows only as sqrt(passes), so the trade keeps paying long after the
    optical time stops being free.

    Unlike `couplings` this does not batch: the leading axis *is* the pass index."""
    T = np.asarray(T, float)
    if T.ndim == 2:
        return couplings(T, modes, n)
    sgn = (-1.0) ** np.arange(T.shape[0])
    return (sgn[:, None] * couplings(T, modes, n)).sum(0)


def _anchor_weight(m: int) -> float:
    """How hard to lean on the J = 0 pseudo-observation, given m measured couplings.

    It buys a determined fit at m = 1 and costs accuracy everywhere else. `b` is not a
    nuisance: it sets how much uniform ferromagnetic term `config_energies` subtracts, so a
    prior on it is a prior on the answer -- and nothing makes the mesh put the hosted block's
    mean at 1/NMODE. At m = 1 there is no choice; at m >= 3 there is, and the measurements
    are better evidence than the guess.

    It also *costs contrast*, which is what makes this worth doing rather than tidy. Held at
    weight 1 the fit is pulled toward b = 1/NMODE, so a design whose hosted block sits well
    away from uniform scores badly however cleanly it carries J -- and those are the bright
    ones. Switching it off at n = 3, where it carried a quarter of the fit, took the reachable
    contrast from 0.263 to 0.592 and the ground-state rate at bench noise from 0.911 to 0.957
    over 96 instances. At n = 4 it is one observation in seven and worth about a tenth of
    that."""
    return ANCHOR_WEIGHT if m < 2 else 0.0


def _affine(x, y, w: float | None = None, anchor: float = ANCHOR):
    """Least-squares y ~ a x + b with one anchored pseudo-observation at (0, `anchor`).

    Written with `.sum(-1)` only, so the same closed form serves the numpy decode and the
    batched torch encode without a second implementation drifting away from the first."""
    m = x.shape[-1]
    w = _anchor_weight(m) if w is None else w
    sw = m + w
    sx, sxx = x.sum(-1), (x * x).sum(-1)
    sy = y.sum(-1) + w * anchor
    sxy = (y * x).sum(-1)
    det = sw * sxx - sx * sx
    return (sw * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det


def host(T, J, modes=None):
    """Fit the transfer matrix's couplings onto the target's.

    Returns (a, b, rel_err, contrast). `contrast` is the RMS hosted swing in doubly
    stochastic units -- what has to beat the readout noise -- and `rel_err` is the residual
    as a fraction of it, which is the part of J the chip failed to host."""
    J = np.asarray(J, float)
    x = offdiag(J)
    y = hosted(T, modes, len(J))
    a, b = _affine(x, y)
    a, b = float(a), float(b)
    resid = y - (a * x + b)
    contrast = abs(a) * float(np.sqrt(np.mean(x * x)))
    rel = float(np.sqrt(np.mean(resid * resid))) / max(contrast, 1e-15)
    return a, b, rel, contrast


def config_energies(T, J, fit=None, cfgs=None, modes=None) -> np.ndarray:
    """Ising energy of every configuration, read out of one measured transfer matrix.

    No further optical measurement: the 2^n landscape is 2^n dot products against the same
    n(n-1)/2 couplings the four shots already returned."""
    J = np.asarray(J, float)
    n = len(J)
    a, b = host(T, J, modes)[:2] if fit is None else fit
    S = hosted(T, modes, n)
    cfgs = configs(n) if cfgs is None else np.asarray(cfgs, float)
    iu = np.triu_indices(n, 1)
    q = 2.0 * (cfgs[:, iu[0]] * cfgs[:, iu[1]]) @ S
    ferro = cfgs.sum(1) ** 2 - n  # what a non-zero b adds, and nothing else
    return -0.5 * (q - b * ferro) / a


def normalise(T, iters: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Strip per-port coupling and per-detector gain by projecting onto doubly stochastic.

    Exactly the right tool, not an approximation: |U|^2 is doubly stochastic and the whole
    nuisance is two diagonals, so the projection removes all of it and nothing else. With
    per-port and per-detector gains of 0.35 in log units injected, the decode returns the
    identical spins and a hosting error identical to four decimals; skip it and that error
    is five to thirteen times worse. `_selftest` asserts both.

    Safe here and only here: the Ising probe is a *bright* one by construction, because
    dimness enters `encode`'s objective as sigma/contrast and a design that the photodiodes
    cannot resolve loses to one that hosts J badly. On a dim probe this is the wrong tool and
    `theory.drift.fit_gains` against a reference is the right one -- see SINKHORN_MIN_SNR
    there.

    Sinkhorn converges linearly, losing about a factor 7 per ten sweeps on a 4x4, so it
    stops when the row and column sums are unity far below the read noise rather than
    burning the full `iters` every call. `decode` is invoked once per configuration study
    and the study is where the time goes."""
    S = np.clip(np.asarray(T, float), 1e-12, None)
    for _ in range(iters):
        S = S / S.sum(-1, keepdims=True)
        S = S / S.sum(-2, keepdims=True)
        if np.abs(S.sum(-1) - 1).max() < tol:
            break
    return S


def decode(T, J, normalised: bool = True, modes=None) -> dict:
    """Measured transfer matrix -> the spin configuration the chip calls the ground state.

    `T` is one probe: a (NMODE, NMODE) matrix, or a stack of them for a multi-pass design,
    in which case each pass is projected onto doubly stochastic separately -- the nuisance
    gains are the same two diagonals but the light went to different places."""
    T = normalise(T) if normalised else np.asarray(T, float)
    a, b, rel, contrast = host(T, J, modes)
    cfgs = configs(len(J))
    E = config_energies(T, J, fit=(a, b), cfgs=cfgs, modes=modes)
    k = int(np.argmin(E))
    return {
        "spins": cfgs[k],
        "index": k,
        "energies": E,
        "configs": cfgs,
        "modes": modes,
        "a": a,
        "b": b,
        "rel_err": rel,
        "contrast": contrast,
        "T": T,
    }


def anneal(
    energy_fn, n: int, rng, restarts: int = 8, sweeps: int = 40, t0: float = 1.0, t1: float = 0.01
):
    """Metropolis single-flip descent against a chip-derived energy oracle.

    At n <= 4 the oracle is 2^n dot products and exhaustion is cheaper, so this exists to
    be the shape the loop keeps when the oracle is not free -- a larger mesh, or a J that
    has to be re-hosted between sweeps. Returns (spins, energy)."""
    best = None
    for _ in range(restarts):
        s = rng.choice([-1.0, 1.0], n)
        e = float(energy_fn(s))
        for k in range(sweeps):
            temp = t0 * (t1 / t0) ** (k / max(sweeps - 1, 1))
            for i in rng.permutation(n):
                t = s.copy()
                t[i] = -t[i]
                et = float(energy_fn(t))
                if et <= e or rng.random() < np.exp(-(et - e) / temp):
                    s, e = t, et
        if best is None or e < best[1]:
            best = (s.copy(), e)
    return best


@dataclass
class Encoding:
    """A programmed design: the volts to hold, and what the twin says they host."""

    volts: np.ndarray
    phases: np.ndarray
    T: np.ndarray
    a: float
    b: float
    rel_err: float
    contrast: float
    n: int
    modes: tuple | None = None
    sigma: float = SIGMA_DESIGN
    passes: int = 1

    @property
    def eps(self) -> float:
        """Relative error of the coupling matrix the decode will recover: what the reachable
        set could not host and what the readout will destroy, added in quadrature because
        they are the same error in the same units. This is what `encode` minimises and what
        `noise_curve` converts into a success rate."""
        return float(np.hypot(self.rel_err, self.sigma / max(self.contrast, 1e-15)))

    @property
    def ok(self) -> bool:
        return self.contrast >= MIN_CONTRAST

    def __str__(self) -> str:
        return (
            f"n={self.n} host err {self.rel_err:.3f} contrast {self.contrast:.4f} "
            f"eps {self.eps:.3f}{'' if self.ok else '  BELOW NOISE'}"
        )


def twin_transfer(twin, phases) -> np.ndarray:
    """|U|^2 for a phase vector: what the four-port probe would return, noise-free."""
    import torch

    ph = (
        phases
        if torch.is_tensor(phases)
        else torch.as_tensor(np.asarray(phases, float), dtype=torch.float32)
    )
    with torch.no_grad():
        return (twin.matrix(ph).abs() ** 2).numpy().astype(float)


def encode(
    J,
    twin,
    calib: Calibration | None = None,
    *,
    trainable=None,
    vmax=VOLTAGE_MAX,
    restarts: int = 24,
    steps: int = 300,
    lr: float = 0.15,
    sigma: float = SIGMA_DESIGN,
    passes: int = 1,
    seed: int = 0,
) -> Encoding:
    """Search the reachable set for heater volts whose |U|^2 hosts J.

    `vmax` may be a scalar or a per-channel array; a channel at 0 is pinned dark, which is
    how the three heaters with no confirmed resistance stay out of a design. The restarts
    run as one batch through the twin, which is why this costs a second and not a minute.

    `passes` designs that many heater states whose signed combination hosts J -- see
    `hosted`. It multiplies the optical cost by the same factor and is off by default, so
    the four-shot claim in the module docstring stays true unless someone asks for more.

    The objective is the relative error of the coupling matrix the decoder will end up
    inverting,

        eps = sqrt( rel^2 + (sigma / contrast)^2 )

    and both terms are the same error in the same units: `rel` is the part of J the
    reachable set could not host, over the hosted swing, and `sigma / contrast` is the part
    the readout will destroy, over the same swing. Neither means anything alone. A perfect
    hosting at an amplitude the photodiodes cannot resolve decodes to noise; a bright design
    that hosts the wrong matrix decodes confidently to the wrong answer. Adding them in
    quadrature is what fixes the exchange rate between them, and `noise_curve` shows the
    success rate is a function of the sum and very nearly not of the split.

    This replaces a hinge that defended `MIN_CONTRAST` and was flat above it, so the
    optimiser bought no brightness it was not forced to and left most of the reachable swing
    on the table. Over 64 paired instances under the bench's own per-channel ceilings, the
    hinge reached 0.117 of contrast at n = 4 where this reaches 0.201, and 0.171 at n = 3
    against 0.795 -- and it is not a trade, because the hosting error fell too, 0.115 to
    0.033 at n = 4. The ground-state rate at the bench's own noise went 0.677 to 0.813 at
    n = 4 and 0.911 to 0.981 at n = 3, and the energy correlation 0.912 to 0.976.

    Most of that is not the noise term. Switching the anchor off and searching the mode
    assignment account for it at n = 3, where the anchor alone was worth 2.2x the contrast;
    at n = 4 the three contribute more evenly.

    `sigma` is a design decision, not a photodiode property -- see `SIGMA_DESIGN`."""
    import torch

    J = np.asarray(J, float)
    n = len(J)
    calib = Calibration.load_or_nominal() if calib is None else calib
    vmax = np.broadcast_to(np.asarray(vmax, float), (N_HEATERS,)).astype(float)
    mask = np.ones(N_HEATERS, bool) if trainable is None else np.asarray(trainable, bool)
    mask = mask & (vmax > 0)

    xj = offdiag(J)
    rms_x = float(np.sqrt(np.mean(xj**2)))
    x = torch.as_tensor(xj, dtype=torch.float32)
    vt = torch.as_tensor(vmax, dtype=torch.float32)
    mt = torch.as_tensor(mask.astype(np.float32))
    vpi = torch.as_tensor(calib.vpi, dtype=torch.float32)
    phi0 = torch.as_tensor(calib.phi0, dtype=torch.float32)
    rows, cols = mode_maps(n)
    ri, ci = torch.as_tensor(rows), torch.as_tensor(cols)
    # every pass carries its own noise into the difference, so the budget the design has to
    # beat grows as sqrt(passes) while the contrast it can reach grows faster
    sig = float(sigma) * np.sqrt(passes)
    sgn = torch.as_tensor([(-1.0) ** k for k in range(passes)])

    g = torch.Generator().manual_seed(seed)
    # not a lever, and measured rather than assumed: against this, a wider gaussian, uniform
    # in voltage and uniform in phase all landed within one percent of each other on eps over
    # 48 instances at both n. `restarts` covers the basins; where they start does not.
    u = torch.randn(restarts, passes, N_HEATERS, generator=g) * 1.5
    u.requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    # the reachable set is a thin curved sheet and Adam at a fixed step orbits its optimum
    # rather than settling on it; annealing the step is worth a few percent of contrast free
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 20)

    def evaluate(u):
        v = mt * vt * torch.sigmoid(u)
        ph = np.pi * (v / vpi) ** 2 + phi0
        T = twin.matrix(ph).abs() ** 2
        S = (T + T.transpose(-1, -2)) / 2
        y = (sgn[None, :, None, None] * S[..., ri, ci]).sum(1)  # (restarts, maps, couplings)
        a, b = _affine(x, y)
        resid = y - (a[..., None] * x + b[..., None])
        contrast = a.abs() * rms_x
        # clamp inside the sqrt, not after. At n = 2 the fit is exact, and sqrt has an
        # infinite derivative at zero: the unclamped form sent every voltage to NaN and the
        # design silently became whatever argmin does with an all-NaN array.
        rel = resid.pow(2).mean(-1).clamp_min(1e-24).sqrt() / contrast.clamp_min(1e-12)
        eps = (rel**2 + (sig / contrast.clamp_min(1e-12)) ** 2).clamp_min(1e-24).sqrt()
        return v, a, b, rel, contrast, eps

    for _ in range(steps):
        opt.zero_grad()
        *_, eps = evaluate(u)
        eps.min(-1).values.sum().backward()  # each restart keeps its own best assignment
        opt.step()
        sched.step()

    with torch.no_grad():
        v, _, _, _, _, eps = evaluate(u)
        eps = torch.where(torch.isfinite(eps), eps, torch.as_tensor(float("inf")))
        r, p = divmod(int(torch.argmin(eps)), eps.shape[1])
        volts = v[r].numpy().astype(float)

    modes = (rows[p], cols[p])
    phases = np.stack([calib.phases(w) for w in volts])
    T = np.stack([twin_transfer(twin, ph) for ph in phases])
    if passes == 1:
        volts, phases, T = volts[0], phases[0], T[0]
    a, b, rel_err, contrast = host(T, J, modes)
    return Encoding(volts, phases, T, a, b, rel_err, contrast, n, modes, sig, passes)


def reach_dim(
    twin,
    calib,
    trainable=None,
    vmax=VOLTAGE_MAX,
    samples: int = 4000,
    seed: int = 0,
    n: int = NSPIN_MAX,
) -> dict:
    """How much of coupling-shape space the heaters can actually steer through.

    The readout gauge (a, b) makes only the *direction* of the off-diagonal vector matter,
    so the target of an n-spin problem is a point on a sphere of dimension m - 2 with
    m = n(n-1)/2: one angle at n = 3, four at n = 4. This samples the voltage box, strips
    the gauge, and returns the participation dimension of what is left -- the number of
    independent directions the mesh can actually swing through. It is the single number the
    n = 3 / n = 4 split falls out of, and it needs no target and no optimisation.

    Every mode assignment is pooled in, because `encode` may pick any of them and a target
    is reachable if *some* assignment reaches it. That is not double counting: permuting the
    coordinates of the reachable set and asking whether it covers x is the same question as
    asking whether x's orbit meets the unpermuted set."""
    import torch

    rng = np.random.default_rng(seed)
    vm = np.broadcast_to(np.asarray(vmax, float), (N_HEATERS,)).astype(float)
    m = np.ones(N_HEATERS, bool) if trainable is None else np.asarray(trainable, bool)
    V = rng.uniform(0, 1, (samples, N_HEATERS)) * vm * m
    with torch.no_grad():
        T = (twin.matrix(torch.as_tensor(calib.phases(V), dtype=torch.float32)).abs() ** 2).numpy()
    rows, cols = mode_maps(n)
    Z = np.concatenate([couplings(T, (rows[p], cols[p])) for p in range(len(rows))])
    Z = Z - Z.mean(-1, keepdims=True)  # b is a shift, so only the residual counts
    nrm = np.linalg.norm(Z, axis=-1)
    D = Z[nrm > 1e-6] / nrm[nrm > 1e-6, None]  # a is a scale, so only the direction counts
    var = np.linalg.svd(D, compute_uv=False) ** 2
    var = var / var.sum()
    return {
        "swing": float(nrm.mean()),
        "spectrum": var,
        "n": n,
        "dim": float(np.exp(-(var * np.log(var + 1e-300)).sum())),
        "of": n * (n - 1) // 2 - 1,
    }


def noise_curve(
    twin,
    calib,
    *,
    n: int = NSPIN_MAX,
    trainable=None,
    vmax=VOLTAGE_MAX,
    instances: int = 8,
    sigmas=(0.001, 0.01, 0.02, 0.05, 0.08, 0.12),
    design_sigmas=(SIGMA_READ, 0.1),
    draws: int = 16,
    bins: int = 6,
    targets=(0.95, 0.9, 0.8),
    steps: int = 300,
    restarts: int = 24,
    seed: int = 0,
) -> dict:
    """Ground-state rate against the recovered coupling error -- and the contrast it demands.

    A design is only ever wrong by `eps = hypot(rel_err, sigma / contrast)`, and the point
    of the sweep is that the success rate collapses onto that one number: designs reached by
    trading hosting fidelity for brightness sit on the same curve as designs reached by
    turning the noise down. That is what licenses `encode`'s objective, and it is what makes
    a bare contrast figure meaningless -- 0.09 is excellent at SIGMA_READ and hopeless at
    the bench's own 0.08.

    `design_sigmas` spreads the designs along the trade, so the collapse is measured rather
    than assumed. The chip here is the twin itself, so every error is readout: this is the
    curve for the error budget, not a prediction of a bench run.

    Returns the binned curve, `threshold[t]` = the largest eps still holding rate `t`, and
    `contrast_at[t]` = the contrast that demands once the hosting residual has taken its
    share of the same budget. Read it as: to hold n = 4 at 90 percent the design needs eps
    below X, which at the bench's own 0.08 per entry means this much contrast."""
    pts = []
    for i in range(instances):
        J = random_ising(np.random.default_rng(seed + 977 * n + i), n=n)
        cfgs_true, E_true = brute_force(J)
        for d, sd in enumerate(design_sigmas):
            enc = encode(
                J,
                twin,
                calib,
                trainable=trainable,
                vmax=vmax,
                sigma=sd,
                steps=steps,
                restarts=restarts,
                seed=seed + i,
            )
            rng = np.random.default_rng(seed + 31 * i)
            for s in sigmas:
                hit = sum(
                    _score(
                        np.clip(enc.T + rng.normal(0, s, enc.T.shape), 0.0, None),
                        J,
                        cfgs_true,
                        E_true,
                        modes=enc.modes,
                    )["found_gs"]
                    for _ in range(draws)
                )
                pts.append(
                    (
                        float(np.hypot(enc.rel_err, s / max(enc.contrast, 1e-15))),
                        hit / draws,
                        enc.rel_err,
                        enc.contrast,
                        d,
                    )
                )
    pts = np.array(pts)
    eps, rate, which = pts[:, 0], pts[:, 1], pts[:, 4]
    edges = np.quantile(eps, np.linspace(0, 1, bins + 1))
    cells, spread = [], 0.0
    for i in range(bins):
        m = (eps >= edges[i]) & (eps <= edges[i + 1])
        if not m.any():
            continue
        cells.append((float(edges[i]), float(edges[i + 1]), int(m.sum()), float(rate[m].mean())))
        # the collapse claim, checked rather than asserted: designs that reached this eps by
        # trading fidelity for brightness must score the same as designs that reached it by
        # a quieter readout
        per = [
            rate[m & (which == d)].mean()
            for d in range(len(design_sigmas))
            if (m & (which == d)).sum() >= 4
        ]
        if len(per) > 1:
            spread = max(spread, float(np.ptp(per)))
    # monotone by construction: the threshold is the last eps at which the rate has held,
    # not the first bin that happens to dip. A ten-bin curve off a few instances is noisy
    # enough to dip and recover, and reading the dip as a threshold reports a bin, not a limit
    thr = {}
    for t in targets:
        held = float(edges[0])
        for lo, hi, _k, r in cells:
            if r < t:
                break
            held = hi
        thr[t] = held
    rel = float(pts[:, 2].mean())
    # eps^2 = rel^2 + (sigma/contrast)^2, so the contrast an eps target demands is what is
    # left of the budget once the hosting residual has taken its share -- and a target below
    # `rel` is unreachable at any brightness, which is what the infinity says.
    room = {t: np.sqrt(max(v * v - rel * rel, 0.0)) for t, v in thr.items()}
    return {
        "n": n,
        "eps": eps,
        "rate": rate,
        "bins": cells,
        "threshold": thr,
        "collapse_spread": spread,
        "saturated": min(r for *_x, r in cells) >= max(targets),
        "contrast_at": {t: (SIGMA_BENCH / v if v > 0 else np.inf) for t, v in room.items()},
        "design": {"rel_err": rel, "contrast": float(pts[:, 3].mean())},
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval on a success rate of k out of n.

    The score interval, not k/n +- z sqrt(pq/n). At the instance counts an optical run can
    afford the normal interval runs off the end of [0, 1] and collapses to zero width at 0
    and at n, which is exactly where an Ising run lands most often. This one does neither,
    and at 4 instances it says what four instances are worth: 2 of 4 is [0.15, 0.85]."""
    if n <= 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def instances_for(
    p0: float, p1: float, power: float = 0.8, alpha: float = 0.05, arms: int = 1
) -> int:
    """Instances needed to see a true rate of `p1` against `p0`, at `power`.

    `arms=1` tests one run against a *known* constant -- blind guessing, 1/2^n up to ground
    state degeneracy -- and is one-sided, because a chip below chance is not a result. Use
    it before claiming a run computed anything. `arms=2` is the two-sided count per arm for
    comparing two runs, and it is the expensive one: separating 50 percent from 75 percent
    needs 55 instances an arm, so the four-instance A/B that motivated this function could
    not have resolved the effect it was looking for whatever the chip did. Running the same
    problems through both arms makes the test paired and cuts that, but only in proportion
    to how correlated the arms are; nothing makes 4 enough."""
    z = NormalDist().inv_cdf
    d = abs(p1 - p0)
    if d < 1e-12:
        raise ValueError("no difference to detect")
    if arms == 1:
        num = z(1 - alpha) * np.sqrt(p0 * (1 - p0)) + z(power) * np.sqrt(p1 * (1 - p1))
        return int(np.ceil(num**2 / d**2))
    return int(np.ceil((z(1 - alpha / 2) + z(power)) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / d**2))


def _score(T, J, cfgs_true, E_true, normalised: bool = True, modes=None) -> dict:
    """One instance: did the chip's own energy ranking find the true ground state?"""
    res = decode(T, J, normalised=normalised, modes=modes)
    E_chip = res["energies"]
    order = {tuple(c): i for i, c in enumerate(res["configs"].tolist())}
    E_ord = np.array([E_chip[order[tuple(c)]] for c in cfgs_true.tolist()])
    k = int(np.argmin(E_ord))
    span = E_true.max() - E_true.min()
    gs = set(np.flatnonzero(np.isclose(E_true, E_true.min())))
    r = np.corrcoef(E_ord, E_true)[0, 1] if len(E_true) > 2 else np.nan
    return {
        "found_gs": bool(k in gs),
        "excess": float((E_true[k] - E_true.min()) / (span + 1e-18)),
        "pearson": float(r),
        "rel_err": res["rel_err"],
        "contrast": res["contrast"],
        "chance": len(gs) / len(E_true),
    }


def _selftest(
    instances: int = 8,
    ns=(2, 3, 4),
    seed: int = 0,
    steps: int = 300,
    restarts: int = 24,
    sigma: float = SIGMA_BENCH,
    verbose: bool = False,
):
    """Small instances with known ground states, scored against the twin.

    Three chips, in order of honesty: a heater law that reaches 2 pi (what the mesh could do
    if the thermo-optic coefficient were the 6x6's), the measured law at the 3 V ceiling
    (what this die does), and the measured law on a mesh with 2 percent coupler error
    (what a fabricated instance does, with the design still fitted against the ideal twin).
    The gap between the first two is the price of the reachable set; between the second and
    third, the price of model error.

    Scored at `SIGMA_BENCH`, not at `SIGMA_READ`. At the photodiode's own noise every case
    scored 100 percent and the test measured nothing; the bench's own error budget is where
    the design decisions actually show up."""
    from .twin import MeshError, Twin

    ideal = Twin()
    fab = Twin(MeshError.sample(sigma_kappa=0.02, seed=1))
    free = Calibration()  # Vpi 1.5 V, phi0 0: every phase reachable
    real = Calibration.load_or_nominal()
    cases = [
        ("free phases", free, ideal, None),
        ("measured heaters", real, ideal, measured(real)),
        ("measured + 2% couplers", real, fab, measured(real)),
    ]

    rows = []
    for label, calib, chip, mask in cases:
        for n in ns:
            rng = np.random.default_rng(seed + 17 * n)
            acc = []
            for i in range(instances):
                J = random_ising(rng, n=n)
                cfgs_true, E_true = brute_force(J)
                enc = encode(
                    J, ideal, calib, trainable=mask, steps=steps, restarts=restarts, seed=seed + i
                )
                T = twin_transfer(chip, enc.phases)
                T = np.clip(T + rng.normal(0, sigma, T.shape), 0.0, None)
                s = _score(T, J, cfgs_true, E_true, modes=enc.modes)
                s["design_err"] = enc.rel_err
                s["design_contrast"] = enc.contrast
                s["eps"] = enc.eps
                s["finite"] = bool(np.isfinite(enc.volts).all())
                acc.append(s)
            row = {
                "case": label,
                "n": n,
                "instances": instances,
                "hits": int(sum(a["found_gs"] for a in acc)),
                **{
                    k: float(np.nanmean([a[k] for a in acc]))
                    for k in (
                        "design_err",
                        "design_contrast",
                        "eps",
                        "rel_err",
                        "contrast",
                        "excess",
                        "pearson",
                        "chance",
                    )
                },
                "found_gs": float(np.mean([a["found_gs"] for a in acc])),
                "finite": all(a["finite"] for a in acc),
            }
            row["ci"] = wilson(row["hits"], instances)
            rows.append(row)
            if verbose:
                lo, hi = row["ci"]
                print(f"  {label:<24} n={n} gs {row['found_gs']:.0%} [{lo:.2f}, {hi:.2f}]")

    # the anneal must agree with exhaustion on the chip's own landscape
    rng = np.random.default_rng(seed)
    J = random_ising(rng, n=4)
    enc = encode(J, ideal, free, steps=steps, restarts=restarts)
    res = decode(enc.T, J, modes=enc.modes)
    lut = {tuple(c): e for c, e in zip(res["configs"].tolist(), res["energies"])}
    _s_a, e_a = anneal(lambda s: lut[tuple(s)], 4, rng)
    assert np.isclose(e_a, res["energies"].min()), (e_a, res["energies"].min())

    # the answer must not depend on launch power, fibre coupling or photodiode gain. Those
    # enter as T -> diag(g) T diag(c) and `normalise` is the whole defence against them, so
    # a decode with them injected must return the same spins as one without -- and the same
    # decode with `normalise` switched off must not, or the projection is doing nothing.
    gains = np.exp(rng.normal(0, 0.35, 2 * NMODE))
    dirty = gains[:NMODE, None] * enc.T * gains[NMODE:][None, :]
    assert np.array_equal(
        decode(dirty, J, modes=enc.modes)["spins"], decode(enc.T, J, modes=enc.modes)["spins"]
    ), "gain invariance"
    raw = decode(dirty, J, normalised=False, modes=enc.modes)
    assert raw["rel_err"] > decode(dirty, J, modes=enc.modes)["rel_err"], raw["rel_err"]

    # every design must be a number. At n = 2 the affine fit is exact, and sqrt(0) has an
    # infinite derivative: the objective used to return NaN there and the design became
    # whatever argmin does with an all-NaN array, silently.
    assert all(r["finite"] for r in rows), [r for r in rows if not r["finite"]]
    # a free-phase mesh must host the largest instance essentially exactly -- asked at
    # SIGMA_READ, because at the default budget the objective deliberately spends hosting
    # error on contrast and the rows above no longer answer a reachability question
    reach = encode(
        random_ising(np.random.default_rng(seed), n=NSPIN_MAX),
        ideal,
        free,
        sigma=SIGMA_READ,
        steps=steps,
        restarts=restarts,
    )
    assert reach.rel_err < 0.01, reach

    # a second pass must pay for itself after its own sqrt(2) of extra noise, or `hosted`
    # is subtracting two matrices for nothing. Deterministic, so it needs no instances.
    two = encode(J, ideal, real, trainable=measured(real), passes=2, steps=steps, restarts=restarts)
    one = encode(J, ideal, real, trainable=measured(real), steps=steps, restarts=restarts)
    assert two.eps < one.eps, (one.eps, two.eps)
    assert two.volts.shape == (2, N_HEATERS) and two.T.shape == (2, NMODE, NMODE), two.volts
    # contrast floors the fidelity-only objective could not clear. Over 160 instances at this
    # same ceiling and mask it averaged 0.151 at n = 3 and 0.093 at n = 4, where this
    # objective averages 0.68 and 0.20; the floors sit between, well clear of both. This is
    # the regression guard on the whole change and it needs no statistics, because contrast
    # is a property of the design rather than of a run.
    for n, floor in ((3, 0.25), (4, 0.12)):
        cell = [r for r in rows if r["case"] == "measured heaters" and r["n"] == n]
        if cell:
            assert cell[0]["design_contrast"] > floor, (n, cell[0]["design_contrast"])
    # and the whole thing must beat blind guessing on the *interval*, not the point
    # estimate. Pooled over every case and every n, because a single cell of a few instances
    # cannot separate 50 percent from chance -- which is exactly the lesson the bench's own
    # four-instance runs taught. See `instances_for`.
    hits = sum(r["hits"] for r in rows)
    trials = sum(r["instances"] for r in rows)
    chance = float(np.mean([r["chance"] for r in rows]))
    assert wilson(hits, trials)[0] > chance, (hits, trials, chance)
    for label, _c, _chip, _m in cases:
        cell = [r for r in rows if r["case"] == label]
        got = float(np.mean([r["found_gs"] for r in cell]))
        exp = float(np.mean([r["chance"] for r in cell]))
        assert got > exp, (label, got, exp)
    return rows


def digest(rows) -> str:
    w = (
        f"{'case':<24}{'n':>2}{'design err':>12}{'contrast':>10}{'eps':>7}{'GS found':>10}"
        f"{'95% CI':>14}{'chance':>8}{'excess':>8}{'E-corr':>8}"
    )
    out = [w]
    for r in rows:
        lo, hi = r.get("ci", (float("nan"), float("nan")))
        out.append(
            f"{r['case']:<24}{r['n']:>2}{r['design_err']:>12.3f}"
            f"{r['contrast']:>10.4f}{r.get('eps', float('nan')):>7.2f}"
            f"{r['hits']:>6}/{r['instances']:<3}"
            f"{f'[{lo:.2f}, {hi:.2f}]':>14}{r['chance']:>8.0%}"
            f"{r['excess']:>8.3f}"
            + (f"{r['pearson']:>8.3f}" if np.isfinite(r["pearson"]) else f"{'--':>8}")
        )
    return "\n".join(out)


if __name__ == "__main__":
    from .twin import Twin

    _tw, _real = Twin(), Calibration.load_or_nominal()
    # only n = 4: at n = 3 the target is one angle on a circle and every calibration steers
    # the whole of it, so the number is 2.00 of 2 and says nothing
    for _lbl, _c, _m in (
        ("free phases", Calibration(), None),
        ("measured heaters", _real, measured(_real)),
    ):
        _r = reach_dim(_tw, _c, _m, n=4)
        print(f"{_lbl:<24} n=4 steers {_r['dim']:.2f} of {_r['of']} coupling directions")
    print()
    print(digest(_selftest()))

    print(
        f"\nground-state rate vs recovered coupling error eps, readout noise only "
        f"(the bench's own residual is {SIGMA_BENCH:.2f} per entry)"
    )
    for _n in (3, 4):
        _q = noise_curve(_tw, _real, n=_n, trainable=measured(_real))
        print(
            f"  n={_n}  "
            + " ".join(f"{lo:.2f}-{hi:.2f}:{r:.2f}" for lo, hi, _k, r in _q["bins"])
            + f"   (design-sigma spread within a bin {_q['collapse_spread']:.2f})"
        )
        if _q["saturated"]:
            print(
                f"        holds every target out to eps "
                f"{max(_q['threshold'].values()):.2f}, the widest this sweep reached"
            )
        else:
            print(
                "        "
                + "  ".join(
                    f"{t:.0%} needs eps<{_q['threshold'][t]:.2f} -> contrast>"
                    f"{_q['contrast_at'][t]:.2f}"
                    for t in (0.95, 0.9, 0.8)
                )
            )

    print(
        f"\ninstances needed: beat chance at n=3 with a true 50% -> "
        f"{instances_for(0.25, 0.5)}; at n=4 -> {instances_for(0.125, 0.5)}; "
        f"separate 50% from 75% between two runs -> {instances_for(0.5, 0.75, arms=2)} each"
    )
