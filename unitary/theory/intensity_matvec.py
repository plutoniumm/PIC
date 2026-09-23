"""Signed matrix-vector product on the 4x4 mesh, read out in intensity. No local oscillator.

A photodiode reports |y|^2, so a single coherent read is quadratic in the input. Three steps
make it a signed linear map. Two are ported unchanged from the 6x6; the third is forced by
this mesh being unitary, and is the only new idea here.

**0. Unitarity costs generality; Sinkhorn buys it back.** The intensity transfer of a
lossless mesh, A = |U|^2, is doubly stochastic -- every row and every column sums to 1. The
6x6's meshes were contractions, so |M|^2 was merely substochastic and one positive scale was
enough to match a target up to it. One scale is not enough here: a generic nonnegative B has
unequal row sums, and no multiple of it is doubly stochastic. An *ideal* 4x4 mesh with all
twelve phases free still floors at 0.3-0.5 relative error on a random 4x4 target and
0.05-0.10 on a 3x3 block -- that is geometry, not optimisation. Sinkhorn factors any B
with total support as B = diag(1/r) A diag(1/s), A doubly stochastic, and both diagonals
are free:
diag(1/s) rescales the four input powers, diag(1/r) rescales the four photodiode readings,
and neither is a mesh phase. Scale the target first and the same ideal mesh reaches 0.000.

**1. Linearisation.** Encode the input as intensities p_j >= 0 and add the output powers. On
the 6x6 that needed a phase dither, because every rail is lit at once and the cross terms
have to be averaged away:

    E_phi | sum_j M_kj sqrt(p_j) e^{i phi_j} |^2 = sum_j |M_kj|^2 p_j = (A p)_k.

Here the 1x4 switch lights ONE port at a time, so there are no cross terms to average: the
same sum comes out exactly, in four switch positions, at zero dither variance. Time
multiplexing is the dither's noise-free limit, and it is cheaper -- four passes against the
6x6's sixty-four. The accumulation over j is digital because the switch admits one port at a
time; the multiply p_j |U_kj|^2 stays optical as long as p_j is a laser setpoint rather than
a number in software. `dither_matvec` keeps the coherent form for the day a splitter exists.

**2. Differential encoding for signs.** An input power cannot be negative and neither can a
detected one, so a signed B and a signed x are reached by differencing nonnegative passes.
The 6x6 split B = B+ - B-, two mesh programs and four passes. That is `mode="split"` and it
is kept, but it is not the default here: each half is mostly zeros, and Sinkhorn scaling a
sparse matrix produces enormous diagonals -- 1057 on the first random 2x2 tried -- which
multiply the read noise by the same factor when they are undone. `mode="shift"` writes
B = (B + cJ) - cJ instead: the hosted half is dense, its diagonals stay of order one, and
the all-ones half is rank one, so its matvec is c times the sum of x and needs no light.
One program, two passes, and 7x less read noise on the twin. Either way, no local
oscillator.

**What the heaters actually reach.** Nothing above says the mesh can host the A it is asked
for. Every heater only ADDS phase, phi = phi0 + pi (V/Vpi)^2, and on this chip the measured
spans run 0.08 to 0.54 pi -- not one heater reaches pi, and a free phase needs 2 pi. Eight
channels steer |U|^2 at all, and only inside that patch. So a target is approached by
optimising the *voltages* through the twin (`fit_nonneg`), never by inverting a
decomposition, and the honest report is the residual, not the plan.

What survives is a **2x2 block**, and it survives exactly: on the measured calibration a
random signed 2x2 is hosted to ~3e-3 and read back at 0.006 relative error with every sign
right. 3x3 and 4x4 do not -- 0.38 and 0.52 -- and that is heater span, since the same fits
reach 0.000 on the mesh as designed. `scan_rails` is worth more than the optimiser: which
sub-block carries the problem sets both the residual (3e-4 to 1.5) and the block brightness
(0.007 to 0.96), and brightness divides straight into the read noise.

    python -m theory.intensity_matvec        # the reachability ladder and a twin end-to-end
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

from .calib import VOLTAGE_MAX, VPI_NOMINAL, Calibration
from .clements import COLUMN, NMODE, NMZI
from .layout import N_HEATERS, PHI_IDX, THETA_IDX
from .twin import Twin

# Intensity is blind to any phase screen on the input or the output rails, so three classes
# of heater cannot move |U|^2 at all: the output trimmers, aux, and the external phases of
# the first mesh column, which sit on the input rails. `_selftest` checks this rather than
# assuming it. Fitting them wastes optimiser dimensions and reads as if they mattered.
_INPUT_PHI = {int(PHI_IDX[k]) for k in range(NMZI) if COLUMN[k] == 0}
VISIBLE_IDX = np.array(sorted((set(THETA_IDX.tolist()) | set(PHI_IDX.tolist())) - _INPUT_PHI), int)

# From `scan_rails` against pic_data/calib.json and the board's per-channel ceilings. Two
# numbers pick a rail pair and both matter: the residual, which spans 3e-4 to 1.5 across the
# 36 pairs, and the block brightness, which spans 0.007 to 0.99 among the pairs that fit
# equally well. Brightness divides into the read noise, so ranking on residual alone throws
# away more than the residual was worth -- the first version of this constant picked a pair
# that fitted perfectly and carried 4 percent of the light, at 23x the noise. This pair is
# the one that stays good as the ceilings move. Re-run `scan_rails` after a
# re-characterization or a change to VOLTAGE_MAX_CH; nothing else here depends on it.
BEST_RAILS = ((1, 2), (1, 2))


def block_rails(k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A default k x k sub-block. Only a starting point -- use `scan_rails`."""
    return BEST_RAILS if k == 2 else (tuple(range(k)), tuple(range(k)))


@dataclass
class HeaterBox:
    """The reachable phase interval of every DAC channel, and the map into it.

    phi_i(V) = phi0_i + pi (V/Vpi_i)^2 for V in [0, vmax_i], so channel i covers exactly
    [phi0_i, phi0_i + pi (vmax_i/Vpi_i)^2] and nothing else. A channel that is not
    `trainable` is held at 0 V, which is phi0_i, not 0.

    `vmax` is per channel because the board's ceiling is: the 60 ohm heaters clamp at 1.5 V
    to stay inside their current rating (`pic.config.VOLTAGE_MAX_CH`), and channels with no
    confirmed resistance are dark at 0.0. `theory` does not know the board, so the rig side
    passes that array in; the default is the uniform design ceiling.
    """

    vpi: np.ndarray
    phi0: np.ndarray
    vmax: np.ndarray
    trainable: np.ndarray = field(default=None)

    def __post_init__(self):
        n = N_HEATERS
        self.vpi = np.where(
            np.isfinite(self.vpi) & (np.asarray(self.vpi, float) > 0), self.vpi, VPI_NOMINAL
        ) * np.ones(n)
        self.phi0 = np.nan_to_num(np.asarray(self.phi0, float) * np.ones(n))
        self.vmax = np.clip(np.asarray(self.vmax, float) * np.ones(n), 0.0, None)
        if self.trainable is None:
            self.trainable = np.zeros(n, bool)
            self.trainable[VISIBLE_IDX] = True
        self.trainable = np.asarray(self.trainable, bool) & (self.vmax > 0)

    @classmethod
    def from_calibration(
        cls, calib: Calibration, vmax=VOLTAGE_MAX, known=None, visible_only: bool = True
    ) -> "HeaterBox":
        """The box the bench is actually in.

        `known` defaults to "Vpi is not still sitting exactly at VPI_NOMINAL", which is what
        an unfitted channel looks like in a merged calibration -- an uncharacterized channel
        has no law, so driving it moves the chip in a direction the twin cannot predict and
        it is safer held at 0 V.
        """
        vpi = np.asarray(calib.vpi, float)
        if known is None:
            known = vpi != VPI_NOMINAL
        t = np.asarray(known, bool).copy()
        if visible_only:
            vis = np.zeros(N_HEATERS, bool)
            vis[VISIBLE_IDX] = True
            t &= vis
        return cls(vpi, np.asarray(calib.phi0, float), vmax, t)

    @classmethod
    def ideal(cls) -> "HeaterBox":
        """Every intensity-visible phase free over 4 pi: the mesh as designed, for the
        ablation that separates "the geometry cannot host this" from "the heaters cannot
        reach it"."""
        return cls(np.full(N_HEATERS, 0.5), np.zeros(N_HEATERS), np.ones(N_HEATERS))

    @property
    def span_pi(self) -> np.ndarray:
        """Phase span of each channel in units of pi. Below 2 the channel is not free."""
        return (self.vmax / self.vpi) ** 2

    def phases(self, volts) -> np.ndarray:
        v = np.clip(np.asarray(volts, float), 0.0, self.vmax)
        return self.phi0 + np.pi * (v / self.vpi) ** 2

    def describe(self) -> str:
        rows = [f"{'dac':>3} {'Vpi':>6} {'Vmax':>5} {'span/pi':>8}  free"]
        for i in range(N_HEATERS):
            rows.append(
                f"{i:>3} {self.vpi[i]:>6.2f} {self.vmax[i]:>5.2f} "
                f"{self.span_pi[i]:>8.3f}  {'y' if self.trainable[i] else 'n'}"
            )
        rows.append(
            f"{int(self.trainable.sum())} of {N_HEATERS} channels free; widest span "
            f"{self.span_pi[self.trainable].max(initial=0.0):.2f} pi"
        )
        return "\n".join(rows)


def sinkhorn(B, iters: int = 2000, floor: float = 1e-3):
    """Nonnegative B -> (A, r, s) with A = diag(r) B diag(s) doubly stochastic.

    Inverted, B = diag(1/r) A diag(1/s), so B x = (A @ (x / s)) / r: the mesh hosts A and
    the two diagonals live in the input powers and the photodiode readings, both of which
    are already scaled in software. This is what makes a general nonnegative target
    representable on a mesh whose intensity transfer must be doubly stochastic.

    `floor` lifts exact zeros. A matrix with an all-zero row or column has no doubly
    stochastic scaling at all, and the sign-split B+ is full of zeros by construction; the
    lift costs a relative error of order `floor` on entries that were zero, and nothing on
    the rest.
    """
    B = np.asarray(B, float)
    if B.min() < 0:
        raise ValueError("sinkhorn needs a nonnegative matrix")
    A = np.maximum(B, floor * max(B.max(), 1e-30))
    r = np.ones(A.shape[0])
    s = np.ones(A.shape[1])
    for _ in range(iters):
        r = 1.0 / np.maximum(A @ s, 1e-300)
        s = 1.0 / np.maximum(A.T @ r, 1e-300)
    return np.diag(r) @ A @ np.diag(s), r, s


def block(U, rails):
    """|U|^2 restricted to (out rails, in rails). Batched over any leading dimensions."""
    out, inp = rails
    P = U.abs() ** 2 if torch.is_tensor(U) else np.abs(U) ** 2
    return P[..., list(out), :][..., :, list(inp)]


def abs2_loss(U, A, rails, eps: float = 1e-18):
    """Scale-invariant squared error of the hosted block against nonnegative A.

    One positive scale is free -- the laser sets it -- so the residual is measured after
    projecting out the best s in |U|^2 ~ s A."""
    P = block(U, rails)
    Pf = P.reshape(*P.shape[:-2], -1)
    Af = torch.as_tensor(np.asarray(A, float), dtype=Pf.dtype).reshape(-1)
    s = (Pf * Af).sum(-1) / ((Af * Af).sum() + eps)
    resid = ((Pf - s.unsqueeze(-1) * Af) ** 2).sum(-1)
    return resid / ((s.unsqueeze(-1) * Af) ** 2).sum(-1).clamp(min=eps)


@dataclass
class Program:
    """One mesh setting: the volts to command, and what they actually host."""

    volts: np.ndarray
    phases: np.ndarray
    scale: float  # the mesh hosts scale * A, so divide every reading by it
    hosted: np.ndarray  # |U|^2 on the rails, as fitted
    err: float  # relative, scale-invariant

    def __str__(self):
        return (
            f"Program(err={self.err:.4f}, scale={self.scale:.4f}, " f"Vmax={self.volts.max():.2f})"
        )


def fit_nonneg(
    A,
    box: HeaterBox,
    twin: Twin | None = None,
    rails=BEST_RAILS,
    restarts: int = 16,
    steps: int = 400,
    lr: float = 0.15,
    tol: float = 0.01,
    seed: int = 0,
):
    """Heater VOLTS whose hosted |U|^2 block matches nonnegative A up to one scale.

    Optimised in voltage space, not phase space: every value the fit can produce is a
    voltage the driver will accept, so nothing has to be clamped afterwards. The 6x6 learnt
    this the hard way -- free-phase fitting there returned phases the chip could not make,
    and clamping them wrecked |M|^2. Here the restriction is far tighter (no heater reaches
    pi), so the fit is a search over a small non-convex patch and multi-start is doing real
    work; `restarts` is the parameter that matters.

    Among the restarts that land inside `tol` the BRIGHTEST is kept, not the most accurate.
    They are all the same matrix -- the loss is scale-invariant -- but the scale is the
    fraction of injected light that reaches the block, and every reading is divided by it,
    so the brightest solution is the one with the best signal-to-noise for free. On the
    measured calibration that is worth about 1.7x.
    """
    twin = twin or Twin()
    A = np.asarray(A, float)
    vpi = torch.as_tensor(box.vpi, dtype=torch.float32)
    phi0 = torch.as_tensor(box.phi0, dtype=torch.float32)
    vmax = torch.as_tensor(box.vmax, dtype=torch.float32)
    free = torch.as_tensor(box.trainable)

    g = torch.Generator().manual_seed(int(seed) * 1000 + 17)
    raw = (torch.rand(restarts, N_HEATERS, generator=g) * 4 - 2).requires_grad_(True)

    def volts():
        return torch.where(free, vmax * torch.sigmoid(raw), torch.zeros_like(raw))

    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        ph = phi0 + np.pi * (volts() / vpi) ** 2
        abs2_loss(twin.matrix(ph), A, rails).sum().backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        v = volts()
        ph = phi0 + np.pi * (v / vpi) ** 2
        errs = torch.sqrt(abs2_loss(twin.matrix(ph), A, rails)).numpy()
        hosted = block(twin.matrix(ph), rails).numpy()
    scales = (hosted * A).sum((-2, -1)) / max((A * A).sum(), 1e-18)
    inside = np.flatnonzero(errs <= max(errs.min(), tol))
    k = int(inside[np.argmax(scales[inside])])
    return Program(
        v[k].numpy().astype(float),
        ph[k].numpy().astype(float),
        float(scales[k]),
        hosted[k],
        float(errs[k]),
    )


def scan_rails(
    A, box: HeaterBox, twin: Twin | None = None, k: int | None = None, tol: float = 0.01, **kw
):
    """Every (out, in) rail pair of size k: feasible ones first, brightest among those.

    Worth the passes, and worth ranking on both numbers. On the measured calibration the
    best 2x2 block is hosted to 3e-4 and the worst misses by 1.5; among the pairs that fit
    equally well the brightness spans 0.007 to 0.96, and brightness divides straight into
    the read noise. Nothing about the target predicts either -- both are set by which
    heaters steer which rails and how far they reach."""
    A = np.asarray(A, float)
    k = A.shape[0] if k is None else k
    rows = []
    for o in itertools.combinations(range(NMODE), k):
        for i in itertools.combinations(range(NMODE), k):
            fit = fit_nonneg(A, box, twin, rails=(o, i), **kw)
            rows.append({"err": fit.err, "scale": fit.scale, "out": o, "in": i, "prog": fit})
    return sorted(rows, key=lambda r: (r["err"] > tol, -r["scale"], r["err"]))


def twin_probe(twin: Twin, box: HeaterBox, noise: float = 0.0, repeats: int = 1, rng=None):
    """The bench's one primitive, in the twin: set the volts, light one input port at some
    power, read the four output intensities.

    `power` is an argument rather than a software multiply because on the rig it is a laser
    setpoint -- that is what keeps the multiply p_j |U_kj|^2 optical.

    `noise` is the photodiode read noise as a fraction of FULL SCALE, which is the injected
    power, not the reading: the bench's measured read noise is 0.6-4.4 mV against a 173-261
    mV full scale (`pic.sim`), so 0.004 to 0.025 is the range to test at. `repeats` averages
    reads, and the vector error falls as 1/sqrt(repeats) -- which is how a usable answer is
    bought at this chip's block brightness."""
    rng = np.random.default_rng() if rng is None else rng

    def probe(volts, port: int, power: float = 1.0):
        ph = torch.as_tensor(box.phases(volts), dtype=torch.float32)
        with torch.no_grad():
            col = (twin.matrix(ph).abs() ** 2)[:, int(port)].numpy()
        y = float(power) * col
        if noise:
            sd = noise * max(float(power), 1e-12) / np.sqrt(max(repeats, 1))
            y = y + rng.normal(0.0, sd, y.shape)
        return y

    return probe


def nonneg_matvec(probe, volts, p, rails=BEST_RAILS):
    """(A p) on the hosted block, by lighting each input rail in turn.

    Exact, not an average: with one port lit there is no cross term to dither away. Ports
    carrying no power are skipped, which is why a signed matvec costs 8 reads and not 16."""
    out, inp = rails
    p = np.asarray(p, float).ravel()
    acc = np.zeros(len(out))
    for j, port in enumerate(inp):
        if p[j] <= 0:
            continue
        acc += np.asarray(probe(volts, port, float(p[j])))[list(out)]
    return acc


def dither_matvec(twin: Twin, phases, p, rails=BEST_RAILS, n_dither: int = 64, rng=None):
    """The 6x6's coherent form: all input rails lit at once with random phases, averaged.

    This bench has a 1x4 switch and no splitter, so nothing can run it today. It is kept
    because it is the scheme that takes over when a splitter arrives -- the accumulation
    becomes optical -- and because its 1/sqrt(n_dither) convergence to the same answer is
    what shows the switched version is that limit rather than a different measurement."""
    rng = np.random.default_rng() if rng is None else rng
    out, inp = rails
    with torch.no_grad():
        U = twin.matrix(torch.as_tensor(np.asarray(phases, float), dtype=torch.float32)).numpy()
    amp = np.sqrt(np.clip(np.asarray(p, float).ravel(), 0.0, None))
    acc = np.zeros(len(out))
    for _ in range(n_dither):
        x = np.zeros(NMODE, complex)
        x[list(inp)] = amp * np.exp(2j * np.pi * rng.random(len(inp)))
        acc += np.abs(U @ x)[list(out)] ** 2
    return acc / n_dither


@dataclass
class Term:
    """One mesh program in a plan: the chip hosts `scale * A`, and B_term is recovered as
    diag(1/r) (hosted/scale) diag(1/s). `coeff` is the sign it enters the sum with."""

    coeff: float
    prog: Program
    r: np.ndarray
    s: np.ndarray

    def matrix(self) -> np.ndarray:
        return np.diag(1 / self.r) @ (self.prog.hosted / self.prog.scale) @ np.diag(1 / self.s)

    def apply(self, probe, v, rails) -> np.ndarray:
        """One nonnegative pass: B_term @ v, undoing the hosted scale and both diagonals."""
        y = nonneg_matvec(probe, self.prog.volts, np.asarray(v, float) / self.s, rails)
        return y / self.prog.scale / self.r


@dataclass
class MatvecPlan:
    """Everything needed to compute y = B x on the chip.

    `terms` are the mesh programs; `offset` is a rank-1 piece computed in software, since
    the all-ones matrix times x is just (sum x) and no light is needed to work that out."""

    B: np.ndarray
    rails: tuple
    mode: str
    terms: tuple
    offset: float = 0.0

    @property
    def err(self) -> float:
        """The worst hosted residual in the plan -- the honest bound on it."""
        return max(t.prog.err for t in self.terms)

    @property
    def passes(self) -> int:
        """Photodiode reads per vector, at full input support."""
        return 2 * len(self.terms) * len(self.rails[1])

    def matvec(self, probe, x) -> np.ndarray:
        """Signed y = B x. Each term contributes a differential pair, because an input
        power cannot be negative: B_t x = B_t x+ - B_t x-."""
        x = np.asarray(x, float).ravel()
        xp, xm = np.clip(x, 0, None), np.clip(-x, 0, None)
        y = np.zeros(len(self.rails[0]))
        for t in self.terms:
            y = y + t.coeff * (t.apply(probe, xp, self.rails) - t.apply(probe, xm, self.rails))
        return y - self.offset * x.sum()

    def predict(self) -> np.ndarray:
        """The signed matrix this plan actually realises, noise aside. Compare it to `B`
        before trusting any vector: the residual is in the matrix, not the readout."""
        return sum(t.coeff * t.matrix() for t in self.terms) - self.offset


def plan_matvec(
    B,
    box: HeaterBox,
    twin: Twin | None = None,
    rails=None,
    mode: str = "shift",
    floor: float = 1e-3,
    margin: float = 0.25,
    seed: int = 0,
    **fit_kw,
) -> MatvecPlan:
    """Signed target B -> the mesh programs that compute it.

    Two nonnegative decompositions, and the choice is a read-noise decision, not a
    correctness one -- both realise B exactly in the twin.

    `split` is the 6x6's: B = B+ - B-, two programs, four passes. It is badly conditioned
    here and the reason is new. Each half is mostly zeros, and Sinkhorn scaling a sparse
    matrix produces enormous diagonals -- 1057 on the first random 2x2 tried -- which
    multiply the read noise by the same factor when they are undone.

    `shift` avoids that: B = (B + cJ) - cJ with c just past -min(B), so the hosted half is
    dense and its diagonals stay of order one, and the cJ half is rank one and needs no
    light at all. One program, two passes, and about half the noise amplification. The cost
    is the cancellation of the constant c, which is mild while c is of order |B|.
    """
    twin = twin or Twin()
    B = np.asarray(B, float)
    rails = block_rails(B.shape[0]) if rails is None else rails
    if mode == "split":
        Ap, rp, sp = sinkhorn(np.clip(B, 0, None), floor=floor)
        Am, rm, sm = sinkhorn(np.clip(-B, 0, None), floor=floor)
        terms = (
            Term(+1.0, fit_nonneg(Ap, box, twin, rails, seed=seed, **fit_kw), rp, sp),
            Term(-1.0, fit_nonneg(Am, box, twin, rails, seed=seed + 1, **fit_kw), rm, sm),
        )
        return MatvecPlan(B, rails, mode, terms, 0.0)
    if mode == "shift":
        c = max(0.0, -float(B.min())) + margin * float(np.abs(B).mean())
        A, r, s = sinkhorn(B + c, floor=floor)
        terms = (Term(+1.0, fit_nonneg(A, box, twin, rails, seed=seed, **fit_kw), r, s),)
        return MatvecPlan(B, rails, mode, terms, c)
    raise ValueError(f"mode={mode!r}; use 'shift' or 'split'")


def score(y_hat, y):
    """(relative 2-norm error, fraction of components with the right sign)."""
    y_hat, y = np.asarray(y_hat, float), np.asarray(y, float)
    err = np.linalg.norm(y_hat - y) / max(np.linalg.norm(y), 1e-18)
    return float(err), float(np.mean(np.sign(y_hat) == np.sign(y)))


def validate(
    box: HeaterBox,
    twin: Twin | None = None,
    k: int = 2,
    rails=None,
    trials: int = 6,
    noise: float = 0.0,
    repeats: int = 1,
    mode: str = "shift",
    seed: int = 0,
    plan=None,
    **fit_kw,
) -> dict:
    """Twin end-to-end: a random signed B, planned and then read back vector by vector.

    Pass `plan` to re-measure one plan at several noise levels without refitting it."""
    twin = twin or Twin()
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(k, k)) if plan is None else plan.B
    plan = plan or plan_matvec(B, box, twin, rails=rails, mode=mode, seed=seed, **fit_kw)
    probe = twin_probe(twin, box, noise=noise, repeats=repeats, rng=rng)

    errs, signs = [], []
    for _ in range(trials):
        x = rng.normal(size=k)
        e, s = score(plan.matvec(probe, x), B @ x)
        errs.append(e)
        signs.append(s)
    m_err, m_sign = score(plan.predict().ravel(), B.ravel())
    return {
        "k": k,
        "rails": plan.rails,
        "mode": plan.mode,
        "fit_err": plan.err,
        "matrix_err": m_err,
        "matrix_sign": m_sign,
        "vec_err": float(np.mean(errs)),
        "sign_acc": float(np.mean(signs)),
        "noise": noise,
        "repeats": repeats,
        "trials": trials,
        "passes": plan.passes,
        "plan": plan,
    }


def reachability(
    box: HeaterBox,
    twin: Twin | None = None,
    sizes=(2, 3, 4),
    trials: int = 2,
    seed: int = 0,
    **fit_kw,
) -> list[dict]:
    """What the restriction costs, split into its two causes.

    `ideal` is the mesh as designed -- every intensity-visible phase free over 4 pi -- so the
    gap between `ideal_raw` and `ideal_sink` is pure geometry (the doubly stochastic
    constraint) and the gap between `ideal_sink` and `box_sink` is pure heater span."""
    twin = twin or Twin()
    rng = np.random.default_rng(seed)
    ideal = HeaterBox.ideal()
    rows = []
    for k in sizes:
        rails = block_rails(k)
        got = {"ideal_raw": [], "ideal_sink": [], "box_raw": [], "box_sink": []}
        for t in range(trials):
            B = np.abs(rng.normal(size=(k, k))) + 0.05
            A = sinkhorn(B)[0]
            got["ideal_raw"].append(fit_nonneg(B, ideal, twin, rails, seed=t, **fit_kw).err)
            got["ideal_sink"].append(fit_nonneg(A, ideal, twin, rails, seed=t, **fit_kw).err)
            got["box_raw"].append(fit_nonneg(B, box, twin, rails, seed=t, **fit_kw).err)
            got["box_sink"].append(fit_nonneg(A, box, twin, rails, seed=t, **fit_kw).err)
        rows.append({"k": k, "rails": rails, **{n: float(np.mean(v)) for n, v in got.items()}})
    return rows


def _selftest(seed: int = 0):
    """Every claim the module makes, against the twin, on the calibration in force."""
    twin = Twin()
    box = HeaterBox.from_calibration(Calibration.load_or_nominal())
    rng = np.random.default_rng(seed)

    # the heaters intensity cannot see must not move |U|^2 at all
    ph = torch.as_tensor(rng.uniform(0, 2 * np.pi, N_HEATERS), dtype=torch.float64)
    twin64 = Twin(dtype=torch.complex128)
    A0 = (twin64.matrix(ph).abs() ** 2).numpy()
    blind = sorted(set(range(N_HEATERS)) - set(VISIBLE_IDX.tolist()))
    ph2 = ph.clone()
    ph2[blind] += 1.0
    blind_move = float(np.abs((twin64.matrix(ph2).abs() ** 2).numpy() - A0).max())
    ds = float(max(np.abs(A0.sum(0) - 1).max(), np.abs(A0.sum(1) - 1).max()))

    # the switched matvec is exact, and the dither converges to it
    prog = fit_nonneg(
        sinkhorn(np.abs(rng.normal(size=(2, 2))) + 0.05)[0],
        box,
        twin,
        BEST_RAILS,
        restarts=12,
        steps=300,
        seed=seed,
    )
    p = np.abs(rng.normal(size=2)) + 0.1
    exact = (
        block(twin.matrix(torch.as_tensor(prog.phases, dtype=torch.float32)), BEST_RAILS).numpy()
        @ p
    )
    sw = nonneg_matvec(twin_probe(twin, box), prog.volts, p, BEST_RAILS)
    switch_err = float(np.abs(sw - exact).max() / max(np.abs(exact).max(), 1e-18))
    dith = {
        n: float(
            np.linalg.norm(
                dither_matvec(twin, prog.phases, p, BEST_RAILS, n, np.random.default_rng(1)) - exact
            )
            / np.linalg.norm(exact)
        )
        for n in (64, 1024)
    }

    # Sinkhorn must actually produce a doubly stochastic factorisation and invert exactly
    Braw = np.abs(rng.normal(size=(3, 3))) + 0.05
    A, r, s = sinkhorn(Braw)
    sink_ds = float(max(np.abs(A.sum(0) - 1).max(), np.abs(A.sum(1) - 1).max()))
    xv = rng.normal(size=3)
    sink_inv = float(np.abs((A @ (xv / s)) / r - Braw @ xv).max())

    # end to end, noiseless and at the bench's measured read noise, for both decompositions
    fit_kw = dict(restarts=12, steps=300)
    clean = validate(box, twin, k=2, trials=6, seed=seed, **fit_kw)
    noisy = validate(box, twin, k=2, trials=6, noise=0.005, seed=seed, plan=clean["plan"])
    split = validate(box, twin, k=2, trials=6, noise=0.005, mode="split", seed=seed, **fit_kw)

    assert blind_move < 1e-9, blind_move
    assert ds < 1e-9, ds
    assert switch_err < 1e-5, switch_err  # complex64 twin; the scheme itself is exact
    assert dith[1024] < dith[64], dith
    assert sink_ds < 1e-9 and sink_inv < 1e-9, (sink_ds, sink_inv)
    assert clean["vec_err"] < 0.05, clean["vec_err"]
    assert clean["sign_acc"] == 1.0, clean["sign_acc"]
    return {
        "blind_move": blind_move,
        "doubly_stochastic": ds,
        "switch_err": switch_err,
        "dither": dith,
        "sinkhorn_ds": sink_ds,
        "sinkhorn_inv": sink_inv,
        "clean": clean,
        "noisy": noisy,
        "split": split,
        "box": box,
        "calibrated": bool(Calibration.load_or_nominal().meta),
    }


if __name__ == "__main__":
    r = _selftest()
    src = "measured calibration" if r["calibrated"] else "NOMINAL (uncharacterized)"
    print(
        f"heater box: {src}, uniform {VOLTAGE_MAX:.0f} V ceiling "
        f"(the board clamps four channels lower -- see pic.matvec)"
    )
    print(r["box"].describe())
    print(f"\nblind heaters move |U|^2 by  {r['blind_move']:.1e}  (they must not)")
    print(f"|U|^2 doubly stochastic to   {r['doubly_stochastic']:.1e}")
    print(f"switched matvec vs A p       {r['switch_err']:.1e}  (exact: no cross terms)")
    print(
        f"coherent dither vs the same  {r['dither'][64]:.3f} at 64, "
        f"{r['dither'][1024]:.3f} at 1024 shots"
    )
    print(
        f"sinkhorn doubly stochastic   {r['sinkhorn_ds']:.1e}, inverts to "
        f"{r['sinkhorn_inv']:.1e}"
    )

    c, n, s = r["clean"], r["noisy"], r["split"]
    plan = c["plan"]
    print(
        f"\n2x2 signed matvec on rails out{c['rails'][0]} in{c['rails'][1]}, "
        f"{c['mode']} decomposition, {c['passes']} reads/vector"
    )
    print(
        f"  hosted residual / block brightness   {c['fit_err']:.4f} / "
        f"{plan.terms[0].prog.scale:.3f}"
    )
    print(f"  realised matrix vs target B          {c['matrix_err']:.4f}")
    print(f"  vector error / sign, noiseless       {c['vec_err']:.4f} / {c['sign_acc']:.0%}")
    print(f"  vector error / sign, 0.5% read noise {n['vec_err']:.4f} / {n['sign_acc']:.0%}")
    print(
        f"  same by the 6x6 split, {s['passes']} reads      "
        f"{s['vec_err']:.4f} / {s['sign_acc']:.0%}"
    )
    print(
        "  read noise, amplified by 1/brightness and the Sinkhorn diagonals "
        "(0.004-0.025 is the bench's measured range; `repeats` buys 1/sqrt(N) back)"
    )
    for nz in (0.004, 0.012, 0.025):
        v = validate(r["box"], plan=plan, noise=nz, trials=12, seed=1)
        print(
            f"    noise {nz:.3f}   vec err {v['vec_err']:.3f}  sign {v['sign_acc']:.0%}"
            f"   -> {int(np.ceil((v['vec_err'] / 0.05) ** 2))} reads for 5%"
        )

    print("\nwhat the restriction costs (relative residual of the hosted block)")
    print(f"{'k':>2} {'ideal raw':>10} {'ideal sink':>11} {'box raw':>9} {'box sink':>9}")
    for row in reachability(r["box"], restarts=12, steps=300):
        print(
            f"{row['k']:>2} {row['ideal_raw']:>10.3f} {row['ideal_sink']:>11.3f} "
            f"{row['box_raw']:>9.3f} {row['box_sink']:>9.3f}"
        )
