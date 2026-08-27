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
phase, phi = pi (V/Vpi)^2 + phi0, and the measured spans run 0.08 to 0.54 pi: not one of
the eleven characterised heaters reaches pi, where a free phase needs 2 pi. So a target
cannot be inverted -- `theory.program` hands back phases the chip refuses -- and `encode`
searches the reachable set by gradient descent on the heater voltages through the
differentiable twin. That is what the 6x6 did through its surrogate, for the same reason
and a different cause.

    enc = encode(J, twin, calib)              # design: volts the chip can actually hold
    T   = measure_the_four_ports(enc.volts)   # or twin_transfer(twin, enc.phases)
    res = decode(T, J)                        # res["spins"], res["energies"], res["fit"]

n is at most NMODE = 4. There are four optical modes and no way to embed a fifth spin.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .calib import VOLTAGE_MAX, VPI_NOMINAL, Calibration
from .clements import NMODE
from .layout import N_HEATERS

NSPIN_MAX = NMODE

# The uniform doubly stochastic matrix is the image of J = 0: every coupling equal is no
# coupling at all, once the (1.s)^2 term it produces has been subtracted. Carrying it as one
# extra pseudo-observation makes the (a, b) fit determined at n = 2, where a single measured
# coupling cannot otherwise separate scale from offset, and is negligible at n = 4 where six
# real observations outvote it.
ANCHOR = 1.0 / NMODE
ANCHOR_WEIGHT = 1.0

# Per-entry noise on a normalised T. The bench reads ~0.3 mV RMS into a ~0.33 V full scale
# (pic.sim, MockPIC), and the four-port probe averages three port cycles, so ~1e-3 of full
# scale per entry is the honest default.
SIGMA_READ = 1e-3

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
    return ~(np.isclose(np.asarray(calib.vpi, float), VPI_NOMINAL)
             & (np.asarray(calib.phi0, float) == 0.0))


def movers(twin, phases, trainable=None, tol: float = 1e-6) -> np.ndarray:
    """Channels whose phase actually changes |U|^2 at this operating point.

    Intensity is blind to the input and output phase screens, so some wired, characterized
    heaters move nothing a photodiode can see. Measured rather than read off the role
    labels, because `theory.layout` says itself that the phi/alpha split is provisional."""
    import torch

    def grad_at(offset):
        # sum |U|^2 is the constant NMODE for any unitary, so it has no gradient at all;
        # the sum of squares is the cheapest scalar that actually tracks where the power went
        ph = torch.as_tensor(np.asarray(phases, float) + offset,
                             dtype=torch.float32).clone().requires_grad_(True)
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


def _affine(x, y, w: float = ANCHOR_WEIGHT, anchor: float = ANCHOR):
    """Least-squares y ~ a x + b with one anchored pseudo-observation at (0, `anchor`).

    Written with `.sum(-1)` only, so the same closed form serves the numpy decode and the
    batched torch encode without a second implementation drifting away from the first."""
    m = x.shape[-1]
    sw = m + w
    sx, sxx = x.sum(-1), (x * x).sum(-1)
    sy = y.sum(-1) + w * anchor
    sxy = (y * x).sum(-1)
    det = sw * sxx - sx * sx
    return (sw * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det


def host(T, J):
    """Fit the transfer matrix's couplings onto the target's.

    Returns (a, b, rel_err, contrast). `contrast` is the RMS hosted swing in doubly
    stochastic units -- what has to beat the readout noise -- and `rel_err` is the residual
    as a fraction of it, which is the part of J the chip failed to host."""
    J = np.asarray(J, float)
    x = offdiag(J)
    y = offdiag(T, len(J))
    a, b = _affine(x, y)
    a, b = float(a), float(b)
    resid = y - (a * x + b)
    contrast = abs(a) * float(np.sqrt(np.mean(x * x)))
    rel = float(np.sqrt(np.mean(resid * resid))) / max(contrast, 1e-15)
    return a, b, rel, contrast


def config_energies(T, J, fit=None, cfgs=None) -> np.ndarray:
    """Ising energy of every configuration, read out of one measured transfer matrix.

    No further optical measurement: the 2^n landscape is 2^n dot products against the same
    n(n-1)/2 couplings the four shots already returned."""
    J = np.asarray(J, float)
    n = len(J)
    a, b = (host(T, J)[:2] if fit is None else fit)
    S = offdiag(T, n)
    cfgs = configs(n) if cfgs is None else np.asarray(cfgs, float)
    iu = np.triu_indices(n, 1)
    q = 2.0 * (cfgs[:, iu[0]] * cfgs[:, iu[1]]) @ S
    ferro = cfgs.sum(1) ** 2 - n          # what a non-zero b adds, and nothing else
    return -0.5 * (q - b * ferro) / a


def normalise(T, iters: int = 200) -> np.ndarray:
    """Strip per-port coupling and per-detector gain by projecting onto doubly stochastic.

    Safe here and only here: the Ising probe is a *bright* one by construction, because
    `encode` refuses a design whose hosted swing does not beat the readout noise. On a dim
    probe this is the wrong tool and `theory.drift.fit_gains` against a reference is the
    right one -- see SINKHORN_MIN_SNR there."""
    S = np.clip(np.asarray(T, float), 1e-12, None)
    for _ in range(iters):
        S = S / S.sum(-1, keepdims=True)
        S = S / S.sum(-2, keepdims=True)
    return S


def decode(T, J, normalised: bool = True) -> dict:
    """Measured transfer matrix -> the spin configuration the chip calls the ground state."""
    T = normalise(T) if normalised else np.asarray(T, float)
    a, b, rel, contrast = host(T, J)
    cfgs = configs(len(J))
    E = config_energies(T, J, fit=(a, b), cfgs=cfgs)
    k = int(np.argmin(E))
    return {"spins": cfgs[k], "index": k, "energies": E, "configs": cfgs,
            "a": a, "b": b, "rel_err": rel, "contrast": contrast, "T": T}


def anneal(energy_fn, n: int, rng, restarts: int = 8, sweeps: int = 40,
           t0: float = 1.0, t1: float = 0.01):
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

    @property
    def ok(self) -> bool:
        return self.contrast >= MIN_CONTRAST

    def __str__(self) -> str:
        return (f"n={self.n} host err {self.rel_err:.3f} contrast {self.contrast:.4f}"
                f"{'' if self.ok else '  BELOW NOISE'}")


def twin_transfer(twin, phases) -> np.ndarray:
    """|U|^2 for a phase vector: what the four-port probe would return, noise-free."""
    import torch

    ph = phases if torch.is_tensor(phases) else torch.as_tensor(np.asarray(phases, float),
                                                                dtype=torch.float32)
    with torch.no_grad():
        return (twin.matrix(ph).abs() ** 2).numpy().astype(float)


def encode(J, twin, calib: Calibration | None = None, *, trainable=None,
           vmax=VOLTAGE_MAX, restarts: int = 16, steps: int = 500, lr: float = 0.15,
           min_contrast: float = MIN_CONTRAST, seed: int = 0) -> Encoding:
    """Search the reachable set for heater volts whose |U|^2 hosts J.

    `vmax` may be a scalar or a per-channel array; a channel at 0 is pinned dark, which is
    how the three heaters with no confirmed resistance stay out of a design. The restarts
    run as one batch through the twin, which is why this costs a second and not a minute.

    The objective is the affine residual as a fraction of the hosted swing -- scale-free,
    because the readout gauge (a, b) is free -- plus a hinge that only bites when the swing
    drops under `min_contrast`. Without that hinge the optimiser is happy to host J
    perfectly at an amplitude the photodiodes cannot see."""
    import torch

    J = np.asarray(J, float)
    n = len(J)
    calib = Calibration.load_or_nominal() if calib is None else calib
    vmax = np.broadcast_to(np.asarray(vmax, float), (N_HEATERS,)).astype(float)
    mask = np.ones(N_HEATERS, bool) if trainable is None else np.asarray(trainable, bool)
    mask = mask & (vmax > 0)

    x = torch.as_tensor(offdiag(J), dtype=torch.float32)
    vt = torch.as_tensor(vmax, dtype=torch.float32)
    mt = torch.as_tensor(mask.astype(np.float32))
    vpi = torch.as_tensor(calib.vpi, dtype=torch.float32)
    phi0 = torch.as_tensor(calib.phi0, dtype=torch.float32)
    iu = np.triu_indices(n, 1)
    ri = torch.as_tensor(iu[0])
    ci = torch.as_tensor(iu[1])

    g = torch.Generator().manual_seed(seed)
    u = torch.randn(restarts, N_HEATERS, generator=g) * 1.5
    u.requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)

    def evaluate(u):
        v = mt * vt * torch.sigmoid(u)
        ph = np.pi * (v / vpi) ** 2 + phi0
        U = twin.matrix(ph)
        T = (U.abs() ** 2)[..., :n, :n]
        y = ((T + T.transpose(-1, -2)) / 2)[..., ri, ci]
        a, b = _affine(x, y)
        resid = y - (a[..., None] * x + b[..., None])
        contrast = a.abs() * float(np.sqrt(np.mean(offdiag(J) ** 2)))
        rel = resid.pow(2).mean(-1).sqrt() / contrast.clamp_min(1e-12)
        return v, T, a, b, rel, contrast

    for _ in range(steps):
        opt.zero_grad()
        _, _, _, _, rel, contrast = evaluate(u)
        hinge = torch.relu(1 - contrast / min_contrast)
        (rel + hinge).sum().backward()
        opt.step()

    with torch.no_grad():
        v, _, a, b, rel, contrast = evaluate(u)
        # a design that cannot be seen is not a design: prefer the bright ones, and only
        # fall back on brightness itself when none of the restarts cleared the floor.
        bright = contrast >= min_contrast
        score = torch.where(bright, rel, torch.as_tensor(float("inf")))
        k = int(torch.argmin(score)) if bool(bright.any()) else int(torch.argmax(contrast))
        volts = v[k].numpy().astype(float)

    phases = calib.phases(volts)
    T = twin_transfer(twin, phases)
    a, b, rel_err, contrast = host(T, J)
    return Encoding(volts, phases, T, a, b, rel_err, contrast, n)


def reach_dim(twin, calib, trainable=None, vmax=VOLTAGE_MAX, samples: int = 4000,
              seed: int = 0) -> dict:
    """How much of coupling-shape space the heaters can actually steer through.

    The readout gauge (a, b) makes only the *direction* of the off-diagonal vector matter,
    so the target of an n-spin problem is a point on a sphere of dimension m - 2 with
    m = n(n-1)/2: one angle at n = 3, four at n = 4. This samples the voltage box, strips
    the gauge, and returns the participation dimension of what is left -- the number of
    independent directions the mesh can actually swing through. It is the single number the
    n = 3 / n = 4 split falls out of, and it needs no target and no optimisation."""
    import torch

    rng = np.random.default_rng(seed)
    vm = np.broadcast_to(np.asarray(vmax, float), (N_HEATERS,)).astype(float)
    m = np.ones(N_HEATERS, bool) if trainable is None else np.asarray(trainable, bool)
    V = rng.uniform(0, 1, (samples, N_HEATERS)) * vm * m
    with torch.no_grad():
        T = (twin.matrix(torch.as_tensor(calib.phases(V), dtype=torch.float32)
                         ).abs() ** 2).numpy()
    Z = offdiag(T)
    Z = Z - Z.mean(-1, keepdims=True)          # b is a shift, so only the residual counts
    nrm = np.linalg.norm(Z, axis=-1)
    D = Z[nrm > 1e-6] / nrm[nrm > 1e-6, None]  # a is a scale, so only the direction counts
    var = np.linalg.svd(D, compute_uv=False) ** 2
    var = var / var.sum()
    return {"swing": float(nrm.mean()), "spectrum": var,
            "dim": float(np.exp(-(var * np.log(var + 1e-300)).sum())),
            "of": NMODE * (NMODE - 1) // 2 - 1}


def _score(T, J, cfgs_true, E_true, normalised: bool = True) -> dict:
    """One instance: did the chip's own energy ranking find the true ground state?"""
    res = decode(T, J, normalised=normalised)
    E_chip = res["energies"]
    order = {tuple(c): i for i, c in enumerate(res["configs"].tolist())}
    E_ord = np.array([E_chip[order[tuple(c)]] for c in cfgs_true.tolist()])
    k = int(np.argmin(E_ord))
    span = E_true.max() - E_true.min()
    gs = set(np.flatnonzero(np.isclose(E_true, E_true.min())))
    r = np.corrcoef(E_ord, E_true)[0, 1] if len(E_true) > 2 else np.nan
    return {"found_gs": bool(k in gs), "excess": float((E_true[k] - E_true.min())
                                                       / (span + 1e-18)),
            "pearson": float(r), "rel_err": res["rel_err"], "contrast": res["contrast"],
            "chance": len(gs) / len(E_true)}


def _selftest(instances: int = 8, ns=(2, 3, 4), seed: int = 0, steps: int = 400,
              restarts: int = 16, sigma: float = SIGMA_READ, verbose: bool = False):
    """Small instances with known ground states, scored against the twin.

    Three chips, in order of honesty: a heater law that reaches 2 pi (what the mesh could do
    if the thermo-optic coefficient were the 6x6's), the measured law at the 3 V ceiling
    (what this die does), and the measured law on a mesh with 2 percent coupler error
    (what a fabricated instance does, with the design still fitted against the ideal twin).
    The gap between the first two is the price of the reachable set; between the second and
    third, the price of model error."""
    import torch

    from .twin import MeshError, Twin

    ideal = Twin()
    fab = Twin(MeshError.sample(sigma_kappa=0.02, seed=1))
    free = Calibration()                      # Vpi 1.5 V, phi0 0: every phase reachable
    real = Calibration.load_or_nominal()
    cases = [("free phases", free, ideal, None),
             ("measured heaters", real, ideal, measured(real)),
             ("measured + 2% couplers", real, fab, measured(real))]

    rows = []
    for label, calib, chip, mask in cases:
        for n in ns:
            rng = np.random.default_rng(seed + 17 * n)
            acc = []
            for i in range(instances):
                J = random_ising(rng, n=n)
                cfgs_true, E_true = brute_force(J)
                enc = encode(J, ideal, calib, trainable=mask, steps=steps,
                             restarts=restarts, seed=seed + i)
                T = twin_transfer(chip, enc.phases)
                T = np.clip(T + rng.normal(0, sigma, T.shape), 0.0, None)
                s = _score(T, J, cfgs_true, E_true)
                s["design_err"] = enc.rel_err
                s["design_contrast"] = enc.contrast
                acc.append(s)
            row = {"case": label, "n": n,
                   **{k: float(np.nanmean([a[k] for a in acc]))
                      for k in ("design_err", "design_contrast", "rel_err", "contrast",
                                "excess", "pearson", "chance")},
                   "found_gs": float(np.mean([a["found_gs"] for a in acc]))}
            rows.append(row)
            if verbose:
                print(f"  {label:<24} n={n} gs {row['found_gs']:.0%}")

    # the anneal must agree with exhaustion on the chip's own landscape
    rng = np.random.default_rng(seed)
    J = random_ising(rng, n=4)
    enc = encode(J, ideal, free, steps=steps, restarts=restarts)
    res = decode(enc.T, J)
    lut = {tuple(c): e for c, e in zip(res["configs"].tolist(), res["energies"])}
    s_a, e_a = anneal(lambda s: lut[tuple(s)], 4, rng)
    assert np.isclose(e_a, res["energies"].min()), (e_a, res["energies"].min())

    # a free-phase mesh must host the largest instance essentially exactly
    top = [r for r in rows if r["case"] == "free phases"][-1]
    assert top["design_err"] < 0.05, top
    assert top["found_gs"] >= 0.9, top
    # every case must beat blind guessing, pooled over n: a single (case, n) cell of a few
    # instances is a coin flip at n = 4 and asserting on one would make this test flaky
    for label, _c, _chip, _m in cases:
        cell = [r for r in rows if r["case"] == label]
        got = float(np.mean([r["found_gs"] for r in cell]))
        exp = float(np.mean([r["chance"] for r in cell]))
        assert got > exp, (label, got, exp)
    return rows


def digest(rows) -> str:
    w = f"{'case':<24}{'n':>2}{'design err':>12}{'contrast':>10}{'GS found':>10}" \
        f"{'chance':>8}{'excess':>8}{'E-corr':>8}"
    out = [w]
    for r in rows:
        out.append(f"{r['case']:<24}{r['n']:>2}{r['design_err']:>12.3f}"
                   f"{r['contrast']:>10.4f}{r['found_gs']:>9.0%}{r['chance']:>8.0%}"
                   f"{r['excess']:>8.3f}"
                   + (f"{r['pearson']:>8.3f}" if np.isfinite(r["pearson"]) else f"{'--':>8}"))
    return "\n".join(out)


if __name__ == "__main__":
    from .twin import Twin

    _tw, _real = Twin(), Calibration.load_or_nominal()
    for _lbl, _c, _m in (("free phases", Calibration(), None),
                         ("measured heaters", _real, measured(_real))):
        _r = reach_dim(_tw, _c, _m)
        print(f"{_lbl:<24} reachable coupling directions {_r['dim']:.2f} of {_r['of']}"
              f"   (n=3 needs 1, n=4 needs 4)")
    print()
    print(digest(_selftest()))
