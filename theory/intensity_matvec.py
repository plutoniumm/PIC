"""Signed matrix-vector product on PIC B with INTENSITY-ONLY detection -- no homodyne.

A photodiode reads power |y|^2, so a single coherent read is quadratic in the input
(that is the Ising route). To get a LINEAR matvec we linearise two ways:

1. **Phase-dithering.** Encode the input as intensities p_j >= 0 (field amplitude sqrt(p_j)),
   randomise the per-rail input phases, and AVERAGE the output power. Cross-terms vanish:
       E_phi[ |sum_j M_kj sqrt(p_j) e^{i phi_j}|^2 ] = sum_j |M_kj|^2 p_j = (A p)_k,
   with A = |M|^2 (entrywise) -- a nonnegative matvec, done optically on the monitor PDs.

2. **Differential (4-pass) encoding for signs.** A signed matrix B = B+ - B- and signed
   input x = x+ - x- (nonnegative parts). Then
       B x = (B+ x+ + B- x-) - (B+ x- + B- x+),
   four nonnegative matvecs (two mesh programs x two input vectors), combined in software.
   Full signed matvec from intensity only, at 4x the passes -- and no LO calibration.

The reachable A = |M|^2 is nonnegative but substochastic (mesh losses), so targets are matched
up to a positive scale. `python -m theory.intensity_matvec` runs the twin validation.
"""
from __future__ import annotations

import numpy as np
import torch

from .twin import NH, NSIG, Twin


def abs2(M):
    return M.abs() ** 2


def abs2_loss(M, A, eps=1e-18):
    """Scale-invariant error of the entrywise power |M|^2 against nonnegative target A."""
    P = abs2(M)
    Pf = P.reshape(*P.shape[:-2], -1)
    Af = torch.as_tensor(np.asarray(A), dtype=Pf.dtype).reshape(-1)
    s = (Pf * Af).sum(-1) / ((Af * Af).sum() + eps)
    resid = ((Pf - s.unsqueeze(-1) * Af) ** 2).sum(-1)
    return resid / ((s.unsqueeze(-1) * Af) ** 2).sum(-1).clamp(min=eps)


def fit_abs2(A, twin=None, trainable=None, seed=0, steps=1500, lr=0.08, restarts=12):
    """Heater phases whose realised power matrix |M|^2 matches nonnegative A up to scale."""
    twin = twin or Twin()
    At = np.asarray(A, float)
    mask = np.ones(NH, bool) if trainable is None else np.asarray(trainable, bool)
    tmask = torch.as_tensor(mask).unsqueeze(0)

    g = torch.Generator().manual_seed(seed * 1000 + 5)
    base = torch.rand(restarts, NH, generator=g) * 2 * torch.pi
    free = base.clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        ph = torch.where(tmask, free, base)
        loss = abs2_loss(twin.matrix(ph), At)
        loss.sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        ph = torch.where(tmask, free, base)
        errs = torch.sqrt(abs2_loss(twin.matrix(ph), At)).numpy()
        r = int(np.argmin(errs))
        M = twin.matrix(ph[r])
        return {"phases": ph[r].detach().clone(), "err": float(errs[r]),
                "A_hosted": abs2(M).detach().numpy(), "M": M.detach().clone()}


def fit_abs2_hw(A, hw, twin=None, seed=0, steps=1500, lr=0.08, restarts=12, vmax=4.0):
    """Like fit_abs2 but REACHABILITY-CONSTRAINED: optimise in VOLTAGE space so every phase
    the chip is asked for is commandable. Trainable = characterised heaters, parametrised
    v in (0, vmax) via sigmoid -> phase = phi0 + pi (v/Vpi)^2 (== the twin's law, and
    quantize/volts can reproduce it exactly). Uncharacterised heaters are held at phase 0
    (their nominal 0 V), matching how the runner programs them. Free-phase fitting produces
    unreachable phases that clamp on hardware and wreck |M|^2 -- this avoids that."""
    twin = twin or Twin()
    At = np.asarray(A, float)
    vpi = torch.tensor(np.where(np.isfinite(hw.vpi), hw.vpi, 1.0), dtype=torch.float32)
    phi0 = torch.tensor(np.where(np.isfinite(hw.phi0), hw.phi0, 0.0), dtype=torch.float32)
    kmask = torch.tensor(hw.known)
    g = torch.Generator().manual_seed(seed * 1000 + 9)
    raw = (torch.rand(restarts, NH, generator=g) * 4 - 2).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def phases():
        v = vmax * torch.sigmoid(raw)
        ph = phi0 + np.pi * (v / vpi) ** 2
        return torch.where(kmask, ph, torch.zeros_like(ph))

    for _ in range(steps):
        opt.zero_grad()
        abs2_loss(twin.matrix(phases()), At).sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        ph = phases()
        errs = torch.sqrt(abs2_loss(twin.matrix(ph), At)).numpy()
        r = int(np.argmin(errs))
        M = twin.matrix(ph[r])
        return {"phases": ph[r].detach().clone(), "err": float(errs[r]),
                "A_hosted": abs2(M).detach().numpy(), "M": M.detach().clone()}


def dithered_matvec(twin, phases, p, n_dither, rng, rails=None):
    """Nonnegative matvec (A p) with A = |M|^2, via phase-dithered average output power.
    `p` is the input intensity vector (len NSIG); `rails` restricts the active inputs."""
    p = np.asarray(p, float)
    rails = range(NSIG) if rails is None else rails
    acc = np.zeros(NSIG)
    for _ in range(n_dither):
        x = np.zeros(NSIG, complex)
        for r in rails:
            x[r] = np.sqrt(max(p[r], 0.0)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        out = twin.forward(phases, x_ext=x)
        acc += (out["mon"].detach().numpy() / twin.tap)   # per-rail power |Es|^2
    return acc / n_dither


def signed_matvec(twin, ph_plus, ph_minus, x, n_dither, rng, rails=None, sp=1.0, sm=1.0):
    """Full signed y = B x from four nonnegative intensity passes. ph_plus programs
    |M|^2 = sp.B+, ph_minus programs sm.B-; x is the signed input. Each pass is divided by
    its OWN hosted scale so B+ and B- combine on a common scale (they are hosted at
    different scales -- averaging one scale over both is wrong)."""
    x = np.asarray(x, float)
    xp, xm = np.clip(x, 0, None), np.clip(-x, 0, None)
    Bp_xp = dithered_matvec(twin, ph_plus, xp, n_dither, rng, rails) / sp
    Bm_xm = dithered_matvec(twin, ph_minus, xm, n_dither, rng, rails) / sm
    Bp_xm = dithered_matvec(twin, ph_plus, xm, n_dither, rng, rails) / sp
    Bm_xp = dithered_matvec(twin, ph_minus, xp, n_dither, rng, rails) / sm
    return (Bp_xp + Bm_xm) - (Bp_xm + Bm_xp)


def _score(y_hat, y):
    err = np.linalg.norm(y_hat - y) / max(np.linalg.norm(y), 1e-18)
    sign = float(np.mean(np.sign(y_hat) == np.sign(y)))
    return float(err), sign


def validate(n=5, n_dither=64, trials=4, seed=0, trainable=None, steps=1200):
    """Twin end-to-end: random signed B (n x n), fit B+/B-, recover B x for random x."""
    from .program import inner_rails
    twin = Twin()
    rng = np.random.default_rng(seed)
    rails = inner_rails(n)
    idx = np.array(rails)

    # (1) linearisation check: does the dithered average reproduce the hosted A p?
    A0 = np.abs(rng.normal(size=(NSIG, NSIG)))
    fit0 = fit_abs2(A0, twin, trainable, seed, steps)
    A_host = fit0["A_hosted"]
    p = np.abs(rng.normal(size=NSIG))
    lin = dithered_matvec(twin, fit0["phases"], p, n_dither, rng)
    lin_err = np.linalg.norm(lin - A_host @ p) / np.linalg.norm(A_host @ p)

    # (2) signed matvec on inner rails
    B = rng.normal(size=(n, n))
    Bp = np.zeros((NSIG, NSIG)); Bm = np.zeros((NSIG, NSIG))
    Bp[np.ix_(idx, idx)] = np.clip(B, 0, None)
    Bm[np.ix_(idx, idx)] = np.clip(-B, 0, None)
    fp = fit_abs2(Bp, twin, trainable, seed + 1, steps)
    fm = fit_abs2(Bm, twin, trainable, seed + 2, steps)
    # calibrate the single positive scale each fit hosts (|M|^2 ~ s.B), on inner block
    def scale(fit, T):
        Ah = fit["A_hosted"][np.ix_(idx, idx)]; Tt = T[np.ix_(idx, idx)]
        return float((Ah * Tt).sum() / max((Tt * Tt).sum(), 1e-18))
    sp, sm = scale(fp, Bp), scale(fm, Bm)

    errs, signs = [], []
    for _ in range(trials):
        x6 = np.zeros(NSIG); x6[idx] = rng.normal(size=n)
        y = signed_matvec(twin, fp["phases"], fm["phases"], x6, n_dither, rng, rails,
                          sp=sp, sm=sm)
        y_cal = np.array([y[r] for r in idx])
        e, s = _score(y_cal, B @ x6[idx])
        errs.append(e); signs.append(s)
    return {"n": n, "rails": rails, "linearisation_err": float(lin_err),
            "fit_plus_err": fp["err"], "fit_minus_err": fm["err"],
            "vec_err": float(np.mean(errs)), "sign_acc": float(np.mean(signs)),
            "n_dither": n_dither, "trials": trials}


if __name__ == "__main__":
    from .hw import Hardware
    known = Hardware().known
    for n in (3, 4, 5):
        r = validate(n=n, trainable=known)
        print(f"n={n} rails={r['rails']} | fit+ {r['fit_plus_err']:.3f} fit- {r['fit_minus_err']:.3f}"
              f" | dither-lin err {r['linearisation_err']:.3f} | vec err {r['vec_err']:.3f}"
              f" | sign {r['sign_acc']:.0%}  (dither={r['n_dither']})")
