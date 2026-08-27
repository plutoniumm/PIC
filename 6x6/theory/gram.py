"""Programme the chip for Ising by matching the GRAM matrix, not the transfer matrix.

For the power route the measured quantity is |M s|^2 = s^T Re(M^H M) s, so the chip only
has to reproduce Re(M^H M) — not M. Everything a left-unitary can change is invisible:
the missing output phase shifters, the whole U-mesh output basis, the global phase. That
removes roughly half the constraints of a matrix match, which is why Ising is far more
programmable on this chip than a signed matrix-vector product is.

Reachable problems are therefore  J = Re(M^H M) - c I  with the diagonal removed, for M
in the mesh's image. The chip's singular values decay fast (edge drops bleed the outer
rails), so the hosted J inherits a skewed spectrum — that, not the readout, is the real
limit on which Ising instances fit.
"""

from __future__ import annotations

import numpy as np
import torch

from .twin import NH, NSIG, Twin


def gram_of(M):
    """Re(M^H M) — the quadratic form the total output power implements."""
    return torch.real(torch.conj(M).transpose(-2, -1) @ M)


def hosted_J(M):
    """The zero-diagonal coupling matrix a given transfer matrix hosts."""
    G = gram_of(M) if torch.is_tensor(M) else np.real(np.conj(M).T @ M)
    G = G.detach().numpy() if torch.is_tensor(G) else G
    J = G - np.diag(np.diag(G))
    return J


def gram_loss(M, J, eps=1e-18):
    """Scale-invariant error of the hosted coupling matrix against target J.

    The diagonal of the Gram matrix is a free additive shift (the constant c), so it is
    excluded — only off-diagonal couplings are scored, exactly what the Ising energy sees.
    """
    G = gram_of(M)
    n = G.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool)
    g = G[..., off]
    t = torch.as_tensor(J, dtype=g.dtype)[off]
    s = (g * t).sum(-1) / (torch.sum(t * t) + eps)
    resid = ((g - s.unsqueeze(-1) * t) ** 2).sum(-1)
    return resid / ((s.unsqueeze(-1) * t) ** 2).sum(-1).clamp(min=eps)


def gram_scale(M, J, eps=1e-18):
    """The positive scale relating the hosted coupling matrix to target J.

    Matching J's *shape* leaves the amplitude free, and amplitude is what decides whether
    config-to-config power differences clear the ADC floor. Maximising this alongside the
    shape match is the difference between a working bench demo and a simulation-only one.
    """
    G = gram_of(M)
    n = G.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool)
    g, t = G[..., off], torch.as_tensor(J, dtype=G.dtype)[off]
    return (g * t).sum(-1) / (torch.sum(t * t) + eps)


def fit_gram(J, twin=None, trainable=None, seed=0, steps=2000, lr=0.1, restarts=12,
             throughput=0.0):
    """Heater phases whose hosted coupling matrix matches J up to a positive scale.

    throughput > 0 adds -w*log(scale) to the objective, trading a little shape fidelity
    for coupling amplitude. The shape match has huge slack (it solves exactly with only
    the 82 characterized heaters), so this is close to free.
    """
    twin = twin or Twin()
    Jt = torch.as_tensor(np.asarray(J), dtype=torch.float32)
    mask = np.ones(NH, bool) if trainable is None else np.asarray(trainable, bool)
    tmask = torch.as_tensor(mask).unsqueeze(0)

    g = torch.Generator().manual_seed(seed * 1000 + 3)
    base = torch.rand(restarts, NH, generator=g) * 2 * torch.pi
    free = base.clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for _ in range(steps):
        opt.zero_grad()
        ph = torch.where(tmask, free, base)
        M = twin.matrix(ph)
        loss = gram_loss(M, Jt)
        if throughput:
            # |scale|: a negative scale means the chip hosts -J, whose ground state is the
            # same one after the readout's linear calibration flips the slope. Only the
            # magnitude is physically meaningful, and log of a negative is a dead gradient.
            loss = loss - throughput * torch.log(gram_scale(M, Jt).abs().clamp(min=1e-12))
        loss.sum().backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        ph = torch.where(tmask, free, base)
        M = twin.matrix(ph)
        errs = torch.sqrt(gram_loss(M, Jt)).numpy()
        amp = np.abs(gram_scale(M, Jt).numpy())
        # among restarts that matched the shape, keep the loudest
        good = np.flatnonzero(errs <= max(errs.min() * 3, 1e-3))
        r = int(good[np.argmax(amp[good])]) if len(good) else int(np.argmin(errs))
        return {"phases": ph[r].detach().clone(), "err": float(errs[r]),
                "scale": float(amp[r]), "M": M[r].detach().clone(),
                "J_hosted": hosted_J(M[r])}
