"""Inverse design: find heater phases that make the chip realise a target 6x6 map.

There is no Clements nulling here — the mesh is not unitary (edge drops dump light), so
the target is matched numerically through the differentiable twin. The chip is a
contraction, so a target is only ever realisable up to a global scale `s`; `s` is solved
in closed form at every step and never counted as error.

Only *characterized* heaters are optimisable. Everything else is frozen at a fixed random
phase, which is the honest model of an uncalibrated heater: it is not zero, and you
cannot command it.
"""

from __future__ import annotations

import numpy as np
import torch

from .twin import NH, NSIG, Twin


def best_scale(M, T):
    """Complex scale s minimising ||M - s T||_F, in closed form."""
    num = torch.sum(torch.conj(T) * M)
    den = torch.sum(torch.conj(T) * T).real + 1e-18
    return num / den


def match_error(M, T):
    """Scale-invariant relative Frobenius error."""
    s = best_scale(M, T)
    return (torch.linalg.norm(M - s * T) / torch.linalg.norm(s * T)).item()


def diag_match_loss(M, T, eps=1e-18):
    """Relative error of M against T up to a global scale AND a left diagonal phase.

    The mesh has no output phase shifters after U, so the chip physically realises
    D @ T for some diagonal unitary D. D is a fixed consequence of the programmed
    phases and is directly measurable by homodyne with a known input, so it is
    correctable in post-processing — it should not be scored as programming error.

    Closed form: per row, the optimal phase aligns M_k with T_k, leaving
      loss = sum_k ||M_k||^2 - (sum_k |<T_k, M_k>|)^2 / sum_k ||T_k||^2.
    """
    ip = torch.sum(torch.conj(T) * M, dim=-1).abs()      # (..., 6) per-row overlap
    nt = torch.sum(T.abs() ** 2)
    nm = torch.sum(M.abs() ** 2, dim=(-2, -1))
    num = ip.sum(-1)
    resid = torch.clamp(nm - num ** 2 / (nt + eps), min=0.0)
    return resid / (num ** 2 / (nt + eps) + eps)


def diag_match_error(M, T):
    return float(torch.sqrt(diag_match_loss(M, T)))


def fit_matrix(target, twin=None, trainable=None, seed=0, steps=1500, lr=0.05,
               restarts=3, verbose=False, up_to_diag=True):
    """Optimise heater phases so `twin.matrix(phases)` matches `target`.

    up_to_diag: score (and optimise) modulo a left diagonal output phase, which the mesh
    cannot set and the readout can measure — see `diag_match_loss`. Set False for the
    strict match.
    trainable: bool[120] of commandable heaters (default all). Frozen heaters take a
    fixed random phase drawn per restart — the honest model of an uncalibrated heater.
    Returns dict(phases, err, err_strict, scale, M).
    """
    twin = twin or Twin()
    T = torch.as_tensor(np.asarray(target), dtype=torch.complex64)
    mask = np.ones(NH, bool) if trainable is None else np.asarray(trainable, bool)
    tmask = torch.as_tensor(mask).unsqueeze(0)

    g = torch.Generator().manual_seed(seed * 1000 + 7)
    base = torch.rand(restarts, NH, generator=g) * 2 * torch.pi   # all restarts at once
    free = base.clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for it in range(steps):
        opt.zero_grad()
        ph = torch.where(tmask, free, base)
        M = twin.matrix(ph)
        if up_to_diag:
            per = diag_match_loss(M, T)
        else:
            s = (torch.sum(torch.conj(T) * M, dim=(-2, -1)) /
                 (torch.sum(torch.conj(T) * T).real + 1e-18))
            D = M - s[:, None, None] * T
            per = (torch.sum(D.abs() ** 2, dim=(-2, -1)) /
                   (torch.sum((s[:, None, None] * T).abs() ** 2, dim=(-2, -1)) + 1e-18))
        per.sum().backward()          # restarts are independent -> one backward
        opt.step()
        sched.step()
        if verbose and it % 300 == 0:
            print(f"  it {it:5d}  best rel {per.min().item()**0.5:.4f}", flush=True)

    with torch.no_grad():
        ph = torch.where(tmask, free, base)
        M = twin.matrix(ph)
        errs = [diag_match_error(M[r], T) if up_to_diag else match_error(M[r], T)
                for r in range(restarts)]
        r = int(np.argmin(errs))
        return {"phases": ph[r].detach().clone(), "err": errs[r],
                "err_strict": match_error(M[r], T),
                "err_diag": diag_match_error(M[r], T),
                "scale": complex(best_scale(M[r], T)), "M": M[r].detach().clone()}


def inner_rails(n):
    """The n most central signal rails. Rails 0 and 5 are the ones the odd-column edge
    drops bleed, so a smaller problem placed in the middle should fare better — this is
    what makes "what size can this chip actually do" a measurable question."""
    order = [2, 3, 1, 4, 0, 5]
    return sorted(order[:n])


def fit_submatrix(target, rails=None, twin=None, trainable=None, seed=0, steps=2000,
                  lr=0.1, restarts=12, up_to_diag=True):
    """Realise an n x n target on a chosen subset of rails, scoring ONLY that sub-block.

    Off-block entries are genuinely irrelevant: the encoder only lights the chosen input
    rails and the readout only reads the chosen output rails, so leakage elsewhere costs
    throughput but not correctness.
    """
    twin = twin or Twin()
    T = torch.as_tensor(np.asarray(target), dtype=torch.complex64)
    n = T.shape[0]
    rails = rails if rails is not None else inner_rails(n)
    ri = torch.as_tensor(rails, dtype=torch.long)
    mask = np.ones(NH, bool) if trainable is None else np.asarray(trainable, bool)
    tmask = torch.as_tensor(mask).unsqueeze(0)

    g = torch.Generator().manual_seed(seed * 1000 + 11)
    base = torch.rand(restarts, NH, generator=g) * 2 * torch.pi
    free = base.clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def sub(M):
        return M.index_select(-2, ri).index_select(-1, ri)

    for _ in range(steps):
        opt.zero_grad()
        ph = torch.where(tmask, free, base)
        S = sub(twin.matrix(ph))
        per = (diag_match_loss(S, T) if up_to_diag else
               torch.stack([torch.tensor(match_error(S[r], T)) for r in range(restarts)]))
        per.sum().backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        ph = torch.where(tmask, free, base)
        S = sub(twin.matrix(ph))
        errs = [diag_match_error(S[r], T) for r in range(restarts)]
        r = int(np.argmin(errs))
        return {"phases": ph[r].detach().clone(), "err": errs[r], "rails": rails,
                "err_strict": match_error(S[r], T), "err_diag": errs[r],
                "M": S[r].detach().clone()}


def realized_matvec(twin, phases, x):
    """Complex output field for an *externally supplied* signal vector x (bypasses the
    encode stage, so this isolates the mesh's map from the encoder's reachability)."""
    out = twin.forward(phases, x_ext=torch.as_tensor(np.asarray(x), dtype=torch.complex64))
    return out["field"].detach().numpy()


def encode_phases(x, twin=None):
    """Heater phases (H0..H23) that encode a complex signal vector x[6] on the signal
    rails, leaving the two reference arms at full amplitude for the LO.

    Per rail: mzi_1x1 gives amplitude cos(d) and phase (t1+t2)/2, then H16+i adds phase.
    x is normalised by its max modulus, since the encoder can only attenuate.
    """
    x = np.asarray(x, complex)
    a = np.abs(x)
    a = a / max(a.max(), 1e-18)
    ph = np.zeros(24)
    for r in range(8):
        if r in (0, 7):  # reference arms: full transmission, zero phase
            ph[2 * r], ph[2 * r + 1], ph[16 + r] = 0.0, 0.0, 0.0
            continue
        k = r - 1
        d = np.arccos(np.clip(a[k], 0, 1))       # amplitude via arm difference
        ph[2 * r], ph[2 * r + 1] = d, -d          # sum = 0 -> no extra phase
        ph[16 + r] = np.angle(x[k])
    return ph
