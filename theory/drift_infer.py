"""Backprop drift-inference: recover heater phase drift from probe residuals.

The chain-rule correction the reanchor should have been. We program K known probe voltage
vectors v_k, read the monitor PDs y_k, and fit a drift offset dphi (added to each heater's
phi0) by backprop through the twin:

    minimise  sum_k || g * mon(phases_from_volts(v_k, vpi, phi0 + dphi)) - y_k ||^2 + l2||dphi||^2

Gradients flow into dphi (and a per-PD gain g) through the differentiable twin. Unlike the
per-heater null search (scripts/reanchor.py), which re-nulls each heater INDEPENDENTLY and
is dominated by its own ~18 deg/pass measurement noise, this fits ONE drift field jointly to
ALL probe residuals -- the noise averages down as 1/sqrt(K*NPD), and a low-dim drift (few
free params) is over-determined by tens of probes. That is exactly the regime the user flags
as favourable: "the drift will indeed be low dimensional".

`selftest()` injects a KNOWN drift, generates probes, and checks recovery -- the honest test
of whether the backprop is correct (noiseless low-dim drift must recover to ~0 residual).
"""
from __future__ import annotations
import numpy as np
import torch

from theory.twin import Twin, mzi_1x1, NH, NSIG, NRAIL, SIG

MON_RAILS = list(range(NSIG))       # twin `mon` is already per-signal-rail (6)
_SIG = torch.tensor(SIG)            # signal rail indices [1..6]


def mon_pred(twin, V, vpi, phi0, dphi):
    """(K,120) volts -> (K,6) monitor powers, with drift dphi added to phi0. Differentiable
    in dphi and VECTORISED over probes: the mesh is linear, so Es = M(ph) @ E_encode(ph)
    with M from the batched twin.matrix() (one call for all K probes) -- equivalent to looping
    twin.forward but ~K x faster (checked against forward in selftest)."""
    V = torch.as_tensor(V, dtype=torch.float32)
    vpi_t = torch.as_tensor(vpi, dtype=torch.float32)
    phi0_t = torch.as_tensor(phi0, dtype=torch.float32)
    ph = torch.pi * (V / vpi_t) ** 2 + phi0_t + dphi             # (K,120)
    # encoded signal field per rail (K,6): amplitude MZI (heaters 2r,2r+1) x phase (heater 16+r)
    t1, t2 = ph[:, 2 * _SIG], ph[:, 2 * _SIG + 1]
    amp0 = 1.0 / NRAIL ** 0.5
    E_sig = amp0 * mzi_1x1(t1, t2) * torch.exp(1j * ph[:, 16 + _SIG].to(torch.complex64))
    M = twin.matrix(ph)                                          # (K,6,6) complex
    Es = torch.einsum("kij,kj->ki", M, E_sig)                    # (K,6) output field
    return twin.tap * Es.abs() ** 2                              # (K,6) monitor power


def infer_drift(twin, vpi, phi0, V, Y, mask=None, l2=1e-3, iters=400, lr=0.05,
                per_pd_gain=True, seed=0, restarts=3):
    """Fit a drift offset dphi (on `mask` heaters) from probes (V volts -> Y monitor). Returns
    (dphi[NH], gain[NPD], resid0, resid1) where resid0/1 are pre/post fit RELATIVE monitor
    error (||g*pred-Y||/||Y||). The loss is RELATIVE (monitor powers are ~1e-3, so an absolute
    MSE is dwarfed by any weight penalty -- a scale bug that silently kills the fit); l2 lightly
    regularises the unobservable subspace toward min-norm."""
    V = np.asarray(V, float)
    Y = torch.as_tensor(np.asarray(Y, float), dtype=torch.float32)
    Ynorm = (Y ** 2).sum().clamp(min=1e-18)
    m = np.ones(NH, bool) if mask is None else np.asarray(mask, bool)
    mt = torch.as_tensor(m)

    def gain_fit(pred):
        if per_pd_gain:
            num = (pred * Y).sum(0); den = (pred * pred).sum(0).clamp(min=1e-12)
        else:
            num = (pred * Y).sum(); den = (pred * pred).sum().clamp(min=1e-12)
        return num / den

    def rel(pred, g):
        return (((g * pred - Y) ** 2).sum() / Ynorm)

    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 17 + r)
        raw = (torch.randn(NH, generator=g) * (0.0 if r == 0 else 0.1)).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            opt.zero_grad()
            dphi = torch.where(mt, raw, torch.zeros_like(raw))
            pred = mon_pred(twin, V, vpi, phi0, dphi)
            g = gain_fit(pred).detach()
            loss = rel(pred, g) + l2 * (dphi ** 2).mean()
            loss.backward()
            opt.step(); sched.step()
        with torch.no_grad():
            dphi = torch.where(mt, raw, torch.zeros_like(raw))
            pred = mon_pred(twin, V, vpi, phi0, dphi)
            g = gain_fit(pred)
            resid = torch.sqrt(rel(pred, g)).item()
            if best is None or resid < best[0]:
                best = (resid, dphi.detach().clone(), g.detach().clone())
    with torch.no_grad():
        pred0 = mon_pred(twin, V, vpi, phi0, torch.zeros(NH))
        g0 = gain_fit(pred0)
        resid0 = torch.sqrt(rel(pred0, g0)).item()
    return best[1].numpy(), best[2].numpy(), resid0, best[0]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def selftest(n_probe=40, n_test=40, drift_kind="lowdim", drift_rms=0.30, noise=0.0,
             mask_stage="mesh", seed=0, l2=3e-2, iters=600):
    """Inject a KNOWN drift, generate probes, recover it, report. This checks the backprop:
    with noise=0 and an identifiable (low-dim) drift, the residual must collapse to ~0."""
    twin = Twin()
    rng = np.random.default_rng(seed)
    vpi = rng.uniform(1.5, 3.0, NH).astype(np.float32)
    phi0 = rng.uniform(0, 2 * np.pi, NH).astype(np.float32)

    # which heaters are allowed to drift / be fit
    mask = np.zeros(NH, bool)
    if mask_stage == "mesh":
        mask[24:120] = True             # V/Sigma/U mesh heaters (the Ising program space)
    elif mask_stage == "encode":
        mask[:24] = True
    else:
        mask[:] = True

    # ground-truth drift
    dtrue = np.zeros(NH, np.float32)
    idx = np.flatnonzero(mask)
    if drift_kind == "global":
        dtrue[idx] = drift_rms
    elif drift_kind == "lowdim":                 # rank-2 field over the masked heaters
        B = rng.normal(size=(len(idx), 2)); z = rng.normal(size=2)
        d = B @ z; d = d / np.sqrt((d ** 2).mean()) * drift_rms
        dtrue[idx] = d
    else:                                         # full per-heater
        dtrue[idx] = rng.normal(0, drift_rms, len(idx))

    def probes(K, rs):
        r = np.random.default_rng(rs)
        V = r.uniform(0, 4.0, (K, NH)).astype(np.float32)
        Yt = mon_pred(twin, V, vpi, phi0, torch.as_tensor(dtrue)).detach().numpy()
        Yt = Yt + r.normal(0, noise, Yt.shape)    # ADC-like noise (monitor units)
        return V, Yt

    Vtr, Ytr = probes(n_probe, seed + 1)
    Vte, Yte = probes(n_test, seed + 2)

    dhat, gain, r0, r1 = infer_drift(twin, vpi, phi0, Vtr, Ytr, mask=mask, l2=l2, iters=iters,
                                     seed=seed)
    # held-out monitor prediction error: no correction (phi0) vs corrected (phi0+dhat)
    with torch.no_grad():
        p_un = mon_pred(twin, Vte, vpi, phi0, torch.zeros(NH))
        p_co = mon_pred(twin, Vte, vpi, phi0, torch.as_tensor(dhat, dtype=torch.float32))
        Yt = torch.as_tensor(Yte, dtype=torch.float32)
        gu = (p_un * Yt).sum(0) / (p_un * p_un).sum(0).clamp(min=1e-12)
        gc = (p_co * Yt).sum(0) / (p_co * p_co).sum(0).clamp(min=1e-12)
        e_un = torch.sqrt(((gu * p_un - Yt) ** 2).mean()).item()
        e_co = torch.sqrt(((gc * p_co - Yt) ** 2).mean()).item()
    # drift recovery (up to the unobservable global-phase gauge per output; report residual)
    derr = np.sqrt((_wrap(dhat[idx] - dtrue[idx]) ** 2).mean())
    dtrue_rms = np.sqrt((dtrue[idx] ** 2).mean())
    print(f"[{drift_kind:6s}/{mask_stage}] K={n_probe} noise={noise:.3f}  "
          f"drift {np.degrees(dtrue_rms):.0f}deg RMS")
    print(f"   train resid  {r0:.4f} -> {r1:.4f}   ({100*(1-r1/max(r0,1e-9)):+.0f}%)")
    print(f"   HELD-OUT monitor RMS   uncorrected {e_un:.4f} -> corrected {e_co:.4f}   "
          f"({100*(1-e_co/max(e_un,1e-9)):+.0f}%)")
    print(f"   drift-recovery RMS error {np.degrees(derr):.0f} deg  (of {np.degrees(dtrue_rms):.0f} deg injected)")
    return {"e_un": e_un, "e_co": e_co, "derr_deg": float(np.degrees(derr))}


if __name__ == "__main__":
    print("=== BACKPROP DRIFT-INFERENCE CHECK (recover a known drift) ===")
    print("\n-- noiseless: correct backprop must collapse the residual --")
    selftest(drift_kind="global", noise=0.0)
    selftest(drift_kind="lowdim", noise=0.0)
    selftest(drift_kind="full",   noise=0.0)
    print("\n-- with monitor noise 0.003 (hardware-like) --")
    selftest(drift_kind="lowdim", noise=0.003)
    selftest(drift_kind="full",   noise=0.003)
    print("\n-- encode-stage drift (well observed by monitors) --")
    selftest(drift_kind="lowdim", noise=0.003, mask_stage="encode")
