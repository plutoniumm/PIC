"""Target unitary -> heater voltages.

Two paths, and the difference between them is the whole story of a real chip:

  `phases_for(U)`  exact and instant. Valid for an ideal mesh: the Clements decomposition
                   is a formula, so a target becomes phases in microseconds with no data,
                   no model and no search.

  `refine(U, twin)` the correction for a fabricated one. Coupler error makes the physical
                   mesh differ from the ideal one, so the exact phases land near the
                   target rather than on it. Eighteen parameters warm-started from the
                   exact answer close that gap by gradient descent in well under a second.

On the 6x6 only the second path existed, it started from nothing, and it ran through a
25k-parameter surrogate. Here the formula does most of the work and the optimiser only
cleans up what fabrication got wrong.
"""

from __future__ import annotations

import numpy as np

from .calib import VOLTAGE_MAX
from .clements import NMODE, decompose, reconstruct
from .layout import pack, unpack


def phases_for(U) -> np.ndarray:
    """Target 4x4 unitary -> the 18-long phase vector, exactly. Ideal mesh.

    The decomposition's global phase is dropped: no heater makes it and no detector sees
    it, so the realised matrix equals the target up to that factor, which is what
    `fidelity` measures."""
    t, p, a, _g = decompose(U)
    return pack(np.pi - 2 * t, p, a)  # theta heaters carry the internal arm phase


def unitary_for(phases) -> np.ndarray:
    """Inverse of `phases_for`: the unitary an ideal mesh realises at these phases."""
    theta_int, phi, alpha = unpack(phases)
    return reconstruct((np.pi - theta_int) / 2, phi, alpha)


def volts_for(U, calib, vmax: float = VOLTAGE_MAX):
    """Target unitary -> (18 DAC volts, reachable mask), through a calibration."""
    return calib.volts(phases_for(U), vmax=vmax)


def fidelity(U, V) -> float:
    """|tr(U^H V)| / N: 1 when V equals U up to a global phase, which no detector sees."""
    U, V = np.asarray(U, complex), np.asarray(V, complex)
    return float(abs(np.trace(U.conj().T @ V)) / NMODE)


def refine(U, twin, phases0=None, steps: int = 400, lr: float = 0.05, restarts: int = 1,
           seed: int = 0):
    """Fit the phases so `twin` realises `U` as closely as its hardware error allows.

    Warm-started from the exact ideal decomposition, which is already close, so this is a
    local polish rather than a search. `restarts` > 1 adds random perturbations of the
    warm start and keeps the best, for the occasional target where coupler error moves
    the optimum away from the exact answer.

    Returns (phases, fidelity)."""
    import torch

    Ut = torch.as_tensor(np.asarray(U, complex), dtype=twin.dtype)
    base = phases_for(U) if phases0 is None else np.asarray(phases0, float)
    rng = np.random.default_rng(seed)
    starts = [base] + [base + rng.normal(0, 0.3, base.size) for _ in range(restarts - 1)]

    best = (-1.0, base)
    for s in starts:
        ph = torch.tensor(s, dtype=torch.float64 if twin.dtype == torch.complex128
                          else torch.float32, requires_grad=True)
        opt = torch.optim.Adam([ph], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            V = twin.matrix(ph)
            loss = 1 - (torch.abs(torch.trace(Ut.conj().T @ V)) / NMODE)
            loss.backward()
            opt.step()
        with torch.no_grad():
            f = float(torch.abs(torch.trace(Ut.conj().T @ twin.matrix(ph))) / NMODE)
        if f > best[0]:
            best = (f, ph.detach().numpy().astype(float))
    return best[1], best[0]


def _selftest(n: int = 20, seed: int = 0):
    """Exact on an ideal mesh; measurably degraded then recovered on a fabricated one."""
    import torch

    from .clements import random_unitary
    from .twin import MeshError, Twin

    rng = np.random.default_rng(seed)
    ideal = Twin(dtype=torch.complex128)
    real = Twin(MeshError.sample(sigma_kappa=0.02, seed=1), dtype=torch.complex128)

    exact, raw, fixed = [], [], []
    for _ in range(n):
        U = random_unitary(rng)
        ph = phases_for(U)
        exact.append(fidelity(U, ideal.matrix(torch.as_tensor(ph)).numpy()))
        raw.append(fidelity(U, real.matrix(torch.as_tensor(ph)).numpy()))
        ph2, f = refine(U, real, steps=300)
        fixed.append(f)
    return float(np.min(exact)), float(np.mean(raw)), float(np.mean(fixed))


if __name__ == "__main__":
    e, r, f = _selftest()
    print(f"exact decomposition on the ideal mesh : fidelity {e:.6f} (worst of 20)")
    print(f"same phases on a 2% coupler-error mesh: fidelity {r:.4f} (mean)")
    print(f"after 18-parameter refinement         : fidelity {f:.4f} (mean)")
