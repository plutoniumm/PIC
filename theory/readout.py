"""Homodyne readout: recover the complex output field from photodiode powers.

Each signal rail carries two PDs (`layout.SIG_ROUTE`): a monitor tap and a homodyne PD
fed by the LO reference arm. The reference arms have their own encode phase shifters
(H16 top, H23 bottom), so the LO phase is commandable — step it and you get both
quadratures.

Per rail, with tap fraction `t` and LO combiner fraction `f`:

    mon        = t |E|^2
    homo(psi)  = (1-f)(1-t)|E|^2 + f|LO|^2 + 2 sqrt(f(1-f)(1-t)) |LO| Re(E e^{-i psi})

Two acquisitions (psi = 0, pi/2) plus the monitor and the LO-power PDs determine E
completely. That is the whole trick: **2 shots -> full complex output vector**.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .twin import NSIG, REF_TOP, REF_BOT

LO_PHASE_HEATER = {REF_TOP: 16, REF_BOT: 23}
LO_OF_RAIL = [REF_TOP] * 3 + [REF_BOT] * 3  # layout: rails 0-2 use top LO, 3-5 bottom


def acquire(twin, phases, psi, rng=None, noise=None):
    """One acquisition at LO phase `psi` (radians). Returns (mon[6], homo[6], lo_pow[2]).

    `psi` is applied to BOTH reference arms' encode phase shifters, leaving every signal
    heater untouched — which is exactly how it would be done on hardware.
    """
    ph = phases.clone() if torch.is_tensor(phases) else torch.as_tensor(phases, dtype=torch.float32)
    ph = ph.clone()
    for h in LO_PHASE_HEATER.values():
        ph[h] = ph[h] + psi
    out = twin.forward(ph)
    mon = out["mon"].detach().numpy()
    homo = out["homo"].detach().numpy()
    lo = out["lo_power"].detach().numpy()
    if noise is not None:
        mon, homo, lo = (noise(a, rng) for a in (mon, homo, lo))
    return mon, homo, lo


def open_reference(phases):
    """Hold both reference arms at full transmission (their split MZIs balanced), so the
    LO is bright. Without this the LO can land near a null and the homodyne term vanishes
    — an operational requirement on hardware, not a modelling detail."""
    ph = phases.clone() if torch.is_tensor(phases) else torch.as_tensor(
        np.asarray(phases), dtype=torch.float32).clone()
    for r in (REF_TOP, REF_BOT):
        ph[2 * r] = 0.0
        ph[2 * r + 1] = 0.0
    return ph


def recover(twin, phases, rng=None, noise=None, derotate=True):
    """Complex output field per signal rail, from two LO phases.

    Homodyne is inherently *relative to the LO*: the raw estimate is E * e^{-i arg(LO)}.
    Rails 0-2 reference the top arm and rails 3-5 the bottom one, so there are two
    unknown global phases. Both LOs come from the same laser through the same splitter
    tree, so those two offsets are fixed chip constants — calibrate once, reuse.
    `derotate=True` applies the twin's known LO phases (i.e. assumes that calibration).
    """
    t, f = twin.tap, twin.lo_frac
    m0, h0, lo0 = acquire(twin, phases, 0.0, rng, noise)
    _, h1, _ = acquire(twin, phases, math.pi / 2, rng, noise)

    p_sig = np.clip(m0, 0, None) / t
    lo_pow = np.array([lo0[0] if LO_OF_RAIL[k] == REF_TOP else lo0[1] for k in range(NSIG)])
    lo_pow = np.clip(lo_pow, 1e-18, None)

    C = (1 - f) * (1 - t) * p_sig + f * lo_pow
    A = 2 * math.sqrt(f * (1 - f) * (1 - t)) * np.sqrt(lo_pow)
    E_hat = ((h0 - C) + 1j * (h1 - C)) / A

    if derotate:
        lo = twin.forward(phases if torch.is_tensor(phases) else torch.as_tensor(
            np.asarray(phases), dtype=torch.float32))["lo"].detach().numpy()
        E_hat = E_hat * np.exp(1j * np.angle(lo))
    return E_hat


def lo_health(twin, phases):
    """LO power reaching each signal rail's combiner — the thing that sets homodyne SNR."""
    out = twin.forward(phases if torch.is_tensor(phases) else torch.as_tensor(
        np.asarray(phases), dtype=torch.float32))
    return np.abs(out["lo"].detach().numpy()) ** 2


def true_field(twin, phases):
    out = twin.forward(phases if torch.is_tensor(phases)
                       else torch.as_tensor(np.asarray(phases), dtype=torch.float32))
    return out["field"].detach().numpy()


def recovery_error(twin, phases, rng=None, noise=None, derotate=True):
    """Relative error of the homodyne-recovered field against ground truth."""
    E = true_field(twin, phases)
    Eh = recover(twin, phases, rng, noise, derotate=derotate)
    return float(np.linalg.norm(Eh - E) / max(np.linalg.norm(E), 1e-18)), E, Eh
