"""6-spin Ising energy evaluated optically on the chip.

Two routes, and the difference matters:

**power route (no homodyne needed).** Shift J positive-definite, J + cI = A^T A with A
real symmetric. Then for a spin vector s in {+-1}^n

    |A s|^2 = s^T (J + cI) s = s^T J s + c n

so the *total output power* is the Ising energy plus a constant. Programme the mesh to A,
launch s as 0/pi input phases at full amplitude, sum the monitor PDs. No LO, no phase
reference, and it uses the bright monitor taps rather than the homodyne PDs.

**homodyne route.** Programme the mesh to J itself, read the complex output y = Js, and
form E = -s . y. Needs the LO and its phase calibration, but gives the local field s_i-wise
— i.e. the per-spin gradient, which is what drives a real annealing loop.

`NUS/Solution to the problems.docx` proposes the reference beam precisely because
intensity-only detection loses the sign. The power route sidesteps it; the homodyne route
uses the hardware the chip already has.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from .program import encode_phases, fit_matrix
from .readout import open_reference, recover
from .twin import NSIG, Twin


def random_ising(rng, n=NSIG, kind="sk"):
    """Symmetric zero-diagonal coupling matrix. 'sk' = Sherrington-Kirkpatrick gaussian."""
    J = rng.normal(size=(n, n)) if kind == "sk" else rng.choice([-1.0, 1.0], (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    return J / np.linalg.norm(J, 2)


def energy(J, s):
    return -0.5 * float(s @ J @ s)


def brute_force(J, n=NSIG):
    """All 2^n configurations with their energies, sorted ascending."""
    cfgs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.array([energy(J, s) for s in cfgs])
    order = np.argsort(E)
    return cfgs[order], E[order]


def psd_factor(J, margin=1.05):
    """A with A^T A = J + cI, c chosen just past -lambda_min so the shift is PSD."""
    w = np.linalg.eigvalsh(J)
    c = margin * max(-w.min(), 0.0) + 1e-6
    S = J + c * np.eye(len(J))
    v, U = np.linalg.eigh(S)
    A = U @ np.diag(np.sqrt(np.clip(v, 0, None))) @ U.T
    return A, c


def spin_phases(s):
    """Encode s in {+-1} as full-amplitude input rails with phase 0 or pi."""
    return encode_phases(np.asarray(s, float).astype(complex))


def chip_energy_power(twin, phases, s, c, scale, rng=None, noise=None):
    """Ising energy from total monitor-PD power (no homodyne)."""
    ph = phases.clone()
    ph[:24] = torch.as_tensor(spin_phases(s), dtype=torch.float32)
    mon = twin.forward(ph)["mon"].detach().numpy()
    if noise is not None:
        mon = noise(mon, rng)
    p = mon / twin.tap
    return -0.5 * (p.sum() / scale - c * NSIG)


def chip_energy_homodyne(twin, phases, s, scale, rng=None, noise=None):
    """Ising energy from the homodyne-recovered field: E = -0.5 s . (J s)."""
    ph = phases.clone()
    ph[:24] = torch.as_tensor(spin_phases(s), dtype=torch.float32)
    ph = open_reference(ph)
    y = recover(twin, ph, rng, noise)
    return -0.5 * float(np.real(np.dot(s, y)) / scale)


def calibrate_scale(vals_chip, vals_true):
    """Single real scale mapping chip readings onto true energies (fitted once per J)."""
    a = np.asarray(vals_chip, float)
    b = np.asarray(vals_true, float)
    A = np.stack([a, np.ones_like(a)], 1)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef


def run(route="power", n_problems=6, trainable=None, twin=None, seed=0, noise=None,
        steps=1200, restarts=2):
    """Programme each J (or its PSD factor), sweep all 64 spin configs, score."""
    twin = twin or Twin()
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_problems):
        J = random_ising(rng)
        cfgs, Etrue = brute_force(J)
        if route == "power":
            A, c = psd_factor(J)
            fit = fit_matrix(A, twin=twin, trainable=trainable, seed=seed * 100 + p,
                             steps=steps, restarts=restarts)
            raw = np.array([chip_energy_power(twin, fit["phases"], s, c, 1.0, rng, noise)
                            for s in cfgs])
        else:
            fit = fit_matrix(J, twin=twin, trainable=trainable, seed=seed * 100 + p,
                             steps=steps, restarts=restarts)
            raw = np.array([chip_energy_homodyne(twin, fit["phases"], s, 1.0, rng, noise)
                            for s in cfgs])

        k, b = calibrate_scale(raw, Etrue)
        Echip = k * raw + b
        gs_true = set(np.flatnonzero(np.isclose(Etrue, Etrue.min())))
        gs_chip = int(np.argmin(Echip))
        gap = (Etrue[gs_chip] - Etrue.min()) / (Etrue.max() - Etrue.min() + 1e-18)
        rows.append({
            "problem": p, "matrix_err": fit["err"],
            "pearson": float(np.corrcoef(Echip, Etrue)[0, 1]),
            "found_gs": bool(gs_chip in gs_true),
            "excess": float(gap),
            "top1_of_true_top5": bool(gs_chip in set(range(5))),
        })
    return rows


def digest(rows, label=""):
    r = float(np.mean([x["pearson"] for x in rows]))
    g = float(np.mean([x["found_gs"] for x in rows]))
    e = float(np.mean([x["excess"] for x in rows]))
    m = float(np.mean([x["matrix_err"] for x in rows]))
    print(f"{label:<34} matrix_err {m:6.3f}   E-corr {r:6.3f}   "
          f"ground state {g:5.0%}   excess {e:6.3f}")
    return {"label": label, "matrix_err": m, "pearson": r, "found_gs": g, "excess": e}
