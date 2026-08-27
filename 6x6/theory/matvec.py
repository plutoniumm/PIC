"""Signed matrix-vector product on the chip, read out by homodyne.

The point of the homodyne arms: a power-only chip cannot represent a negative weight or
report a negative result. With the LO you read the field, so both the matrix and the
result carry sign. This module measures how well that actually works under the real
chip's constraints (uncharacterized heaters, ADC floor, non-unitary mesh).

Pipeline per trial: pick a signed target M -> program the mesh (theory.program.fit_matrix)
-> encode a signed vector x on the input rails -> homodyne-recover y -> compare to Mx.
"""

from __future__ import annotations

import numpy as np
import torch

from .program import encode_phases, fit_matrix
from .readout import open_reference, recover
from .twin import NSIG, Twin


def random_signed_matrix(rng, n=NSIG, kind="gauss"):
    if kind == "gauss":
        M = rng.normal(size=(n, n))
    elif kind == "pm1":
        M = rng.choice([-1.0, 1.0], size=(n, n))
    else:
        raise ValueError(kind)
    return M / np.linalg.norm(M, 2)


def program_target(M, trainable=None, twin=None, seed=0, steps=1200, restarts=3):
    """Phases realising M up to scale, plus the achieved matrix and error."""
    twin = twin or Twin()
    fit = fit_matrix(M, twin=twin, trainable=trainable, seed=seed, steps=steps,
                     restarts=restarts)
    return fit


def measure_matvec(twin, mesh_phases, x, rng=None, noise=None):
    """Encode x, run the chip, homodyne-recover the output field."""
    ph = mesh_phases.clone()
    ph[:24] = torch.as_tensor(encode_phases(x), dtype=torch.float32)
    ph = open_reference(ph)
    return recover(twin, ph, rng, noise)


def run(n_trials=8, n_vecs=12, trainable=None, twin=None, seed=0, noise=None,
        steps=1200, restarts=2, kind="gauss"):
    """Full signed-matvec benchmark. Returns per-trial dicts."""
    twin = twin or Twin()
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_trials):
        M = random_signed_matrix(rng, kind=kind)
        fit = program_target(M, trainable=trainable, twin=twin, seed=seed * 100 + t,
                             steps=steps, restarts=restarts)
        ph = fit["phases"]

        errs, sgn, ref, got = [], [], [], []
        for _ in range(n_vecs):
            x = rng.choice([-1.0, 1.0], size=NSIG)
            y_true = M @ x
            y = measure_matvec(twin, ph, x, rng, noise)
            # the chip realises s*M and the encoder normalises x, so compare up to a
            # single real scale fitted across this trial's vectors
            ref.append(y_true)
            got.append(y)
        ref, got = np.array(ref), np.array(got)
        # best real scale + global phase aligning measurement to truth
        s = np.vdot(ref.ravel(), got.ravel()) / max(np.vdot(ref.ravel(), ref.ravel()).real, 1e-18)
        pred = (got / s).real
        errs = np.linalg.norm(pred - ref, axis=1) / np.linalg.norm(ref, axis=1)
        sgn = (np.sign(pred) == np.sign(ref)).mean()

        out.append({"trial": t, "matrix_err": fit["err"], "vec_err": float(errs.mean()),
                    "sign_acc": float(sgn), "cos": float(
                        np.vdot(pred.ravel(), ref.ravel()).real /
                        (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-18))})
    return out


def digest(rows, label=""):
    me = np.mean([r["matrix_err"] for r in rows])
    ve = np.mean([r["vec_err"] for r in rows])
    sa = np.mean([r["sign_acc"] for r in rows])
    cs = np.mean([r["cos"] for r in rows])
    print(f"{label:<34} matrix_err {me:6.3f}   vec_err {ve:6.3f}   "
          f"sign_acc {sa:6.3f}   cos {cs:6.3f}")
    return {"label": label, "matrix_err": me, "vec_err": ve, "sign_acc": sa, "cos": cs}
