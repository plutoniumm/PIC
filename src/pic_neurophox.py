"""Neurophox simulation of the full PIC as an optical SVD: V mesh → Σ → U mesh → PDs.

M = U·Σ·V (the chip applies M to the encoded input). U and V are 6×6 Clements
(rectangular) MZI meshes — built here with neurophox's numpy backend — and Σ is the
diagonal attenuator bank (the singular values), realised as a field multiply between the
two unitary meshes (neurophox meshes are unitary; the loss/attenuation stage is not).

neurophox's numpy backend only touches tensorflow for two dtype constants, so a tiny
module stub satisfies it on Python 3.14 (no tf wheel). The shim + neurophox import are
LAZY — done on the first mesh build — so plain `import src` stays tf-free for the
pure-numpy modelling code.

    from src import pic_neurophox as nx
    V, U = nx.build_mesh(1), nx.build_mesh(2)
    sigma = nx.sigma_from_matrix(A_target)        # Σ ← target singular values (passive, ≤1)
    M = nx.chip_matrix(V, sigma, U)               # the operator the chip realises
    field, intensity = nx.run_optical(V, sigma, U, x)   # propagate one input vector

Self-check demo:
    /usr/local/Caskroom/miniconda/base/envs/pic/bin/python -m src.pic_neurophox
"""
from __future__ import annotations
import sys
import types
import numpy as np

UNITS = 6

_RMNumpy = None


def _rmnumpy():
    """Lazily install the tf-dtype shim and return neurophox's `RMNumpy` class.

    A real module stub (not MagicMock — its auto-attrs recurse in isinstance/Generator
    checks). Only the first call pays the import; the class is cached thereafter.
    """
    global _RMNumpy
    if _RMNumpy is None:
        if "tensorflow" not in sys.modules:
            _tf = types.ModuleType("tensorflow")
            _tf.complex64, _tf.float32 = np.complex64, np.float32
            _tf.__getattr__ = lambda name: (_ for _ in ()).throw(AttributeError(f"tf stub: no {name}"))
            sys.modules["tensorflow"] = _tf
        from neurophox.numpy import RMNumpy
        _RMNumpy = RMNumpy
    return _RMNumpy


def build_mesh(seed: int, units: int = UNITS):
    """A `units`×`units` Clements (rectangular) MZI mesh = one unitary half (U or V).

    The Haar-random phases stand in for one configured chip state; replace by loading
    real heater phases once the heater→phase calibration exists.
    """
    np.random.seed(seed)
    return _rmnumpy()(units=units)


def sigma_from_matrix(A: np.ndarray) -> np.ndarray:
    """Σ stage for a target matrix A: its singular values, scaled to ≤1.

    The attenuators are passive, so the realisable spectrum is normalised by its max
    (σ_max → 1). U/V then set the singular *vectors*; Σ sets the singular *values*.
    """
    sv = np.linalg.svd(A, compute_uv=False)
    return sv / sv.max()


def chip_matrix(V, sigma: np.ndarray, U) -> np.ndarray:
    """The linear operator the configured PIC implements: M = Uᵀ · diag(σ) · Vᵀ.

    (neurophox convention: `transform(x) == matrix.T @ x`, so the propagated operator is
    built from the transposed mesh matrices.)
    """
    return U.matrix.T @ np.diag(sigma) @ V.matrix.T


def run_optical(V, sigma: np.ndarray, U, x: np.ndarray):
    """Propagate x through the physical stages V → Σ → U.

    Returns (output_field, photodiode_intensities). The intensities are |field|² at the
    U-mesh outputs (what an ideal lossless power-monitor would read).
    """
    f_after_V = V.transform(np.atleast_2d(x))[0]    # after V mesh
    f_after_S = sigma * f_after_V                    # after attenuators
    f_out = U.transform(np.atleast_2d(f_after_S))[0]  # after U mesh → at photodiodes
    return f_out, np.abs(f_out) ** 2


def demo():
    """Build a PIC, encode a target's Σ-spectrum, run an input, verify the SVD + power."""
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    V, U = build_mesh(seed=1), build_mesh(seed=2)
    print("PIC set up: two 6x6 Clements MZI meshes (V, U).")
    print(f"  V unitary err |Vᴴ V - I| = {np.abs(V.matrix.conj().T @ V.matrix - np.eye(UNITS)).max():.1e}")
    print(f"  U unitary err |Uᴴ U - I| = {np.abs(U.matrix.conj().T @ U.matrix - np.eye(UNITS)).max():.1e}")

    rng = np.random.default_rng(0)
    A_target = rng.standard_normal((UNITS, UNITS))
    sigma = sigma_from_matrix(A_target)
    print("\nEncoded Σ to the target's (normalised) singular-value spectrum:")
    print(f"  σ = {sigma}")

    M = chip_matrix(V, sigma, U)
    sv_M = np.linalg.svd(M, compute_uv=False)
    print("  realized M = U·Σ·V  →  its singular values reproduce Σ exactly:")
    print(f"  sv(M) = {sv_M}")
    print(f"  max|sv(M) - σ| = {np.abs(np.sort(sv_M)[::-1] - np.sort(sigma)[::-1]).max():.1e}")

    x = rng.standard_normal(UNITS).astype(np.complex128)
    x = x / np.linalg.norm(x)
    f_out, intensity = run_optical(V, sigma, U, x)
    print("\nRan input x through V → Σ → U:")
    print(f"  photodiode |field|² = {intensity}")
    print(f"  field vs M·x : max|err| = {np.abs(f_out - M @ x).max():.1e}")

    pin, pout = np.linalg.norm(x) ** 2, intensity.sum()
    print("\nPower accounting (Σ≤1 ⇒ sub-unitary, never gain):")
    print(f"  power in = {pin:.4f}   power out = {pout:.4f}   σ_max(M) = {sv_M.max():.4f} ≤ 1")


if __name__ == "__main__":
    demo()
