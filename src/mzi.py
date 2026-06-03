"""MZI transfer matrices for the two-phase-shifter mesh, and Clements composition.

Derivation (two arm phase shifters theta1, theta2 between two 50:50 couplers
C = (1/sqrt2)[[1, i], [i, 1]]):

    M = C @ diag(e^{i*theta1}, e^{i*theta2}) @ C
      = i * e^{i*(theta1+theta2)/2} * [[ sin D,  cos D],
                                       [ cos D, -sin D]],     D = (theta1 - theta2)/2

The common mode (theta1+theta2)/2 is a global phase; the differential D sets the
splitting ratio. The 1-in/1-out attenuator (other output dumped) has transmission
t = i*e^{i*(theta1+theta2)/2}*sin D, i.e. amplitude |sin D| -- a singular-value (E)
element. Heater voltage -> phase is quadratic: phi = phi2 * v**2 + phi0.
"""
from __future__ import annotations
import numpy as np

# 50:50 directional coupler
COUPLER = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)


def phase(v, phi2, phi0=0.0):
    """Thermo-optic phase from heater voltage: phi = phi2 * v**2 + phi0."""
    return phi2 * np.asarray(v, float) ** 2 + phi0


def mzi(theta1, theta2):
    """2x2 unitary of one MZI with arm phases theta1, theta2 (numeric composition)."""
    P = np.array([[np.exp(1j * theta1), 0], [0, np.exp(1j * theta2)]], dtype=complex)
    return COUPLER @ P @ COUPLER


def mzi_closed(theta1, theta2):
    """Closed-form equivalent of :func:`mzi` (used to verify the derivation)."""
    s = (theta1 + theta2) / 2.0
    d = (theta1 - theta2) / 2.0
    return 1j * np.exp(1j * s) * np.array(
        [[np.sin(d), np.cos(d)], [np.cos(d), -np.sin(d)]], dtype=complex
    )


def attenuator(theta1, theta2):
    """Complex transmission of a 1-in/1-out MZI attenuator (a Sigma/E element)."""
    return mzi(theta1, theta2)[0, 0]


def clements_layers(N):
    """Mode-pair schedule for an N-mode Clements rectangular mesh.

    Returns a list of layers; each layer is a list of acting mode indices ``m``
    (the MZI couples modes ``m`` and ``m+1``). Even layers start at 0, odd at 1.
    """
    layers = []
    for col in range(N):
        start = 0 if col % 2 == 0 else 1
        layers.append([m for m in range(start, N - 1, 2)])
    return layers


def mesh_unitary(N, params, layers=None, out_phases=None):
    """Compose the NxN unitary from per-MZI ``(theta1, theta2)``.

    params:  dict ``{(layer_index, m): (theta1, theta2)}``
    layers:  from :func:`clements_layers` (default) -- the mesh topology
    out_phases: optional length-N output single-mode phases (the diagonal D matrix)

    NOTE: this composes a *given* parametrisation forward. The inverse (target U ->
    per-MZI angles via Clements nulling) is TODO -- see references/1603.08788.
    """
    layers = layers or clements_layers(N)
    U = np.eye(N, dtype=complex)
    for li, layer in enumerate(layers):
        L = np.eye(N, dtype=complex)
        for m in layer:
            t1, t2 = params[(li, m)]
            L[np.ix_([m, m + 1], [m, m + 1])] = mzi(t1, t2)
        U = L @ U
    if out_phases is not None:
        U = np.diag(np.exp(1j * np.asarray(out_phases, float))) @ U
    return U
