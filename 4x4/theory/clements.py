"""Exact Clements decomposition of a 4x4 unitary into the physical mesh.

This is the whole reason a unitary chip is worth having. On the 6x6 the meshes are
contractions (odd columns dump an edge output), no formula inverts them, and a target
matrix has to be reached by gradient descent through a surrogate. Here the map from a
target unitary to heater phases is closed form, exact to machine precision, and costs
microseconds.

    theta, phi, alpha = decompose(U)     # U (4x4 unitary) -> 6 MZI pairs + 4 output phases
    assert np.allclose(reconstruct(theta, phi, alpha), U)

Convention (Clements et al., Optica 3, 1460 (2016)). One MZI on modes (m, n):

    T(m, n, theta, phi) = [[e^{i phi} cos t, -sin t],
                           [e^{i phi} sin t,  cos t]]

`phi` is the external phase on the upper input arm, `theta` the internal (arm-difference)
phase. The device is the ordered product, input on the right:

    U = e^{i g} . D(alpha) . T_5 . T_4 . T_3 . T_2 . T_1 . T_0

with the six mode pairs fixed by the rectangular layout (`MESH`) and D a diagonal output
phase screen. `decompose` gauge-fixes `alpha[0] = 0` and hands the leftover back as the
scalar `g`: a global phase is invisible to any detector, so the chip carries three output
trimmers, not four, and 12 + 3 = 15 phase shifters is the whole 4x4 unitary. That is
exactly what the packaged part has (UH1..UH15).
"""

from __future__ import annotations

import numpy as np

NMODE = 4
NMZI = NMODE * (NMODE - 1) // 2  # 6

# Mode pairs in propagation order (input -> output). Derived, not assumed: the nulling
# order in `decompose` produces exactly this sequence, and `_selftest` asserts it.
MESH = ((0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2))

# Which rectangular column each MZI sits in (greedy layering of MESH). Two MZIs in the
# same column act on disjoint mode pairs, so they are driven simultaneously.
COLUMN = (0, 0, 1, 2, 2, 3)
NCOL = 4


def T(m, n, theta, phi, N=NMODE):
    """The MZI transfer matrix embedded in an N-mode identity."""
    t = np.eye(N, dtype=complex)
    c, s, e = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    t[m, m], t[m, n] = e * c, -s
    t[n, m], t[n, n] = e * s, c
    return t


def _Ti(m, n, theta, phi, N=NMODE):
    """T(m, n, theta, -phi) transposed: the factor that nulls an element by acting on the
    right. Its inverse is T(m, n, theta, phi), which is what the mesh actually contains."""
    return T(m, n, theta, -phi, N).T


def _null_right(row, col, U):
    """Parameters of the `_Ti` on modes (col, col+1) that zeroes U[row, col] from the right."""
    if U[row, col + 1] == 0:
        return col, col + 1, np.pi / 2, 0.0
    r = U[row, col] / U[row, col + 1]
    return col, col + 1, float(np.arctan(abs(r))), float(np.angle(r))


def _null_left(row, col, U):
    """Parameters of the `T` on modes (row-1, row) that zeroes U[row, col] from the left."""
    if U[row - 1, col] == 0:
        return row - 1, row, np.pi / 2, 0.0
    r = -U[row, col] / U[row - 1, col]
    return row - 1, row, float(np.arctan(abs(r))), float(np.angle(r))


def _commute(m, n, theta, phi, d):
    """Move one inverse MZI through the diagonal: T(m,n,theta,phi)^-1 . D = D' . T(m,n,theta,phi').

    Both sides are 2x2 on modes (m, n) and the identity holds exactly, which is what lets
    the algorithm's mid-mesh diagonal be pushed all the way out to the output screen.
    Returns (phi', d') with d' a copy of d carrying the two updated entries."""
    a, b = np.angle(d[m]), np.angle(d[n])
    d = d.copy()
    d[m] = np.exp(1j * (b - phi - np.pi))
    d[n] = np.exp(1j * b)
    return float(a - b + np.pi), d


def decompose(U, tol: float = 1e-9):
    """4x4 unitary -> (theta[6], phi[6], alpha[4], global_phase) in `MESH` order.

    `alpha[0]` is always 0: the overall phase is pulled out into `global_phase`, which no
    detector can see and no heater needs to make.

    Raises ValueError if U is not unitary to `tol`; the whole point of this chip is that
    the target is a unitary, and silently decomposing something else would hand back
    phases that program a matrix nobody asked for."""
    U = np.asarray(U, dtype=complex)
    if U.shape != (NMODE, NMODE):
        raise ValueError(f"expected a {NMODE}x{NMODE} matrix, got {U.shape}")
    err = np.abs(U.conj().T @ U - np.eye(NMODE)).max()
    if err > tol:
        raise ValueError(f"matrix is not unitary (max |U^H U - I| = {err:.2e} > {tol:.0e})")

    V = U.copy()
    right, left = [], []  # `right` in application order, `left` outermost-last
    for k, i in enumerate(range(NMODE - 2, -1, -1)):
        if k % 2 == 0:
            for j in reversed(range(NMODE - 1 - i)):
                p = _null_right(i + j + 1, j, V)
                right.append(p)
                V = V @ _Ti(*p)
        else:
            for j in range(NMODE - 1 - i):
                p = _null_left(i + j + 1, j, V)
                left.append(p)
                V = T(*p) @ V

    # V is now diagonal:  (prod left) . U . (prod right) = D, so
    # U = left_1^-1 .. left_k^-1 . D . T(right_n) .. T(right_1).
    # Push each inverse through D, innermost first, until D reaches the output.
    d = np.diag(V).copy()
    pushed = []
    for m, n, th, ph in reversed(left):
        ph2, d = _commute(m, n, th, ph, d)
        pushed.append((m, n, th, ph2))
    pushed.reverse()  # back to left's original order: T'_1 .. T'_k

    # Propagation order (input -> output): the right factors as applied, then the pushed
    # left factors reversed, then D.
    order = list(right) + list(reversed(pushed))
    pairs = tuple((m, n) for m, n, _, _ in order)
    if pairs != MESH:
        raise AssertionError(f"decomposition produced mesh order {pairs}, expected {MESH}")

    theta = np.array([p[2] for p in order], float)
    phi = np.array([p[3] for p in order], float)
    alpha = np.angle(d).astype(float)
    g = float(alpha[0])
    return theta, phi, alpha - g, g


def reconstruct(theta, phi, alpha=None, global_phase: float = 0.0):
    """(theta[6], phi[6], alpha[4]) in `MESH` order -> the 4x4 unitary the mesh realises.

    `global_phase` is only ever needed to reproduce a decomposition exactly; leave it at 0
    for anything the hardware has to make."""
    theta = np.asarray(theta, float).ravel()
    phi = np.asarray(phi, float).ravel()
    if theta.size != NMZI or phi.size != NMZI:
        raise ValueError(f"expected {NMZI} theta and {NMZI} phi, got {theta.size}/{phi.size}")
    U = np.eye(NMODE, dtype=complex)
    for k, (m, n) in enumerate(MESH):
        U = T(m, n, theta[k], phi[k]) @ U
    if alpha is not None:
        U = np.diag(np.exp(1j * np.asarray(alpha, float).ravel())) @ U
    return U * np.exp(1j * global_phase)


def random_unitary(rng=None):
    """Haar-random 4x4 unitary, for round-trip tests and random target programming."""
    rng = np.random.default_rng() if rng is None else rng
    z = (rng.normal(size=(NMODE, NMODE)) + 1j * rng.normal(size=(NMODE, NMODE))) / 2**0.5
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _columns():
    """Greedy layering of MESH into rectangular columns; the source of `COLUMN`."""
    cols, busy = [], []
    for m, n in MESH:
        for c, used in enumerate(busy):
            if m not in used and n not in used:
                cols.append(c)
                used.update((m, n))
                break
        else:
            busy.append({m, n})
            cols.append(len(busy) - 1)
    return tuple(cols)


def _selftest(n: int = 500, seed: int = 0):
    """Round-trip every claim this module makes. No hardware, no torch."""
    rng = np.random.default_rng(seed)
    assert _columns() == COLUMN, (_columns(), COLUMN)
    worst = 0.0
    for _ in range(n):
        U = random_unitary(rng)
        th, ph, al, g = decompose(U)
        assert abs(al[0]) < 1e-12
        worst = max(worst, float(np.abs(reconstruct(th, ph, al, g) - U).max()))
    # degenerate cases the nulling guards exist for
    for U in (np.eye(NMODE, dtype=complex), np.roll(np.eye(NMODE, dtype=complex), 1, 0)):
        th, ph, al, g = decompose(U)
        worst = max(worst, float(np.abs(reconstruct(th, ph, al, g) - U).max()))
    assert worst < 1e-10, f"round-trip error {worst:.2e}"
    return worst


if __name__ == "__main__":
    print(f"clements 4x4: {NMZI} MZIs in {NCOL} columns {COLUMN}")
    print(f"round-trip worst error {_selftest():.2e}  OK")
