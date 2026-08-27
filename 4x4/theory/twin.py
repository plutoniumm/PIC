"""Differentiable twin of the 4x4 mesh: heater phases in, complex 4x4 transfer out.

Unitary by construction. With `MeshError.ideal()` the twin reproduces
`clements.reconstruct` to machine precision, so the exact decomposition and the
simulation are the same object rather than two models that have to be kept in agreement.

Fabrication error enters as a physical perturbation, not a fudge factor: each MZI is
built from two directional couplers with their own power splitting ratios, and a coupler
that is not exactly 50:50 caps the extinction that MZI can reach. That is the dominant
static error on a mesh like this, and it is what stops a nominally unitary chip from
hitting an arbitrary target. Excess loss is separate and makes the mesh sub-unitary --
set it to zero and `U^H U = I` holds to 1e-6 in complex64.

    twin = Twin()                         # ideal
    U = twin.matrix(phases)               # phases: (18,) or (B, 18) -> (4,4) or (B,4,4)
    twin = Twin(MeshError.sample(seed=0)) # a plausible fabricated instance

The 18-vector is in heater order (`theory.layout`): theta heaters carry the *internal*
arm phase, not the Clements angle. `theory.program` owns that conversion; get it wrong
and everything downstream is quietly off by pi - 2t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .clements import MESH, NMODE, NMZI
from .layout import ALPHA_IDX, ALPHA_RAIL, PHI_IDX, THETA_IDX

IDEAL_KAPPA = 0.5


@dataclass
class MeshError:
    """Static fabrication error of one physical chip.

    kappa   (NMZI, 2) power splitting ratio of each MZI's two couplers; 0.5 is ideal.
    loss_db (NMZI,)   excess loss per MZI in dB; 0 keeps the mesh exactly unitary.
    """

    kappa: np.ndarray
    loss_db: np.ndarray

    @classmethod
    def ideal(cls) -> "MeshError":
        return cls(np.full((NMZI, 2), IDEAL_KAPPA), np.zeros(NMZI))

    @classmethod
    def sample(cls, sigma_kappa: float = 0.02, loss_db: float = 0.0, seed=None) -> "MeshError":
        """A plausible fabricated instance. sigma_kappa=0.02 is a 2 percent-point coupler
        spread, the order a mature C-band process holds across a die."""
        rng = np.random.default_rng(seed)
        k = np.clip(IDEAL_KAPPA + rng.normal(0, sigma_kappa, (NMZI, 2)), 0.02, 0.98)
        return cls(k, np.full(NMZI, float(loss_db)))

    @property
    def extinction_db(self) -> np.ndarray:
        """Best extinction each MZI can reach, given its couplers. An ideal pair is
        infinite; this is the number that decides which targets are actually reachable."""
        a = np.sqrt(self.kappa[:, 0] * self.kappa[:, 1])
        b = np.sqrt((1 - self.kappa[:, 0]) * (1 - self.kappa[:, 1]))
        floor = np.abs(a - b) ** 2
        return -10 * np.log10(np.maximum(floor, 1e-18))

    def as_tensors(self, dtype=torch.float32):
        return (torch.as_tensor(self.kappa, dtype=dtype),
                torch.as_tensor(self.loss_db, dtype=dtype))


def _coupler(kappa, cdtype):
    """Directional coupler as a 2x2 block: [[t, i r], [i r, t]], t^2 + r^2 = 1."""
    t = torch.sqrt(1 - kappa).to(cdtype)
    r = torch.sqrt(kappa).to(cdtype)
    return torch.stack([torch.stack([t, 1j * r], -1),
                        torch.stack([1j * r, t], -1)], -2)


def _mzi_block(theta, phi, k1, k2, amp, cdtype):
    """One MZI as a 2x2 block: coupler, arm phase, coupler, external input phase.

    The leading gauge factor -i e^{-i theta/2} is a common optical phase, invisible in
    intensity, kept only so the ideal limit equals `clements.T` exactly."""
    theta_c, phi_c = theta.to(cdtype), phi.to(cdtype)
    arm = torch.zeros(theta.shape + (2, 2), dtype=cdtype)
    arm[..., 0, 0] = torch.exp(1j * theta_c)
    arm[..., 1, 1] = 1
    ext = torch.zeros_like(arm)
    ext[..., 0, 0] = torch.exp(1j * phi_c)
    ext[..., 1, 1] = 1
    flip = torch.zeros_like(arm)
    flip[..., 0, 0] = 1
    flip[..., 1, 1] = -1

    m = _coupler(k2, cdtype) @ arm @ _coupler(k1, cdtype) @ flip @ ext
    gauge = (-1j * torch.exp(-0.5j * theta_c) * amp.to(cdtype))[..., None, None]
    return gauge * m


class Twin:
    """The 4x4 mesh as a differentiable function of the heater phases."""

    def __init__(self, error: MeshError | None = None, dtype=torch.complex64):
        self.error = MeshError.ideal() if error is None else error
        self.dtype = dtype
        rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.kappa, loss_db = self.error.as_tensors(rdtype)
        self.amp = 10 ** (-loss_db / 20)

    def matrix(self, phases):
        """(18,) or (B, 18) heater phases -> (4,4) or (B,4,4) complex transfer.

        Column j of the result is the output field for light injected in port j."""
        ph = phases if torch.is_tensor(phases) else torch.as_tensor(phases, dtype=torch.float32)
        theta, phi, alpha = ph[..., THETA_IDX], ph[..., PHI_IDX], ph[..., ALPHA_IDX]

        lead = ph.shape[:-1]
        U = torch.eye(NMODE, dtype=self.dtype).expand(*lead, NMODE, NMODE).clone()
        for k, (m, n) in enumerate(MESH):
            blk = _mzi_block(theta[..., k], phi[..., k],
                             self.kappa[k, 0], self.kappa[k, 1], self.amp[k], self.dtype)
            step = torch.eye(NMODE, dtype=self.dtype).expand(*lead, NMODE, NMODE).clone()
            step[..., m, m], step[..., m, n] = blk[..., 0, 0], blk[..., 0, 1]
            step[..., n, m], step[..., n, n] = blk[..., 1, 0], blk[..., 1, 1]
            U = step @ U
        # the output screen has 3 trimmers; rail 0 is the phase reference
        d = torch.zeros(*lead, NMODE, NMODE, dtype=self.dtype)
        idx = torch.arange(NMODE)
        scr = torch.zeros(*lead, NMODE, dtype=alpha.dtype)
        scr[..., torch.as_tensor(ALPHA_RAIL)] = alpha
        d[..., idx, idx] = torch.exp(1j * scr.to(self.dtype))
        return d @ U

    def outputs(self, phases, x=None):
        """Output intensities |U x|^2 for input field `x` (default: all light in port 0)."""
        U = self.matrix(phases)
        if x is None:
            xt = torch.zeros(NMODE, dtype=self.dtype)
            xt[0] = 1
        else:
            xt = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=self.dtype)
            xt = xt.to(self.dtype)
        return (U @ xt.unsqueeze(-1)).squeeze(-1).abs() ** 2


def _selftest(seed: int = 0):
    """The ideal twin must equal the exact decomposition, and stay unitary."""
    from .clements import decompose, random_unitary, reconstruct
    from .layout import pack

    rng = np.random.default_rng(seed)
    twin = Twin(dtype=torch.complex128)
    worst_ideal = worst_unitary = 0.0
    for _ in range(200):
        U = random_unitary(rng)
        t, p, a, _g = decompose(U)
        ph = pack(np.pi - 2 * t, p, a)  # theta heaters carry the internal arm phase
        M = twin.matrix(torch.as_tensor(ph)).numpy()
        worst_ideal = max(worst_ideal, float(np.abs(M - reconstruct(t, p, a)).max()))
        worst_unitary = max(worst_unitary, float(np.abs(M.conj().T @ M - np.eye(NMODE)).max()))

    # batching must agree with the loop, and a fabricated instance must stay unitary
    def heater_phases(U):
        t, p, a, _g = decompose(U)
        return pack(np.pi - 2 * t, p, a)

    phs = np.stack([heater_phases(random_unitary(rng)) for _ in range(7)])
    Ub = twin.matrix(torch.as_tensor(phs)).numpy()
    worst_batch = max(float(np.abs(Ub[i] - twin.matrix(torch.as_tensor(phs[i])).numpy()).max())
                      for i in range(len(phs)))
    err = Twin(MeshError.sample(seed=1), dtype=torch.complex128)
    Ue = err.matrix(torch.as_tensor(phs[0])).numpy()
    worst_err_unitary = float(np.abs(Ue.conj().T @ Ue - np.eye(NMODE)).max())

    assert worst_ideal < 1e-10, worst_ideal
    assert worst_unitary < 1e-10, worst_unitary
    assert worst_batch < 1e-12, worst_batch
    assert worst_err_unitary < 1e-10, worst_err_unitary
    return worst_ideal, worst_unitary, worst_batch, worst_err_unitary


if __name__ == "__main__":
    a, b, c, d = _selftest()
    print(f"ideal twin vs exact decomposition : {a:.2e}")
    print(f"unitarity (lossless)              : {b:.2e}")
    print(f"batched vs looped                 : {c:.2e}")
    print(f"unitarity with coupler error      : {d:.2e}")
    e = MeshError.sample(sigma_kappa=0.02, seed=0)
    print(f"sampled couplers -> extinction {np.round(e.extinction_db, 1)} dB")
