"""Identify the mesh from four-port transfer matrices, where the phases are observable.

`learn.unitary_fit` fits the instrument to single-port readings, and its own docstring says
why that is not enough: with light in one port a whole family of phase settings produces
identical readings, so the fit reproduces the data and still programs the wrong matrix.
Every phi0 in `pic_data/calib.json` came from single-port fringe sweeps. Vpi came out of
those cleanly -- it sets the *frequency* of a fringe, which one port sees perfectly well --
but phi0 was never observable from the data that set it.

This module fits the other object: the whole intensity transfer matrix, one column per
input port, measured at many heater states. Four columns break the degeneracy that one
column cannot.

Three things follow from |U|^2 being the observable, and all three are structure rather
than preference.

**Column normalisation is the measurement, not a convenience.** A unitary conserves power,
so every photon entering port k leaves through one of the four outputs and the column sums
to 1. Dividing each column by its own sum removes launch power, fibre coupling and switch
insertion loss *at that heater state* -- which matters here, because the detected power per
port moves by up to 6x across heater states, and a lossless twin has nowhere to put that.
Asking a state-independent PD gain to absorb it drags it to a compromise that fits nothing.
The residual per-detector gain survives normalisation and is fitted, applied *before* the
column division so the model passes through the same normalisation the data did.

**Six of the fifteen mesh phases are gauge and are not fitted.** |D_L U D_R|^2 = |U|^2 for
diagonal unitaries, so the output trimmers, the two column-0 external phases (input phases
on rails 0 and 2) and the common part of the column-1/2 external pair are invisible.
`identifiable` finds them as the rank deficiency of the probe Jacobian rather than
asserting which they are, and freezes them at zero. A fitted value for a gauge coordinate
is a large confident meaningless number, and reporting one as measured is the failure mode
this whole file exists to avoid.

**Which DAC drives which MZI is testable here and nowhere else.** A fringe sweep says a
channel modulates; it does not say which mesh slot it sits in. `search_roles` scores
assignments by held-out fit quality. On the 2026-08-27 sixteen-state set that search says
`theory.twin` is currently consuming the theta channels in the wrong order -- see
`ROLES_LAYOUT` below.

    tr = load("pic_data/sessions/2026-08-27/raw_transfers.json")
    res = fit(tr, roles=ROLES_LAYOUT, calib0=Calibration.load("pic_data/calib.json"))
    print(res.r2_val, res.calib.vpi)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from theory.calib import Calibration
from theory.clements import COLUMN, MESH, NMODE, NMZI
from theory.layout import ALPHA_IDX, N_HEATERS, PHI_IDX, THETA_DAC, THETA_IDX
from theory.twin import MeshError, Twin

KAPPA_HALF_RANGE = 0.45  # kappa stays in 0.5 +/- this, so a coupler can never invert
MIN_SWING_V = 0.2        # a channel this static carries no Vpi information at all
# Prior widths, both measured rather than chosen. 12 percent is what a shared-Vpi fit of all
# eight of a heater's (port, base) fringe traces achieves on this chip -- no heater spans
# even 0.55 pi at its ceiling, so within one trace amplitude and period trade off and the
# information simply is not there to do better. 0.02 is the coupler spread a mature process
# holds across a die, the same number `MeshError.sample` draws from.
VPI_LOG_SD = 0.12
KAPPA_SD = 0.02
# RMS scatter of a normalised transfer entry across repeats at +8 dBm. Only the ratio of
# this to the prior widths matters, and it is what turns two Gaussians into one loss.
SIGMA_T = 0.01


@dataclass(frozen=True)
class RoleMap:
    """Which DAC channel drives each mesh phase. `theta[k]`/`phi[k]` belong to `MESH[k]`.

    The output trimmers are absent on purpose: a left diagonal phase cancels out of
    |U|^2, so no transfer measurement can place them and this model does not pretend to.
    """

    theta: tuple
    phi: tuple

    def __post_init__(self):
        if len(self.theta) != NMZI or len(self.phi) != NMZI:
            raise ValueError(f"expected {NMZI} theta and {NMZI} phi channels")
        if len(set(self.theta) | set(self.phi)) != 2 * NMZI:
            raise ValueError("a DAC channel cannot drive two mesh phases")

    @property
    def dacs(self) -> np.ndarray:
        return np.array(self.theta + self.phi, int)

    def label(self, dac: int) -> str:
        if dac in self.theta:
            return f"theta{self.theta.index(dac)}"
        if dac in self.phi:
            return f"phi{self.phi.index(dac)}"
        return "unused"


# What `theory.twin` implements today: it reads `ph[THETA_IDX]` and uses element k for
# MESH[k], so position in the sorted heater list is the MZI number and the `index` field
# of `theory.layout.HEATERS` is ignored.
ROLES_TWIN = RoleMap(tuple(int(i) for i in THETA_IDX), tuple(int(i) for i in PHI_IDX))

# What `theory.layout` says was MEASURED: report-MZI k at mesh index k-1, matched to the
# H-names on (pins, resistance), giving DAC k -> theta of mesh index 5-k. `THETA_DAC` maps
# mesh index -> DAC, which is the reverse of the order `ROLES_TWIN` assumes.
ROLES_LAYOUT = RoleMap(tuple(int(THETA_DAC[k]) for k in range(NMZI)),
                       tuple(int(i) for i in PHI_IDX))


@dataclass
class Transfers:
    """Column-normalised |U|^2, one 4x4 per heater state, plus the volts that made it."""

    volts: np.ndarray        # (n, N_HEATERS)
    T: np.ndarray            # (n, NMODE detector, NMODE input port), columns sum to 1
    dark: np.ndarray         # the photodiode floor that was subtracted
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.volts)

    def split(self, val_frac: float = 0.25, seed: int = 0):
        """Held out by *state*, never by port: the four ports of one state are one
        measurement of one matrix, and splitting inside it leaks the answer."""
        rng = np.random.default_rng(seed)
        va = rng.choice(len(self), max(1, int(round(val_frac * len(self)))), replace=False)
        tr = np.setdiff1d(np.arange(len(self)), va)
        return tr, va

    @property
    def swing(self) -> np.ndarray:
        return self.volts.max(0) - self.volts.min(0)

    def defect(self) -> np.ndarray:
        """Per-state row-sum defect. Columns are 1 by construction; the rows are the part
        of double stochasticity the normalisation did *not* impose, so this is the honest
        measure of how unitary the chip looks."""
        return np.abs(self.T.sum(2) - 1).mean(1)


def load(path, dark: str = "dark_switch_off") -> Transfers:
    """Read a raw four-port capture. `raw[port][pd]` is volts, so the array transposes.

    The switch-off floor is the right dark: it is measured with the optical path blocked
    but everything else running, so what it removes is TIA offset and ADC bias and nothing
    optical. On this bench it agrees with the laser-off floor to 0.2 mV, which is what
    proves there is no thermal pedestal hiding inside it."""
    d = json.loads(Path(path).read_text())
    V = np.array([s["volts"] for s in d["states"]], float)
    raw = np.array([s["raw"] for s in d["states"]], float)      # (n, port, pd)
    dk = np.array(d[dark], float)
    P = np.clip(raw - dk[None, None, :], 0.0, None)
    T = np.transpose(P / np.maximum(P.sum(2, keepdims=True), 1e-12), (0, 2, 1))
    meta = {k: v for k, v in d.items() if k != "states"}
    meta["path"] = str(path)
    return Transfers(V, T, dk, meta)


class TransferModel(torch.nn.Module):
    """Volts -> column-normalised intensity transfer, differentiable end to end.

    Free parameters, and only the free ones: Vpi for channels that actually moved, phi0 for
    the phase coordinates `identifiable` says a four-port probe separates, the twelve
    coupler ratios, and three detector gain ratios. Everything else is pinned, because a
    parameter that cannot change the prediction will still happily wander during a fit.

    `restarts` independent parameter sets are carried as a leading dimension and trained in
    one backward pass. The loss is periodic in every phi0, so a single descent lands in
    whichever basin it started next to and sequential restarts are the only cure; the twin
    already batches over arbitrary leading dimensions, so paying for thirty-two of them
    costs one matmul instead of thirty-two runs (measured: 32 restarts in 1.15x the wall
    time of 1). The restarts never interact -- the loss is a plain sum over them."""

    def __init__(self, roles: RoleMap, calib0: Calibration | None = None,
                 free_vpi=None, free_phi0=None, error0: MeshError | None = None,
                 fit_couplers: bool = True, restarts: int = 1, seed: int = 0):
        super().__init__()
        c = Calibration() if calib0 is None else calib0
        e = MeshError.ideal() if error0 is None else error0
        g = torch.Generator().manual_seed(seed)
        self.roles, self.R = roles, int(restarts)
        self.register_buffer("theta_dac", torch.as_tensor(np.asarray(roles.theta, int)))
        self.register_buffer("phi_dac", torch.as_tensor(np.asarray(roles.phi, int)))

        vpi0 = torch.log(torch.as_tensor(c.vpi, dtype=torch.float64))
        self.log_vpi = torch.nn.Parameter(vpi0.expand(self.R, N_HEATERS).clone())
        # restart 0 keeps the bootstrap phases; the rest are fresh uniform draws
        ph0 = torch.rand(self.R, N_HEATERS, generator=g, dtype=torch.float64) * 2 * np.pi
        ph0[0] = torch.as_tensor(c.phi0, dtype=torch.float64)
        self.phi0 = torch.nn.Parameter(ph0)
        k = np.clip((e.kappa - 0.5) / KAPPA_HALF_RANGE, -0.999, 0.999)
        kr = torch.atanh(torch.as_tensor(k, dtype=torch.float64))
        self.kappa_raw = torch.nn.Parameter(kr.expand(self.R, NMZI, 2).clone(),
                                            requires_grad=fit_couplers)
        self.log_gain = torch.nn.Parameter(torch.zeros(self.R, NMODE, dtype=torch.float64))

        self.register_buffer("vpi_ref", vpi0.expand(self.R, N_HEATERS).clone())
        self.register_buffer("mask_vpi", _mask(free_vpi))
        self.register_buffer("mask_phi", _mask(free_phi0))
        self.twin = Twin(dtype=torch.complex128)

    @property
    def kappa(self):
        return 0.5 + KAPPA_HALF_RANGE * torch.tanh(self.kappa_raw)

    @property
    def vpi(self):
        return torch.exp(self.vpi_ref + self.mask_vpi * (self.log_vpi - self.vpi_ref))

    @property
    def gains(self):
        """Detector gains, normalised to unit geometric mean: column normalisation removes
        the overall scale, so only three of the four ratios are measurable."""
        return torch.exp(self.log_gain - self.log_gain.mean(-1, keepdim=True))

    def phases(self, V):
        """(B, 16) volts -> (R, B, 16) optical phases, one plane per restart."""
        V = V if torch.is_tensor(V) else torch.as_tensor(V, dtype=torch.float64)
        return (np.pi * (V[None] / self.vpi[:, None]) ** 2
                + self.mask_phi * self.phi0[:, None])

    def forward(self, V):
        ph = self.phases(V)
        canon = torch.zeros(ph.shape, dtype=ph.dtype)
        canon[..., THETA_IDX] = ph[..., self.theta_dac]
        canon[..., PHI_IDX] = ph[..., self.phi_dac]
        self.twin.kappa = self.kappa.permute(1, 2, 0).unsqueeze(-1)  # broadcast over states
        P = self.twin.matrix(canon).abs() ** 2 * self.gains[:, None, :, None]
        return P / P.sum(-2, keepdim=True)

    def mse(self, V, Y):
        """Per-restart mean squared error; summing it trains every restart at once."""
        return ((self(V) - Y[None]) ** 2).mean(dim=(1, 2, 3))

    def prior(self) -> torch.Tensor:
        """Per-restart Gaussian penalty on Vpi and the couplers, in units of variance.

        Not regularisation for its own sake -- both terms are measurements. Vpi came out of
        the shared-trace fringe fits to about `VPI_LOG_SD`, which is real information the
        transfer data should not be allowed to overwrite; and a coupler is a fabricated
        directional coupler with `KAPPA_SD` of spread, not a free knob. Without these the
        fit walks kappa to 0.05 and 0.95 -- couplers with 1 dB of achievable extinction,
        which would mean the MZIs cannot switch at all, while the same channels are
        producing clean fringes on the bench. What it is really doing there is using the
        couplers to absorb the chip's configuration-dependent loss, which a unitary twin
        has no other way to represent."""
        dv = (self.log_vpi - self.vpi_ref) * self.mask_vpi
        dk = self.kappa - 0.5
        return ((dv / VPI_LOG_SD) ** 2).sum(-1) + ((dk / KAPPA_SD) ** 2).sum((-2, -1))

    def n_params(self) -> int:
        return int(self.mask_vpi.sum() + self.mask_phi.sum()) + NMZI * 2 + NMODE - 1

    def pick(self, r: int) -> "TransferModel":
        """The single-restart model holding restart `r`'s parameters."""
        out = TransferModel(self.roles, restarts=1)
        with torch.no_grad():
            for name in ("log_vpi", "phi0", "kappa_raw", "log_gain"):
                getattr(out, name).copy_(getattr(self, name)[r: r + 1])
            out.vpi_ref.copy_(self.vpi_ref[r: r + 1])
            out.mask_vpi.copy_(self.mask_vpi)
            out.mask_phi.copy_(self.mask_phi)
        return out

    def extract(self, calib0: Calibration | None = None, meta: dict | None = None):
        """Pull the fit back out as a (Calibration, MeshError) pair, DAC-indexed.

        The calibration is only meaningful together with the role map, which goes into
        `meta`: `theory.twin` consumes phases in mesh order, and on this chip that is not
        DAC order. Use `phases_for_twin`."""
        if self.R != 1:
            raise ValueError("extract one restart at a time: model.pick(r).extract()")
        with torch.no_grad():
            scale = 1.0 if calib0 is None else float(np.mean(calib0.pd_gain))
            gains = self.gains[0].numpy() * scale
            offs = np.zeros(NMODE) if calib0 is None else calib0.pd_offset
            c = Calibration(self.vpi[0].numpy(),
                            np.mod(self.phi0[0].numpy() * self.mask_phi.numpy(), 2 * np.pi),
                            gains, offs, dict(meta or {}))
            err = MeshError(self.kappa[0].numpy(), np.zeros(NMZI))
        return c, err


def _mask(idx) -> torch.Tensor:
    m = torch.zeros(N_HEATERS, dtype=torch.float64)
    if idx is None:
        m += 1
    else:
        m[torch.as_tensor(np.asarray(idx, int))] = 1
    return m


def phases_for_twin(phases_by_dac, roles: RoleMap) -> np.ndarray:
    """DAC-indexed optical phases -> the vector `theory.twin.Twin.matrix` expects.

    The twin reads `ph[THETA_IDX]` positionally, so element k of that slice has to be
    MZI k's internal phase. Under `ROLES_TWIN` that is the identity; under `ROLES_LAYOUT`
    it is a reversal, and getting it wrong costs held-out R^2 0.84 -> 0.29."""
    ph = np.asarray(phases_by_dac, float)
    out = np.zeros(ph.shape[:-1] + (N_HEATERS,))
    out[..., THETA_IDX] = ph[..., np.asarray(roles.theta, int)]
    out[..., PHI_IDX] = ph[..., np.asarray(roles.phi, int)]
    return out


def identifiable(roles: RoleMap, n_points: int = 8, seed: int = 0, rcond: float = 1e-8):
    """Which DAC phase offsets a four-port intensity probe can separate, and which it cannot.

    Derived, not asserted. The observable is vec(|U|^2) over all four input ports; its
    Jacobian with respect to the twelve mesh phases is rank deficient by exactly the gauge
    group that survives an intensity measurement. A column-pivoted QR picks a maximal
    independent subset of the *coordinates* -- so the frozen ones are genuine coordinates
    of the model rather than an abstract null direction, which is what a fit needs.

    Returns (free, frozen, rank), all DAC channels."""
    dacs = np.asarray(roles.dacs, int)
    J = _jacobian(roles, dacs, n_points, seed)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > rcond * sv[0]).sum())
    _q, _r, piv = _qr_pivot(J)
    free = np.sort(dacs[piv[:rank]])
    return free, np.sort(np.setdiff1d(dacs, free)), rank


def _jacobian(roles: RoleMap, dacs, n_points: int = 8, seed: int = 0) -> np.ndarray:
    """d vec(|U|^2) / d(phase of `dacs`), stacked over random operating points.

    Several points, because a rank deficiency at one operating point can be an accident of
    that point while the gauge is flat everywhere."""
    dacs = torch.as_tensor(np.asarray(dacs, int))
    ti, pi = (torch.as_tensor(np.asarray(x, int)) for x in (roles.theta, roles.phi))
    twin = Twin(dtype=torch.complex128)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_points):
        ph = torch.tensor(rng.uniform(0, 2 * np.pi, N_HEATERS))

        def T(p, ph=ph):
            full = ph.clone()
            full[dacs] = p
            canon = torch.zeros(N_HEATERS, dtype=torch.float64)
            canon[THETA_IDX], canon[PHI_IDX] = full[ti], full[pi]
            return (twin.matrix(canon).abs() ** 2).reshape(-1)

        rows.append(torch.autograd.functional.jacobian(T, ph[dacs]).numpy())
    return np.concatenate(rows, 0)


def _qr_pivot(J):
    """Column-pivoted QR, Businger-Golub, on the small Jacobians this module builds."""
    A = J.copy().astype(float)
    m, n = A.shape
    piv = np.arange(n)
    for k in range(min(m, n)):
        norms = (A[k:, k:] ** 2).sum(0)
        j = int(np.argmax(norms)) + k
        if j != k:
            A[:, [k, j]] = A[:, [j, k]]
            piv[[k, j]] = piv[[j, k]]
        x = A[k:, k]
        if np.linalg.norm(x) < 1e-14:
            continue
        v = x.copy()
        v[0] += np.sign(x[0] or 1.0) * np.linalg.norm(x)
        v /= np.linalg.norm(v)
        A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
    return None, A, piv


@dataclass
class FitResult:
    model: TransferModel
    calib: Calibration
    error: MeshError
    roles: RoleMap
    r2_val: float
    r2_train: float
    r_val: float          # pearson, the figure the twin-vs-chip comparison is quoted in
    mae_val: float
    # Held-out R^2 per input port. Reported separately because the ports are not equally
    # steerable: on 2026-08-27 the dominant entry of port 3 moved over 0.30-0.55 against
    # 0.03-0.90 for port 0, so the phases that would move port 3 are weakly determined and
    # a single pooled number hides that.
    r2_val_by_port: np.ndarray
    r2_val_by_port_modulation: np.ndarray
    free_vpi: np.ndarray
    free_phi0: np.ndarray
    frozen_phi0: np.ndarray
    rank: int


def scores(P, T, ports=None, modulation: bool = False):
    """(R^2, pearson r, mean |entry error|) over the chosen input ports.

    R^2 is against the grand mean of the block being scored: the object predicted is one
    distribution per column, and 0.25 everywhere is the honest null model for it.

    `modulation=True` scores against the per-entry mean *across states* instead. That asks
    the only question a programming model cares about -- does the model predict what the
    heaters changed -- and it is a much harder bar on a port the mesh barely steers, where
    the pooled figure is carried by a static routing pattern any constant reproduces. On
    2026-08-27 that is the whole story of input port 3."""
    if ports is not None:
        P, T = P[..., ports], T[..., ports]
    ref = T.mean(0, keepdims=True) if modulation else T.mean()
    ss = float(((T - P) ** 2).sum())
    st = float(((T - ref) ** 2).sum())
    return (1 - ss / st, float(np.corrcoef(P.ravel(), T.ravel())[0, 1]),
            float(np.abs(P - T).mean()))


_scores = scores


def predict(calib: Calibration, roles: RoleMap, volts, error: MeshError | None = None,
            gains=None) -> np.ndarray:
    """Twin prediction of the column-normalised transfer, for any calibration.

    This is the function the before/after comparison runs on both calibrations, so the
    original calibration is judged through exactly the same normalisation as the fit."""
    twin = Twin(MeshError.ideal() if error is None else error, dtype=torch.complex128)
    ph = phases_for_twin(calib.phases(np.atleast_2d(volts)), roles)
    P = twin.matrix(torch.as_tensor(ph)).abs().numpy() ** 2
    if gains is not None:
        P = P * np.asarray(gains, float)[None, :, None]
    return P / P.sum(1, keepdims=True)


def fit(tr: Transfers, roles: RoleMap = ROLES_LAYOUT, *, calib0: Calibration | None = None,
        error0: MeshError | None = None, steps: int = 2000, lr: float = 0.05,
        restarts: int = 32, val_frac: float = 0.25, seed: int = 0,
        fit_couplers: bool = True, priors: bool = True, split=None,
        verbose: bool = False) -> FitResult:
    """Fit the instrument to measured four-port transfers.

    Warm-start Vpi from the fringe sweeps: they set it well, and a cold joint fit has every
    heater oscillating at the wrong frequency at once (`learn.unitary_fit` measured that
    plateau at R^2 0.42). What this fit is really for is phi0, which single-port sweeps
    could not see. Vpi is refined only for channels that moved by `MIN_SWING_V`; the rest
    keep the bootstrap value, because there is no information about them in this data and a
    free parameter with no information in it is noise with a name.

    The winning restart is picked on the *training* states, so the held-out states are
    never used to choose anything and `r2_val` means what it says. The restart count is not
    a tuning knob to be trimmed: on synthetic data with a known answer 8 restarts recover
    the instrument in roughly half of seeds and 32 in all of them, and the failures are
    silent -- a wrong-basin fit reports a plausible R^2 and a wrong calibration."""
    calib0 = Calibration() if calib0 is None else calib0
    free_phi0, frozen, rank = identifiable(roles, seed=seed)
    moved = np.flatnonzero(tr.swing > MIN_SWING_V)
    free_vpi = np.intersect1d(moved, roles.dacs)
    itr, iva = tr.split(val_frac, seed) if split is None else split

    V = torch.as_tensor(tr.volts, dtype=torch.float64)
    Y = torch.as_tensor(tr.T, dtype=torch.float64)
    m = TransferModel(roles, calib0, free_vpi, free_phi0, error0,
                      fit_couplers=fit_couplers, restarts=restarts, seed=seed + 1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n_obs = float(Y[itr].numel())
    for _ in range(steps):
        opt.zero_grad()
        loss = m.mse(V[itr], Y[itr]) * (n_obs / SIGMA_T ** 2)
        if priors:
            loss = loss + m.prior()
        loss.sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():  # chosen on the training states, so `r2_val` stays held out
        sel = m.mse(V[itr], Y[itr]).numpy()
    best = int(np.argmin(sel))
    if verbose:
        val = m.mse(V[iva], Y[iva]).detach().numpy()
        print(f"  restart {best}: train mse {sel[best]:.5f}, val {val[best]:.5f}; "
              f"across restarts train median {np.median(sel):.5f}, worst {sel.max():.5f}")
    m = m.pick(best)

    with torch.no_grad():
        P = m(V).numpy()[0]
    r2v, rv, mae = scores(P[iva], tr.T[iva])
    r2t, _, _ = scores(P[itr], tr.T[itr])
    by_port = np.array([scores(P[iva], tr.T[iva], [k])[0] for k in range(NMODE)])
    by_port_mod = np.array([scores(P[iva], tr.T[iva], [k], modulation=True)[0]
                            for k in range(NMODE)])
    meta = {"source": "learn.transfer_fit.fit", "data": tr.meta.get("path"),
            "n_states": len(tr), "roles": {"theta": list(roles.theta), "phi": list(roles.phi)},
            "free_vpi": free_vpi.tolist(), "free_phi0": free_phi0.tolist(),
            "gauge_phi0": frozen.tolist(), "observable_rank": rank, "priors": priors,
            "r2_val": r2v, "r2_train": r2t, "pearson_val": rv, "mae_val": mae,
            "r2_val_by_port": by_port.tolist(),
            "r2_val_by_port_modulation": by_port_mod.tolist(),
            "r2_val_ports012": scores(P[iva], tr.T[iva], [0, 1, 2])[0],
            "r2_val_modulation": scores(P[iva], tr.T[iva], modulation=True)[0],
            "val_states": sorted(int(i) for i in iva),
            "note": "phi0 is DAC-indexed and only valid with `roles`; gauge entries are 0 "
                    "because intensity cannot see them, not because they were measured. "
                    "pd_gain carries measured ratios on an arbitrary absolute scale."}
    calib, err = m.extract(calib0, meta)
    return FitResult(m, calib, err, roles, r2v, r2t, rv, mae, by_port, by_port_mod,
                     free_vpi, free_phi0, frozen, rank)


def search_roles(tr: Transfers, candidates=None, *, calib0=None, steps: int = 800,
                 restarts: int = 12, seed: int = 0, top: int = 5, verbose: bool = False):
    """Score DAC-to-mesh-slot assignments by held-out fit quality.

    A fringe sweep proves a channel modulates something; it cannot say which slot. Only the
    channels that actually moved are worth searching -- a static channel contributes a
    constant phase, and where a constant sits is a question this data cannot answer.

    Returns the candidates sorted best first as (r2_val, RoleMap)."""
    cands = list(candidates) if candidates is not None else list(theta_permutations())
    split = tr.split(0.25, seed)
    out = []
    for i, rm in enumerate(cands):
        res = fit(tr, rm, calib0=calib0, steps=steps, restarts=restarts, seed=seed,
                  split=split)
        out.append((res.r2_val, rm))
        if verbose:
            print(f"  [{i + 1}/{len(cands)}] {rm.theta} -> R2 {res.r2_val:+.4f}")
    out.sort(key=lambda t: -t[0])
    return out[:top]


def theta_permutations(base: RoleMap = ROLES_LAYOUT):
    """Every ordering of the six internal-phase channels onto the six MZI slots.

    The channels are known -- six 110-120 ohm heaters that each produce a clean fringe --
    but which MZI each one sits in comes from matching two vendor tables on pins and
    resistance, and that is the kind of claim a transfer measurement can check."""
    from itertools import permutations

    for p in permutations(base.theta):
        yield RoleMap(tuple(p), base.phi)


def _selftest(n: int = 40, seed: int = 0, verbose: bool = True):
    """Recover a known instrument through the real pipeline, and check the gauge claims.

    The synthetic chip is measured the way the bench measures: per-port coupling and
    per-state loss thrown in on top of the true transfer, then column-normalised back out.
    If the normalisation were wrong those nuisances would not cancel and the fit would not
    recover Vpi."""
    rng = np.random.default_rng(seed)
    roles = ROLES_LAYOUT
    truth = Calibration.sample(seed=seed)
    truth.vpi = np.clip(truth.vpi * 3, 1.0, None)          # this die's Vpi are 3-6 V
    truth.phi0 = np.where(_mask(identifiable(roles)[0]).numpy() > 0, truth.phi0, 0.0)
    terr = MeshError.sample(sigma_kappa=0.02, seed=seed)
    gains = np.array([1.1, 1.35, 0.95, 0.72])

    dacs = np.asarray(roles.dacs, int)
    moved = np.array([0, 1, 2, 3, 4, 5, 9, 10])
    V = np.zeros((n, N_HEATERS))
    V[:, moved] = rng.uniform(0, 4.5, (n, moved.size))
    P = predict(truth, roles, V, terr, gains) * rng.uniform(0.2, 1.0, (n, 1, NMODE))
    P = P * rng.uniform(0.5, 1.5, (n, 1, 1))               # per-state loss, must cancel
    T = P / P.sum(1, keepdims=True)
    T = np.clip(T + rng.normal(0, 2e-3, T.shape), 0, None)
    tr = Transfers(V, T / T.sum(1, keepdims=True), np.zeros(NMODE), {"path": "_selftest"})

    free, frozen, rank = identifiable(roles)
    assert rank == 9, rank
    assert frozen.size == 3, frozen
    # freezing is only a legitimate gauge fixing if what is left still spans everything the
    # probe can see; the frozen coordinates are allowed to be non-flat individually, and
    # two of the three here are (only their sum is flat).
    assert identifiable(RoleMap(roles.theta, roles.phi))[2] == rank
    Jfree = _jacobian(roles, free, seed=seed)
    assert np.linalg.matrix_rank(Jfree, tol=1e-8 * np.linalg.norm(Jfree, 2)) == rank

    # the exact symmetries, at finite amplitude rather than to first order: a diagonal
    # phase on either side of U cancels out of |U|^2, so the output trimmers and the
    # external phases of the two column-0 MZIs (which are input phases on rails 0 and 2)
    # can take any value at all.
    exact = [int(roles.phi[k]) for k in range(NMZI) if COLUMN[k] == 0] + list(ALPHA_IDX)
    base = rng.uniform(0, 2 * np.pi, N_HEATERS)
    Vg = rng.uniform(0, 4.5, (3, N_HEATERS))
    c = Calibration(np.full(N_HEATERS, 4.0), base, np.ones(NMODE), np.zeros(NMODE))
    T0 = predict(c, roles, Vg)
    worst = 0.0
    for d in exact:
        b2 = base.copy()
        b2[d] += 1.0
        c2 = Calibration(np.full(N_HEATERS, 4.0), b2, np.ones(NMODE), np.zeros(NMODE))
        worst = max(worst, float(np.abs(predict(c2, roles, Vg) - T0).max()))
    assert worst < 1e-9, f"a gauge coordinate moved the transfer by {worst:.2e}"

    boot = Calibration(truth.vpi * (1 + 0.05 * rng.normal(size=N_HEATERS)),
                       np.zeros(N_HEATERS), np.ones(NMODE), np.zeros(NMODE))
    res = fit(tr, roles, calib0=boot, steps=2000, restarts=32, seed=seed)
    vpi_err = float(np.abs(res.calib.vpi[moved] / truth.vpi[moved] - 1).max())
    gain_err = float(np.abs(res.model.gains[0].detach().numpy()
                            / (gains / gains.prod() ** 0.25) - 1).max())
    if verbose:
        print(f"  identifiable: rank {rank}, free phi0 {free.tolist()}, "
              f"gauge {frozen.tolist()} (+ trimmers {ALPHA_IDX.tolist()})")
        print(f"  recovered a known instrument: held-out R2 {res.r2_val:+.4f}, "
              f"worst Vpi error {100 * vpi_err:.1f}%, worst gain ratio error "
              f"{100 * gain_err:.1f}%")
    assert res.r2_val > 0.97, res.r2_val
    assert vpi_err < 0.10, vpi_err
    assert gain_err < 0.15, gain_err

    # the role search must prefer the truth over the ordering the twin currently assumes
    got = search_roles(tr, [roles, ROLES_TWIN], calib0=boot, steps=800, restarts=12, seed=seed,
                       top=2)
    if verbose:
        print(f"  role search: {[(round(s, 4), rm.theta) for s, rm in got]}")
    assert got[0][1].theta == roles.theta, got
    return {"rank": rank, "r2": res.r2_val, "vpi_err": vpi_err}


if __name__ == "__main__":
    print(f"mesh {MESH}")
    print(f"roles as theory.twin consumes them : theta {ROLES_TWIN.theta}")
    print(f"roles as theory.layout measured    : theta {ROLES_LAYOUT.theta}")
    _selftest()
