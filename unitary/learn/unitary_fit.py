"""The efficient surrogate: fit the instrument itself, not a function of it.

Fifty-six parameters describe this chip completely.

    Vpi, phi0   per heater            32
    coupler kappa   per MZI, two each 12
    gain, offset    per photodiode     8
    gain            per input port     4

(36 + 12 + 8 = 56 was the count at 18 heaters; the DAC81416 drives 16.)

Everything else is fixed by the fact that a lossless mesh is unitary. That constraint is
not a regulariser bolted on afterwards, it is the model: `theory.twin` composes MZIs, so
whatever the parameters do the forward map stays a valid physical transfer. A generic
network of the same size can represent maps this chip cannot produce, and spends its
capacity ruling them out from data.

What that buys, against the black-box surrogate in `learn.dpnn`:

  * it extrapolates, because the law is right outside the sampled region too;
  * it inverts, so `theory.program` turns a target into voltages with no search;
  * every parameter is a measurable property of a component, so a fit that goes wrong
    points at the part of the chip that is wrong;
  * it needs far fewer samples, because it is not spending them learning V^2.

Vpi first, then everything else. A joint fit from a nominal start does not converge: with
16 unknown Vpi the forward map oscillates at the wrong frequency in every heater at once,
and no number of restarts finds its way back (measured: R^2 plateaus at 0.42 whether you
allow 4 restarts or 12). Hand it the Vpi that `pic.characterize`'s single-heater fringe
sweeps recover and the same fit reaches R^2 = 1.0000. Bootstrap first; this is a refiner,
not a search.

Observability, stated because it decides the experiment design. Intensity detection sees
|U x|^2 for whatever input x you drive. With light in one port only, a whole family of
phase settings produces identical readings and the fit cannot separate them; the fitted
parameters then reproduce the data and still program the wrong matrix. Drive several input
ports -- this chip has four -- and pass them in as `X`. The single-port default here is for
bootstrap only.
"""

from __future__ import annotations

import numpy as np
import torch

from pic.config import NUM_OUT
from theory.calib import Calibration
from theory.clements import NCOL, NMODE, NMZI
from theory.layout import COLUMN_OF_HEATER, N_HEATERS, PHI_IDX, THETA_IDX
from theory.twin import MeshError, Twin

KAPPA_HALF_RANGE = 0.45  # kappa stays in 0.5 +/- this, so a coupler can never invert
T_REF_C = 26.5  # die temperature the fitted phi0 refers to; dphi_dT carries the rest


class InstrumentModel(torch.nn.Module):
    """The whole rig as one differentiable module: volts in, photodiode volts out."""

    def __init__(self, calib: Calibration | None = None, error: MeshError | None = None,
                 fit_couplers: bool = True, fit_readout: bool = True, xtalk_rank: int = 0):
        super().__init__()
        c = Calibration() if calib is None else calib
        e = MeshError.ideal() if error is None else error

        self.log_vpi = torch.nn.Parameter(torch.log(torch.as_tensor(c.vpi, dtype=torch.float32)))
        self.phi0 = torch.nn.Parameter(torch.as_tensor(c.phi0, dtype=torch.float32))
        k = np.clip((e.kappa - 0.5) / KAPPA_HALF_RANGE, -0.999, 0.999)
        self.kappa_raw = torch.nn.Parameter(torch.atanh(torch.as_tensor(k, dtype=torch.float32)),
                                            requires_grad=fit_couplers)
        self.log_gain = torch.nn.Parameter(
            torch.log(torch.as_tensor(c.pd_gain, dtype=torch.float32)), requires_grad=fit_readout)
        self.offset = torch.nn.Parameter(torch.as_tensor(c.pd_offset, dtype=torch.float32),
                                         requires_grad=fit_readout)
        # Gain on the input side too: the switch and the four fibre-to-chip couplings launch
        # different powers, which a per-detector gain cannot absorb. Without it the two dim
        # ports fitted at R^2 -2.1..0.5 and the whole fit capped at 0.66 (bench, 2026-09-25).
        self.log_gain_in = torch.nn.Parameter(torch.zeros(NMODE), requires_grad=fit_readout)
        # Thermal crosstalk, low rank: heater j's power V_j^2 shifts heater i's phase by
        # (u v^T)_ij V_j^2, in units of heater i's own V^2. u starts at 0, so a fresh model is
        # crosstalk-free; `fit` frees it only after the rest has converged.
        r = max(int(xtalk_rank), 1)
        self.xtalk_u = torch.nn.Parameter(torch.zeros(N_HEATERS, r), requires_grad=False)
        # v random, not constant: identical columns get identical gradients and never split,
        # so a constant start is rank 1 whatever r says
        g = torch.Generator().manual_seed(0)
        self.xtalk_v = torch.nn.Parameter(0.1 * torch.rand(N_HEATERS, r, generator=g),
                                          requires_grad=False)
        # The die moves (the TEC cannot hold it under load: 26.4..28 C in one run), and every
        # phase moves with it. Per heater, rad per C, from the die temperature of each reading.
        self.dphi_dT = torch.nn.Parameter(torch.zeros(N_HEATERS))
        # Heater law past V^2: resistance rises as the heater warms, so at the ceiling --
        # where a run pins most heaters -- phase bends away from V^2. Per heater, on V^4.
        self.bend = torch.nn.Parameter(torch.zeros(N_HEATERS))
        self.twin = Twin()

    @property
    def kappa(self):
        return 0.5 + KAPPA_HALF_RANGE * torch.tanh(self.kappa_raw)

    def phases(self, V, chip=None):
        P = V**2
        P = P + (P @ self.xtalk_v) @ self.xtalk_u.T + self.bend * P**2 / 25.0
        ph = torch.pi * P / torch.exp(self.log_vpi) ** 2 + self.phi0
        if chip is not None:
            ph = ph + self.dphi_dT * (torch.as_tensor(chip, dtype=ph.dtype) - T_REF_C)[..., None]
        return ph

    def forward(self, V, X=None, chip=None):
        """V: (B, 16) volts. X: (B, 4) complex input fields, default all light in port 0.
        chip: (B,) die temperature, C; None leaves the phases at T_REF_C."""
        self.twin.kappa = self.kappa  # live parameter, so gradients reach the couplers
        U = self.twin.matrix(self.phases(V, chip))
        if X is None:
            field = U[..., :, 0]
        else:
            Xt = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=self.twin.dtype)
            Xt = Xt.to(self.twin.dtype) * torch.exp(0.5 * self.log_gain_in).to(self.twin.dtype)
            field = (U @ Xt.unsqueeze(-1)).squeeze(-1)
        return field.abs() ** 2 * torch.exp(self.log_gain) + self.offset

    def fractions(self, V, X=None, chip=None):
        """The share of the light on each detector, as `fractions` makes of a reading."""
        p = self(V, X, chip) - self.offset
        return p / p.sum(-1, keepdim=True)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


    def extract(self):
        """Pull the fitted physics back out as a (Calibration, MeshError) pair."""
        with torch.no_grad():
            calib = Calibration(torch.exp(self.log_vpi).numpy(),
                                np.mod(self.phi0.numpy(), 2 * np.pi),
                                torch.exp(self.log_gain).numpy(), self.offset.numpy(),
                                {"source": "learn.unitary_fit"})
            err = MeshError(self.kappa.numpy(), np.zeros(NMZI))
        return calib, err


def fractions(Y, dark):
    """Photodiode volts -> the share of the light on each detector, per reading.

    One port is lit per reading, so the laser power and that port's coupling onto the chip
    are one scale on the whole row and dividing by the row sum removes them, drift included.
    What is left is where the light goes, which is what a transfer matrix is."""
    y = np.clip(np.asarray(Y, float) - np.asarray(dark, float), 0.0, None)
    return y / np.maximum(y.sum(-1, keepdims=True), 1e-12)


def rel_err(y, yh) -> float:
    """||yh - y|| / ||y||: the number the Run page shows for a predicted transfer."""
    return float(np.linalg.norm(np.asarray(yh) - y) / max(np.linalg.norm(y), 1e-12))


def fit(V, Y, X=None, *, calib0=None, error0=None, steps: int = 3000, lr: float = 0.03,
        val_frac: float = 0.2, restarts: int = 8, seed: int = 0, fit_couplers: bool = True,
        xtalk_rank: int = 2, init=None, chip=None, dark=None,
        verbose: bool = False):
    """Fit the instrument to measured (volts -> photodiode volts) pairs.

    `calib0` is the bootstrap from `pic.characterize` when there is one; without it the
    starting phi0 is unknown and `restarts` random draws are what get past the local minima
    that unknown offsets create.

    The defaults are the budget this actually needs, measured on the mock instrument at
    n = 300 with a clean bootstrap: 2 restarts x 1200 steps reaches R^2 0.42, 4 x 2000
    reaches 0.67, 8 x 3000 reaches 1.0000. Warm-started from a previous round's calibration
    one restart is enough. Returns (model, calibration, error, val_r2).

    `init`, a previously fitted model, is the warm start: the first restart begins from all of
    it -- couplers, both gains and the crosstalk, which a Calibration does not carry -- so a
    refit after drift starts from the last stable point and needs one restart.

    With `dark` (each detector's offset) the fit is on `fractions`, not volts, and
    `model.val_err` is the held-out `rel_err` on them. `chip` is the die temperature of
    each reading, which `dphi_dT` is fitted against."""
    V = np.atleast_2d(np.asarray(V, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    if V.shape[1] != N_HEATERS or Y.shape[1] != NUM_OUT:
        raise ValueError(f"expected V (n,{N_HEATERS}) and Y (n,{NUM_OUT}), "
                         f"got {V.shape} and {Y.shape}")
    rng = np.random.default_rng(seed)
    n = len(V)
    vi = rng.choice(n, max(1, int(val_frac * n)), replace=False)
    tr = np.ones(n, bool)
    tr[vi] = False

    if dark is not None:
        Y = fractions(Y, dark)
    Vt = torch.as_tensor(V, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    Xt = None if X is None else torch.as_tensor(np.atleast_2d(np.asarray(X)))
    Ct = None if chip is None else torch.as_tensor(np.asarray(chip, float), dtype=torch.float32)

    def sub(t, m):
        return None if t is None else t[m]

    def out(m, k):
        f = m.fractions if dark is not None else m
        return f(Vt[k], sub(Xt, k), sub(Ct, k))

    best = (-np.inf, None)
    for r in range(restarts):
        c0 = calib0
        if r > 0 or calib0 is None:  # a fresh random phi0 draw per restart
            base = Calibration() if calib0 is None else calib0
            c0 = Calibration(base.vpi, rng.uniform(0, 2 * np.pi, N_HEATERS),
                             base.pd_gain, base.pd_offset)
        if r == 0 and init is not None:
            import copy

            model = copy.deepcopy(init)
            model.xtalk_u.requires_grad_(False)  # crosstalk moves in its own stage, below
            model.xtalk_v.requires_grad_(False)
        else:
            model = InstrumentModel(c0, error0, fit_couplers=fit_couplers, xtalk_rank=xtalk_rank)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((out(model, tr) - Yt[tr]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            pv = out(model, ~tr).numpy()
        r2 = float(np.mean(1 - ((Y[~tr] - pv) ** 2).sum(0)
                           / (((Y[~tr] - Y[~tr].mean(0)) ** 2).sum(0) + 1e-12)))
        if verbose:
            print(f"  restart {r}: val R2 {r2:+.4f}")
        if r2 > best[0]:
            best = (r2, model)

    r2, model = best
    if xtalk_rank:
        # Crosstalk second, from the converged fit: freed with everything else from a random
        # restart it lets the fit wander off. Kept only if it helps on the held-out split.
        import copy

        xt = copy.deepcopy(model)
        xt.xtalk_u.requires_grad_(True)
        xt.xtalk_v.requires_grad_(True)
        opt = torch.optim.Adam([p for p in xt.parameters() if p.requires_grad], lr=lr / 3)
        for _ in range(steps):
            opt.zero_grad()
            ((out(xt, tr) - Yt[tr]) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            pv = out(xt, ~tr).numpy()
        r2x = float(np.mean(1 - ((Y[~tr] - pv) ** 2).sum(0)
                            / (((Y[~tr] - Y[~tr].mean(0)) ** 2).sum(0) + 1e-12)))
        if verbose:
            print(f"  crosstalk rank {xtalk_rank}: val R2 {r2:+.4f} -> {r2x:+.4f}")
        if r2x > r2:
            r2, model = r2x, xt
    with torch.no_grad():
        model.val_err = rel_err(Y[~tr], out(model, ~tr).numpy())
    calib, err = model.extract()
    return model, calib, err, r2


def column_masks() -> list[tuple[int, np.ndarray]]:
    """(column, gradient mask over DAC channels) for each rectangular column of the mesh.

    One place, so `fit_staged` and `_selftest` cannot disagree about which space the mask is
    in. `phi0`, `COLUMN_OF_HEATER` and the mask are all keyed by DAC channel; `THETA_IDX` and
    `PHI_IDX` are the DAC channels driving mesh element k, in mesh order, so concatenating
    them gives the twelve channels that carry a mesh phase and nothing else."""
    dacs = np.concatenate([THETA_IDX, PHI_IDX])
    out = []
    for c in sorted(set(COLUMN_OF_HEATER[dacs].tolist())):
        m = np.zeros(N_HEATERS)
        m[dacs[COLUMN_OF_HEATER[dacs] == c]] = 1.0
        out.append((int(c), m))
    return out


def fit_staged(V, Y, X=None, *, calib0, steps: int = 600, lr: float = 0.05,
               restarts: int = 4, val_frac: float = 0.2, polish: int = 1500, seed: int = 0,
               verbose: bool = False):
    """Fit phi0 column by column, then polish everything jointly.

    Twelve unknown mesh phases at once is a bad optimisation: the loss is periodic in every
    one of them, so Adam finds a local basin and restarts only re-roll the same dice. The mesh
    itself says how to break it up. Light crosses the columns in order, so column 0's phases
    are determined by data that column 3 cannot touch; fit column 0 alone (four unknowns,
    where a handful of restarts really is exhaustive), freeze it, move to column 1, and each
    stage stays small. The output screen never enters, being invisible in intensity.

    Everything here stays in DAC-channel space, which is the space `phi0` and the gradient
    mask are indexed in. The previous version tested `h in MESH_IDX` with h a DAC channel and
    MESH_IDX a DAC-ordered array of *mesh element numbers*: the two spaces are 0..15 and
    0..5, so the test silently reduced to `h < 6` and staged only the six theta channels,
    handing all six phi channels to the joint polish from a cold start -- the exact failure
    staging exists to avoid -- while running a whole column of Adam against an all-zero mask.

    Returns (model, calibration, error, val_r2), same as `fit`."""
    V = np.atleast_2d(np.asarray(V, float))
    Y = np.atleast_2d(np.asarray(Y, float))
    rng = np.random.default_rng(seed)
    n = len(V)
    vi = rng.choice(n, max(1, int(val_frac * n)), replace=False)
    tr = np.ones(n, bool)
    tr[vi] = False
    Vt = torch.as_tensor(V, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    Xt = None if X is None else torch.as_tensor(np.atleast_2d(np.asarray(X)))

    def val_r2(model):
        with torch.no_grad():
            pv = model(Vt[~tr], None if Xt is None else Xt[~tr]).numpy()
        return float(np.mean(1 - ((Y[~tr] - pv) ** 2).sum(0)
                             / (((Y[~tr] - Y[~tr].mean(0)) ** 2).sum(0) + 1e-12)))

    def run(model, params, mask, n_steps):
        """Adam over `params`, with phi0 updates confined to `mask`."""
        h = None
        if mask is not None:
            m = torch.as_tensor(mask, dtype=torch.float32)
            h = model.phi0.register_hook(lambda g: g * m)
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = ((model(Vt[tr], None if Xt is None else Xt[tr]) - Yt[tr]) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        if h is not None:
            h.remove()
        return model

    model = InstrumentModel(calib0, fit_couplers=False, fit_readout=True)
    for c, active in column_masks():
        best = (-np.inf, None)
        for r in range(restarts):
            trial = InstrumentModel(*model.extract(), fit_couplers=False, fit_readout=True)
            with torch.no_grad():  # re-roll only this column's phases
                idx = np.flatnonzero(active)
                trial.phi0[idx] = torch.as_tensor(
                    rng.uniform(0, 2 * np.pi, idx.size) if r else model.phi0[idx].numpy(),
                    dtype=torch.float32)
            trial = run(trial, [trial.phi0, trial.log_gain, trial.offset], active, steps)
            s = val_r2(trial)
            if s > best[0]:
                best = (s, trial)
        model = best[1]
        if verbose:
            print(f"  column {c}: val R2 {best[0]:+.4f}")

    model = InstrumentModel(*model.extract(), fit_couplers=True, fit_readout=True)
    model = run(model, list(model.parameters()), None, polish)
    calib, err = model.extract()
    return model, calib, err, val_r2(model)


def bootstrap(V_sweeps, Y_sweeps, calib0=None, min_visibility: float = 0.10):
    """Per-heater fringe fits -> a Calibration carrying the Vpi the joint fit needs.

    `V_sweeps[h]` are the voltages swept on heater h with the rest held at a reference,
    `Y_sweeps[h]` the (levels, detectors) block that came back; the detector that saw the
    heater most strongly is the one fitted. The phi0 this recovers is offset by whatever
    the rest of the mesh contributes at that detector, so treat it as a starting point;
    Vpi is the number that matters here and it comes out clean."""
    from pic.characterize import NO_FIT, best_fringe, better

    base = Calibration() if calib0 is None else calib0
    vpi, phi0 = base.vpi.copy(), base.phi0.copy()
    seen = []
    for h, (v, blocks) in enumerate(zip(V_sweeps, Y_sweeps)):
        best = dict(NO_FIT)
        for y in (blocks if isinstance(blocks, (list, tuple)) else [blocks]):
            best = better(best_fringe(v, y, min_visibility=min_visibility), best,
                          min_visibility)
        if best["visibility"] >= min_visibility:
            vpi[h], phi0[h] = best["vpi"], best["phi0"]
            seen.append(h)
    return Calibration(vpi, phi0, base.pd_gain, base.pd_offset,
                       {"source": "learn.unitary_fit.bootstrap", "identified": seen})


def _selftest(n: int = 300, seed: int = 0, verbose: bool = True):
    """Run the real pipeline against a known instrument: sweep, bootstrap, joint fit.

    Also fits the black-box surrogate on the same random data, so the sample-efficiency
    claim in the docstring is measured rather than asserted."""
    from pic.config import DRIVE_MAX_V, drive_to_volts

    from . import dpnn

    # `fit_staged` is never exercised end to end here, so its one index-space hazard is
    # checked directly: the masks live in DAC space and must partition the twelve mesh
    # channels. Ordering them by mesh element instead is what mirrored the twin once already.
    cm = column_masks()
    stages = [np.flatnonzero(m) for _, m in cm]
    assert all(s.size for s in stages), cm
    assert sum(s.size for s in stages) == 2 * NMZI, [s.tolist() for s in stages]
    assert set(np.concatenate(stages).tolist()) == set(THETA_IDX.tolist()) | set(PHI_IDX.tolist())
    assert not any(COLUMN_OF_HEATER[s].max() >= NCOL for s in stages), "output screen staged"

    rng = np.random.default_rng(seed)
    truth_c = Calibration.sample(seed=seed)
    truth_e = MeshError.sample(sigma_kappa=0.02, seed=seed)
    truth = InstrumentModel(truth_c, truth_e)

    def measure(V, X=None):
        with torch.no_grad():
            Y = truth(torch.as_tensor(np.atleast_2d(V), dtype=torch.float32),
                      None if X is None else torch.as_tensor(np.atleast_2d(X))).numpy()
        return Y + rng.normal(0, 3e-4, Y.shape)

    # stage 1: one heater at a time, from a couple of base biases and through every probe
    # input, keeping the cleanest fringe each heater produced anywhere in that set
    from pic.acquisition import grid
    from pic.characterize import probe_inputs, random_bases

    levels = grid()
    bases = random_bases(2)
    inputs = probe_inputs()
    sweeps = []
    for h in range(N_HEATERS):
        blocks = []
        for base in bases:
            for x in inputs:
                V = np.tile(base, (levels.size, 1))
                V[:, h] = levels
                blocks.append(measure(V, np.tile(x, (levels.size, 1))))
        sweeps.append(blocks)
    calib0 = bootstrap([levels] * N_HEATERS, sweeps)
    ident = calib0.meta["identified"]
    vpi_err = float(np.abs(calib0.vpi[ident] - truth_c.vpi[ident]).max())

    # stage 2: random operating points across the input ports, joint fit from the bootstrap.
    # Drawn in the uniform drive command and expanded per channel, which is what the bench
    # does -- so the three channels with no confirmed resistance sit at 0 V here too and
    # their Vpi is correctly left unidentified rather than fitted to data that cannot exist.
    D = rng.uniform(0, DRIVE_MAX_V, (n, N_HEATERS))
    V = drive_to_volts(D)
    X = np.stack([inputs[i] for i in rng.integers(0, len(inputs), n)])
    Y = measure(V, X)

    out = {"vpi_err": vpi_err, "identified": ident,
           "sweep_reads": int(levels.size * N_HEATERS * len(bases) * len(inputs))}
    if verbose:
        print(f"  bootstrap: {out['sweep_reads']} sweep reads -> {len(ident)}/{N_HEATERS} "
              f"heaters identified, worst Vpi error {vpi_err:.5f} V")
        print("  joint fit at the production budget, this takes a few minutes ...")
    phys, c, e, r2 = fit(V, Y, X, calib0=calib0, seed=seed)
    # the network works in drive commands and needs to be told which port is lit; `inputs`
    # includes the two-port splitter probes, which port_features takes as launched intensity
    tel = np.tile([13.0, 0.8, 25.0, 14.0, 25.0], (n, 1))
    _, _, r2d, dmeta = dpnn.fit(D, X, tel, Y, epochs=300, seed=seed)
    out["r2_physics"], out["r2_dpnn"] = r2, float(np.mean(r2d))
    out["n_params_physics"] = int(phys.n_params())
    out["n_params_dpnn"] = int(dmeta["n_params"])
    if verbose:
        print(f"  n={n}   physics ({phys.n_params()}p) R2 {r2:+.4f}   "
              f"dpnn ({dmeta['n_params']}p) R2 {float(np.mean(r2d)):+.4f}")
    return out


if __name__ == "__main__":
    m = InstrumentModel()
    print(f"instrument model: {m.n_params()} parameters "
          f"({N_HEATERS} Vpi + {N_HEATERS} phi0 + {NMZI * 2} kappa + {NUM_OUT * 2} readout)")
    print("recovering a known instrument, via the real two-stage pipeline:")
    _selftest()
