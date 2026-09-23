"""The reduced DPNN: a dynamically pruned MLP surrogate for the 4x4 chip.

    features = [ V^2 on the modelled heaters (6) | input-port one-hot (4) | telemetry (5) ]
    targets  = the four output photodiodes

Heaters enter as the uniform *drive* command (`pic.config.drive_to_volts`): every channel
is asked for 0..DRIVE_MAX_V and its own scale opens that to its real ceiling, so nothing
above here carries a per-channel limit and the input block stays rectangular. What the
features hold is the volts that command lands as, squared -- the physical quantity.

The input port is a feature, not a separate model. Intensity detection sees |U x|^2 for
whatever x is launched, so the port is half the input: one heater vector reads four
different ways through the 1x4 switch. One net across all four ports learns from four
times the data about the same mesh, and it is the same encoding `learn.unitary_fit`
already carries.

Three things shrink this against the 6x6 version. Width: six modelled heaters against 112
and four detectors against fourteen, so (64, 32) covers what needed (128, 64, 32) and
25,063 parameters there. Drift: on the 6x6 the substrate was unmeasured and unheld, so
laser telemetry stood in as a proxy the net had to factor out; here the TEC holds the chip
and reports it, so temperature is an input with a known scale rather than a nuisance
inferred from four correlated channels. And the channel count -- ten of the sixteen DAC
channels cannot move a photodiode at all (`MODEL_DACS`), which is worth more than any
architecture change.

Linear voltage stays out, but for a weaker reason than on the 6x6. Thermo-optic phase is
*exactly* affine in V^2 (phi = pi V^2/Vpi^2 + phi0), so a linear first layer maps V^2
straight onto the phase sums and differences the mesh interferes, and cannot get there
from V by any weights at all -- an argument this mesh makes structurally where that one
only measured it. What has changed is the penalty for getting it wrong. Measured Vpi is
4.24-5.32 V against a 3.0 V ceiling, so each channel covers 0.32-0.50 pi: monotonic
segments, not fringes, and nothing wraps. On 28,000 bench rows
(`mrunal/28k_Data_points .xlsx`, six channels x four ports) V^2 beat raw V at every sample
count, by 0.013 mean R^2 at n=500, 0.004 at n=2000 and 0.001 at n=8000, and adding V on top
of V^2 never helped at any n. So V^2 is still right and still free, but it now buys sample
efficiency rather than the 0.10 R^2 the 6x6 lost -- because over a monotone segment a net
can learn the squaring, which it could not do across that chip's many wrapped periods.
`_selftest` re-measures it on a known instrument.

The physics surrogate in `learn.unitary_fit` does the same job in 52 parameters and
extrapolates. Reach for this one for whatever the physics does not capture.

Self-test: ``python -m learn.dpnn``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from pic.config import (
    DRIVE_MAX_V, DRIVE_SCALE, NUM_OUT, OUT_PDS, TEC_SETPOINT_C, drive_to_volts,
    drive_volts_raw,
)
from theory.clements import NMODE
# theory.layout, not pic.layout: theory.layout is already indexed by DAC channel (its own
# docstring says so, and `pic.config.DAC_HEATER` is the same decode), while pic.layout
# re-permutes through a WIRED_MAP written for the old 18-pad numbering and now disagrees
# with it about every role. Roles decide which channels get a column, so take them from
# the side that is keyed the way the firmware is.
from theory.layout import HEATERS, N_HEATERS

from .prune import DynamicPrunedMLP, finetune_fixed, train_pruned

TELEMETRY = ("dbm", "bfm", "diode_temp", "mA", "chip_c")
HIDDEN = (64, 32)
CKPT = "runs/dpnn"

# Which DAC channels earn a feature column. Ten of sixteen do not, and each of those would
# be capacity the net can only overfit with -- an input that moves while the target does
# not is indistinguishable from an input that explains noise.
#
#   dark       VOLTAGE_MAX_CH == 0: no commanded drive reaches the chip. None today --
#              the three channels whose resistance was never confirmed are STAGED at
#              V_UNMEASURED = 1.5 V rather than held at 0 -- but the filter stays, because
#              holding a channel dark is still how an unsafe one is taken out.
#   out_phase  a diagonal phase screen after the mesh cannot change |U x|^2. Structural,
#              not a measurement -- the same fact that makes this mesh 15 phases and not
#              16 -- so no amount of data will ever make that column mean anything.
#   aux        real and wired, but outside the mesh model and held at 0 V.
#
# `phi` is the judgement call, and it is excluded by measurement rather than by theory: an
# input-side external phase is observable in principle, but on this board a live external
# channel driven to its ceiling moved all four detectors by 0.0 mV (bench, 2026-08-27).
# That is evidence the provisional phi assignment in `theory.layout` is wrong, not that
# phi is blind -- but either way the column varies while the target does not, and the 6x6
# measured what one useless feature family costs: 0.10 of held-out R^2.
#
# `learn.train_hw` still drives the external channels while collecting, so the buffer stays
# general. The day a sweep finds an external channel that modulates, put "phi" in
# VISIBLE_ROLES and refit the same data; nothing else has to change.
VISIBLE_ROLES = ("theta",)
# Ordered by DAC channel, because `HEATERS` is, and every consumer here reads it that way:
# it selects feature COLUMNS (`V[:, MODEL_DACS]`) and is scaled channel by channel in
# `compute_norm`, all in the same order. It holds the same six channels as
# `theory.layout.THETA_IDX` in the REVERSE order, since THETA_IDX is ordered by the mesh
# element each channel drives. Never hand this to anything mesh-ordered -- that swap is what
# `theory.layout._by_index` documents mirroring the twin.
MODEL_DACS = np.array([h.h for h in HEATERS
                       if h.role in VISIBLE_ROLES and DRIVE_SCALE[h.h] > 0], int)

# Fixed analytic scales for the inputs whose ranges are known ahead of any data. A
# data-driven std degenerates whenever a regime is narrow -- with the TEC working, chip_c
# barely moves, and normalising it by its own std would amplify sensor noise into the
# dominant input feature.
_SCALES = {
    "dbm": (7.0, 8.0),                    # -2 .. +15 dBm
    "bfm": (0.8, 0.6),                    # monitor optical power
    "diode_temp": (25.0, 2.0),
    "mA": (14.0, 13.0),
    "chip_c": (TEC_SETPOINT_C, 0.5),      # held, so a wide-ish fixed scale, never its own std
}


def port_features(ports, n: int | None = None) -> np.ndarray:
    """Input excitation -> (n, NMODE) launched intensity per port.

    An integer per row is the switch position and becomes a one-hot. A (n, NMODE) array is
    taken as launched intensity directly, so the external-splitter probes that light two
    ports at once (`pic.characterize.probe_inputs`) fit the same slot; complex amplitudes
    are squared on the way in, which is the only thing intensity detection can see."""
    p = np.asarray(ports)
    if p.ndim >= 2 or (p.ndim == 1 and p.size == NMODE and p.dtype.kind in "fc"):
        X = np.atleast_2d(p)
        if X.shape[1] != NMODE:
            raise ValueError(f"port matrix must have {NMODE} columns, got {X.shape[1]}")
        return np.abs(X) ** 2 if np.iscomplexobj(X) else X.astype(float)
    idx = np.atleast_1d(p).astype(int)
    if n is not None and idx.size == 1 and n > 1:
        idx = np.repeat(idx, n)
    return np.eye(NMODE)[idx]


def make_features(D, ports, tel, channels=None) -> np.ndarray:
    """(drive commands, input port, telemetry) -> the feature matrix.

    `D` is (n, N_HEATERS) in the uniform drive variable; the per-channel scale turns it
    into the volts the chip actually sees and V^2 is the feature. A structurally dark
    channel scales to 0 V, so if a caller does put one in `channels` its column comes out
    identically zero -- a dead input contributing nothing past the bias -- rather than a
    live one for the net to chase."""
    channels = MODEL_DACS if channels is None else np.asarray(channels, int)
    D = np.atleast_2d(np.asarray(D, float))
    tel = np.atleast_2d(np.asarray(tel, float))
    if D.shape[1] != N_HEATERS:
        raise ValueError(f"drive must have {N_HEATERS} columns, got {D.shape[1]}")
    if tel.shape[1] != len(TELEMETRY):
        raise ValueError(f"telemetry must have {len(TELEMETRY)} columns {TELEMETRY}")
    # drive_volts_raw, not drive_to_volts: see its docstring. Identical for any buffer
    # collected through the board, which cannot exceed the ceiling in the first place.
    V = drive_volts_raw(D)
    return np.hstack([V[:, channels] ** 2, port_features(ports, len(D)), tel])


def feature_names(channels=None) -> list[str]:
    channels = MODEL_DACS if channels is None else np.asarray(channels, int)
    return ([f"v2_dac{c}" for c in channels] + [f"port{p}" for p in range(NMODE)]
            + list(TELEMETRY))


def compute_norm(F, Y, channels=None):
    """Feature and target scaling. Only the targets are data-driven.

    Each heater column is normalised by its *own* ceiling rather than a shared one: the
    channels come in two resistance groups whose V^2 ranges differ 4:1, and one global
    scale would hand the 1.5 V group a quarter of the numeric swing for no physical
    reason. This is where uniform drive pays off -- after it the input block is square.
    The port one-hot is left alone; it is already O(1), and its own std would blow up
    whichever port happened to be rare in a round."""
    channels = MODEL_DACS if channels is None else np.asarray(channels, int)
    nh = channels.size
    n = F.shape[1]
    fm, fs = np.zeros(n), np.ones(n)
    g = np.linspace(0.0, DRIVE_MAX_V, 11)
    for i, c in enumerate(channels):
        q = (g * DRIVE_SCALE[c]) ** 2
        fm[i], fs[i] = q.mean(), max(q.std(), 1e-9)   # a dark channel stays 0/1: dead, not amplified
    for k, name in enumerate(TELEMETRY):
        fm[nh + NMODE + k], fs[nh + NMODE + k] = _SCALES[name]
    return [fm, fs, Y.mean(0), Y.std(0) + 1e-4]


def r2_vec(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return 1 - ((y - yh) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def build_model(din, dout=NUM_OUT, hidden=HIDDEN, act="relu", min_neurons=8, seed=None):
    if seed is not None:   # initial weights come from the global torch RNG, so seed it here
        torch.manual_seed(seed)
    return DynamicPrunedMLP(din, dout, hidden=hidden, activation=act, min_neurons=min_neurons)


def train_round(model, F, Y, norm, mode, epochs, seed=0, verbose=False):
    """mode: 'dense' full-width with no pruning, 'prune' establish the pruned width once,
    'finetune' warm-start at the fixed width. Returns (model, per-target held-out R^2)."""
    fm, fs, ym, ys = norm
    n = len(F)
    # torch too, not just the split: the batch shuffle comes from the global torch RNG, and
    # without seeding it (and the weights, in `build_model`) two runs of the same `fit`
    # differ by ~0.005 R^2 -- enough to swamp the feature comparisons this module makes.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vi = rng.choice(n, max(1, int(0.15 * n)), replace=False)
    tr = np.ones(n, bool)
    tr[vi] = False
    Xn = ((F - fm) / fs).astype(np.float32)
    Yn = ((Y - ym) / ys).astype(np.float32)
    if mode == "finetune":
        model, _ = finetune_fixed(model, Xn[tr], Yn[tr], Xn[~tr], Y[~tr], ym, ys, epochs=epochs)
    else:
        warmup = 100 if mode == "dense" else 20  # dense schedules no prune epochs at all
        model, _ = train_pruned(model, Xn[tr], Yn[tr], Xn[~tr], Y[~tr], ym, ys,
                                total_epochs=epochs, warmup_pct=warmup, verbose=verbose)
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xn[~tr])).numpy() * ys + ym
    return model, r2_vec(Y[~tr], pv)


def make_predict(model, norm, buf=None, op_telemetry=None, op_port=None, channels=None):
    """A closure ``f(drive) -> photodiode volts`` at a pinned operating point.

    Both the telemetry and the input port are held: the port because the switch is not part
    of the drive vector and a prediction has to be told which one is lit, the telemetry
    because it defaults to the training buffer's median."""
    fm, fs, ym, ys = norm
    channels = MODEL_DACS if channels is None else np.asarray(channels, int)
    if op_telemetry is None:
        op_telemetry = (np.median(buf["tel"], axis=0) if buf is not None and len(buf["tel"])
                        else np.array([_SCALES[k][0] for k in TELEMETRY]))
    if op_port is None:
        p = None if buf is None else np.asarray(buf.get("ports", []), int)
        op_port = int(np.bincount(p).argmax()) if p is not None and p.size else 0
    tel = np.asarray(op_telemetry, float).reshape(1, -1)

    def predict(d, port=None):
        F = make_features(np.asarray(d, float).reshape(1, -1),
                          op_port if port is None else port, tel, channels)
        x = torch.tensor(((F - fm) / fs).astype(np.float32))
        with torch.no_grad():
            return (model(x).numpy() * ys + ym).ravel()

    return predict


def save_ckpt(path, model, norm, buf, meta):
    os.makedirs(path, exist_ok=True)
    fm, fs, ym, ys = norm
    torch.save({"state_dict": model.state_dict(), "widths": model.widths(),
                "din": model.din, "dout": model.dout, "meta": meta,
                "norm": [fm.tolist(), fs.tolist(), ym.tolist(), ys.tolist()]},
               os.path.join(path, "ckpt.pt"))
    np.savez(os.path.join(path, "buffer.npz"), D=buf["D"], ports=buf["ports"],
             tel=buf["tel"], Y=buf["Y"])
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=2)
    return path


def load_ckpt(path=CKPT, act="relu", min_neurons=8):
    ck = torch.load(os.path.join(path, "ckpt.pt"), weights_only=False)
    model = build_model(ck["din"], ck["dout"], tuple(ck["widths"]), act, min_neurons)
    model.load_state_dict(ck["state_dict"])
    norm = [np.array(x, float) for x in ck["norm"]]
    # The buffer is the training data, not the model. It is what `--resume` needs and what
    # a refit needs; inference does not, and a checkpoint distributed beside the capture it
    # was fitted from has no reason to carry a second copy of it.
    bp = os.path.join(path, "buffer.npz")
    buf = None
    if os.path.exists(bp):
        b = np.load(bp)
        buf = {k: b[k] for k in ("D", "ports", "tel", "Y")}
    return model, norm, buf, ck["meta"]


def fit(D, ports, tel, Y, *, channels=None, epochs: int = 200, hidden=HIDDEN, seed: int = 0,
        verbose: bool = False):
    """Train from scratch on a collected dataset: dense, then one pruning pass."""
    channels = MODEL_DACS if channels is None else np.asarray(channels, int)
    F = make_features(D, ports, tel, channels)
    norm = compute_norm(F, Y, channels)
    model = build_model(F.shape[1], Y.shape[1], hidden, seed=seed)
    model, _ = train_round(model, F, Y, norm, "dense", epochs, seed)
    model, r2 = train_round(model, F, Y, norm, "prune", epochs, seed, verbose=verbose)
    meta = {"n_samples": int(len(F)), "widths": model.widths(),
            "n_params": model.n_params(), "pds": list(OUT_PDS),
            "channels": [int(c) for c in channels], "n_features": int(F.shape[1]),
            "features": feature_names(channels)}
    return model, norm, r2, meta


# Measured on the six internal phase shifters, 2026-08-27. Against a 3.0 V ceiling this is
# 0.32-0.50 pi per channel: monotonic segments, not fringes.
VPI_MEASURED = (4.24, 5.32)
# Per-photodiode dark noise, same session. PD0 is a genuinely bad detector and sets its own
# R^2 ceiling; the self-test reports that ceiling so a low PD0 number reads as the detector.
DARK_NOISE_V = (0.044, 0.0006, 0.0006, 0.0006)


def _selftest(n: int = 800, epochs: int = 150, seed: int = 0, verbose: bool = True):
    """Fit a known instrument in the regime this chip is actually in, and re-measure the
    two feature claims the docstring makes.

    The truth is `learn.unitary_fit.InstrumentModel` -- the same twin the physics fit uses
    -- with the measured heater law and the measured per-PD dark noise, so the reported
    R^2 has a real ceiling rather than being limited only by the net."""
    from theory.calib import Calibration
    from theory.twin import MeshError

    from .unitary_fit import InstrumentModel

    rng = np.random.default_rng(seed)
    truth = InstrumentModel(
        Calibration(rng.uniform(*VPI_MEASURED, N_HEATERS), rng.uniform(0, 2 * np.pi, N_HEATERS),
                    np.full(NUM_OUT, 0.55), np.array([0.158, 0.125, 0.111, 0.147])),
        MeshError.sample(sigma_kappa=0.02, seed=seed))

    D = np.zeros((n, N_HEATERS))
    D[:, MODEL_DACS] = rng.uniform(0.0, DRIVE_MAX_V, (n, MODEL_DACS.size))
    ports = rng.integers(0, NMODE, n)
    tel = np.tile([8.0, 0.8, 25.0, 14.0, TEC_SETPOINT_C], (n, 1))
    with torch.no_grad():
        Y = truth(torch.as_tensor(drive_to_volts(D), dtype=torch.float32),
                  torch.as_tensor(np.eye(NMODE, dtype=complex)[ports])).numpy()
    noise = np.array(DARK_NOISE_V)
    Y = Y + rng.normal(0, noise, Y.shape)
    ceiling = 1 - noise**2 / Y.var(0)

    model, norm, r2, meta = fit(D, ports, tel, Y, epochs=epochs, seed=seed)

    # A channel at zero drive must arrive as a dead column, not a live input the net can
    # chase. This used to be checked on the channels with DRIVE_SCALE == 0; there are none
    # left -- `pic.config.V_UNMEASURED` stages the three unmeasured channels at 1.5 V rather
    # than holding them at 0 V -- so `dark` went empty and the max over it RAISED instead of
    # checking anything. `held` is everything outside MODEL_DACS and is never empty, and the
    # invariant is the same one: the feature is (DRIVE_SCALE[c] * d)^2, zero when either is.
    allch = np.arange(N_HEATERS)
    held = np.setdiff1d(allch, MODEL_DACS)
    Dz = D + 0.5 * DRIVE_MAX_V
    Dz[:, held] = 0.0
    Fall = make_features(Dz, ports, tel, allch)
    n_dark = int(sum(DRIVE_SCALE[c] == 0.0 for c in allch))
    dark_max = float(np.abs(Fall[:, held]).max())

    # the port must actually be doing work: one drive vector, four switch positions
    predict = make_predict(model, norm, {"tel": tel, "ports": ports}, channels=MODEL_DACS)
    d0 = np.zeros(N_HEATERS)
    d0[MODEL_DACS] = 0.5 * DRIVE_MAX_V
    spread = float(np.ptp(np.stack([predict(d0, p) for p in range(NMODE)]), axis=0).max())

    # V^2 against V and against both, same budget, same split. Telemetry is left out of all
    # three: it is a constant tile here, so it carries nothing and its presence would only
    # dilute the comparison. Data-driven scales, since the whole point is that the heater
    # block is a different variable in each variant.
    V = drive_to_volts(D)
    P = port_features(ports)
    variants = {"V^2": V[:, MODEL_DACS] ** 2, "V": V[:, MODEL_DACS],
                "V+V^2": np.hstack([V[:, MODEL_DACS], V[:, MODEL_DACS] ** 2])}
    abl = {}
    for name, H in variants.items():
        F = np.hstack([H, P])
        nrm = [np.concatenate([H.mean(0), np.zeros(NMODE)]),
               np.concatenate([H.std(0) + 1e-9, np.ones(NMODE)]),
               Y.mean(0), Y.std(0) + 1e-4]
        m = build_model(F.shape[1], NUM_OUT, HIDDEN, seed=seed)
        _, rv = train_round(m, F, Y, nrm, "dense", epochs, seed)
        abl[name] = float(rv.mean())

    out = {"r2": r2, "ceiling": ceiling, "n_params": meta["n_params"],
           "n_features": meta["n_features"], "channels": meta["channels"],
           "dark_max": dark_max, "port_spread": spread, "ablation": abl}
    if verbose:
        print(f"  {n} samples x {NMODE} ports, {meta['n_features']} features "
              f"-> {NUM_OUT} photodiodes, {meta['n_params']} parameters "
              f"(widths {meta['widths']})")
        print("  held-out R^2   " + "  ".join(f"PD{j} {r2[j]:+.4f}/{ceiling[j]:.4f}"
                                              for j in range(NUM_OUT))
              + "   (fit / noise ceiling)")
        print(f"  {held.size} channels at 0 V arrive dead: max |feature| {dark_max:.1e}"
              f"   ({n_dark} channels structurally dark on this board)")
        print(f"  same drive across the four input ports: {spread * 1e3:.0f} mV spread")
        print("  features       " + "  ".join(f"{k} {v:+.4f}" for k, v in abl.items()))

    assert dark_max == 0.0, "a channel held at 0 V produced a non-zero feature"
    assert spread > 10e-3, "the input port is not moving the prediction"
    assert float((r2 / ceiling).min()) > 0.85, f"under-fitted against the noise: {r2}"
    # A loose bar on purpose: over monotonic segments V^2's margin over V is a few
    # thousandths, not the 6x6's 0.10. The load-bearing measurement of that gap is the
    # 28,000 bench rows quoted in the module docstring; this only catches a sign flip.
    assert abl["V^2"] >= abl["V"] - 0.01, f"V clearly beat V^2 on this instrument: {abl}"
    return out


if __name__ == "__main__":
    print(f"modelled DAC channels {MODEL_DACS.tolist()} of {N_HEATERS} "
          f"({', '.join(feature_names())})")
    _selftest()
