"""The reduced DPNN: a dynamically pruned MLP surrogate for the 4x4 chip.

    features = [ active heaters^2 (16) , dbm , bfm , diode_temp , mA , chip_C ]  -> 21
    targets  = the four output photodiodes

Two things shrink it against the 6x6 version. The obvious one is width: 16 heaters
instead of 112, four detectors instead of fourteen, so a (64, 32) net covers it where that
chip needed (128, 64, 32) and 25,063 parameters. The less obvious one is that this net no
longer has to learn the drift. On the 6x6 the substrate temperature was unmeasured and
unheld, so the laser telemetry stood in as a proxy the net had to factor out; here the TEC
holds the chip and reports it, so temperature is an input with a known scale rather than a
nuisance variable inferred from four correlated laser channels.

Linear voltage stays out of the features. Thermo-optic phase goes as V^2, so V^2 carries
the signal and raw V only adds noise dimensions -- confirmed on the 6x6, where including
it cost 0.10 of held-out R^2.

The physics surrogate in `learn.unitary_fit` does the same job in 56 parameters and
extrapolates. Reach for this one for whatever the physics does not capture.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from pic.config import NUM_OUT, OUT_PDS, TEC_SETPOINT_C, VOLTAGE_MAX
from pic.layout import ACTIVE_DACS, N_HEATERS

from .prune import DynamicPrunedMLP, finetune_fixed, train_pruned

TELEMETRY = ("dbm", "bfm", "diode_temp", "mA", "chip_c")
HIDDEN = (64, 32)
CKPT = "runs/dpnn"

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


def make_features(H, tel, channels=None) -> np.ndarray:
    """(heater volts, telemetry) -> the feature matrix. `tel` is [n, len(TELEMETRY)]."""
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    H = np.atleast_2d(np.asarray(H, float))
    tel = np.atleast_2d(np.asarray(tel, float))
    if tel.shape[1] != len(TELEMETRY):
        raise ValueError(f"telemetry must have {len(TELEMETRY)} columns {TELEMETRY}")
    return np.hstack([H[:, channels] ** 2, tel])


def compute_norm(F, Y, n_heaters):
    """Feature and target scaling. Heater and telemetry columns get the analytic scales
    above so every data regime shares one normalisation; only the targets are data-driven."""
    n = F.shape[1]
    fm, fs = np.zeros(n), np.ones(n)
    g2 = np.linspace(0, VOLTAGE_MAX, 11) ** 2
    fm[:n_heaters], fs[:n_heaters] = g2.mean(), g2.std()
    for k, name in enumerate(TELEMETRY):
        fm[n_heaters + k], fs[n_heaters + k] = _SCALES[name]
    return [fm, fs, Y.mean(0), Y.std(0) + 1e-4]


def r2_vec(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return 1 - ((y - yh) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def build_model(din, dout=NUM_OUT, hidden=HIDDEN, act="relu", min_neurons=8):
    return DynamicPrunedMLP(din, dout, hidden=hidden, activation=act, min_neurons=min_neurons)


def train_round(model, F, Y, norm, mode, epochs, seed=0, verbose=False):
    """mode: 'dense' full-width with no pruning, 'prune' establish the pruned width once,
    'finetune' warm-start at the fixed width. Returns (model, per-target held-out R^2)."""
    fm, fs, ym, ys = norm
    n = len(F)
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


def make_predict(model, norm, buf=None, op_telemetry=None, channels=None):
    """A closure ``f(volts18) -> photodiode volts`` with the telemetry pinned to an
    operating point (default: the training buffer's median)."""
    fm, fs, ym, ys = norm
    channels = ACTIVE_DACS if channels is None else np.asarray(channels, int)
    if op_telemetry is None:
        op_telemetry = (np.median(buf["tel"], axis=0) if buf is not None and len(buf["tel"])
                        else np.array([_SCALES[k][0] for k in TELEMETRY]))
    tel = np.asarray(op_telemetry, float).reshape(1, -1)

    def predict(v):
        F = make_features(np.asarray(v, float).reshape(1, -1), tel, channels)
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
    np.savez(os.path.join(path, "buffer.npz"), H=buf["H"], tel=buf["tel"], Y=buf["Y"])
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=2)
    return path


def load_ckpt(path=CKPT, act="relu", min_neurons=8):
    ck = torch.load(os.path.join(path, "ckpt.pt"), weights_only=False)
    model = build_model(ck["din"], ck["dout"], tuple(ck["widths"]), act, min_neurons)
    model.load_state_dict(ck["state_dict"])
    norm = [np.array(x, float) for x in ck["norm"]]
    b = np.load(os.path.join(path, "buffer.npz"))
    return model, norm, {"H": b["H"], "tel": b["tel"], "Y": b["Y"]}, ck["meta"]


def fit(H, tel, Y, *, epochs: int = 200, hidden=HIDDEN, seed: int = 0, verbose: bool = False):
    """Train from scratch on a collected dataset: dense, then one pruning pass."""
    F = make_features(H, tel)
    norm = compute_norm(F, Y, ACTIVE_DACS.size)
    model = build_model(F.shape[1], Y.shape[1], hidden)
    model, _ = train_round(model, F, Y, norm, "dense", epochs, seed)
    model, r2 = train_round(model, F, Y, norm, "prune", epochs, seed, verbose=verbose)
    meta = {"n_samples": int(len(F)), "widths": model.widths(),
            "n_params": model.n_params(), "pds": list(OUT_PDS),
            "features": ACTIVE_DACS.size + len(TELEMETRY)}
    return model, norm, r2, meta


assert N_HEATERS == 18
