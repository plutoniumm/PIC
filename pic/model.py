"""The third swappable rig component: a predictor that maps heater voltages -> PD values,
mirroring how the laser and board swap hardware<->mock.

    from pic import Rig
    rig = Rig(laser="mock", board="mock", model="mock").open()
    y = rig.predict(H)            # H: heater-voltage vector -> predicted PD values

Three backends, one `.predict(H)` interface:

  - MockModel  the chip's phi ~ V^2 fringe forward (the same one MockPIC uses); no torch.
  - DpnnModel  the hardware-trained DPNN surrogate (runs/dpnn_hw); needs torch.
  - TwinModel  the differentiable digital twin (theory/twin.py); needs torch.

IMPORT SAFETY: this module's top level imports ONLY stdlib + numpy + the already-loaded
`pic.interface`/`pic.config`. Every heavy dependency (torch, scripts.pic_gd, theory.*) is
imported LAZILY inside a class __init__/method, so `import pic` stays fast and torch-free
and `make_model("mock")` works with no torch and no hardware.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import NUM_ADC_RAW, NUM_DAC
from .interface import mock_fringe_forward


@runtime_checkable
class Predictor(Protocol):
    """Anything with ``predict(H) -> np.ndarray of PD values`` is a rig model.

    ``H`` is a heater/DAC voltage vector (length ``NUM_DAC``). The return is a 1-D array
    of predicted photodiode values; see each backend for exactly which PDs it populates.
    """

    def predict(self, H) -> np.ndarray: ...


class MockModel:
    """phi ~ V^2 fringe surrogate -- the same forward :class:`pic.interface.MockPIC` uses.

    Fully hardware-free (no torch). With the default ``seed=0``/``num_dac`` it is bit-for-bit
    the forward a ``board="mock"`` rig measures through, so ``predict`` matches ``measure``
    (minus the board's readout noise). Returns all ``NUM_ADC_RAW`` (14) PDs; the four damaged
    PDs (0, 2, 7, 11) read 0.
    """

    def __init__(self, seed: int = 0, num_dac: int = NUM_DAC, **kw):
        self._forward = mock_fringe_forward(seed=seed, num_dac=num_dac, **kw)

    def predict(self, H) -> np.ndarray:
        return np.asarray(self._forward(np.asarray(H, float).ravel()), float)


class DpnnModel:
    """The hardware-trained dynamically-pruned MLP surrogate (checkpoint ``runs/dpnn_hw``).

    Wraps ``scripts.pic_gd.load_surrogate`` + ``make_predict``: [working-heaters^2, laser
    telemetry] -> PD volts, with the telemetry held fixed at an operating point (default the
    training-buffer median dBm, matching pic_gd). ``predict(H)`` takes a full ``NUM_DAC`` heater
    vector, slices out the working-heater columns the net was trained on, and returns the
    checkpoint's PD outputs (14 on PIC B). Needs torch; all heavy imports are deferred to here.
    """

    def __init__(self, ckpt: str = "runs/dpnn_hw", op_dbm: float | None = None,
                 act: str = "relu", min_neurons: int = 8):
        import os
        if not os.path.exists(os.path.join(ckpt, "ckpt.pt")):
            raise FileNotFoundError(
                f"DPNN checkpoint not found at {ckpt}/ckpt.pt -- train it with "
                "scripts/train_hw_dpnn.py (--pic-b) first, or pass ckpt=")
        import torch  # noqa: F401  (imported so a missing torch fails here, not at predict)
        import scripts.pic_gd as G

        model, norm, buf, meta, feat, cfg = G.load_surrogate(ckpt, act, min_neurons)
        self.op_dbm = float(np.median(buf["dbm"])) if op_dbm is None else float(op_dbm)
        tel = G.operating_telemetry(buf, self.op_dbm)
        self._predict = G.make_predict(model, norm, feat, self.op_dbm, tel)
        self._feat = np.asarray(feat, int)
        self._torch = torch
        self.meta, self.feat = meta, feat

    def predict(self, H) -> np.ndarray:
        v_work = np.asarray(H, float).ravel()[self._feat]
        y = self._predict(self._torch.tensor(v_work, dtype=self._torch.float32))
        return y.detach().numpy()


class TwinModel:
    """The differentiable digital twin (``theory/twin.py``) with the real PIC-B calibration
    (``theory/hw.py`` ``Hardware`` reads ``pic_data/pic_b_config.json``).

    ``predict(H)`` converts DAC volts -> per-heater optical phase (phi = pi*(V/Vpi)^2 + phi0
    on characterized heaters; held at phi0 / 0 otherwise), forwards the twin, and returns the
    6 monitor-tap powers packed into a 14-length PD vector at ``MON_PDS`` (matching
    ``scripts.ising_hw.mock_read``), zeros elsewhere. These are dimensionless optical powers,
    not volts. Needs torch; heavy imports are deferred to ``__init__``.
    """

    MON_PDS = (2, 4, 6, 7, 9, 11)   # schematic monitor taps, one per signal rail

    def __init__(self, config: str | None = None, **twin_kw):
        import torch
        from theory.hw import Hardware
        from theory.twin import Twin, NH

        self._torch = torch
        self._hw = Hardware(config) if config is not None else Hardware()
        self._twin = Twin(**twin_kw)
        self._nh = NH

    def predict(self, H) -> np.ndarray:
        torch, hw = self._torch, self._hw
        H = np.asarray(H, float).ravel()
        nh = self._nh

        vh = np.zeros(nh)
        dac = hw.dac_of[:nh]
        valid = (dac >= 0) & (dac < H.size)
        vh[valid] = H[dac[valid]]

        ph = np.where(np.isfinite(hw.phi0[:nh]), hw.phi0[:nh], 0.0).astype(float)
        known = np.isfinite(hw.vpi[:nh]) & np.isfinite(hw.phi0[:nh])
        ph[known] = np.pi * (vh[known] / hw.vpi[:nh][known]) ** 2 + hw.phi0[:nh][known]

        out = self._twin.forward(torch.as_tensor(ph, dtype=torch.float32))
        mon = out["mon"].detach().numpy()
        raw = np.zeros(NUM_ADC_RAW)
        for k, pd in enumerate(self.MON_PDS):
            raw[pd] = mon[k]
        return raw


_MODELS = {"mock": MockModel, "dpnn": DpnnModel, "twin": TwinModel}


def make_model(spec, **kw):
    """``'mock'|'dpnn'|'twin'`` -> a predictor; a predictor instance passes through unchanged.

    Extra ``**kw`` go to the chosen backend's constructor (e.g. ``make_model('dpnn',
    ckpt=...)``, ``make_model('mock', seed=1)``). ``make_model('mock')`` needs no torch and
    no hardware.
    """
    if not isinstance(spec, str):
        return spec
    try:
        cls = _MODELS[spec]
    except KeyError:
        raise ValueError(
            f"model={spec!r}; use {sorted(_MODELS)}, or a predictor instance") from None
    return cls(**kw)
