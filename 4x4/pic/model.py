"""The predictor component: heater volts -> photodiode volts, without touching the chip.

Two families, and the contrast between them is the point of a unitary device.

  TwinModel   the physics. Forty-four parameters -- an (Vpi, phi0) pair per heater plus a
              gain and offset per photodiode -- through a mesh that is unitary by
              construction. It extrapolates, it inverts in closed form, and every
              parameter is a thing you can measure on its own.

  DpnnModel   the pruned MLP, the 6x6 approach carried over. Here it is the fallback for
              whatever the physics does not capture, and it is small: the squared voltages
              of the six channels a photodiode can see, plus the lit input port and the
              laser telemetry, four photodiodes out, a couple of thousand parameters
              against the 25,063 that chip needed. A 6x6 contraction had no closed form to
              lean on, so the black box had to carry everything.

Heavy imports are deferred, so `import pic` stays torch-free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import NUM_ADC_RAW, NUM_DAC, OUT_PDS, volts_to_drive


@runtime_checkable
class Predictor(Protocol):
    """Anything with ``predict(V) -> np.ndarray`` of photodiode values is a rig model."""

    def predict(self, V) -> np.ndarray: ...


class TwinModel:
    """Calibration plus differentiable twin: the instrument written down as physics."""

    def __init__(self, calib=None, error=None, x=None, dark: float = 0.01):
        import torch

        from theory.calib import Calibration
        from theory.twin import Twin

        self._torch = torch
        self.calib = Calibration.load_or_nominal() if calib is None else calib
        self.twin = Twin(error)
        self.x = None if x is None else torch.as_tensor(np.asarray(x), dtype=self.twin.dtype)
        self.dark = dark

    def predict(self, V) -> np.ndarray:
        ph = self.calib.phases(np.asarray(V, float).ravel())
        inten = (
            self.twin.outputs(self._torch.as_tensor(ph, dtype=self._torch.float32), self.x)
            .detach()
            .numpy()
        )
        raw = np.full(NUM_ADC_RAW, self.dark)
        raw[list(OUT_PDS)] = self.calib.to_volts_response(inten)
        return raw

    @property
    def n_params(self) -> int:
        c = self.calib
        return c.vpi.size + c.phi0.size + c.pd_gain.size + c.pd_offset.size


class MockModel(TwinModel):
    """A `TwinModel` on the same fabricated instance `MockPIC` measures through, so
    `rig.predict(v)` matches `rig.measure(v)` up to readout noise."""

    def __init__(self, seed: int = 0, **kw):
        from theory.calib import Calibration
        from theory.twin import MeshError

        super().__init__(
            calib=Calibration.sample(seed=seed), error=MeshError.sample(seed=seed), **kw
        )


class DpnnModel:
    """The reduced pruned MLP surrogate, loaded from a `learn.train_hw` checkpoint.

    The network works in the uniform drive command and needs to be told which input port
    is lit; a `Predictor` takes neither, so this converts volts at the boundary and pins
    the port to whatever the training buffer used most. Pass `op_port` to move it."""

    def __init__(self, ckpt: str = "runs/dpnn", op_telemetry=None, op_port=None):
        import os

        if not os.path.exists(os.path.join(ckpt, "ckpt.pt")):
            raise FileNotFoundError(
                f"no checkpoint at {ckpt}/ckpt.pt -- train one with `python -m learn.train_hw`"
            )
        import torch

        from learn.dpnn import MODEL_DACS, load_ckpt, make_predict

        self._torch = torch
        self.model, self.norm, self.buf, self.meta = load_ckpt(ckpt)
        self.channels = np.asarray(self.meta.get("channels", MODEL_DACS), int)
        self._predict = make_predict(
            self.model, self.norm, self.buf, op_telemetry, op_port, self.channels
        )
        self.pds = list(self.meta.get("pds", OUT_PDS))

    def predict(self, V, port=None) -> np.ndarray:
        y = self._predict(volts_to_drive(np.asarray(V, float).ravel()), port)
        raw = np.zeros(NUM_ADC_RAW)
        raw[self.pds] = y
        return raw

    @property
    def n_params(self) -> int:
        return int(self.meta.get("n_params", 0))


_MODELS = {"mock": MockModel, "twin": TwinModel, "dpnn": DpnnModel}


def make_model(spec, **kw):
    """``'mock'|'twin'|'dpnn'`` -> a predictor; an instance passes through unchanged."""
    if not isinstance(spec, str):
        return spec
    try:
        cls = _MODELS[spec]
    except KeyError:
        raise ValueError(f"model={spec!r}; use {sorted(_MODELS)} or a predictor") from None
    return cls(**kw)


assert NUM_DAC == 16
