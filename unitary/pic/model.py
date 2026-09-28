"""The predictor component: heater volts -> photodiode volts, without touching the chip.

  TwinModel   the physics. Forty-four parameters -- an (Vpi, phi0) pair per heater plus a
              gain and offset per photodiode -- through a mesh that is unitary by
              construction. It extrapolates, it inverts in closed form, and every
              parameter is a thing you can measure on its own.

Heavy imports are deferred, so `import pic` stays torch-free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import NUM_ADC_RAW, NUM_DAC, OUT_PDS


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


_MODELS = {"mock": MockModel, "twin": TwinModel}


def make_model(spec, **kw):
    """``'mock'|'twin'`` -> a predictor; an instance passes through unchanged."""
    if not isinstance(spec, str):
        return spec
    try:
        cls = _MODELS[spec]
    except KeyError:
        raise ValueError(f"model={spec!r}; use {sorted(_MODELS)} or a predictor") from None
    return cls(**kw)


assert NUM_DAC == 16
