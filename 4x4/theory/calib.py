"""The heater law: DAC volts <-> optical phase, plus the photodiode readout scale.

A thermo-optic shifter's phase follows dissipated power, so phase goes as V^2:

    phi_i(V) = pi (V / Vpi_i)^2 + phi0_i

Two numbers per heater. With 18 heaters that is 36 parameters for the whole mesh, and
with the four readout gains and offsets a complete instrument model in 44. The point of
writing it down as a law rather than learning a black box is that the law extrapolates:
`volts` inverts it in closed form, so a target phase becomes a voltage without a search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .clements import NMODE
from .layout import N_HEATERS

CONFIG_PATH = Path("pic_data/calib.json")

NUM_OUT = NMODE
# PROVISIONAL and probably low. 1.5 V is the 6x6 number, and that chip is silicon; this one
# is Ligentec AN800 silicon nitride, whose thermo-optic coefficient is several times smaller,
# so the same phase costs more power. With the board capped at 3 V a heater only reaches a
# full 2 pi if Vpi <= 2.12 V, which is the first thing a sweep has to confirm.
VPI_NOMINAL = 1.5
VOLTAGE_MAX = 3.0   # per-channel ceiling on the bring-up board (mrunal/Setup.ino maxVolt)


@dataclass
class Calibration:
    vpi: np.ndarray = field(default_factory=lambda: np.full(N_HEATERS, VPI_NOMINAL))
    phi0: np.ndarray = field(default_factory=lambda: np.zeros(N_HEATERS))
    pd_gain: np.ndarray = field(default_factory=lambda: np.ones(NUM_OUT))
    pd_offset: np.ndarray = field(default_factory=lambda: np.zeros(NUM_OUT))
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, n in (("vpi", N_HEATERS), ("phi0", N_HEATERS),
                        ("pd_gain", NUM_OUT), ("pd_offset", NUM_OUT)):
            v = np.asarray(getattr(self, name), float).ravel()
            if v.size != n:
                raise ValueError(f"{name}: expected {n} values, got {v.size}")
            setattr(self, name, v)

    def phases(self, volts) -> np.ndarray:
        v = np.asarray(volts, float)
        return np.pi * (v / self.vpi) ** 2 + self.phi0

    def volts(self, phases, vmax: float = VOLTAGE_MAX):
        """Inverse of `phases`, wrapped into the reachable 0..vmax window.

        A phase is 2-pi periodic, so several voltages produce it; this returns the
        *lowest*. Every volt of heater drive is dissipated into the substrate, where it
        raises the neighbours' phases and loads the TEC, so the cheapest image of a target
        is the right one even though any of them realises the same matrix.

        Returns (volts, ok). `ok` is False where no image lies inside the window: that
        heater cannot make that phase, which is a hardware fact and not something to clip
        away silently."""
        ph = np.asarray(phases, float)
        k = np.ceil((self.phi0 - ph) / (2 * np.pi))
        target = ph + 2 * np.pi * k - self.phi0
        u = target / np.pi * self.vpi**2
        ok = (u >= -1e-12) & (u <= vmax**2)
        return np.sqrt(np.clip(u, 0.0, None)), ok

    def to_volts_response(self, intensity) -> np.ndarray:
        """Optical intensity at the four outputs -> photodiode volts."""
        return np.asarray(intensity, float) * self.pd_gain + self.pd_offset

    def to_intensity(self, pd_volts) -> np.ndarray:
        """Photodiode volts -> optical intensity, the inverse of `to_volts_response`."""
        return (np.asarray(pd_volts, float) - self.pd_offset) / self.pd_gain

    @property
    def reach_span(self) -> np.ndarray:
        """Phase span each heater covers over 0..VOLTAGE_MAX, in units of pi. Below 2 the
        heater cannot reach every phase and some targets are unprogrammable."""
        return (VOLTAGE_MAX / self.vpi) ** 2

    def save(self, path=CONFIG_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "vpi": self.vpi.tolist(), "phi0": self.phi0.tolist(),
            "pd_gain": self.pd_gain.tolist(), "pd_offset": self.pd_offset.tolist(),
            "meta": self.meta,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path=CONFIG_PATH):
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["vpi"]), np.array(d["phi0"]),
                   np.array(d.get("pd_gain", np.ones(NUM_OUT))),
                   np.array(d.get("pd_offset", np.zeros(NUM_OUT))),
                   d.get("meta", {}))

    @classmethod
    def load_or_nominal(cls, path=CONFIG_PATH):
        """The measured calibration if one exists, otherwise the nominal law. Callers
        that care can check `meta` -- a nominal calibration has an empty one."""
        p = Path(path)
        return cls.load(p) if p.exists() else cls()

    @classmethod
    def sample(cls, sigma_vpi: float = 0.15, seed=None) -> "Calibration":
        """A plausible fabricated instance: Vpi spread across heaters and unknown static
        offsets. Used to give the mock rig something a characterization run has to find."""
        rng = np.random.default_rng(seed)
        return cls(VPI_NOMINAL * (1 + sigma_vpi * rng.normal(size=N_HEATERS)),
                   rng.uniform(0, 2 * np.pi, N_HEATERS),
                   np.full(NUM_OUT, 0.8) * (1 + 0.05 * rng.normal(size=NUM_OUT)),
                   np.full(NUM_OUT, 0.01),
                   {"source": f"Calibration.sample(seed={seed})"})


def _selftest(seed: int = 0):
    c = Calibration.sample(seed=seed)
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, VOLTAGE_MAX, N_HEATERS)
    assert np.allclose(c.phases(c.volts(c.phases(v))[0]) % (2 * np.pi),
                       c.phases(v) % (2 * np.pi)), "volts/phases round trip"
    tgt = rng.uniform(0, 2 * np.pi, N_HEATERS)
    vv, ok = c.volts(tgt)
    got = c.phases(vv) % (2 * np.pi)
    assert np.allclose(got[ok], tgt[ok] % (2 * np.pi)), "reachable targets must land exactly"
    return int(ok.sum()), N_HEATERS


if __name__ == "__main__":
    n, tot = _selftest()
    c = Calibration.sample(seed=0)
    print(f"volts <-> phases round trip OK; {n}/{tot} random targets reachable at "
          f"{VOLTAGE_MAX:.0f} V")
    print(f"phase span per heater (units of pi): {np.round(c.reach_span, 2)}")
