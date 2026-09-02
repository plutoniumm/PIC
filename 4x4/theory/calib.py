"""The heater law: DAC volts <-> optical phase, plus the photodiode readout scale.

A thermo-optic shifter's phase follows dissipated power, so phase goes as V^2:

    phi_i(V) = pi (V / Vpi_i)^2 + phi0_i

Two numbers per heater. With the DAC81416's 16 channels that is 32 parameters for the whole
mesh, and with the four readout gains and offsets a complete instrument model in 40. The
point of writing it down as a law rather than learning a black box is that the law
extrapolates: `volts` inverts it in closed form, so a target phase becomes a voltage without
a search.
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
VOLTAGE_MAX = 3.0   # hard per-channel ceiling of this design, not a bench setting. Host,
                    # firmware (pic4x4.ino VMAX) and mrunal/Setup.ino all agree at 3.0.


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

    def to_transfer(self, raw, dark=None) -> np.ndarray:
        """A full four-port sweep of photodiode volts -> |U|^2, one column per input port.

        `raw[j, k]` is photodiode j read with input port k lit, so this is the matrix-level
        sibling of `to_intensity` and the only correct way to read the mesh: an absolute
        intensity needs a scale, and the scale is per port and per heater state, so it
        cannot be carried in a constant.

        Two steps, both forced by physics rather than fitted. First remove the blocked-path
        floor: `pd_offset` is measured with the switch dumped, so what it takes out is TIA
        offset and ADC bias and nothing optical. On the 2026-08-27 bench the switch-off and
        laser-off floors agree to 0.2 mV, which is what proves there is no thermal pedestal
        hiding inside it. Then divide each column by its own sum. The mesh is unitary --
        every photon entering port k leaves through one of the four measured outputs -- so
        `sum_j |U[j,k]|^2 = 1` is the definition of the object, not an assumption about it.
        That single division removes launch power, fibre coupling and the switch's
        per-position insertion loss together, and removes them *as they were at this heater
        state*: on that sweep the light leaving ports 0 and 1 moved by 5.1x and 6.0x across
        sixteen heater states, so any scale stored once is wrong by that factor everywhere
        else.

        What it cannot do is tell real loss from routing. Light that genuinely leaves the
        measured set is renormalised as though it had not, so the result is the
        distribution *given detection*. That is the right object for anything scale-free --
        a hosted block, a decoded Ising coupling -- and the wrong one for absolute
        efficiency.

        Columns only, and deliberately: forcing the rows as well has no unique solution
        (`pic.normalise.audit` measures what it invents instead). A negative entry is
        dark-subtraction noise on an extinguished cell and clips to zero, the only value an
        intensity can take."""
        raw = np.asarray(raw, float)
        # `pd_offset` and `pd_gain` are per DETECTOR and go on axis 0 here, where `to_intensity`
        # puts them on the last axis. A stack of sweeps has the same trailing shape as one, so
        # handing this an (n, 4, 4) would broadcast detector gains onto the wrong axis and
        # return something plausible. Map over the stack instead; the callers all do.
        if raw.ndim != 2 or raw.shape[0] != NUM_OUT:
            raise ValueError(f"to_transfer takes one ({NUM_OUT}, ports) sweep with detectors "
                             f"on axis 0, got {raw.shape}")
        off = self.pd_offset if dark is None else np.asarray(dark, float)
        P = np.clip((raw - off[:, None]) / self.pd_gain[:, None], 0.0, None)
        s = P.sum(0, keepdims=True)
        return np.divide(P, s, out=np.zeros_like(P), where=s > 0)

    def reach_span(self, vmax=VOLTAGE_MAX) -> np.ndarray:
        """Phase span each heater covers over 0..vmax, in units of pi. Below 2 the heater
        cannot reach every phase and some targets are unprogrammable.

        `vmax` is per channel on this board and the caller has to say so. The ceiling is
        I*R at the heater's own measured resistance (`pic.config.VOLTAGE_MAX_CH`, 1.50 to
        4.75 V), so the design scalar overstates the 60-ohm channels and understates the
        118-ohm ones -- an error in both directions, which is why defaulting to it read as
        plausible. `theory` must not import `pic`, so the board's table arrives as an
        argument; the scalar default is the design ceiling and nothing measured."""
        return (np.asarray(vmax, float) / self.vpi) ** 2

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


def ds_error(T) -> float:
    """Worst departure of an intensity transfer matrix from doubly stochastic, no free scale.

    `theory.drift.ds_imbalance` rescales the total to NMODE first, which is right when the
    question is drift -- there the overall power is a nuisance shared by every entry. It is
    wrong as a readout check, because a readout that is uniformly 2x out gives a perfect
    imbalance score and a matvec that is uniformly 2x wrong. After `to_transfer` the total
    is NMODE by construction, so on that side the two agree and only this one also catches
    the scale."""
    T = np.asarray(T, float)
    return float(max(np.abs(T.sum(-1) - 1).max(), np.abs(T.sum(-2) - 1).max()))


def fit_shared(levels, traces, vpi_seeds, weights=None):
    """One Vpi and one phi0 for a heater, fitted across ALL its sweeps at once.

    Every (port, base) trace of a heater sees the same physical device: the same Vpi, the
    same phi0. Only the amplitude and offset change, because the upstream mesh routes a
    different amount of light onto its arm. Fitting each trace alone and keeping the best
    throws that away -- and on this chip it matters, because no heater spans even half a
    period, so within any single trace amplitude and period trade off against each other and
    a whole family of (A, B, Vpi) fits the data equally well. The degenerate direction is
    not the same in every trace, so several traces together pin down what one cannot.

    Linear in (A_t, B_t cos phi0, -B_t sin phi0) once Vpi and phi0 are fixed, so the inner
    solve is least squares and only Vpi and phi0 need searching.

    `traces` is a list of 1-D arrays, all sampled at `levels`. Returns
    (vpi, phi0, per_trace_amplitude, rmse)."""
    v = np.asarray(levels, float).ravel()
    ts = [np.asarray(t, float).ravel() for t in traces]
    w = np.ones(len(ts)) if weights is None else np.asarray(weights, float)

    def resid(vpi, phi0):
        th = np.pi * (v / vpi) ** 2 + phi0
        M = np.stack([np.ones_like(th), np.cos(th)], axis=1)
        tot, amps = 0.0, []
        for t, wi in zip(ts, w):
            coef, *_ = np.linalg.lstsq(M, t, rcond=None)
            amps.append(abs(float(coef[1])))
            tot += wi * float(np.mean((M @ coef - t) ** 2))
        return tot / max(w.sum(), 1e-9), amps

    best = None
    for vpi0 in np.asarray(vpi_seeds, float).ravel():
        for f in np.linspace(0.6, 1.6, 21):           # Vpi is the poorly-known one
            vpi = float(vpi0 * f)
            if not 0.4 <= vpi <= 12.0:
                continue
            for phi0 in np.linspace(0, 2 * np.pi, 49, endpoint=False):
                r, amps = resid(vpi, phi0)
                if best is None or r < best[0]:
                    best = (r, vpi, phi0, amps)
    r, vpi, phi0, amps = best
    return float(vpi), float(phi0 % (2 * np.pi)), amps, float(np.sqrt(r))


def refit_phi0(volts, y, vpi, amp=None, offset=None, prior=None, span=np.pi):
    """Re-find one heater's phase offset with everything that does not drift held fixed.

    Vpi is set by heater geometry and does not drift; phi0 follows the waveguide's optical
    path and does. So a re-anchor only has to recover phi0 -- but it cannot recover it from
    scratch on this chip. Each heater covers under 0.5 pi, so the fringe is a monotonic
    segment and amplitude, offset and phase are not separable from a handful of points: a
    free three-parameter solve lands 0.3 rad out at realistic noise.

    Given `amp` and `offset` from the full characterization it becomes a ONE-parameter
    problem, scanned over `span` around `prior`. That is well conditioned on 4 points and
    is the difference between a 25-minute characterization and a 30-second re-anchor.

    With no prior it falls back to the free linear solve, which is honest but weak; callers
    should pass what the last characterization measured.

    Returns (phi0, amplitude, offset, rmse)."""
    v = np.asarray(volts, float).ravel()
    y = np.asarray(y, float).ravel()
    th = np.pi * (v / float(vpi)) ** 2

    if amp is None or offset is None:
        M = np.stack([np.ones_like(th), np.cos(th), -np.sin(th)], axis=1)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        a, c, d = coef
        return (float(np.arctan2(d, c)) % (2 * np.pi), float(np.hypot(c, d)), float(a),
                float(np.sqrt(np.mean((M @ coef - y) ** 2))))

    p0 = 0.0 if prior is None else float(prior)
    grid = p0 + np.linspace(-span, span, 2001)
    resid = y[None, :] - (offset + amp * np.cos(th[None, :] + grid[:, None]))
    rms = np.sqrt((resid ** 2).mean(axis=1))
    k = int(np.argmin(rms))
    return float(grid[k] % (2 * np.pi)), float(amp), float(offset), float(rms[k])


def _selftest(seed: int = 0):
    rng0 = np.random.default_rng(seed)
    for trial in range(20):        # refit_phi0 recovers a planted offset from few points
        vpi = rng0.uniform(3.5, 5.5)
        phi = rng0.uniform(0, 2 * np.pi)
        amp, off = rng0.uniform(0.05, 0.3), rng0.uniform(0.1, 0.5)
        v = np.linspace(0, 3.0, 5)
        y = off + amp * np.cos(np.pi * (v / vpi) ** 2 + phi) + 0.002 * rng0.normal(size=v.size)
        drift = rng0.uniform(-0.4, 0.4)      # what a re-anchor is actually chasing
        y = off + amp * np.cos(np.pi * (v / vpi) ** 2 + phi + drift) \
            + 0.002 * rng0.normal(size=v.size)
        got, _, _, _ = refit_phi0(v, y, vpi, amp=amp, offset=off, prior=phi)
        err = abs((got - phi - drift + np.pi) % (2 * np.pi) - np.pi)
        assert err < 0.06, (trial, err, drift, vpi)

    c = Calibration.sample(seed=seed)
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, VOLTAGE_MAX, N_HEATERS)
    assert np.allclose(c.phases(c.volts(c.phases(v))[0]) % (2 * np.pi),
                       c.phases(v) % (2 * np.pi)), "volts/phases round trip"
    tgt = rng.uniform(0, 2 * np.pi, N_HEATERS)
    vv, ok = c.volts(tgt)
    got = c.phases(vv) % (2 * np.pi)
    assert np.allclose(got[ok], tgt[ok] % (2 * np.pi)), "reachable targets must land exactly"

    # to_transfer must invert an arbitrary per-port launch through the readout it knows.
    # The per-port factors are the ones the old static `input_scale` guessed once; here they
    # are drawn fresh, which is the whole point -- the answer must not depend on them.
    U = np.linalg.qr(rng.normal(size=(NUM_OUT, NUM_OUT))
                     + 1j * rng.normal(size=(NUM_OUT, NUM_OUT)))[0]
    P = np.abs(U) ** 2
    for _ in range(20):
        launch = rng.uniform(0.05, 3.0, NUM_OUT)
        raw = c.to_volts_response((P * launch[None, :]).T).T
        assert ds_error(c.to_transfer(raw)) < 1e-9, launch
    # a dead port must come back as a zero column, not NaN: dividing by its sum is the one
    # place this transform can produce a number out of nothing
    raw = c.to_volts_response((P * np.array([1.0, 1.0, 1.0, 0.0])[None, :]).T).T
    T = c.to_transfer(raw)
    assert np.all(np.isfinite(T)) and T[:, 3].sum() == 0.0, T
    return int(ok.sum()), N_HEATERS


if __name__ == "__main__":
    n, tot = _selftest()
    c = Calibration.sample(seed=0)
    print(f"volts <-> phases round trip OK; {n}/{tot} random targets reachable at "
          f"{VOLTAGE_MAX:.0f} V")
    print(f"phase span per heater at the {VOLTAGE_MAX:.0f} V design ceiling (units of pi): "
          f"{np.round(c.reach_span(), 2)}   -- the board's real per-channel ceilings are in "
          f"pic.config.VOLTAGE_MAX_CH")
