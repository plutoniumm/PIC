"""Heater resistance from the TEC's own control effort, by substitution.

Three heaters (DAC 6, 8, 11) have no measured resistance, and that one gap locked them out
of everything: no resistance meant a 0 V ceiling, a 0 V ceiling meant no characterization
sweep, and an unfitted Vpi meant `HeaterBox.from_calibration` dropped them for good. Two of
the three are intensity-visible, so the gap costs a quarter of the mesh's steering.

There is no current sense anywhere on this rig -- the DAC81416 is a voltage source and the
board's only ADC reads photodiodes. But the TEC is a calorimeter nobody was using. Holding
the die at a setpoint, its steady-state drive voltage is a monotone function of the heat
load, and a heater at V dissipates V^2/R into that load. So resistance is readable without
an ammeter:

    drive(V_u^2 / R_u) == drive(V_k^2 / R_k)   =>   R_u = R_k (V_u / V_k)^2

which is a substitution measurement against a known channel. The calorimeter's gain, its
offset and its nonlinearity all cancel at the balance point -- only its REPEATABILITY has to
hold, and only over the minutes between the two readings. That is a much weaker requirement
than calibrating it, and it is why this reads a balance rather than a curve.

Position does not enter it. A heater near the die edge runs hotter locally than one in the
middle at the same power, but at steady state every watt still leaves through the Peltier,
so the total load the TEC fights is the same. `holdout` tests that claim rather than
asserting it: it infers a resistance the rig already knows and reports the error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import HEATER_OHMS, N_HEATERS, V_UNMEASURED, VOLTAGE_MAX_CH

# Thermal, not electrical. A step in heater power reaches the TEC through the die and the
# submount, and the loop then has to re-integrate against it; `pic-thermal-settling` puts the
# chip's own t99 at 3-5 s but the substrate tail runs to tens of seconds and the TEC's
# integrator is slower still. Sampling before it lands biases every reading the same way --
# toward the previous state -- which a substitution measurement cannot cancel because the two
# arms are taken at different times.
SETTLE_S = 45.0
READS = 20
POLL_S = 0.5

# A channel whose full-scale power cannot move the TEC drive by this much is not measurable
# this way, and the routine says so instead of returning a number. Set from the drive noise
# measured by `noise_floor`, not from a guess: the check is `signal > SNR_MIN * sigma`.
SNR_MIN = 5.0


@dataclass
class Balance:
    """One substitution: an unknown at `v_u` matched against a known at `v_k`."""

    dac_u: int
    dac_k: int
    v_u: float
    v_k: float
    r_k: float
    drive_u: float
    drive_k: float
    drive_0: float
    sigma: float

    @property
    def ohms(self) -> float:
        return self.r_k * (self.v_u / self.v_k) ** 2

    @property
    def snr(self) -> float:
        """How far the unknown moved the calorimeter, in noise widths."""
        return abs(self.drive_u - self.drive_0) / max(self.sigma, 1e-12)

    @property
    def residual(self) -> float:
        """How far from balanced the match ended up, in noise widths. Small is good, and it
        is a different quantity from `snr` -- a strong signal badly matched and a weak one
        matched perfectly both give a wrong resistance, for opposite reasons."""
        return abs(self.drive_u - self.drive_k) / max(self.sigma, 1e-12)


def _drive(rig, volts, settle_s=SETTLE_S, reads=READS, poll_s=POLL_S) -> tuple[float, float]:
    """Set the heaters, wait for the die to settle, return (mean drive_v, its sem).

    The photodiodes are read and thrown away: `rig.measure` is the only path that writes the
    DACs, and going through it keeps the watchdog and calibration gates in front of this
    routine like every other measurement."""
    rig.measure(np.asarray(volts, float))
    time.sleep(settle_s)
    xs = []
    for _ in range(reads):
        s = rig.tec.status()
        if "drive_v" not in s:
            raise RuntimeError("this TEC reports no drive_v, so it cannot be used as a "
                               "calorimeter; resistance by substitution needs the hardware "
                               "controller, not the mock or the open-loop stub")
        xs.append(float(s["drive_v"]))
        time.sleep(poll_s)
    x = np.asarray(xs, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def noise_floor(rig, n: int = 3, **kw) -> float:
    """Drive-voltage repeatability with the heaters cold, in volts.

    The floor that matters is not the scatter within one dwell -- that averages down -- but
    the scatter BETWEEN dwells minutes apart, which is what a substitution actually differences
    across. Measuring it any other way flatters the method."""
    v0 = np.zeros(N_HEATERS)
    return float(np.std([_drive(rig, v0, **kw)[0] for _ in range(n)], ddof=1))


def _single(dac: int, v: float) -> np.ndarray:
    out = np.zeros(N_HEATERS)
    out[dac] = v
    return out


def balance(rig, dac_u: int, dac_k: int, v_u: float | None = None, sigma: float | None = None,
            tol: float = 0.02, max_steps: int = 4, **kw) -> Balance:
    """Match `dac_u` at `v_u` against known `dac_k`, and return the balance.

    Interleaved u,k,u,k rather than u then k: the die drifts over the minutes this takes, and
    a drift that is common to both arms cancels out of the difference only if the arms are
    adjacent in time. That is the whole reason this is cheap enough to be worth doing -- the
    calorimeter does not have to be stable for an hour, only between two neighbouring dwells.

    Secant, not bisection. Drive is close to affine in power over the ~100 mW a single heater
    spans, so two points bracket well and the third usually lands inside `tol`; bisection
    would spend the same settles to less effect."""
    r_k = HEATER_OHMS[dac_k]
    if r_k is None:
        raise ValueError(f"DAC {dac_k} is not a known reference: its resistance is unmeasured")
    v_u = float(VOLTAGE_MAX_CH[dac_u] if v_u is None else v_u)
    sigma = noise_floor(rig, **kw) if sigma is None else float(sigma)

    d0, _ = _drive(rig, np.zeros(N_HEATERS), **kw)
    du, _ = _drive(rig, _single(dac_u, v_u), **kw)
    if abs(du - d0) < SNR_MIN * sigma:
        raise RuntimeError(
            f"DAC {dac_u} at {v_u:.2f} V moves the TEC drive by {abs(du-d0)*1e3:.1f} mV "
            f"against a {sigma*1e3:.1f} mV floor -- below the {SNR_MIN}x gate. The heater is "
            f"too weak to weigh against this calorimeter; measure it with an ohmmeter.")

    # Seed from the nominal pairing, then correct. Starting at the far end of the reference's
    # range instead would put the first probe where drive is least linear.
    v_k = min(VOLTAGE_MAX_CH[dac_k], v_u * np.sqrt(r_k / 114.0))
    hist = []
    for _ in range(max_steps):
        dk, _ = _drive(rig, _single(dac_k, v_k), **kw)
        hist.append((v_k, dk))
        if abs(dk - du) < tol * abs(du - d0):
            break
        if len(hist) >= 2:
            (a, fa), (b, fb) = hist[-2], hist[-1]
            if abs(fb - fa) > 1e-9:
                v_k = b + (du - fb) * (b - a) / (fb - fa)
            else:
                v_k = b * float(np.sqrt(abs(du - d0) / max(abs(fb - d0), 1e-9)))
        else:
            # power is the lever, so scale the SQUARE of the voltage by the drive ratio
            v_k = v_k * float(np.sqrt(abs(du - d0) / max(abs(dk - d0), 1e-9)))
        v_k = float(np.clip(v_k, 0.05, VOLTAGE_MAX_CH[dac_k]))
    return Balance(dac_u, dac_k, v_u, v_k, float(r_k), du, hist[-1][1], d0, sigma)


def holdout(rig, dac: int, ref: int, **kw) -> tuple[float, float, float]:
    """Infer a resistance the rig already knows. Returns (true, inferred, relative error).

    The gate, not a formality. If a heater at the other end of the die cannot be weighed
    against this reference to a few percent, then position DOES enter the measurement and the
    unknowns must not be trusted to it either."""
    truth = HEATER_OHMS[dac]
    if truth is None:
        raise ValueError(f"DAC {dac} has no known resistance to hold out")
    b = balance(rig, dac, ref, **kw)
    return float(truth), b.ohms, abs(b.ohms - truth) / truth


def unmeasured() -> list[int]:
    return [i for i, r in enumerate(HEATER_OHMS) if r is None]


def references() -> list[int]:
    return [i for i, r in enumerate(HEATER_OHMS) if r is not None]


def _selftest():
    """A TEC whose drive is a nonlinear, offset, drifting function of total heater power.

    Every one of those is a property the substitution is supposed to be immune to, so the
    model has them on purpose: an affine calorimeter would pass a method that only works on
    an affine calorimeter."""

    class FakeTEC:
        def __init__(self, rng):
            self.rng, self.load, self.t = rng, 0.0, 0.0

        def status(self):
            self.t += 1.0
            # offset + gain + curvature + a slow ramp + white noise: nothing here is known
            # to the routine, and none of it may reach the answer
            d = (-0.813 + 4.02 * self.load - 1.7 * self.load ** 2
                 + 2e-5 * self.t + self.rng.normal(0, 3e-4))
            return {"drive_v": d}

    class FakeRig:
        def __init__(self, ohms, rng):
            self.ohms, self.tec = ohms, FakeTEC(rng)

        def measure(self, v):
            self.tec.load = float(sum(vi ** 2 / r for vi, r in zip(v, self.ohms) if r))
            return np.zeros(4)

    rng = np.random.default_rng(0)
    truth = [r if r is not None else t for r, t in
             zip(HEATER_OHMS, [114.1, 62.3, 118.8, 114.1, 57.8, 113.5, 117.4, 57.2,
                               112.6, 114.9, 56.4, 115.8, 116.1, 117.1, 116.8, 114.1])]
    rig = FakeRig(truth, rng)
    fast = dict(settle_s=0.0, reads=8, poll_s=0.0)

    sigma = noise_floor(rig, **fast)
    assert sigma < 1e-2, sigma

    # holdouts: weigh known heaters from both resistance groups against one reference
    errs = []
    for dac in (0, 4, 9, 10):
        t, got, err = holdout(rig, dac, ref=3, sigma=sigma, **fast)
        errs.append(err)
        assert err < 0.05, f"DAC {dac}: {t:.1f} -> {got:.1f} ohm, {err:.1%}"
    print(f"holdout on 4 known channels: max error {max(errs):.2%}")

    # the unknowns, against the same reference
    for dac in unmeasured():
        b = balance(rig, dac, 3, sigma=sigma, **fast)
        err = abs(b.ohms - truth[dac]) / truth[dac]
        assert err < 0.06, f"DAC {dac}: {truth[dac]:.1f} -> {b.ohms:.1f} ohm, {err:.1%}"
        print(f"  DAC {dac:2d}  {b.ohms:6.1f} ohm  (true {truth[dac]:.1f}, "
              f"{err:+.2%}, snr {b.snr:.0f}, residual {b.residual:.1f})")

    # a heater too weak to weigh must be refused, not guessed at
    weak = FakeRig([1e6] * N_HEATERS, np.random.default_rng(1))
    weak.ohms[3] = 114.1
    try:
        balance(weak, 8, 3, v_u=0.01, sigma=1e-3, **fast)
    except RuntimeError as e:
        assert "below the" in str(e)
        print("refuses a sub-threshold channel rather than returning a number")
    else:
        raise AssertionError("a channel below the SNR gate must be refused")

    print("resistance-by-substitution selftest OK")


if __name__ == "__main__":
    _selftest()


def curve(rig, dac_k: int, n: int = 4, volts=None, **kw) -> tuple[np.ndarray, np.ndarray, float]:
    """Drive vs power for a KNOWN channel: the calorimeter's response, measured once.

    `balance` is the better measurement -- its answer survives any monotone calorimeter -- but
    it spends a settle per secant step per unknown. This spends `n` settles ONCE and then
    costs one settle per unknown, which is the difference between eight minutes and thirty.
    The price is that it assumes the response is smooth enough to interpolate over the ~20 mW
    a single 1.5 V heater spans, and that assumption is not free: `holdout` is what decides
    whether it held, and it is the same holdout the balance path answers to."""
    r_k = HEATER_OHMS[dac_k]
    # The curve has to BRACKET the unknowns in POWER, not span the reference's own range:
    # an unknown staged at 1.5 V dissipates 19-40 mW depending on which group it is in, and a
    # curve that starts at a third of a 4.55 V reference starts at 22 mW and leaves the
    # 114 ohm case below its floor. `from_curve` refuses to extrapolate, so getting this
    # wrong costs the whole run rather than silently biasing it.
    if volts is None:
        p_lo, p_hi = 0.5 * V_UNMEASURED ** 2 / 120.0, 2.0 * V_UNMEASURED ** 2 / 56.0
        v = np.sqrt(np.linspace(p_lo, p_hi, n) * r_k)
    else:
        v = np.asarray(volts, float)
    v = np.clip(v, 0.05, VOLTAGE_MAX_CH[dac_k])
    d0, _ = _drive(rig, np.zeros(N_HEATERS), **kw)
    p = v ** 2 / r_k
    d = np.array([_drive(rig, _single(dac_k, vi), **kw)[0] for vi in v])
    return p, d - d0, d0


def from_curve(rig, dac: int, p, dd, d0, v: float | None = None, **kw) -> float:
    """Resistance of `dac` read off a calorimeter curve. R = V^2 / P."""
    v = float(VOLTAGE_MAX_CH[dac] if v is None else v)
    du, _ = _drive(rig, _single(dac, v), **kw)
    y = du - d0
    order = np.argsort(dd)
    if not (dd[order][0] <= y <= dd[order][-1]):
        raise RuntimeError(
            f"DAC {dac} at {v:.2f} V lands at {y*1e3:+.1f} mV, outside the reference curve's "
            f"{dd.min()*1e3:+.1f}..{dd.max()*1e3:+.1f} mV -- extrapolating a calorimeter is "
            f"how a resistance gets invented. Widen the curve or use `balance`.")
    return v ** 2 / float(np.interp(y, dd[order], np.asarray(p)[order]))
