"""Closed-loop drift correction on the live rig.

`theory.drift` owns the maths; this owns the instrument. One probe is four switch
positions and four photodiode reads at a held bias -- about five seconds -- and it returns
the full intensity transfer matrix, which is enough to re-anchor both the per-port coupling
and the nine mesh phase directions intensity can see.

Off by default, and deliberately. Correcting a drift smaller than the probe noise makes the
chip *worse*, and on the bench's own 30-minute set the drift is already down at the noise:
the 6x6 spent a lot of effort learning that a correction has to earn its place. So the
tracker refuses its own estimate unless the fit both looks like a unitary and reduces the
mismatch (`DriftEstimate.ok`), and `--dynamic` is what opts in.

    rig = Rig(board="sim", dynamic=True).open()
    with rig.session(duration_s=60, power_dbm=8):
        rig.drift.anchor()          # reference probe, at the calibration operating point
        ...
        rig.drift.update()          # re-probe; applies if it earns it
        v, ok = rig.program(U)      # now carries the correction
"""

from __future__ import annotations

import time

import numpy as np

from theory.clements import NMODE
from theory.drift import RCOND, infer_drift

from .config import VOLTAGE_MAX
from .layout import N_HEATERS

# A probe at all-zero bias is a poor place to measure from: several MZIs sit at an extremum
# where the phase derivative vanishes, so drift there moves nothing. This is a fixed,
# arbitrary-looking bias chosen only to be off every extremum at once; any reproducible
# mid-range point does the job, and it must not change between anchor and update.
PROBE_BIAS_FRAC = 0.45


class DriftTracker:
    """Holds the reference probe and the correction inferred from the latest one."""

    def __init__(self, rig, *, enabled: bool = False, period_s: float = 180.0,
                 repeats: int = 3, rcond: float = RCOND, verbose: bool = False):
        self.rig = rig
        self.enabled = bool(enabled)
        self.period_s = float(period_s)
        self.repeats = int(repeats)
        self.rcond = float(rcond)
        self.verbose = verbose
        self.bias = None          # the held operating point; fixed at anchor time
        self.T_ref = None
        self.last = None          # most recent DriftEstimate
        self._dphi = np.zeros(N_HEATERS)
        self._t_last = 0.0
        self.history = []

    def default_bias(self) -> np.ndarray:
        v = np.zeros(N_HEATERS)
        v[:] = PROBE_BIAS_FRAC * VOLTAGE_MAX
        return v

    def probe(self, bias=None) -> np.ndarray:
        """Cycle the four input ports at a held bias -> T[j, k], PD j with port k lit.

        Averaging happens here rather than in the firmware because the switch, not the
        readout, is the slow part: repeating the whole port cycle also averages over the
        switch's own repeatability, which is the larger of the two."""
        b = self.default_bias() if bias is None else np.asarray(bias, float)
        acc = np.zeros((NMODE, NMODE))
        for _ in range(max(1, self.repeats)):
            for k in range(NMODE):
                self.rig.select_input(k)
                acc[:, k] += self.rig.outputs(b)
        return acc / max(1, self.repeats)

    def anchor(self, bias=None):
        """Take the reference probe. Everything later is measured relative to this."""
        self.bias = self.default_bias() if bias is None else np.asarray(bias, float)
        self.T_ref = self.probe(self.bias)
        self._dphi = np.zeros(N_HEATERS)
        self._t_last = time.time()
        self.history.clear()
        return self.T_ref

    def due(self) -> bool:
        return self.enabled and (time.time() - self._t_last) >= self.period_s

    def update(self, force: bool = False):
        """Re-probe and, if the estimate earns it, adopt the correction.

        A rejected estimate leaves the previous correction in place rather than reverting
        to zero: the last thing that did explain the chip beats nothing at all."""
        if self.T_ref is None:
            self.anchor()
            return None
        if not (force or self.due()):
            return None

        T = self.probe(self.bias)
        phases0 = self.rig.calib.phases(self.bias)
        est = infer_drift(self.rig.twin, phases0, T, T_ref=self.T_ref, rcond=self.rcond)
        self._t_last = time.time()
        self.last = est
        self.history.append(est)
        if est.ok:
            self._dphi = est.dphi
        if self.verbose:
            print(f"  drift {'applied' if est.ok else 'rejected'}: {est}")
        return est

    @property
    def dphi(self) -> np.ndarray:
        return self._dphi if self.enabled else np.zeros(N_HEATERS)

    def correct(self, phases) -> np.ndarray:
        """Pre-distort a target phase vector by the inferred drift."""
        return np.asarray(phases, float) - self.dphi

    def status(self) -> dict:
        return {"enabled": self.enabled,
                "anchored": self.T_ref is not None,
                "applied_rad": float(np.abs(self.dphi).max()),
                "last": str(self.last) if self.last else None}


class _TwinRig:
    """A rig whose chip *is* the twin, for testing the loop rather than the chip.

    The real bench cannot exercise this path yet: with a nominal calibration the twin and
    the chip disagree by far more than any drift, so `infer_drift` correctly rejects every
    estimate. That rejection is the designed behaviour -- drift correction is gated behind
    characterization -- but it means the plumbing needs a chip the model actually matches."""

    def __init__(self, dphi_true, gains=None, noise=0.0, seed=0):
        from theory.calib import Calibration
        from theory.twin import Twin
        self._twin = Twin()
        self.calib = Calibration.load_or_nominal()
        self.dphi_true = np.asarray(dphi_true, float)
        self.g, self.c = (np.ones(NMODE), np.ones(NMODE)) if gains is None else gains
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self._port = 0

    @property
    def twin(self):
        return self._twin

    def select_input(self, port):
        self._port = int(port)
        return self._port

    def outputs(self, volts):
        import torch
        ph = self.calib.phases(volts) + self.dphi_true
        U = self._twin.matrix(torch.as_tensor(ph, dtype=torch.float32)).detach().numpy()
        col = np.abs(U[:, self._port]) ** 2 * self.g * self.c[self._port]
        return np.clip(col + self.rng.normal(0, self.noise * max(col.mean(), 1e-12), NMODE),
                       1e-12, None)


def _selftest(seed: int = 0):
    """The loop must correct a planted drift, report the gains, and refuse what it cannot see.

    Errors are measured in the de-gained frame. `dphi` corrects phase; the coupling change
    is reported, not pre-distorted away, because no heater can undo a lossy fibre -- so a
    raw |dT| would score the phase fit on a job it is not doing."""
    from theory.calib import Calibration
    from theory.drift import degain, fit_gains

    calib = Calibration.load_or_nominal()
    rng = np.random.default_rng(seed)

    def err(T, T_ref):
        g, c = fit_gains(T, T_ref)
        A = degain(T, g, c)
        return float(np.abs(A / A.sum() - T_ref / T_ref.sum()).mean())

    def trial(scale, noise, gains=None):
        rig = _TwinRig(np.zeros(N_HEATERS), gains=gains, noise=noise, seed=seed)
        tr = DriftTracker(rig, enabled=True, repeats=3)
        T_ref = tr.anchor()
        dphi = np.zeros(N_HEATERS)
        dphi[:12] = rng.normal(0, scale, 12)          # mesh heaters only
        rig.dphi_true = dphi                          # the chip drifts under us
        T_bad = tr.probe(tr.bias)
        est = tr.update(force=True)
        v_corr, _ = calib.volts(tr.correct(calib.phases(tr.bias)))
        return err(T_bad, T_ref), err(tr.probe(v_corr), T_ref), est

    before, after, est = trial(0.10, 0.01)
    g_before, g_after, g_est = trial(0.10, 0.01,
                                     gains=(np.array([1.0, 1.3, 0.9, 1.05]),
                                            np.array([1.0, 0.95, 10 ** -0.6, 1.02])))
    quiet = trial(0.0005, 0.04)

    port3_db = g_est.coupling_db[2] - g_est.coupling_db[0]

    assert est.ok and after < before / 2, (before, after, str(est))
    assert g_est.ok and g_after < g_before / 2, (g_before, g_after, str(g_est))
    assert abs(port3_db + 6.0) < 1.0, port3_db
    assert not quiet[2].ok, f"a drift below the noise must be refused: {quiet[2]}"
    return dict(before=before, after=after, est=est, gain_before=g_before,
                gain_after=g_after, gain_est=g_est, port3_db=port3_db, quiet=quiet)


if __name__ == "__main__":
    r = _selftest()
    print("planted 0.10 rad drift, 1 percent probe noise")
    print(f"  probe error (de-gained) {r['before']:.5f} -> {r['after']:.5f}")
    print(f"  {r['est']}")
    print("same, through a PD-gain change and a -6 dB port-3 coupling change")
    print(f"  probe error (de-gained) {r['gain_before']:.5f} -> {r['gain_after']:.5f}")
    print(f"  port 3 coupling recovered at {r['port3_db']:+.2f} dB (planted -6.00)")
    print(f"drift below the noise floor is refused: {r['quiet'][2]}")
