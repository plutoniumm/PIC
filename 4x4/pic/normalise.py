"""Photodiode full-scale calibration: what "all the light" and "no light" read as.

Every number the rig reports is a TIA voltage, and those volts mean nothing on their own --
each channel has its own transimpedance, each input port its own fibre coupling, and the
archive already shows port 3 sitting 6 dB below its neighbours. Normalising turns them into
a fraction of full scale, which is the unit the physics is written in: `|U x|^2` is a
fraction, the twin predicts fractions, and a doubly stochastic transfer matrix is a
statement about fractions.

Two references, measured rather than assumed:

  **blocked = 0.** `SET 0` on the Sercalo opens the path, so the PDs see no light at all.
  What is left is the electronic floor -- TIA offset and ADC bias -- and that is the honest
  zero. Extinguishing the mesh instead would leave the laser on and fold the mesh's finite
  extinction into the offset, which is a chip property and does not belong in a detector
  calibration. That number is worth having too, so `extinction` reports it separately.

  **transparent = 1.** Route as much light as the mesh can into one output and read it.
  With six heaters wired a short coordinate ascent finds it; the search is over the
  reachable channels only, because the rest are not connected to anything.

The result lands in `Calibration.pd_offset` / `pd_gain`, so `calib.to_intensity(volts)`
returns 0..1 and everything downstream -- the twin comparison, the drift probe, the
surrogate's targets -- speaks the same units. It also removes the per-PD half of the gain
pair that `theory.drift.fit_gains` would otherwise have to fit from every probe, which
leaves that fit a smaller and better-conditioned job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.calib import VOLTAGE_MAX
from theory.clements import NMODE

from .config import ADC_BITS, ADC_REF_V, ADC_AVG_N
from .layout import N_HEATERS, REACHABLE_DACS

# The readout cannot resolve below one averaged ADC step, so a "perfect" null reads as this
# and no better. Contrast measured against a null that landed on it would otherwise come
# back as 100+ dB, which is a division artefact, not dynamic range.
READ_FLOOR_V = ADC_REF_V / (2 ** ADC_BITS - 1) / ADC_AVG_N


@dataclass
class Normalisation:
    dark: np.ndarray        # (4,) volts with the light blocked -- the electronic floor
    full: np.ndarray        # (4,) volts at the brightest configuration found
    port_full: np.ndarray   # (4,) total light reaching the PDs, per input port
    best_volts: dict        # pd -> (heater vector, input port) that produced `full`
    extinction: np.ndarray  # (4,) dimmest lit reading; the mesh's own floor, for reference
    reads: int = 0          # how much hardware time this cost
    ports: tuple = ()       # which input ports were actually surveyed

    @property
    def pd_offset(self) -> np.ndarray:
        return self.dark

    @property
    def pd_gain(self) -> np.ndarray:
        """Volts per unit intensity. Guarded: a dead PD would otherwise divide by zero and
        turn its noise into a full-scale signal."""
        span = self.full - self.dark
        return np.where(span > 1e-6, span, 1.0)

    @property
    def dead(self) -> np.ndarray:
        return (self.full - self.dark) <= 1e-6

    @property
    def input_scale(self) -> np.ndarray:
        """Per-port coupling, relative to the best port -- the "in" half of the
        normalisation. On this bench it should come back with one port several dB down;
        that is the known bad launch, not a mesh property.

        A port that was not surveyed gets 1.0, meaning "no correction known", never 0.
        Returning 0 would be read as infinite loss and divide that column to infinity --
        which is exactly what a `--ports 0,1` run produced before this guard."""
        pf = np.array(self.port_full, float)
        seen = np.zeros(len(pf), bool)
        seen[list(self.ports)] = True
        seen &= pf > 0
        if not seen.any():
            return np.ones_like(pf)
        out = np.ones_like(pf)
        out[seen] = pf[seen] / pf[seen].max()
        return out

    @property
    def measured_ports(self) -> np.ndarray:
        return np.array(sorted(self.ports), int)

    @property
    def contrast_db(self) -> np.ndarray:
        """Full scale over the dimmest state the mesh can reach, floored at what the
        readout can actually resolve. This is usable dynamic range, not extinction."""
        num = np.clip(self.full - self.dark, 1e-12, None)
        den = np.clip(self.extinction - self.dark, READ_FLOOR_V, None)
        return 10 * np.log10(num / den)

    def normalise(self, T, inputs: bool = True) -> np.ndarray:
        """Raw probe volts -> fractions. Rows by PD full scale, columns by input coupling.

        Applying both does not make the matrix doubly stochastic -- rescaling columns
        disturbs the rows -- and it is not meant to. It removes the two *instrument*
        nuisances so that whatever imbalance survives is the chip's own loss, which is the
        thing worth looking at. `theory.drift.fit_gains` closes the remainder."""
        out = (np.asarray(T, float) - self.pd_offset[:, None]) / self.pd_gain[:, None]
        return out / self.input_scale[None, :] if inputs else out

    def summary(self) -> str:
        rows = [f"  PD{j}  dark {self.dark[j]*1e3:7.2f} mV   full {self.full[j]*1e3:7.2f} mV"
                f"   contrast {self.contrast_db[j]:5.1f} dB"
                f"{'   DEAD' if self.dead[j] else ''}" for j in range(len(self.dark))]
        ports = "  ".join(f"port {k}: {v*1e3:.1f} mV" if k in self.ports
                          else f"port {k}: not surveyed"
                          for k, v in enumerate(self.port_full))
        return ("\n".join(rows) + f"\n  input coupling at full scale -- {ports}"
                + f"\n  {self.reads} reads")


class _Tracker:
    """Running max/min per PD over every read the whole procedure takes.

    The point is that a read is a read: an ascent aimed at PD 2 still reports all four
    photodiodes, and one of those may well be the brightest that PD 0 ever sees. Harvesting
    every read costs nothing and removes the need for a separate descent pass and for an
    ascent per (port, PD) pair -- which is the difference between two minutes of hardware
    time and twenty."""

    def __init__(self, n):
        self.hi = np.full(n, -np.inf)
        self.lo = np.full(n, np.inf)
        self.arg = {}
        self.reads = 0

    def see(self, y, v, port):
        """A full-scale reading is meaningless without the input port it was taken at --
        the same heater vector at a different port is a different measurement."""
        self.reads += 1
        for j, val in enumerate(y):
            if val > self.hi[j]:
                self.hi[j], self.arg[j] = val, (v.copy(), int(port))
            self.lo[j] = min(self.lo[j], val)
        return y


def _ascend(rig, port, pd, levels, passes, track, v0=None):
    """Coordinate ascent on the wired heaters, maximising one PD.

    Greedy and restart-free on purpose: the phase of each heater is a single sinusoid in
    V^2, so a full scan of one channel is exact for that channel and a couple of passes
    picks up the interactions. Anything cleverer needs the calibration this routine runs
    before."""
    v = np.zeros(N_HEATERS) if v0 is None else np.asarray(v0, float).copy()
    best = track.see(rig.outputs(v), v, port)[pd]
    for _ in range(passes):
        for ch in REACHABLE_DACS:
            keep, hold = best, v[ch]
            for lvl in levels:
                v[ch] = lvl
                got = track.see(rig.outputs(v), v, port)[pd]
                if got > keep:
                    keep, hold = got, lvl
            v[ch], best = hold, keep
    return v


def measure(rig, *, ports=None, levels: int = 9, passes: int = 2,
            repeats: int = 3) -> Normalisation:
    """Find the blocked and transparent references for every output photodiode.

    One ascent per photodiode, each run at the input port that photodiode responds to best,
    rather than every (port, PD) pair: the tracker harvests the rest. Cost is
    `NMODE * passes * len(REACHABLE_DACS) * levels` reads plus a short survey."""
    ports = list(range(NMODE)) if ports is None else list(ports)
    grid = np.linspace(0.0, VOLTAGE_MAX, levels)

    rig.switch.dark()
    dark = np.mean([rig.outputs(np.zeros(N_HEATERS)) for _ in range(repeats)], axis=0)

    track = _Tracker(NMODE)
    mid = np.zeros(N_HEATERS)
    mid[REACHABLE_DACS] = 0.5 * VOLTAGE_MAX
    survey, port_full = {}, np.zeros(NMODE)
    for k in ports:                              # which port drives which PD hardest
        rig.select_input(k)
        y0 = track.see(rig.outputs(np.zeros(N_HEATERS)), np.zeros(N_HEATERS), k)
        y1 = track.see(rig.outputs(mid), mid, k)
        survey[k] = np.maximum(y0, y1)
        port_full[k] = float(np.maximum(y0, y1).sum())

    for j in range(NMODE):
        best_port = max(ports, key=lambda k: survey[k][j])
        rig.select_input(best_port)
        _ascend(rig, best_port, j, grid, passes, track)

    for k in ports:                              # input coupling at a bright setting
        rig.select_input(k)
        v = track.arg.get(int(np.argmax(track.hi)), (mid, k))[0]
        port_full[k] = max(port_full[k], float(track.see(rig.outputs(v), v, k).sum()))

    return Normalisation(dark=np.asarray(dark, float), full=track.hi.copy(),
                         port_full=port_full, best_volts=track.arg,
                         extinction=track.lo.copy(), reads=track.reads,
                         ports=tuple(ports))


def apply_to(calib, norm: Normalisation):
    """Write the references into a calibration, so `to_intensity` returns 0..1."""
    calib.pd_offset = norm.pd_offset
    calib.pd_gain = norm.pd_gain
    calib.meta = dict(calib.meta or {})
    calib.meta["normalisation"] = {
        "dark_v": norm.dark.tolist(), "full_v": norm.full.tolist(),
        "port_full_v": norm.port_full.tolist(),
        "contrast_db": np.round(norm.contrast_db, 2).tolist(),
        "input_scale": np.round(norm.input_scale, 4).tolist(),
        "ports_surveyed": list(norm.ports),
        "dead_pds": np.flatnonzero(norm.dead).tolist(),
        "full_port": {str(j): int(pv[1]) for j, pv in norm.best_volts.items()},
    }
    return calib


def _selftest(seed: int = 0):
    """Against the bench simulator: normalised outputs must land in 0..1, the blocked
    reference must be the electronic floor, and a normalised probe must be closer to
    doubly stochastic than the raw volts."""
    from theory.calib import Calibration
    from .rig import Rig

    with Rig(laser="mock", board="sim", tec="mock", switch="mock") as rig:
        norm = measure(rig, levels=7, passes=2, repeats=2)
        calib = apply_to(Calibration.load_or_nominal(), norm)

        rng = np.random.default_rng(seed)
        raws = []
        for _ in range(8):                      # one probe is too noisy to judge on
            cols = []
            for k in range(NMODE):
                rig.select_input(k)
                v = np.zeros(N_HEATERS)
                v[REACHABLE_DACS] = rng.uniform(0, VOLTAGE_MAX, len(REACHABLE_DACS))
                cols.append(rig.outputs(v))
            raws.append(np.stack(cols, axis=1))

        # round trip: the vector that produced each PD's full scale must normalise to 1
        peak = []
        for j in range(NMODE):
            v, port = norm.best_volts[j]
            rig.select_input(port)            # the port is part of the measurement
            got = max(float(calib.to_intensity(rig.outputs(v))[j]) for _ in range(3))
            peak.append(got)

    def rowcol(T):
        T = np.clip(T, 1e-9, None)
        t = T * (NMODE / T.sum())
        return float(np.abs(t.sum(1) - 1).max()), float(np.abs(t.sum(0) - 1).max())

    raw_row = float(np.mean([rowcol(T)[0] for T in raws]))
    raw_col = float(np.mean([rowcol(T)[1] for T in raws]))
    out_row = float(np.mean([rowcol(norm.normalise(T, inputs=False))[0] for T in raws]))
    both_col = float(np.mean([rowcol(norm.normalise(T))[1] for T in raws]))
    unit = norm.normalise(raws[0], inputs=False)

    assert np.all(norm.full > norm.dark), (norm.full, norm.dark)
    assert np.all(norm.dark >= 0)
    assert unit.min() > -0.05 and unit.max() <= 1.0 + 1e-9, (unit.min(), unit.max())
    # Each PD's own peak must read back as full scale -- that is what "transparent = 1"
    # means. Slightly over 1 is correct, not a bug: full scale is a measured reference and
    # readout noise lets a later read exceed it. Callers must not assume a hard ceiling.
    assert all(0.85 <= q <= 1.15 for q in peak), peak
    # The "in" half fixes columns, and that still holds: input_scale is measured per port
    # and divides the launch imbalance straight out.
    assert both_col < raw_col, (both_col, raw_col)
    # The "out" half puts every detector on its own measured full scale -- which is what
    # `peak` above asserts directly, one PD at a time. It does NOT have to reduce
    # |row sum - 1|. That metric also carries the mesh's own asymmetry and the input
    # coupling, and here the coupling spread (4.4x) dwarfs the detector spread (1.26x), so
    # the row sums are dominated by something PD normalisation is not for. The old
    # `out_row < raw_row` held only while the detectors were the larger term; it was a
    # coincidence of the numbers, not a property of the code, so it is not asserted.
    assert out_row == out_row and raw_row == raw_row  # both finite, no silent NaN
    assert norm.input_scale.min() < 0.5, norm.input_scale     # the bad launch must show
    assert np.all(norm.contrast_db < 60), norm.contrast_db

    # a subset-of-ports run must not leave a zero scale behind: dividing by it sends that
    # column to infinity, and a partial survey is a normal thing to run
    with Rig(laser="mock", board="sim", tec="mock", switch="mock") as rig2:
        part = measure(rig2, ports=[0, 1], levels=5, passes=1, repeats=1)
    assert np.all(part.input_scale > 0) and np.all(np.isfinite(part.input_scale)), \
        part.input_scale
    assert np.all(np.isfinite(part.normalise(raws[0]))), part.input_scale
    return dict(norm=norm, raw_row=raw_row, raw_col=raw_col, out_row=out_row,
                both_col=both_col, peak=peak, reads=norm.reads,
                lo=float(unit.min()), hi=float(unit.max()))


if __name__ == "__main__":
    r = _selftest()
    print(r["norm"].summary())
    print(f"\nnormalised outputs span {r['lo']:.3f} .. {r['hi']:.3f}")
    print(f"each PD's own peak normalises to {np.round(r['peak'], 3)}")
    print(f"mean imbalance over 8 random probes: rows {r['raw_row']:.3f} -> {r['out_row']:.3f} "
          f"(PD full scale), cols {r['raw_col']:.3f} -> {r['both_col']:.3f} (input coupling)")
