"""Real PIC-B constraints, loaded from `pic_data/pic_b_config.json`.

Turns the live characterization state into the masks and calibration the twin needs:
which heaters can be *driven*, which are *characterized* (so their phase can be
commanded), and the per-heater (Vpi, phi0) that converts phase <-> volts.

Everything is keyed by geometric heater id H0..H119 (== the config's `net` field),
which is the same index `theory.twin` uses for its phase vector.
"""

from __future__ import annotations

import json
import math

import numpy as np

CONFIG = "pic_data/pic_b_config.json"
NH = 120
ADC_LSB = 5.0 / 1023  # 10-bit over 0-5 V ~ 4.9 mV
ADC_AVG_N = 10        # firmware averages this many sweeps


class Hardware:
    """Per-heater state of the real chip.

    vpi[h], phi0[h]   calibration, NaN where uncharacterized
    driveable[h]      DAC reaches it and it is not dead/shorted
    known[h]          driveable AND characterized -> phase is commandable
    """

    def __init__(self, path: str = CONFIG, vmax: float = 4.0, phi0_offset=None):
        cfg = json.load(open(path))
        self.cfg, self.vmax = cfg, vmax
        self.vpi = np.full(NH, np.nan)
        self.phi0 = np.full(NH, np.nan)
        self.driveable = np.zeros(NH, bool)
        self.stage = np.array(["?"] * NH, dtype=object)
        self.dac_of = np.full(NH, -1, int)

        for r in cfg["channels"]:
            net = r.get("net")
            if net in (None, "") or r.get("stage") == "spare":
                continue
            h = int(net)
            self.dac_of[h] = r["dac"]
            self.stage[h] = r["stage"]
            self.driveable[h] = r.get("elec") == "ok"
            vpi, v0 = r.get("Vpi"), r.get("V0")
            if vpi not in (None, "") and v0 not in (None, ""):
                self.vpi[h] = float(vpi)
                # max pass at V0 => phi(V0) == 0 (mod 2pi)
                self.phi0[h] = (-math.pi * (float(v0) / float(vpi)) ** 2) % (2 * math.pi)

        # backprop drift-correction: add an inferred per-heater phi0 offset (radians), applied
        # only to characterised heaters. This is how a "corrected" config differs from baseline
        # -- a pure phase-offset shift, no V0 root-flip. Consumed by scripts/ising_ab.py (--dphi);
        # the drift-inference investigation that produced these offsets is archived in hw.md.
        if phi0_offset is not None:
            off = np.asarray(phi0_offset, float)
            fin = np.isfinite(self.phi0) & np.isfinite(off)
            self.phi0[fin] = (self.phi0[fin] + off[fin]) % (2 * math.pi)

        self.known = self.driveable & np.isfinite(self.vpi)

    def summary(self):
        import collections
        by = collections.Counter()
        for h in range(NH):
            if self.stage[h] == "?":
                continue
            by[(self.stage[h], "known" if self.known[h] else
                "drivable" if self.driveable[h] else "dead")] += 1
        return by

    def reachable_phase(self, h, phase):
        """Can heater h be commanded to `phase` (mod 2pi) within 0..vmax?"""
        if not self.known[h]:
            return False
        span = math.pi * (self.vmax / self.vpi[h]) ** 2
        return (phase - self.phi0[h]) % (2 * math.pi) <= span or span >= 2 * math.pi

    def quantize(self, phases, rng=None):
        """Round-trip a phase vector through volts: clamp to what each heater can reach,
        and hold uncharacterized/dead heaters at their 0 V phase (phi0, or unknown).

        Returns (phases_realized, ok_mask). Unknown heaters keep whatever phase the
        caller supplied only if `rng` is None; otherwise they get a random fixed phase,
        which is the honest model — we cannot command them, and they are not zero.
        """
        ph = np.asarray(phases, float).copy()
        ok = np.zeros(NH, bool)
        for h in range(NH):
            if self.known[h]:
                span = math.pi * (self.vmax / self.vpi[h]) ** 2
                want = (ph[h] - self.phi0[h]) % (2 * math.pi)
                if want <= span:
                    ok[h] = True
                    ph[h] = self.phi0[h] + want
                else:  # unreachable within 0..vmax: clip to the nearest end
                    ph[h] = self.phi0[h] + (0.0 if want > (span + 2 * math.pi) / 2 else span)
            elif self.driveable[h]:
                ph[h] = rng.uniform(0, 2 * math.pi) if rng is not None else ph[h]
            else:
                ph[h] = rng.uniform(0, 2 * math.pi) if rng is not None else ph[h]
        return ph, ok

    def volts(self, phases):
        """Phase vector -> DAC volts (NaN where not commandable)."""
        ph = np.asarray(phases, float)
        u = (ph - self.phi0) % (2 * math.pi) / math.pi * self.vpi ** 2
        v = np.sqrt(np.clip(u, 0, None))
        v[~self.known] = np.nan
        return np.clip(v, 0, self.vmax)


def adc_noise(y, rng, avg_n: int = ADC_AVG_N, lsb: float = ADC_LSB):
    """Quantization + shot-ish readout noise on a PD reading, after on-chip averaging."""
    y = np.asarray(y, float)
    sigma = lsb / math.sqrt(12 * avg_n)
    return np.round((y + rng.normal(0, sigma, y.shape)) / lsb) * lsb
