"""NOTE (2026-08-13): the driver scripts this module names (homodyne_pdmap,
homodyne_calibrate, char_h23_homodyne) were removed in the cleanup -- the branch is closed,
see homodyne.md. Recover them from git history if the bottom LO arm is ever fixed.

Hardware homodyne readout -- the real-chip counterpart of ``theory/readout.py``.

``theory/readout.py`` runs the 2-shot homodyne recovery against the differentiable twin
(fields in, PD banks out, all in torch). This module runs the SAME math against a real
``PIC`` (or ``MockPIC``): program heater volts, set the LO phase on the two reference-arm
encode heaters (H16 top / H23 bottom), read the 14 raw photodiodes, and pull monitor /
homodyne / LO-tap PDs out of the raw read via a **configurable PD-role map** (the map is
schematic-derived and UNVERIFIED on board B -- so it is data, not a hardcoded constant;
verify it with ``scripts/homodyne_pdmap.py``).

Per signal rail, with monitor-tap fraction ``t`` and LO-combiner fraction ``f``:

    mon        = t |E|^2
    homo(psi)  = (1-f)(1-t)|E|^2 + f|LO|^2 + 2 sqrt(f(1-f)(1-t)) |LO| Re(E e^{-i psi})

Two shots (psi = 0, pi/2) plus the monitor and LO-power PDs determine E, up to one global
phase per LO (the two reference arms). Those two phases are fixed chip constants measured
by ``scripts/homodyne_calibrate.py`` and passed here as ``lo_phase`` -- without them
``recover`` is worse than not doing homodyne at all (see that script).

Phase <-> volt is the repo law ``phi = pi (v/Vpi)^2 + phi0`` via ``theory.hw.Hardware``;
net index == geometric heater H0..H119 (== the config's ``net`` field == the twin's phase
index). ``Hardware.dac_of`` scatters a 120-net vector onto the 128 firmware DAC channels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .config import NUM_ADC_RAW, NUM_DAC
from .layout import SIG_ROUTE

NPD = NUM_ADC_RAW
NSIG = 6

LO_PHASE_NET = {"top": 16, "bottom": 23}          # reference-arm encode phase heaters
LO_OF_RAIL = ["top", "top", "top", "bottom", "bottom", "bottom"]  # readout.LO_OF_RAIL
REF_SPLIT_NETS = (0, 1, 14, 15)                    # top/bottom reference split-MZI heaters


@dataclass
class PDMap:
    """Which raw ADC index plays each role. ``monitor``/``homodyne`` are per signal rail
    (index 0..5); ``lo_tap_top``/``lo_tap_bottom`` are the two reference-arm power taps."""

    monitor: tuple           # 6 raw PD indices, one per signal rail
    homodyne: tuple          # 6 raw PD indices, one per signal rail
    lo_tap_top: int
    lo_tap_bottom: int

    def to_dict(self):
        return {"monitor": [int(x) for x in self.monitor],
                "homodyne": [int(x) for x in self.homodyne],
                "lo_tap_top": int(self.lo_tap_top),
                "lo_tap_bottom": int(self.lo_tap_bottom)}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["monitor"]), tuple(d["homodyne"]),
                   int(d["lo_tap_top"]), int(d["lo_tap_bottom"]))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.load(open(path)))


# Schematic default, straight off layout.SIG_ROUTE = [(rail, monitor_pd, homodyne_pd), ...]
DEFAULT_PDMAP = PDMap(
    monitor=tuple(mon for _, mon, _ in SIG_ROUTE),      # (2, 4, 6, 7, 9, 11)
    homodyne=tuple(homo for _, _, homo in SIG_ROUTE),   # (1, 3, 5, 8, 10, 12)
    lo_tap_top=0,
    lo_tap_bottom=13,
)


def open_reference(phases):
    """Return a copy with both reference arms at full transmission (their split MZIs
    balanced) so the LO is bright -- mirrors ``theory.readout.open_reference``."""
    ph = np.asarray(phases, float).copy()
    ph[list(REF_SPLIT_NETS)] = 0.0
    return ph


def patch_lo(hw, cal):
    """Make LO heaters commandable on a ``Hardware`` in place. ``cal`` maps net -> (Vpi, phi0)
    -- e.g. the ``char_h23_homodyne`` result for net 23, or a synthetic pair for --mock.
    After this, ``hw.volts`` can command those nets and the twin can invert their volts."""
    for net, (vpi, phi0) in cal.items():
        hw.vpi[net] = float(vpi)
        hw.phi0[net] = float(phi0) % (2 * math.pi)
        hw.driveable[net] = True
        hw.known[net] = True
    return hw


class Homodyne:
    """Two-shot homodyne acquire/recover over a real (or mock) ``PIC``.

    ``pic``        an open ``PIC``/``MockPIC``; only ``measure_raw(v128)`` is used.
    ``hw``         ``theory.hw.Hardware`` -- phase<->volt calibration + net->dac map.
    ``pdmap``      role map (default schematic; verify with ``homodyne_pdmap``).
    ``t``, ``f``   monitor-tap and LO-combiner power fractions (twin defaults 0.1, 0.5).
    ``lo_phase``   per-rail LO absolute phase (6,) from ``homodyne_calibrate``; ``recover``
                   derotates by it. ``None`` -> ``recover(derotate=...)`` is uncalibrated.
    ``dark``       optional 14-vector dark baseline subtracted from every raw read.
    ``on_tick``    called after every hardware read (laser keepalive under ``laser_session``).
    """

    def __init__(self, pic, hw, pdmap=DEFAULT_PDMAP, t=0.1, f=0.5, lo_phase=None,
                 dark=None, on_tick=None):
        self.pic = pic
        self.hw = hw
        self.pdmap = pdmap
        self.t, self.f = float(t), float(f)
        self.lo_phase = None if lo_phase is None else np.asarray(lo_phase, float)
        self.dark = np.zeros(NPD) if dark is None else np.asarray(dark, float)
        self.on_tick = on_tick
        self.num_dac = getattr(pic.cfg, "num_dac", NUM_DAC)

    def to_dac_volts(self, phases):
        """120-net phase vector -> 128 firmware DAC volts (uncommandable nets stay at 0)."""
        vnet = self.hw.volts(phases)
        v = np.zeros(self.num_dac)
        for h in range(len(vnet)):
            d = self.hw.dac_of[h]
            if d >= 0 and np.isfinite(vnet[h]):
                v[d] = vnet[h]
        return v

    def _require_lo(self, los):
        bad = [name for name in los if not self.hw.known[LO_PHASE_NET[name]]]
        if bad:
            nets = {name: LO_PHASE_NET[name] for name in bad}
            raise RuntimeError(
                f"LO phase heater(s) uncommandable: {nets}. The H23 characterisation "
                "scripts were removed (superseded by pic.compute.matvec, which needs no "
                "LO); recover them from git history to use this path.")

    def read_raw(self, phases):
        """Program ``phases`` and return the dark-subtracted 14 raw PD volts."""
        raw = np.asarray(self.pic.measure_raw(self.to_dac_volts(phases)), float) - self.dark
        if self.on_tick is not None:
            self.on_tick()
        return raw

    def acquire(self, phases, psi, los=("top", "bottom")):
        """One shot at LO phase ``psi`` (rad), added to the given reference arms' encode
        heaters. Returns ``(mon[6], homo[6], lo_pow[2])`` pulled out via the PD-role map.
        Mirrors ``theory.readout.acquire`` but reads real photodiodes."""
        self._require_lo(los)
        ph = np.asarray(phases, float).copy()
        for name in los:
            ph[LO_PHASE_NET[name]] = ph[LO_PHASE_NET[name]] + psi
        raw = self.read_raw(ph)
        mon = raw[list(self.pdmap.monitor)]
        homo = raw[list(self.pdmap.homodyne)]
        lo = np.array([raw[self.pdmap.lo_tap_top], raw[self.pdmap.lo_tap_bottom]])
        return mon, homo, lo

    def recover(self, phases, derotate=True, los=("top", "bottom")):
        """Complex output field per signal rail from two LO phases (psi = 0, pi/2), the
        hardware form of ``theory.readout.recover``. ``derotate`` applies the calibrated
        ``lo_phase``; without it the estimate is E * e^{-i arg(LO)} (an unknown per-arm
        phase). Rails whose LO arm is not in ``los`` are not meaningful."""
        t, f = self.t, self.f
        m0, h0, lo0 = self.acquire(phases, 0.0, los)
        _, h1, _ = self.acquire(phases, math.pi / 2, los)

        p_sig = np.clip(m0, 0, None) / t
        lo_pow = np.array([lo0[0] if LO_OF_RAIL[k] == "top" else lo0[1] for k in range(NSIG)])
        lo_pow = np.clip(lo_pow, 1e-18, None)

        C = (1 - f) * (1 - t) * p_sig + f * lo_pow
        A = 2 * math.sqrt(f * (1 - f) * (1 - t)) * np.sqrt(lo_pow)
        E_hat = ((h0 - C) + 1j * (h1 - C)) / A
        if derotate and self.lo_phase is not None:
            E_hat = E_hat * np.exp(1j * self.lo_phase)
        return E_hat


def dac_to_phase(hw, v, lo_bias=None):
    """Invert 128 DAC volts -> 120 net optical phases through ``hw`` (the inverse of
    ``Homodyne.to_dac_volts``; exact mod 2pi for commandable nets, ``phi0`` fallback for the
    rest -- the honest model of an uncharacterized heater sitting at its built-in offset).

    ``lo_bias`` (dict ``{"top": rad, "bottom": rad}``) adds a hidden phase to the LO-arm
    encode heaters that the recover side does NOT know -- a stand-in for the real chip's
    unmodeled reference-arm propagation, i.e. exactly the constant ``homodyne_calibrate``
    exists to measure. Leave ``None`` for the twin's own (bias-free) LO phase."""
    NH = len(hw.vpi)
    ph = np.zeros(NH)
    for h in range(NH):
        d = hw.dac_of[h]
        phi0 = hw.phi0[h] if np.isfinite(hw.phi0[h]) else 0.0
        if d >= 0 and hw.known[h] and np.isfinite(hw.vpi[h]):
            ph[h] = math.pi * (v[d] / hw.vpi[h]) ** 2 + phi0
        else:
            ph[h] = phi0
    if lo_bias:
        for name, net in LO_PHASE_NET.items():
            ph[net] = ph[net] + lo_bias.get(name, 0.0)
    return ph


def twin_pd_forward(twin, hw, pdmap=DEFAULT_PDMAP, dark=None, lo_bias=None):
    """A ``MockPIC`` forward ``f(v128) -> raw14`` backed by the differentiable twin, so the
    whole acquire/recover path can be exercised with no hardware. Inverts DAC volts -> net
    phases (``dac_to_phase``, same calibration ``Homodyne`` uses), runs ``twin.forward``, and
    packs the twin's mon/homo/lo_power into raw PD slots via the SAME ``pdmap``. ``lo_bias``
    injects a hidden LO-arm phase (see ``dac_to_phase``). Torch imported lazily so the
    hardware path stays torch-free."""
    import torch  # noqa: local, keeps the hardware path torch-free

    base = np.zeros(NPD) if dark is None else np.asarray(dark, float)

    def forward(v):
        ph = dac_to_phase(hw, np.asarray(v, float), lo_bias)
        out = twin.forward(torch.as_tensor(ph, dtype=torch.float32))
        mon = out["mon"].detach().numpy()
        homo = out["homo"].detach().numpy()
        lo = out["lo_power"].detach().numpy()
        raw = base.copy()
        raw[list(pdmap.monitor)] = mon
        raw[list(pdmap.homodyne)] = homo
        raw[pdmap.lo_tap_top] = lo[0]
        raw[pdmap.lo_tap_bottom] = lo[1]
        return raw

    return forward


# a fixed nonzero hidden LO-arm phase for --mock, so recover WITHOUT calibration is wrong
# and homodyne_calibrate has a real constant to find (mirrors the real chip's unknown).
MOCK_LO_BIAS = {"top": 0.9, "bottom": -1.3}


def build_mock(pdmap=DEFAULT_PDMAP, noise=1e-4, t=0.1, f=0.5, lo_patch=None,
               lo_bias=MOCK_LO_BIAS, config="pic_data/pic_b_config.json"):
    """Full no-hardware stack: (MockPIC over the twin forward, Hardware, Twin). ``lo_patch``
    (net -> (Vpi, phi0)) makes extra LO heaters commandable -- e.g. a synthetic net-23 so the
    bottom arm is exercisable before the real H23 characterization exists. ``lo_bias`` is the
    hidden per-arm LO phase the calibration must recover (default ``MOCK_LO_BIAS``)."""
    from theory.hw import Hardware
    from theory.twin import Twin
    from . import MockPIC

    hw = Hardware(config)
    if lo_patch:
        patch_lo(hw, lo_patch)
    twin = Twin(tap=t, lo_frac=f)
    pic = MockPIC(twin_pd_forward(twin, hw, pdmap, lo_bias=lo_bias), noise=noise).open()
    return pic, hw, twin
