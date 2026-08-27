"""Differentiable digital twin of PIC B, built to match `src.pic.layout.build_scene()`
element for element — the same structure the schematic draws.

    input -> 1:8 splitter tree -> encode (amp MZI + phase) -> V mesh -> Sigma -> U mesh
          -> readout (monitor tap PD + LO-combined homodyne PD)

Rails: 8 total. Rail 0 and rail 7 are the straight reference arms (homodyne LO, no
interaction); rails 1..6 are the signal rails that pass through V/Sigma/U.

Heater numbering H0..H119 follows layout.py's draw order exactly, so a phase vector here
indexes the same heaters as `pic_data/pic_b_config.json`. Torch throughout, complex64,
autograd-friendly: `phases` is the only tensor you optimise.

The mesh is NOT unitary — odd columns dump one edge output to a cut-off (layout's
`_edge_drop`), so each mesh is a contraction. That is the whole point; don't "fix" it.
"""

from __future__ import annotations

import torch

NRAIL = 8
NSIG = 6
NH = 120
SIG = list(range(1, 7))  # rail indices carrying signal
REF_TOP, REF_BOT = 0, 7

# heater id blocks, matching layout.HEATER_STAGES
ENC_SPLIT = range(0, 16)
ENC_PHASE = range(16, 24)
V_MESH = range(24, 66)
SIGMA = range(66, 78)
U_MESH = range(78, 120)


def _mesh_slots(base):
    """Heater ids for one 6-column mesh, in layout draw order.

    Even columns: three 2x2 MZIs on rail pairs (0,1) (2,3) (4,5).
    Odd columns: edge-drop on rail 0, 2x2 MZIs on (1,2) (3,4), edge-drop on rail 5.
    Returns [(kind, rails, (h_a, h_b)), ...] with kind in {"mzi", "drop"}.
    """
    slots, h = [], base
    for col in range(6):
        if col % 2 == 0:
            for a, b in ((0, 1), (2, 3), (4, 5)):
                slots.append(("mzi", (a, b), (h, h + 1)))
                h += 2
        else:
            slots.append(("drop", (0,), (h, h + 1)))
            h += 2
            for a, b in ((1, 2), (3, 4)):
                slots.append(("mzi", (a, b), (h, h + 1)))
                h += 2
            slots.append(("drop", (5,), (h, h + 1)))
            h += 2
    return slots


V_SLOTS = _mesh_slots(24)
U_SLOTS = _mesh_slots(78)
assert V_SLOTS[-1][2][1] == 65 and U_SLOTS[-1][2][1] == 119


def mzi_2x2(t1, t2):
    """2x2 MZI transfer (coupler-phase-coupler), matching src.mzi.mzi_closed:
    i * e^{i s} [[sin d, cos d], [cos d, -sin d]], s=(t1+t2)/2, d=(t1-t2)/2."""
    s, d = (t1 + t2) / 2, (t1 - t2) / 2
    pre = 1j * torch.exp(1j * s.to(torch.complex64))
    sd, cd = torch.sin(d).to(torch.complex64), torch.cos(d).to(torch.complex64)
    return pre * sd, pre * cd  # (bar, cross)


def mzi_1x1(t1, t2):
    """1-in/1-out MZI (layout `_mzi1`): split, phase each arm, recombine.
    t = e^{i(t1+t2)/2} cos((t1-t2)/2) -> independent amplitude and phase."""
    s, d = (t1 + t2) / 2, (t1 - t2) / 2
    return torch.exp(1j * s.to(torch.complex64)) * torch.cos(d).to(torch.complex64)


def drop(t1, t2):
    """Edge MZI whose second output is dumped into a cut-off. Modelled as a 2x2 MZI fed
    on one port with the cross port taken; the bar port's light leaves the chip.

    NOTE: layout.py draws a single input coupler, which would make t2 optically dead.
    A 2x2 with one port terminated is the physically sensible reading and keeps both
    heaters live. Flip with `Twin(drop_live=False)` to test the other interpretation.
    """
    _, cross = mzi_2x2(t1, t2)
    return cross


class Twin:
    """The chip as a differentiable function of 120 heater phases.

    forward(phases, x_ext=None) -> dict with the complex rail fields and both PD banks.
    Phases are *optical phases* in radians. Voltage enters via `phases_from_volts`.
    """

    def __init__(self, tap: float = 0.1, lo_frac: float = 0.5, drop_live: bool = True,
                 loss_db_per_stage: float = 0.0, dtype=torch.complex64):
        self.tap = tap                # power fraction split to each monitor PD
        self.lo_frac = lo_frac        # LO power fraction at the homodyne combiner
        self.drop_live = drop_live
        self.stage_amp = 10 ** (-loss_db_per_stage / 20)
        self.dtype = dtype

    def _apply_mesh(self, rails, slots, ph, P=None):
        """Functional (no in-place writes, autograd-safe): `rails` is a list of per-rail
        tensors. `P(h)` fetches heater h's phase, shaped for broadcasting over any
        leading batch axis; defaults to plain indexing."""
        P = P or (lambda h: ph[h])
        rails = list(rails)
        for kind, rr, (ha, hb) in slots:
            t1, t2 = P(ha), P(hb)
            if kind == "mzi":
                a, b = rr
                bar, cross = mzi_2x2(t1, t2)
                ea, eb = rails[a], rails[b]
                rails[a] = bar * ea + cross * eb
                rails[b] = cross * ea - bar * eb
            else:
                r = rr[0]
                g = (drop(t1, t2) if self.drop_live
                     else torch.exp(1j * t1.to(self.dtype)) / 2 ** 0.5)
                rails[r] = g * rails[r]
            rails = [e * self.stage_amp for e in rails]
        return rails

    def forward(self, phases, x_ext=None):
        ph = phases if torch.is_tensor(phases) else torch.as_tensor(phases, dtype=torch.float32)
        amp0 = torch.as_tensor(1.0 / NRAIL ** 0.5, dtype=self.dtype)

        # encode: per-rail amplitude MZI then a dedicated phase shifter
        E = [amp0 * mzi_1x1(ph[2 * i], ph[2 * i + 1]) * torch.exp(1j * ph[16 + i].to(self.dtype))
             for i in range(NRAIL)]
        if x_ext is not None:  # override the encoded signal vector directly
            xt = x_ext if torch.is_tensor(x_ext) else torch.as_tensor(x_ext, dtype=self.dtype)
            for k, r in enumerate(SIG):
                E[r] = xt[k].to(self.dtype)

        rails = self._apply_mesh([E[r] for r in SIG], V_SLOTS, ph)
        rails = [rails[k] * mzi_1x1(ph[66 + 2 * k], ph[67 + 2 * k]) for k in range(NSIG)]
        rails = self._apply_mesh(rails, U_SLOTS, ph)
        Es = torch.stack(rails)

        lo_t, lo_b = E[REF_TOP], E[REF_BOT]
        lo = torch.stack([lo_t if k < 3 else lo_b for k in range(NSIG)])

        mon = self.tap * Es.abs() ** 2
        sig = (1 - self.tap) ** 0.5 * Es
        homo = ((1 - self.lo_frac) ** 0.5 * sig + self.lo_frac ** 0.5 * lo).abs() ** 2
        return {"field": Es, "lo": lo, "mon": mon, "homo": homo,
                "lo_power": torch.stack([lo_t.abs() ** 2, lo_b.abs() ** 2])}

    def matrix(self, phases):
        """The realised 6x6 complex map from encoded signal field to output field
        (mesh only — V, Sigma, U — with the encode stage bypassed).

        All six basis vectors propagate in one pass: each rail carries a length-6 batch,
        so rails[k][j] ends up as M[k, j].

        `phases` may be (120,) -> returns (6, 6), or (R, 120) -> returns (R, 6, 6).
        The batch axis runs R independent parameter sets through the same ~100 python-
        level ops, so R optimiser restarts cost barely more than one. That is the only
        reason inverse design is affordable here.
        """
        ph = phases if torch.is_tensor(phases) else torch.as_tensor(
            phases, dtype=torch.float32)
        batched = ph.ndim == 2
        idx = (slice(None),) if batched else ()

        def P(h):
            c = ph[..., h]
            return c.unsqueeze(-1) if batched else c

        eye = torch.eye(NSIG, dtype=self.dtype)
        rails = [eye[k].expand(ph.shape[0], NSIG) if batched else eye[k]
                 for k in range(NSIG)]
        rails = self._apply_mesh(rails, V_SLOTS, ph, P)
        rails = [rails[k] * mzi_1x1(P(66 + 2 * k), P(67 + 2 * k)) for k in range(NSIG)]
        rails = self._apply_mesh(rails, U_SLOTS, ph, P)
        return torch.stack(rails, dim=-2) if batched else torch.stack(rails)


def phases_from_volts(v, vpi, phi0):
    """Heater volts -> optical phase, the repo's law: phi = pi*(v/Vpi)^2 + phi0."""
    v = torch.as_tensor(v, dtype=torch.float32)
    return torch.pi * (v / torch.as_tensor(vpi, dtype=torch.float32)) ** 2 + \
        torch.as_tensor(phi0, dtype=torch.float32)


def volts_from_phases(ph, vpi, phi0, vmax=4.0):
    """Inverse of `phases_from_volts`, wrapped into the reachable 0..vmax window.
    Returns (volts, ok) where ok flags phases a heater cannot reach."""
    ph = torch.as_tensor(ph, dtype=torch.float32)
    vpi = torch.as_tensor(vpi, dtype=torch.float32)
    k = torch.floor((torch.pi * (vmax / vpi) ** 2 + torch.as_tensor(phi0) - ph)
                    / (2 * torch.pi))
    tgt = ph + 2 * torch.pi * torch.clamp(k, min=0) - torch.as_tensor(phi0)
    u = tgt / torch.pi * vpi ** 2
    ok = (u >= 0) & (u <= vmax ** 2)
    return torch.sqrt(torch.clamp(u, min=0.0)), ok
