"""Analysis of a one-at-a-time heater sweep (produced by ``template.py --sweep``).

Turns one sweep CSV into the things we need to know the chip's reachable map:

  * which heaters are dead (a channel that moves no PD) and which PDs are dead (a PD no
    channel moves) -- the reduced input/output support;
  * the influence map (peak-to-peak of each PD over each channel's sweep) -- coarse routing;
  * per-channel fringe fits (Vpi, phi0, visibility) reusing :func:`characterize.fit_fringe`,
    plus the fringe extrema (max/min and the voltage at max) that renormalisation needs;
  * a marginal renormalisation range per PD.

Everything is importable and composable -- ``load_sweep_csv`` gives you the arrays, the
rest are pure functions over them. Influence here is peak-to-peak, NOT the ``influence.py``
eta^2 (that one is for the *randomised* dataset; on one-hot sweep data a channel's "level 0"
group would wrongly pool every other channel's rows).

The max/min found here is *marginal* (all other heaters at 0). A PD on an off-path can look
dead until light is routed to it -- confirm with the ``acquisition.maximize_pd`` peak-up.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .pic.config import NUM_DAC, NUM_ADC_RAW, DAMAGED_PDS, LIVE_PDS, FIRMWARE_VMAX
from .characterize import fit_fringe


@dataclass
class Sweep:
    channels: list[int]  # channels actually swept
    V: dict  # channel -> level array (with 0 injected from the baseline)
    Y: dict  # channel -> raw-PD array [n_levels, NUM_ADC_RAW]
    base: np.ndarray  # laser-on all-zero-input baseline [NUM_ADC_RAW]
    dark: np.ndarray  # laser-off floor mean [NUM_ADC_RAW]
    dark_std: np.ndarray  # laser-off floor std  [NUM_ADC_RAW]
    n_signal: int = 0
    meta: dict = field(default_factory=dict)


def load_sweep_csv(path: str, num_dac: int = NUM_DAC, num_adc: int = NUM_ADC_RAW) -> Sweep:
    """Reconstruct per-channel sweeps from a ``run_experiment`` CSV. The active channel and
    its level are read straight from each one-hot ``input`` field; the shared all-zero row
    becomes every channel's v=0 point. Repeats at the same (channel, level) are averaged."""
    import csv

    dark, sig_base = [], []
    cell: dict = {}  # (channel, level) -> list of pd arrays
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "elapsed_s":
                continue
            phase = row[1]
            vec = np.array([float(x) for x in row[7].split()], float)
            pds = np.array([float(x) for x in row[8 : 8 + num_adc]], float)
            if phase == "dark":
                dark.append(pds)
                continue
            if phase != "signal":
                continue
            nz = np.flatnonzero(vec)
            if nz.size == 0:
                sig_base.append(pds)
            elif nz.size == 1:
                cell.setdefault((int(nz[0]), float(vec[nz[0]])), []).append(pds)

    dark = np.asarray(dark) if dark else np.zeros((1, num_adc))
    base = np.mean(sig_base, 0) if sig_base else dark.mean(0)
    channels = sorted({c for c, _ in cell})
    V, Y = {}, {}
    for c in channels:
        levels = sorted({lv for cc, lv in cell if cc == c})
        vs = [0.0] + levels
        ys = [base] + [np.mean(cell[(c, lv)], 0) for lv in levels]
        V[c] = np.asarray(vs, float)
        Y[c] = np.asarray(ys, float)
    return Sweep(
        channels=channels,
        V=V,
        Y=Y,
        base=base,
        dark=dark.mean(0),
        dark_std=dark.std(0),
        n_signal=sum(len(v) for v in cell.values()) + len(sig_base),
    )


def influence_ptp(sw: Sweep, num_adc: int = NUM_ADC_RAW) -> np.ndarray:
    """``E[channel, pd]`` = peak-to-peak of that PD over that channel's sweep. Rows for
    channels not swept are 0. This is the coarse routing map for one-hot data."""
    E = np.zeros((NUM_DAC, num_adc))
    for c in sw.channels:
        E[c] = sw.Y[c].max(0) - sw.Y[c].min(0)
    return E


def dead_elements(sw: Sweep, E: np.ndarray | None = None, ch_thresh: float = 0.01,
                  pd_k: float = 5.0, pd_floor: float = 0.01) -> dict:
    """Classify dead heaters (move no PD) and dead PDs (moved by no channel).

    ``ch_thresh`` [V]: a channel whose largest PD swing is below this is a dead heater.
    A PD is dead if no swept channel moves it past ``max(pd_floor, pd_k * dark_std)``.
    """
    E = influence_ptp(sw) if E is None else E
    dead_h = [c for c in sw.channels if E[c].max() < ch_thresh]
    live_h = [c for c in sw.channels if c not in dead_h]
    pd_thresh = np.maximum(pd_floor, pd_k * sw.dark_std)
    per_pd_max = E.max(0)
    dead_pd = [k for k in range(len(per_pd_max)) if per_pd_max[k] < pd_thresh[k]]
    live_pd = [k for k in range(len(per_pd_max)) if k not in dead_pd]
    return {
        "dead_heaters": dead_h,
        "live_heaters": live_h,
        "dead_pds": dead_pd,
        "live_pds": live_pd,
        "pd_response": per_pd_max,
        "pd_thresh": pd_thresh,
    }


def fringe_extrema(fit: dict, vmax: float = FIRMWARE_VMAX) -> dict:
    """From a :func:`characterize.fit_fringe` result, the fringe max/min and the smallest
    non-negative voltage reaching each. ``*_reachable`` flags whether that voltage is within
    the [0, vmax] the firmware allows (the fit still gives the value if it isn't)."""
    A, B, phi2, phi0 = fit["A"], fit["B"], fit["phi2"], fit["phi0"]

    def v_for(angle):  # smallest v>=0 with phi2*v^2 + phi0 == angle (mod 2pi)
        best = None
        for n in range(-4, 6):
            v2 = (angle + 2 * np.pi * n - phi0) / phi2 if phi2 else -1
            if v2 >= 0:
                v = float(np.sqrt(v2))
                if best is None or v < best:
                    best = v
        return best

    v_max = v_for(0.0)  # cos = +1
    v_min = v_for(np.pi)  # cos = -1
    return {
        "pmax": A + abs(B),
        "pmin": A - abs(B),
        "v_at_max": v_max,
        "v_at_min": v_min,
        "max_reachable": v_max is not None and v_max <= vmax,
        "min_reachable": v_min is not None and v_min <= vmax,
    }


def fit_all_fringes(sw: Sweep, vpi0: float = 1.5, live_only: bool = True) -> dict:
    """Fit each swept channel's fringe on its most-responsive live PD, and attach the
    extrema. Returns ``{channel: fit_dict}`` (fit_dict is None where the fit failed)."""
    live = set(LIVE_PDS) if live_only else set(range(sw.base.size))
    out = {}
    for c in sw.channels:
        Y = sw.Y[c]
        ptp = Y.max(0) - Y.min(0)
        cand = [k for k in np.argsort(-ptp) if k in live]
        k = cand[0] if cand else int(np.argmax(ptp))
        fit = fit_fringe(sw.V[c], Y[:, k], vpi0=vpi0)
        if fit is not None:
            fit["pd"] = int(k)
            fit.update(fringe_extrema(fit))
        out[c] = fit
    return out


def renorm_from_sweep(sw: Sweep, num_adc: int = NUM_ADC_RAW) -> np.ndarray:
    """Marginal renormalisation range per PD: ``[:, 0]`` = min, ``[:, 1]`` = max seen across
    the whole sweep (baseline included). Global full-scale needs ``maximize_pd`` per PD."""
    lo = np.full(num_adc, np.inf)
    hi = np.full(num_adc, -np.inf)
    for c in sw.channels:
        lo = np.minimum(lo, sw.Y[c].min(0))
        hi = np.maximum(hi, sw.Y[c].max(0))
    lo = np.minimum(lo, sw.base)
    hi = np.maximum(hi, sw.base)
    return np.stack([lo, hi], axis=1)


def report(sw: Sweep, vpi0: float = 1.5) -> str:
    """Terse scannable digest: floors, dead elements, top influence, fringe summary."""
    E = influence_ptp(sw)
    de = dead_elements(sw, E)
    fits = fit_all_fringes(sw, vpi0=vpi0)
    rn = renorm_from_sweep(sw)
    live = list(LIVE_PDS)
    L = []
    L.append(f"sweep: {len(sw.channels)} channels, {sw.n_signal} signal reads")
    L.append("  dark floor (live) : " + " ".join(f"{sw.dark[k]:.3f}" for k in live))
    L.append("  laser-on base(live): " + " ".join(f"{sw.base[k]:.3f}" for k in live))
    L.append("")
    known = " (matches shipped 0,2,7,11)" if set(de["dead_pds"]) == set(DAMAGED_PDS) else ""
    L.append(f"dead PDs   : {de['dead_pds']}{known}")
    L.append(f"dead heaters: {de['dead_heaters']}")
    L.append("")
    L.append("top influence per live PD  (channel: peak-to-peak V)")
    for k in live:
        order = np.argsort(-E[:, k])
        top = ", ".join(f"{int(c)}:{E[c, k]:.3f}" for c in order[:3] if E[c, k] > 1e-4)
        L.append(f"  pd{k:<2}: {top or '-- flat --'}")
    L.append("")
    L.append("per-channel fringe (best live PD)")
    L.append(f"  {'ch':>3} {'pd':>3} {'Vpi':>5} {'vis':>5} {'v@max':>6} {'rmse':>6}")
    for c in sw.channels:
        f = fits[c]
        if f is None:
            L.append(f"  {c:>3}  -- fit failed --")
            continue
        vam = f"{f['v_at_max']:.2f}" if f["v_at_max"] is not None else " -- "
        L.append(
            f"  {c:>3} {f['pd']:>3} {f['Vpi']:>5.2f} {f['visibility']:>5.2f} "
            f"{vam:>6} {f['rmse']:>6.4f}"
        )
    L.append("")
    L.append("marginal renorm range per live PD  [min, max] V")
    for k in live:
        L.append(f"  pd{k:<2}: [{rn[k, 0]:.3f}, {rn[k, 1]:.3f}]")
    return "\n".join(L)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="analyse a template.py --sweep CSV")
    ap.add_argument("csv", help="sweep CSV from template.py --sweep")
    ap.add_argument("--vpi0", type=float, default=1.5, help="Vpi guess for the fringe fit")
    a = ap.parse_args(argv)
    print(report(load_sweep_csv(a.csv), vpi0=a.vpi0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
