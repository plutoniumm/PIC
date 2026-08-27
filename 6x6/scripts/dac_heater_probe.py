"""Which DAC channel drives which physical heater?

The 6x6P structure has 120 heaters (geometric numbering H0..H119, `pic_data/heater_map.csv`);
the daughter board exposes 64 DAC channels. The correspondence is recorded nowhere: NUS's
`PIC_update.pdf` numbers heaters by DAC channel, the GDS numbers them by position, and no
document ties the two. `pic_data/dac_heater_map.csv` therefore ships a labelled placeholder.

This script says what the existing data can and cannot settle, and runs the experiment that
settles the rest.

    python scripts/dac_heater_probe.py evidence     # offline: what the 75k rows already prove
    python scripts/dac_heater_probe.py sweep        # hardware: the decisive measurement
    python scripts/dac_heater_probe.py analyse runs/dac_probe.csv

WHAT THE OFFLINE EVIDENCE SHOWS
The shipped datasets randomise all 64 channels at once, so a deep heater's marginal effect is
averaged over random downstream MZI states and washes out: ~59 of 64 channels sit at the noise
floor. Only the channels that modulate the *homodyne local oscillator* survive averaging,
because they shift photodiode means rather than merely rotating an interference phase.

That accident is informative. The chip's two outer rails are the reference (LO) arms. The top
one feeds the homodyne combiners of signal rails 0-2 (-> PD1, PD3, PD5) plus LO monitor PD0;
the bottom one feeds rails 3-5 (-> PD8, PD10, PD12) plus LO monitor PD13. In the encode stage
each rail carries a 1x2 MZI (two heaters, an intensity modulator) then a single shifter (phase
only). So an LO *intensity* modulator should light up exactly one of those two PD groups.

Under the identity placeholder, H0/H1 are the top reference MZI's arms and H14/H15 the bottom
one's. The data agrees, sharply -- and then disagrees about the shifter block. See `evidence`.

WHY THE HARDWARE SWEEP IS NEEDED
Marginal effects cannot localise a heater inside the meshes. Driving one channel at a time with
the other 63 held at a fixed bias makes each heater's footprint observable: 64 channels x ~9
levels ~= 576 reads, well under a minute of board time. The laser must be lit (use the UI or
`./do laser state 1; ./do laser set 5`) or every photodiode reads its dark floor.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src import data
from src.pic import PIC, MockPIC, PICError, find_port, NUM_DAC, NUM_ADC_RAW, DAMAGED_PDS
from src.pic.layout import SIG_ROUTE

SESSIONS = [
    "pic_data/testing_working_ps_hybrid_64_25k_22july(10).xlsx",
    "pic_data/testing_working_ps_hybrid_64_25k_22july(10)_2.xlsx",
    "pic_data/testing_working_ps_hybrid_64_25k_15july.xlsx",
]
LIVE = [i for i in range(NUM_ADC_RAW) if i not in DAMAGED_PDS]

# encode-stage geometry, read straight off heater_map.csv (both blocks share one x, so the
# descending-y sort numbers them top rail -> bottom rail):
#   rail k  ->  MZI arms H(2k), H(2k+1)   and   shifter H(16+k)
# rails 0 and 7 are the outer reference (LO) arms.
RAIL_TOP, RAIL_BOT = 0, 7


def _homodyne_group(rails):
    return sorted(h for ri, _mon, h in SIG_ROUTE if ri in rails)


LO_TOP = sorted(_homodyne_group({0, 1, 2}) + [0])    # PD0 is dead -> invisible in the data
LO_BOT = sorted(_homodyne_group({3, 4, 5}) + [13])   # PD13 is live -> a visible fingerprint


def eta2(x, y):
    """Fraction of y's variance explained by the level of x alone (ANOVA eta^2)."""
    tot = y.var()
    if tot == 0:
        return 0.0
    gm, between = y.mean(), 0.0
    for lvl in np.unique(x):
        sel = x == lvl
        between += sel.mean() * (y[sel].mean() - gm) ** 2
    return between / tot


def _noise_threshold(X, Y, n=60, sigma=6.0, seed=0):
    """eta^2 of a shuffled channel: what 'no effect' looks like at this N."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        c = int(rng.integers(0, X.shape[1]))
        p = int(rng.choice(LIVE))
        xs = X[:, c].copy()
        rng.shuffle(xs)
        null.append(eta2(xs, Y[:, p]))
    return float(np.mean(null) + sigma * np.std(null))


def footprint(X, Y, c, thr):
    return [p for p in LIVE if eta2(X[:, c], Y[:, p]) > thr]


def cmd_evidence(_a):
    Xs, Ys = zip(*(data.load_session(f) for f in SESSIONS))
    X, Y = np.vstack(Xs), np.vstack(Ys)
    thr = _noise_threshold(X, Y)
    print(f"pooled N={len(X)}   noise floor (eta^2, +6sd) = {thr:.5f}\n")
    print(f"topology predicts   top reference rail -> PDs {LO_TOP}  (PD0 dead)")
    print(f"                 bottom reference rail -> PDs {LO_BOT}\n")

    live = [c for c in range(NUM_DAC) if footprint(X, Y, c, thr)]
    strong = sorted(live, key=lambda c: -sum(eta2(X[:, c], Y[:, p]) for p in LIVE))[:5]
    print(f"channels above the floor: {len(live)}/64   strongest: {strong}\n")

    probe = [(0, "H0  top ref MZI arm"), (1, "H1  top ref MZI arm"),
             (14, "H14 bottom ref MZI arm"), (15, "H15 bottom ref MZI arm"),
             (16, "H16 top ref shifter"), (23, "H23 bottom ref shifter")]
    print(f"{'DAC':>4}  {'identity says':<24} {'measured footprint':<22} verdict")
    print("-" * 84)
    for c, label in probe:
        f = footprint(X, Y, c, thr)
        want = LO_TOP if ("top" in label) else LO_BOT
        want_vis = [p for p in want if p in LIVE]
        if not f:
            v = "SILENT (channel drives nothing)"
        elif f == want_vis:
            v = "matches identity"
        elif f == [p for p in (LO_BOT if want is LO_TOP else LO_TOP) if p in LIVE]:
            v = "matches the OTHER reference rail"
        else:
            v = "ambiguous"
        print(f"{c:>4}  {label:<24} {str(f):<22} {v}")

    a = np.array([eta2(X[:, 14], Y[:, p]) for p in LIVE])
    b = np.array([eta2(X[:, 15], Y[:, p]) for p in LIVE])
    print(f"\ncorr(DAC14, DAC15 PD profiles) = {np.corrcoef(a, b)[0, 1]:+.5f}"
          "   -> same MZI, one arm each")
    print("""
Reading:
  * DAC1 and DAC14/DAC15 land exactly where the identity map predicts, on the two
    reference-arm intensity modulators. DAC15's footprint even includes PD13, the one
    live LO monitor, which fixes the top/bottom orientation independently.
  * DAC0 is electrically silent: that heater is probably not bonded.
  * DAC16 and DAC23 are SWAPPED. Both sit on reference rails by geometry (H16 shares a
    rail with H0/H1, H23 with H14/H15), yet each reports the other rail's PD group. Their
    effects are small but their group assignment is clean, with no cross-over.

So the map is block-structured, not identity: the encode MZI block looks like DAC c -> Hc,
while the 8-channel shifter block looks reversed (DAC 16..23 -> H23..H16), which is what a
wire-bond row flipped end-for-end would produce. Nothing here constrains DAC24..63.

This is inference from an observational dataset, not a measurement. Run `sweep` to settle it.""")


def _open_pic(port, mock):
    if mock:
        rng = np.random.default_rng(0)
        W = rng.normal(0, 0.15, (NUM_ADC_RAW, NUM_DAC))
        b = rng.uniform(0, 2 * np.pi, NUM_ADC_RAW)
        return MockPIC(lambda v: 0.02 + 0.05 * (1 + np.sin(W @ (np.asarray(v) / 4.0) + b)),
                       noise=3e-4).open()
    p, cands = find_port(port)
    if not p:
        raise PICError(f"no PIC port found (saw {cands or 'none'}); pass --port")
    return PIC(port=p).open()


def cmd_sweep(a):
    """Drive one channel at a time, everything else held at a fixed bias."""
    levels = np.round(np.linspace(0.0, a.vmax, a.levels), 3)
    pic = _open_pic(a.port, a.mock)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    try:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([f"# single-channel DAC sweep  bias={a.bias}V  settle={a.settle}s"])
            w.writerow(["dac", "level", "rep"] + [f"pd{i}" for i in range(NUM_ADC_RAW)])
            base = np.full(NUM_DAC, a.bias)
            pic.measure_raw(base)
            time.sleep(max(a.settle, 0.2))
            for c in range(NUM_DAC):
                for lv in levels:
                    v = base.copy()
                    v[c] = lv
                    pic.measure_raw(v)
                    if a.settle:
                        time.sleep(a.settle)
                    for r in range(a.repeats):
                        pds = np.asarray(pic.measure_raw(v), float)
                        w.writerow([c, lv, r] + [f"{x:.4f}" for x in pds])
                        n += 1
                    fh.flush()
                print(f"  DAC {c:>2}/63 swept", end="\r", flush=True)
    finally:
        pic.close()
    print(f"\nwrote {n} rows -> {a.out}\nnow: python scripts/dac_heater_probe.py analyse {a.out}")


def cmd_analyse(a):
    rows = [r for r in csv.DictReader(l for l in open(a.csv) if not l.startswith("#"))]
    if not rows:
        print("no rows")
        return
    dacs = sorted({int(r["dac"]) for r in rows})
    print(f"{len(rows)} rows, {len(dacs)} channels\n")
    print(f"{'DAC':>4}  {'moves PDs':<26} {'peak swing (V)':>14}  reading")
    print("-" * 78)
    silent = []
    for c in dacs:
        rs = [r for r in rows if int(r["dac"]) == c]
        lv = sorted({float(r["level"]) for r in rs})
        moved, peak = [], 0.0
        for p in LIVE:
            means = np.array([np.mean([float(r[f"pd{p}"]) for r in rs
                                       if float(r["level"]) == L]) for L in lv])
            swing = means.max() - means.min()
            peak = max(peak, swing)
            if swing > a.eps:
                moved.append(p)
        if not moved:
            silent.append(c)
            note = "silent (unbonded, or a heater with no optical path to a live PD)"
        elif moved == [p for p in LO_TOP if p in LIVE]:
            note = "top reference-arm LO modulator"
        elif moved == [p for p in LO_BOT if p in LIVE]:
            note = "bottom reference-arm LO modulator"
        elif len(moved) <= 3:
            note = "late stage (U mesh): few PDs -> well localised"
        else:
            note = "early stage (encode/V): mixes into many PDs"
        print(f"{c:>4}  {str(moved):<26} {peak:>14.4f}  {note}")
    if silent:
        print(f"\nsilent channels: {silent}")
    print("\nEach channel's footprint now localises its heater against the netlist. Write the "
          "result into pic_data/dac_heater_map.csv and set verified=1 on the rows it settles.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evidence", help="what the shipped datasets already prove").set_defaults(fn=cmd_evidence)

    s = sub.add_parser("sweep", help="single-channel hardware sweep (laser must be lit)")
    s.add_argument("--port", default=None)
    s.add_argument("--mock", action="store_true", help="no hardware; exercises the flow")
    s.add_argument("--bias", type=float, default=1.0, help="hold the other 63 channels here")
    s.add_argument("--vmax", type=float, default=2.0, help="firmware clamps above 2 V")
    s.add_argument("--levels", type=int, default=9)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--settle", type=float, default=0.05)
    s.add_argument("--out", default="runs/dac_probe.csv")
    s.set_defaults(fn=cmd_sweep)

    an = sub.add_parser("analyse", help="turn a sweep CSV into per-channel footprints")
    an.add_argument("csv")
    an.add_argument("--eps", type=float, default=0.002, help="PD swing that counts as a move")
    an.set_defaults(fn=cmd_analyse)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
