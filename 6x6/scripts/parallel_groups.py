"""Parallel-characterization scheduler: partition heaters into lockstep-sweep rounds with no
cross-talk confounding, via resource-constrained list-coloring (interference-graph scheduling).

Naive "read each heater on its dominant PD, then colour" caps at ~2x here because one bright PD
(pd10) dominates for ~26 heaters -> a big same-PD clique. The fix (this module): make the
READ-PD a decision variable -- each heater has a ranked set of PDs it can be cleanly read on,
and each round assigns a distinct, uncorrupted read-PD per member. That breaks the clique and
lifts the speedup to ~4-4.5x. Multiplexing (Hadamard/group-testing) does NOT apply: the fringe
is nonlinear and two co-driven heaters interfere on a shared PD, so their fringes can't be
linearly separated -- spatial demux (one heater <-> one clean PD) + scheduling is the way.

    python scripts/parallel_groups.py                       # schedule with default thresholds
    python scripts/parallel_groups.py --swing-read 10 --r2 0.6 --foot 20

Reads the heater x PD swing/r2/Vpi matrix (pic_data/census/fringe_fits_b128.csv), writes the
round schedule to pic_data/parallel_schedule.json for the hardware runner.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict

NPD = 14


def load_matrix(path):
    """(ch, pd) -> {swing, r2, vpi}. Missing pairs default to zero swing."""
    swing = defaultdict(lambda: defaultdict(float))
    r2 = defaultdict(lambda: defaultdict(float))
    vpi = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(path)):
        ch, pd = int(r["ch"]), int(r["pd"])
        swing[ch][pd] = float(r["swing_mV"])
        r2[ch][pd] = float(r.get("r2", 0) or 0)
        vpi[ch][pd] = abs(float(r.get("Vpi", 0) or 0))
    return swing, r2, vpi


def read_set(ch, swing, r2, vpi, swing_read, r2_min):
    """PDs where ch has a strong, clean, in-range fringe -- ranked best-first."""
    ok = [pd for pd in range(NPD)
          if swing[ch][pd] >= swing_read and r2[ch][pd] >= r2_min and 0.8 <= vpi[ch][pd] <= 6.0]
    return sorted(ok, key=lambda pd: -swing[ch][pd])


def footprint(ch, swing, foot_thr):
    """PDs ch corrupts when co-swept (conservative, low threshold)."""
    return {pd for pd in range(NPD) if swing[ch][pd] >= foot_thr}


def schedule(heaters, swing, r2, vpi, swing_read, r2_min, foot_thr):
    R = {c: read_set(c, swing, r2, vpi, swing_read, r2_min) for c in heaters}
    F = {c: footprint(c, swing, foot_thr) for c in heaters}
    weak = [c for c in heaters if not R[c]]
    placeable = [c for c in heaters if R[c]]
    # most-constrained first: fewest read options, then largest footprint (hardest to place)
    order = sorted(placeable, key=lambda c: (len(R[c]), -len(F[c])))

    rounds = []  # each: {"members": [(ch, read_pd), ...], "used": set(read_pds)}
    for c in order:
        placed = False
        for rd in rounds:
            for p in R[c]:
                if p in rd["used"]:
                    continue  # read-PD already taken this round
                # c's read p must be clean of every member's leakage, and c must not
                # leak onto any member's read-PD
                if all(p not in F[m] and pm not in F[c] for m, pm in rd["members"]):
                    rd["members"].append((c, p))
                    rd["used"].add(p)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rounds.append({"members": [(c, R[c][0])], "used": {R[c][0]}})
    return rounds, weak, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="pic_data/census/fringe_fits_b128.csv")
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--only-knobs", action="store_true",
                    help="restrict to analytic ok/weak heaters (skip dead/spare)")
    ap.add_argument("--swing-read", type=float, default=10.0, help="min swing to READ a heater on a PD [mV]")
    ap.add_argument("--r2", type=float, default=0.6, help="min fringe r2 to read")
    ap.add_argument("--foot", type=float, default=20.0, help="footprint (corruptor) swing threshold [mV]")
    ap.add_argument("--out", default="pic_data/parallel_schedule.json")
    a = ap.parse_args()

    swing, r2, vpi = load_matrix(a.fits)
    heaters = sorted(swing)
    if a.only_knobs:
        cfg = json.load(open(a.config))
        knobs = {c["dac"] for c in cfg["channels"] if c.get("analytic") in ("ok", "weak")}
        heaters = [h for h in heaters if h in knobs]

    rounds, weak, R, F = schedule(heaters, swing, r2, vpi, a.swing_read, a.r2, a.foot)
    n = len(heaters); n_solo = len(weak); n_rounds = len(rounds)
    placed = n - n_solo
    sweeps = n_rounds + n_solo
    mean_read = sum(len(R[c]) for c in heaters) / max(n, 1)

    print(f"heaters {n}  |  read-set mean {mean_read:.1f} PDs/heater  |  "
          f"thresholds: read>={a.swing_read}mV r2>={a.r2} foot>={a.foot}mV\n")
    for i, rd in enumerate(rounds):
        mem = "  ".join(f"{c}->pd{p}" for c, p in rd["members"])
        print(f"round {i:>2} ({len(rd['members'])}): {mem}")
    if weak:
        print(f"\nsolo (no clean read-PD, n={n_solo}): {weak}")
    print(f"\n{placed} heaters in {n_rounds} parallel rounds + {n_solo} solo = {sweeps} sweeps")
    print(f"speedup vs {n} solo sweeps: {n/max(sweeps,1):.1f}x  "
          f"(placed-only {placed/max(n_rounds,1):.1f}x)")

    json.dump({"rounds": [rd["members"] for rd in rounds], "solo": weak,
               "thresholds": {"swing_read": a.swing_read, "r2": a.r2, "foot": a.foot}},
              open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
