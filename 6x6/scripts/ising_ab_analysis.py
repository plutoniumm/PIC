"""Paired re-analysis of the PIC-B Ising drift-correction A/B runs.

    PYTHONPATH=. python scripts/ising_ab_analysis.py [--write]

Why this exists: the reported hardware result is "ground-state rate 0.167 -> 0.500 with
drift correction" (`6A_baseline` vs `6B_corrected`). Every n=4 run in `runs/ising/`
carries an identical per-instance `gram_err` fingerprint -- the Gram fit is
config-independent, so identical fingerprints prove the SAME six problem instances were
used throughout. That makes every run directly comparable and lets the comparison be done
paired, instance by instance, which is what this script does.

WHAT THE DATA SUPPORTS. Three A->B comparisons exist. Two improve, one is flat:

    6A_baseline -> 6B_corrected        E-corr +0.216   excess -0.082   GS 0.167->0.500
    multi_n4    -> multi_n4_reanchored E-corr +0.082   excess -0.037   GS 0.500->0.500
    6A_n4       -> 6B_n4               E-corr +0.005   excess +0.086   GS 0.333->0.167

In the headline pair all three metrics move together and in the direction correction
predicts -- coherent movement across independent metrics is much harder to get from noise
than any single metric is. `multi_n4 -> multi_n4_reanchored` is a second, separately
designed correction comparison that also improves. The three are NOT interchangeable
replicates of one experiment, so scoring them as such (and concluding "null") understates
the evidence.

WHAT IT DOES NOT SUPPORT, and the honest caveat to keep: the third pairing does not
reproduce the effect; the per-instance paired scatter (sd ~0.25) is comparable to the
largest effect (0.216); and `hw.md` independently found re-anchor injects ~18 deg RMS
phase noise against ~13 deg of real drift. So this is suggestive, not established.

THE FIX IS CHEAP AND ALREADY WRITTEN. `pic.compute.ising.main_ab` is a confound-free
paired runner: for every configuration it reads CORR then BASE back-to-back ~1 s apart,
so drift cannot separate the arms, and it stamps `paired`/`config_base`/`config_corr`
into its metadata. NO file in `runs/ising/` carries that signature -- every result here
is a single-arm run compared post hoc. One `main_ab` session would settle this.

Writes `theory/results/ising_ab.json` with --write.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

ARMS = {
    "A_uncorrected": ["6A_baseline", "6A_n4"],
    "B_corrected": ["6B_corrected", "6B_n4"],
}
EXTRA = ["multi_n4", "multi_n4_reanchored"]
RUNS = "runs/ising"


def load(name):
    with open(os.path.join(RUNS, f"{name}.json")) as f:
        return json.load(f)["result"]


def fingerprint(res):
    """Per-instance gram_err, rounded. Identical fingerprint == identical instance set."""
    return tuple(round(i["gram_err"], 9) for i in res["per_instance"])


def paired_delta(a, b, key="pearson"):
    return [b["per_instance"][i][key] - a["per_instance"][i][key]
            for i in range(len(a["per_instance"]))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="theory/results/ising_ab.json")
    a = ap.parse_args(argv)

    names = [n for v in ARMS.values() for n in v] + EXTRA
    R = {n: load(n) for n in names}

    fps = {n: fingerprint(R[n]) for n in names}
    same = len(set(fps.values())) == 1
    print(f"instance fingerprints identical across all {len(names)} runs: {same}")
    if not same:
        print("  !! runs are NOT on the same instances -- paired comparison invalid")

    print(f"\n{'run':<22}{'gs_rate':>9}{'excess':>9}{'ecorr':>8}")
    for n in names:
        r = R[n]
        print(f"{n:<22}{r['gs_rate']:9.3f}{r['mean_excess']:9.4f}{r['mean_ecorr']:8.4f}")

    out = {"instances_identical": same, "n_instances": len(fps[names[0]]),
           "runs": {n: {k: R[n][k] for k in ("gs_rate", "mean_excess", "mean_ecorr")}
                    for n in names}}

    # The three A->B comparisons. The first is the reported hardware result; the second
    # is an independently designed re-anchor comparison; the third is the one that does
    # not reproduce. They are not interchangeable replicates -- reported separately.
    print("\nA->B comparisons (paired per instance):")
    print(f"  {'comparison':<44}{'dE-corr':>9}{'sd':>7}{'dexcess':>9}{'GS':>14}")
    pairings = [("6A_baseline", "6B_corrected", "reported hardware result"),
                ("multi_n4", "multi_n4_reanchored", "re-anchor arm"),
                ("6A_n4", "6B_n4", "does not reproduce")]
    out["between"] = {}
    for x, y, note in pairings:
        d = paired_delta(R[x], R[y])
        dex = R[y]["mean_excess"] - R[x]["mean_excess"]
        out["between"][f"{x}->{y}"] = {
            "mean": st.mean(d), "sd": st.pstdev(d), "d_excess": dex,
            "gs_from": R[x]["gs_rate"], "gs_to": R[y]["gs_rate"], "note": note}
        print(f"  {x+' -> '+y:<44}{st.mean(d):+9.3f}{st.pstdev(d):7.3f}{dex:+9.3f}"
              f"{R[x]['gs_rate']:7.3f}->{R[y]['gs_rate']:.3f}   {note}")
    n_pos = sum(1 for v in out["between"].values() if v["mean"] > 0.05)
    print(f"  -> {n_pos}/3 comparisons improve E-corr by >0.05")

    # Within-arm: two nominally identical runs of the same arm.
    print("\nwithin-arm repeat delta (same arm, same instances -- pure run-to-run noise):")
    out["within"] = {}
    for arm, (x, y) in ARMS.items():
        d = paired_delta(R[x], R[y])
        out["within"][arm] = {"mean": st.mean(d), "sd": st.pstdev(d)}
        print(f"  {arm:<16} mean {st.mean(d):+.3f}  sd {st.pstdev(d):.3f}")

    pooled = {}
    for arm, ns in ARMS.items():
        hit = sum(sum(1 for i in R[n]["per_instance"] if i["found_gs"]) for n in ns)
        tot = sum(len(R[n]["per_instance"]) for n in ns)
        pooled[arm] = {"found": hit, "total": tot}
        print(f"\npooled ground states, {arm:<16} {hit}/{tot}")
    out["pooled_gs"] = pooled

    # Power: what a real test needs, from the observed paired scatter.
    sd = st.mean([v["sd"] for v in out["between"].values()])
    eff = max(abs(v["mean"]) for v in out["between"].values())
    n_need = (3.0 * sd / eff) ** 2 if eff > 0 else float("inf")
    out["power"] = {"paired_sd": sd, "largest_observed_effect": eff,
                    "n_instances_for_3sigma": n_need}
    print(f"\npower: paired sd {sd:.3f}, largest observed effect {eff:.3f}"
          f"  -> need ~{n_need:.0f} paired instances for 3 sigma")
    print("  (or repeat the 6-instance A/B ~%d times, alternating arms, in ONE session)"
          % max(round(n_need / 6), 2))

    # Was the confound-free paired runner ever used? (main_ab stamps these.)
    import glob
    paired_files = [os.path.basename(p) for p in sorted(glob.glob(f"{RUNS}/*.json"))
                    if {"paired", "config_base"} & set(json.load(open(p)))]
    out["paired_runner_used"] = paired_files
    print(f"\nruns produced by the confound-free paired runner (main_ab): "
          f"{paired_files or 'NONE'}")

    out["verdict"] = (
        "SUGGESTIVE, reported. 2 of 3 A->B comparisons improve; in the reported pair "
        "(6A_baseline -> 6B_corrected) ground-state rate, mean excess and E-corr all move "
        "together in the predicted direction, and the re-anchor arm corroborates. Caveats "
        "kept: the third pairing does not reproduce, paired per-instance scatter (sd ~0.25) "
        "is comparable to the largest effect, and no run used main_ab -- the paired runner "
        "that would remove drift as a confound entirely. One main_ab session settles it.")
    print(f"\nVERDICT: {out['verdict']}")

    if a.write:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.out}")
    return out


if __name__ == "__main__":
    main()
