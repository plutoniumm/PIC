"""Per-heater fringe fit from a probe_channels census -> V0 (max pass) and Vpi (null).

For each channel: take its most-responsive PD's transmission across the swept levels,
fit ``P(v) = A + B*cos(phi2*v^2 + phi0)`` (src.characterize.fit_fringe), and read off the
two operating points we normalize between (src.sweep_analysis.fringe_extrema):

    V0    = v_at_max        -> brightest (max pass)
    Vnull = v_at_min        -> null (0 light out, pi phase)
    Vpi   = sqrt(pi/phi2)   -> half-wave voltage (fringe period)

Because it is a model fit along the known V^2 law, Vnull comes out even when the null sits
past the 2 V clamp (flagged not-reachable). Reads the census CSV probe_channels writes;
no hardware. The fit primitives live in ``src.census`` (robust_fit / load_census / analyze
/ digest) -- this file is the CLI wrapper.

    python scripts/census_fringes.py pic_data/census/census_YYYYMMDD_HHMMSS.csv [--write]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# re-exported for callers that still import these off this module (back-compat)
from src.census import NPD, VPI_SEEDS, analyze, digest, load_census, robust_fit  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--vmax", type=float, default=2.0, help="drive ceiling for reachability flags")
    ap.add_argument("--min-vis", type=float, default=0.15,
                    help="visibility floor; lower (~0.02) for light-routed bright-baseline runs")
    ap.add_argument("--min-swing", type=float, default=5.0, help="absolute swing floor [mV]")
    ap.add_argument("--write", action="store_true", help="write fringes_<name>.csv beside it")
    a = ap.parse_args()
    recs = analyze(a.census, vmax=a.vmax, min_vis=a.min_vis, min_swing=a.min_swing)
    digest(recs)
    if a.write:
        p = Path(a.census)
        out = p.with_name("fringes_" + p.stem.split("_", 1)[-1] + ".csv")
        cols = ["ch", "pd", "swing_mV", "reliable", "Vpi", "V0", "Vnull", "V0_reach",
                "Vnull_reach", "phi0", "visibility", "rmse_mV", "ok"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(recs)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
