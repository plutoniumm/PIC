"""What fraction of unitary space survives the 3 V ceiling?

`Calibration.volts` already refuses a phase it cannot make; this asks how often that
refusal bites over random targets, and which heater is responsible. A heater whose span
is under 2 pi cannot reach every phase, so the question is not "is the mesh universal"
-- it is not -- but "how much of it is left, and is the shortfall concentrated or spread".
"""
import sys

import numpy as np

from theory.calib import VOLTAGE_MAX, Calibration
from theory.clements import NMODE, random_unitary
from theory.program import phases_for


def survey(calib, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    blocked = np.zeros(len(calib.vpi), int)
    n_ok = 0
    for _ in range(n):
        ph = phases_for(random_unitary(rng))
        _, ok = calib.volts(ph, vmax=VOLTAGE_MAX)
        blocked += ~ok
        n_ok += bool(ok.all())
    return n_ok / n, blocked / n


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "pic_data/calib.json"
    c = Calibration.load(path)
    wired = [i for i, v in enumerate(c.vpi) if np.isfinite(v)]
    span = c.reach_span
    print(f"{'dac':>4} {'Vpi':>6} {'span/pi':>8}  reaches")
    for i in wired:
        print(f"{i:>4} {c.vpi[i]:>6.2f} {span[i]:>8.2f}  "
              f"{'2pi (free)' if span[i] >= 2 else ('pi' if span[i] >= 1 else 'under pi')}")
    frac, blocked = survey(c)
    print(f"\nrandom unitaries fully reachable: {100 * frac:.1f}%")
    order = np.argsort(-blocked)
    print("blocked most often by:")
    for i in order[:6]:
        if blocked[i]:
            print(f"  dac {i:>2}  blocks {100 * blocked[i]:5.1f}% of targets  (span {span[i]:.2f} pi)")
