"""Live-hardware PD learnability probe -- which of the 14 photodiodes carry a signal we
can actually MOVE with the heaters, now that the chip is drivable. This decides which PDs
become the DPNN's outputs.

Randomize the drivable heaters (config `elec == "ok"`) over the {0,0.5,..,4} V grid, read
all 14 PDs a few times per config, and split each PD's variance into SIGNAL (variance of
the per-config mean across configs) vs NOISE (mean within-config variance across repeats).
A PD is learnable when the signal dominates -- frac = signal/(signal+noise) is high -- and
its per-config swing clears the noise floor.

Laser held at +15 dBm (the PIC-B operating power; NEVER exceed +15 dBm -- PD damage risk).
Every raw read + laser telemetry is streamed to pic_data/census/ as measured (crash loses
<= one config). The per-PD verdict is also written to pic_data/pd_learnable.json for the
DPNN to read which PDs to target.

    PYTHONPATH=. python scripts/pd_learn_test.py --mock --n 40           # no hardware
    PYTHONPATH=. python scripts/pd_learn_test.py --n 150 --repeats 3     # real hardware
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.pic import NUM_ADC_RAW
from template import laser_session

GRID = np.round(np.arange(0.0, 4.0 + 1e-9, 0.5), 3)  # host heater grid; firmware clamps at 4 V
FRAC_MIN = 0.5        # signal must be >= half the total per-PD variance ...
SWING_MIN_MV = 3.0    # ... and the per-config span must clear the noise floor
JSON_OUT = "pic_data/pd_learnable.json"


def load_config(path):
    cfg = json.load(open(path))
    num_dac = int(cfg.get("firmware", {}).get("num_dac", 128))
    drivable = [c["dac"] for c in cfg["channels"] if c.get("elec") == "ok"]
    dbm_cap = float(cfg.get("laser", {}).get("max_dbm", 15))
    return num_dac, drivable, dbm_cap


def open_devices(mock, port, num_dac):
    if mock:
        from src.pic import MockPIC, mock_fringe_forward
        from template import _FakeLaser
        return (_FakeLaser().open(),
                MockPIC(mock_fringe_forward(num_dac=num_dac), noise=3e-4, num_dac=num_dac).open())
    from laser.laser import Laser
    from src.pic import PIC
    from template import _resolve_pic_port
    laser = Laser().open()
    pic = PIC(port=_resolve_pic_port(port, laser.dev.port), num_dac=num_dac).open()
    return laser, pic


def analyze(means, within):
    """means/within: (n_cfg, 14). signal = variance of per-config means ACROSS configs;
    noise = mean WITHIN-config variance. Returns {pd: metrics}."""
    means, within = np.asarray(means, float), np.asarray(within, float)
    sig = means.var(axis=0)                       # across-config signal power
    noise = within.mean(axis=0)                   # within-config noise power
    swing_mV = (means.max(0) - means.min(0)) * 1000
    frac = sig / (sig + noise + 1e-12)
    snr = sig / (noise + 1e-12)
    out = {}
    for p in range(means.shape[1]):
        learn = bool(frac[p] >= FRAC_MIN and swing_mV[p] >= SWING_MIN_MV)
        out[p] = dict(swing_mV=float(swing_mV[p]),
                      signal_sd_mV=float(np.sqrt(sig[p]) * 1000),
                      noise_sd_mV=float(np.sqrt(noise[p]) * 1000),
                      snr=float(snr[p]), frac_learnable=float(frac[p]), learnable=learn)
    return out


def report(metrics):
    order = sorted(metrics, key=lambda p: -metrics[p]["frac_learnable"])
    print("\n pd  swing_mV  sig_sd  noise_sd     snr   frac   verdict")
    for p in order:
        m = metrics[p]
        print(f" {p:2d}  {m['swing_mV']:8.1f}  {m['signal_sd_mV']:6.2f}  "
              f"{m['noise_sd_mV']:8.2f}  {m['snr']:6.1f}  {m['frac_learnable']:5.2f}   "
              f"{'LEARNABLE' if m['learnable'] else '-'}")
    learnable = sorted(p for p in metrics if metrics[p]["learnable"])
    print(f"\nlearnable PDs ({len(learnable)}): {learnable}")
    return learnable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150, help="random heater configs")
    ap.add_argument("--repeats", type=int, default=3, help="reads per config (noise estimate)")
    ap.add_argument("--dbm", type=float, default=15.0, help="laser power; PIC-B ceiling +15 dBm")
    ap.add_argument("--settle", type=float, default=0.4, help="dwell before reading, s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", default=os.environ.get("PIC_PORT"))
    ap.add_argument("--mock", action="store_true", help="run with no hardware")
    ap.add_argument("--config", default="pic_data/pic_b_config.json")
    ap.add_argument("--out", default=None, help="raw-stream CSV (default pic_data/census/pdlearn_*.csv)")
    ap.add_argument("--json-out", default=JSON_OUT)
    a = ap.parse_args(argv)

    num_dac, drivable, dbm_cap = load_config(a.config)
    drivable = np.asarray(drivable, int)
    if a.dbm > dbm_cap and not a.mock:
        print(f"** {a.dbm:+.1f} dBm ABOVE the +{dbm_cap:.0f} dBm PD-safe ceiling -- PD damage risk **")
    rng = np.random.default_rng(a.seed)
    print(f"{drivable.size} drivable heaters (elec=ok) of {num_dac}; {a.n} configs x "
          f"{a.repeats} reads @ {a.dbm:+.1f} dBm{' [MOCK]' if a.mock else ''}")

    out = a.out or time.strftime("pic_data/census/pdlearn_%Y%m%d_%H%M%S.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    laser, pic = open_devices(a.mock, a.port, num_dac)
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "cfg", "rep", "dbm", "bfm", "temp_c", "mA"]
               + [f"pd{i}" for i in range(NUM_ADC_RAW)] + ["input"])

    means, within = [], []
    dur = a.n * (a.settle + a.repeats * 0.15 + 0.25) * 1.4 + 25.0
    try:
        with laser_session(laser, duration_s=dur, power_dbm=a.dbm) as ls:
            print(f"emitted={ls.emitted}  bfm {ls.bfm_off:.3f}->{ls.bfm_on:.3f}  "
                  f"mA={ls.mA:.1f}  -> {out}", flush=True)
            if not ls.emitted:
                print("  ** monitor PD did not rise -- laser may be READY not lasing; "
                      "results suspect (task #9). **", flush=True)
            for c in range(a.n):
                if ls.expired():
                    print("  watchdog reached -- keeping partial run.", flush=True)
                    break
                v = np.zeros(num_dac)
                v[drivable] = rng.choice(GRID, size=drivable.size)
                ls.keepalive()
                tel = ls.telemetry()
                pic.measure_raw(v)                       # apply + prime
                if a.settle > 0:
                    time.sleep(a.settle)
                istr = " ".join(f"{x:.1f}" for x in v)
                reads = []
                for r in range(max(1, a.repeats)):
                    y = np.asarray(pic.measure_raw(v), float)
                    reads.append(y)
                    w.writerow([f"{time.time():.2f}", c, r, f"{a.dbm:.2f}",
                                f"{tel[0]:.3f}", f"{tel[1]:.2f}", f"{tel[2]:.2f}"]
                               + [f"{x:.4f}" for x in y] + [istr])
                f.flush()
                R = np.array(reads)
                means.append(R.mean(0))
                within.append(R.var(0, ddof=1) if len(R) > 1 else np.zeros(R.shape[1]))
                if (c + 1) % 25 == 0 or c + 1 == a.n:
                    print(f"  {c + 1}/{a.n} configs", flush=True)
    finally:
        f.flush(); f.close()
        pic.set_zero(); pic.close(); laser.close()

    if len(means) < 2:
        print("too few configs measured for a variance estimate.")
        return 1
    metrics = analyze(means, within)
    learnable = report(metrics)
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "source_csv": out,
               "n_configs": len(means), "repeats": a.repeats, "dbm": a.dbm, "mock": a.mock,
               "frac_min": FRAC_MIN, "swing_min_mV": SWING_MIN_MV,
               "learnable": learnable, "all": {str(p): metrics[p] for p in metrics}}
    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump(payload, open(a.json_out, "w"), indent=2)
    print(f"wrote {out}  and  {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
