"""Standard PIC end-to-end experiment template.

Coordinates BOTH devices at once from one host process -- the AeroDiode laser (FTDI) and
the PIC's Arduino (CH340): power the laser to a set level for a BOUNDED time, apply one or
a series of heater-voltage vectors, measure the photodiodes for each, and log everything
to a CSV for post-processing.

This is the baseline every specific test is built on. Either import it:

    from template import open_devices, run_experiment
    laser, pic = open_devices(mock=False)
    try:
        run_experiment(laser, pic, duration_s=20, power_dbm=5,
                       heater_vectors=my_vectors, out_path="runs/my_test.csv")
    finally:
        laser.close(); pic.close()

or copy this file and edit main(). Run the built-in demo / self-test:

    python template.py --duration 20 --power 5 --volts-file vecs.csv --out runs/exp.csv
    python template.py --duration 20 --power 5 --zero      # zero-vector baseline (M.0)
    python template.py --mock --duration 5 --power 5 --zero # no hardware

SAFETY (all learned the hard way in this project -- do not weaken):
  * a background threading.Timer HARD-OFFs the laser after `duration_s` no matter what the
    measurement loop is doing; try/finally drives it safe on every other exit path too;
  * every laser-serial touch is under one lock, so the watchdog can never collide with the
    keepalive/telemetry on the FTDI port;
  * `laser_status = 1` does NOT prove emission -- we baseline the built-in monitor PD
    (bfm_optical_power) laser-off vs laser-on and FLAG it if the light never rose (task #9);
  * the driver self-disables after a few idle seconds, so we re-poke the setpoint between
    PD reads to hold the laser lit across the measurement.

Two independent serial devices: the laser is the FTDI, the PIC is the numeric CH340. We
open the laser first, learn its port, and pick the PIC port explicitly EXCLUDING it. Note
`ui.py` owns the Arduino port while running -- stop it before running a test here.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laser.laser import Laser, LaserError
from laser.pdmv5 import PDMv5Error
from src.pic import PIC, MockPIC, PICError, find_port, NUM_DAC, NUM_ADC_RAW, LIVE_PDS

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------- heater-vector parsing
def parse_vector(text: str) -> np.ndarray:
    """One heater vector from a comma/space string; must be NUM_DAC long."""
    parts = [p for p in text.replace(",", " ").split() if p]
    v = np.array([float(p) for p in parts], float)
    if v.size != NUM_DAC:
        raise ValueError(f"expected {NUM_DAC} heater volts, got {v.size}")
    return v


def load_vectors(volts, volts_file, zero) -> list[np.ndarray]:
    """Resolve the CLI heater-vector spec into a list of NUM_DAC-vectors."""
    if zero:
        return [np.zeros(NUM_DAC)]
    if volts:
        return [parse_vector(volts)]
    if volts_file:
        out = []
        with open(volts_file) as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    out.append(parse_vector(s))
                except ValueError as e:
                    raise ValueError(f"{volts_file}:{ln}: {e}")
        if not out:
            raise ValueError(f"no vectors found in {volts_file}")
        return out
    return [np.zeros(NUM_DAC)]  # default: the zero-vector baseline


# --------------------------------------------------------------------- device opening
def _resolve_pic_port(explicit, laser_port):
    """PIC (CH340) port, EXCLUDING the laser FTDI that find_port's glob also matches."""
    if explicit:
        return explicit
    env = os.environ.get("PIC_PORT")
    if env:
        return env
    _, cands = find_port(None)
    cands = [c for c in cands if c != laser_port]
    if not cands:
        raise PICError(
            "no PIC serial port found (after excluding the laser). "
            "Pass --pic-port or set $PIC_PORT; is ui.py holding the port?"
        )
    return cands[0]


def open_devices(mock=False, laser_port=None, pic_port=None):
    """Return (laser, pic), both open. Real hardware unless mock=True."""
    if mock:
        return _FakeLaser().open(), MockPIC(_mock_forward(), noise=3e-4).open()
    laser = Laser(port=laser_port).open()
    pic = PIC(port=_resolve_pic_port(pic_port, laser.dev.port)).open()
    return laser, pic


# ------------------------------------------------------------------------ the procedure
def run_experiment(
    laser,
    pic,
    *,
    duration_s: float,
    power_dbm: float,
    heater_vectors,
    out_path: str,
    settle_s: float = 0.3,
    repeats: int = 3,
    dark_repeats: int = 5,
    keepalive_s: float = 0.4,
    emit_eps: float = 0.05,
) -> dict:
    """Laser-on-for-duration end-to-end run. Returns a summary dict and writes out_path.

    Sequence: dark baseline (laser off) -> laser ON at power_dbm (watchdog armed) ->
    for each heater vector: apply, settle, sample the 14 PDs `repeats`x -> laser OFF.
    Every sample is one CSV row so post-processing can average / filter offline.
    """
    vectors = [np.asarray(v, float).ravel() for v in heater_vectors]
    lock = threading.Lock()  # serialises all laser-FTDI access
    stopped = threading.Event()  # set once the watchdog has forced the laser off

    def force_off():
        with lock:
            try:
                print(f"\n[watchdog] {duration_s:.0f}s reached -> forcing laser OFF")
                laser.hw_off()
            except Exception as e:  # a watchdog must never die silently
                print(f"[watchdog] hw_off error: {e}")
        stopped.set()

    def bfm():
        return float(laser.dev.measure("bfm_optical_power"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    started = datetime.now()
    t0 = time.time()
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(
        [f"# PIC end-to-end experiment  {started.isoformat(timespec='seconds')}"]
    )
    w.writerow(
        [
            f"# duration_s={duration_s} power_dbm={power_dbm} "
            f"settle_s={settle_s} repeats={repeats}"
        ]
    )
    w.writerow(
        [
            "elapsed_s",
            "phase",
            "vec_id",
            "sample",
            "laser_dbm",
            "laser_mA",
            "bfm",
            "input",
        ]
        + [f"pd{i}" for i in range(NUM_ADC_RAW)]
    )

    def log(phase, vec_id, sample, dbm, mA, b, vec, pds):
        w.writerow(
            [
                f"{time.time()-t0:.2f}",
                phase,
                vec_id,
                sample,
                f"{dbm:.2f}",
                f"{mA:.1f}",
                f"{b:.5g}",
                " ".join(f"{x:.3f}" for x in vec),
            ]
            + [f"{p:.4f}" for p in pds]
        )
        f.flush()

    def sample_pds(vec):
        pic.measure_raw(vec)  # apply + prime
        if settle_s > 0:
            time.sleep(settle_s)
        return [np.asarray(pic.measure_raw(vec), float) for _ in range(max(1, repeats))]

    summary = {"out_path": out_path, "n_rows": 0}
    wd = None
    try:
        # 1. dark baseline -- laser OFF (proves the floor + gives the bfm reference)
        print("dark baseline (laser off) ...")
        bfm_off = bfm()
        dark = []
        for s in range(max(1, dark_repeats)):
            pds = np.asarray(pic.measure_raw(vectors[0]), float)
            dark.append(pds)
            log("dark", 0, s, 0.0, 0.0, bfm_off, vectors[0], pds)
        summary["dark_mean_live"] = np.mean(dark, 0)[list(LIVE_PDS)].tolist()

        # 2. laser ON at power, watchdog armed from this instant
        print(f"laser ON -> {power_dbm:+.1f} dBm ...")
        with lock:
            laser.hw_on()
            laser.set(power_dbm)
            sp = float(laser.dev.read_setting("cw_current"))
            mA = laser.measured_mA()
        wd = threading.Timer(duration_s, force_off)
        wd.daemon = True
        wd.start()
        laser_deadline = time.time() + duration_s
        print(f"[watchdog] armed: laser forced OFF at t+{duration_s:.0f}s")

        bfm_on = bfm()
        emitted = (bfm_on - bfm_off) > emit_eps
        summary.update(
            bfm_off=bfm_off,
            bfm_on=bfm_on,
            emitted=bool(emitted),
            laser_mA=mA,
            laser_setpoint=sp,
        )
        edbm, emW = laser.setpoint_to_dbm(sp)
        print(
            f"  setpoint {sp:.1f} (~{emW:.2f} mW), {mA:.0f} mA, "
            f"monitor {bfm_off:.3g}->{bfm_on:.3g}"
        )
        if not emitted:
            print(
                "  ** WARNING: monitor PD did not rise -- laser may be armed (READY) "
                "but NOT lasing. Treat outputs as suspect (see task #9). **"
            )

        # 3. signal phase -- apply each heater vector, sample the PDs
        for vec_id, vec in enumerate(vectors):
            if stopped.is_set() or time.time() >= laser_deadline:
                print("  duration reached before all vectors ran -- stopping early.")
                break
            for s, pds in enumerate(sample_pds(vec)):
                log("signal", vec_id, s, power_dbm, mA, bfm_on, vec, pds)
                summary["n_rows"] += 1
            with lock:  # keepalive: hold the laser lit against the idle watchdog
                laser.dev.write_setting("cw_current", sp, verify=False)
            time.sleep(keepalive_s)
    finally:
        if wd is not None:
            wd.cancel()
        with lock:
            if not stopped.is_set():
                laser.hw_off()
        f.close()

    print(f"  wrote {summary['n_rows']} signal rows -> {out_path}")
    return summary


# --------------------------------------------------------------------- no-hardware mock
def _mock_forward():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, (NUM_ADC_RAW, NUM_DAC))
    b = rng.uniform(0, 6.28, NUM_ADC_RAW)

    def fwd(v):
        return 0.02 + 0.05 * (1 + np.sin(W @ (np.asarray(v) / 4.0) + b))

    return fwd


class _FakeDev:
    port = "mock-laser"

    def __init__(self, parent):
        self.p = parent

    def read_setting(self, name):
        return self.p.sp if name == "cw_current" else 0

    def write_setting(self, name, val, *, verify=True, apply=True):
        if name == "cw_current":
            self.p.sp = float(val)

    def measure(self, name):
        if name == "bfm_optical_power":
            return 0.2 + (0.04 * self.p.sp if self.p.on else 0.0)
        if name == "diode_cw_current":
            return self.p.measured_mA()
        return 0


class _FakeLaser:
    """Minimal stand-in exercising the exact calls run_experiment makes."""

    def __init__(self):
        self.on = False
        self.sp = 0.0
        self.dev = _FakeDev(self)

    def open(self):
        return self

    def close(self):
        pass

    def hw_on(self):
        self.on = True
        self.sp = 5.0
        print("[mock] laser ON (floor)")

    def hw_off(self):
        self.sp = 0.0
        self.on = False
        print("[mock] laser OFF (safe)")

    def set(self, dbm, *, raw=False):
        mW = min(10.0 ** (float(dbm) / 10.0), 32.8)
        self.sp = max(5.0, (mW + 1.44) / 0.3437)

    def measured_mA(self):
        return 2.5 * self.sp - 7 if self.on else 1.6

    def setpoint_to_dbm(self, sp):
        import math

        mW = max(0.3437 * sp - 1.44, 1e-4)
        return 10 * math.log10(mW), mW


# ----------------------------------------------------------------------------- CLI demo
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--duration", type=float, default=20.0, help="laser-on cap, seconds"
    )
    ap.add_argument("--power", type=float, default=0.0, help="laser output, dBm")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--volts", help="one heater vector, 64 comma/space-separated volts")
    g.add_argument("--volts-file", help="file with one 64-value heater vector per line")
    g.add_argument(
        "--zero", action="store_true", help="single zero vector (M.0 baseline)"
    )
    ap.add_argument(
        "--out", default=None, help="output CSV (default runs/exp_<ts>.csv)"
    )
    ap.add_argument("--settle", type=float, default=0.3, help="dwell before reading, s")
    ap.add_argument("--repeats", type=int, default=3, help="PD samples per vector")
    ap.add_argument("--dark-repeats", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="run with no hardware")
    ap.add_argument("--laser-port", default=None)
    ap.add_argument("--pic-port", default=None)
    a = ap.parse_args(argv)

    if a.power > 5:
        print(
            f"note: {a.power:+.1f} dBm is above the +5 dBm PD-safe policy for this rig."
        )

    try:
        vectors = load_vectors(a.volts, a.volts_file, a.zero)
    except ValueError as e:
        print(f"bad heater vectors: {e}")
        return 2

    out = a.out or os.path.join(
        ROOT, "runs", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    try:
        laser, pic = open_devices(a.mock, a.laser_port, a.pic_port)
    except (LaserError, PICError, PDMv5Error) as e:
        print(f"device open failed: {e}")
        return 1
    try:
        s = run_experiment(
            laser,
            pic,
            duration_s=a.duration,
            power_dbm=a.power,
            heater_vectors=vectors,
            out_path=out,
            settle_s=a.settle,
            repeats=a.repeats,
            dark_repeats=a.dark_repeats,
        )
    except (LaserError, PICError, PDMv5Error) as e:
        print(f"experiment error: {e}")
        return 1
    finally:
        laser.close()
        pic.close()

    # a monitor rise is necessary but NOT sufficient for lasing (it can rise on ASE while
    # the panel still reads READY) -- confirm real emission on the panel/OPM (task #9).
    verdict = (
        "monitor PD rose (emission likely -- confirm on panel/OPM)"
        if s.get("emitted")
        else "monitor PD FLAT -- no light out"
    )
    print(f"done: {len(vectors)} vector(s), {s['n_rows']} rows; {verdict}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
