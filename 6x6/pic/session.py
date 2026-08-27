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
open the laser first, learn its port, and pick the PIC port explicitly EXCLUDING it. Only
one process may hold the Arduino port -- stop any other reader before running a test here.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .devices.laser import Laser, LaserError
from .devices.mock import MockLaser
from .devices.pdmv5 import PDMv5Error
from .interface import PIC, MockPIC, PICError, find_port, mock_fringe_forward
from .config import (
    NUM_DAC,
    NUM_ADC_RAW,
    LIVE_PDS,
    FIRMWARE_VMAX,
    VPI_NOMINAL,
    ADC_AVG_MS,
)
from .acquisition import poll_pds, settling_time

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


def parse_channels(spec, num_dac=NUM_DAC) -> list[int]:
    """'all' / '0-63' / '0,1,5' / '0-15,32-47' -> sorted unique channel list."""
    if not spec or spec.strip() in ("all", "*"):
        return list(range(num_dac))
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [c for c in out if not 0 <= c < num_dac]
    if bad:
        raise ValueError(f"channel(s) out of 0..{num_dac - 1}: {sorted(bad)}")
    return sorted(out)


def parse_levels(spec) -> list[float]:
    """'0.25,0.5,2' or 'start:stop:step' -> sorted unique positive levels."""
    spec = (spec or "").strip()
    if ":" in spec:
        a, b, s = (float(x) for x in spec.split(":"))
        n = int(round((b - a) / s)) + 1
        vals = [round(a + i * s, 6) for i in range(n)]
    else:
        vals = [float(x) for x in spec.replace(" ", "").split(",") if x]
    return sorted({v for v in vals if v > 0})


def sweep_vectors(channels, levels, num_dac=NUM_DAC, include_base=True):
    """One-at-a-time sweep program: a single shared all-zero baseline (tagged channel -1),
    then one one-hot vector per (channel, level). Others held at 0 -- the marginal fringe.
    Returns ``(vectors, meta)`` with ``meta`` a list of ``(channel, level)`` per vector;
    the active channel and its level are also recoverable straight from each one-hot input,
    so the CSV needs no extra columns. Feed ``vectors`` to :func:`run_experiment`."""
    vectors, meta = [], []
    if include_base:
        vectors.append(np.zeros(num_dac))
        meta.append((-1, 0.0))
    for c in channels:
        for v in levels:
            x = np.zeros(num_dac)
            x[c] = float(v)
            vectors.append(x)
            meta.append((int(c), float(v)))
    return vectors, meta


def estimate_sweep_seconds(n_vectors, settle_s, repeats, keepalive_s=0.4, read_s=0.15):
    """Rough wall-clock for a sweep of ``n_vectors``, to size the laser-on watchdog."""
    per = settle_s + repeats * read_s + keepalive_s
    return n_vectors * per * 1.3 + 5.0


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
            "Pass --pic-port or set $PIC_PORT; is another process holding the port?"
        )
    return cands[0]


def open_devices(mock=False, laser_port=None, pic_port=None):
    """Return (laser, pic), both open. Real hardware unless mock=True."""
    if mock:
        return MockLaser().open(), MockPIC(mock_fringe_forward(seed=0), noise=3e-4).open()
    laser = Laser(port=laser_port).open()
    pic = PIC(port=_resolve_pic_port(pic_port, laser.dev.port)).open()
    return laser, pic


# --------------------------------------------------------------- laser safety scaffolding
class _Session:
    """Handle yielded by :func:`laser_session`; see there for the fields/methods."""


@contextmanager
def laser_session(laser, *, duration_s, power_dbm, bfm_off=None, emit_eps=0.05):
    """Bounded, watchdog-guarded laser-on session -- the ONE copy of the laser safety
    logic every experiment here shares. On enter: baseline the monitor PD (laser still
    off) unless ``bfm_off`` is supplied, turn the laser ON to ``power_dbm``, and ARM a
    background hard-off Timer at ``duration_s``. Yields a :class:`_Session` with
    ``.sp .mA .bfm_off .bfm_on .emitted .deadline``, ``.keepalive(sleep_s=0)`` (re-pokes
    the setpoint so the driver's idle self-disable can't drop the beam), and
    ``.expired()``. On EVERY exit path (normal, exception, or watchdog): cancel the timer
    and force the laser OFF unless the watchdog already did. All laser-FTDI access is
    under one lock so the watchdog can never collide with keepalive/telemetry. This is the
    safety contract from the module docstring -- do not weaken it."""
    lock = threading.Lock()
    stopped = threading.Event()
    wd = None

    def _bfm():
        return float(laser.dev.measure("bfm_optical_power"))

    def force_off():  # watchdog target -- a watchdog must never die silently
        with lock:
            try:
                print(f"\n[watchdog] {duration_s:.0f}s reached -> forcing laser OFF")
                laser.off()
            except Exception as e:
                print(f"[watchdog] hw_off error: {e}")
        stopped.set()

    s = _Session()
    s.stopped = stopped
    s.bfm_off = _bfm() if bfm_off is None else bfm_off  # laser still off here
    try:
        with lock:
            laser.on(power_dbm)  # open (idempotent) + enable + ramp, one call
            s.sp = float(laser.dev.read_setting("cw_current"))
            s.mA = laser.measured_mA()
        wd = threading.Timer(duration_s, force_off)
        wd.daemon = True
        wd.start()
        s.deadline = time.time() + duration_s
        s.bfm_on = _bfm()
        s.emitted = (s.bfm_on - s.bfm_off) > emit_eps

        def keepalive(sleep_s: float = 0.0):
            with lock:
                laser.dev.write_setting("cw_current", s.sp, verify=False)
            if sleep_s:
                time.sleep(sleep_s)

        def set_power(dbm):
            """Change the live output power mid-session and update the keepalive target so
            it holds the NEW setpoint (not the one the session opened at). Lets one lit
            session sweep several dbm levels without re-latching the laser each time."""
            with lock:
                laser.set(dbm)
                s.sp = float(laser.dev.read_setting("cw_current"))
                s.mA = laser.measured_mA()

        def telemetry():
            """(bfm optical power, diode temp, measured mA) read under the lock -- the
            laser-side state to log alongside a measurement."""
            with lock:
                bfm = float(laser.dev.measure("bfm_optical_power"))
                temp = float(laser.dev.measure("diode_temperature"))
                mA = float(laser.measured_mA())
            return bfm, temp, mA

        s.keepalive = keepalive
        s.set_power = set_power
        s.telemetry = telemetry
        s.expired = lambda: stopped.is_set() or time.time() >= s.deadline
        yield s
    finally:
        if wd is not None:
            wd.cancel()
        with lock:
            if not stopped.is_set():
                laser.off()


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
    try:
        # 1. dark baseline -- laser OFF (proves the floor + gives the bfm reference)
        print("dark baseline (laser off) ...")
        bfm_off = float(laser.dev.measure("bfm_optical_power"))
        dark = []
        for s in range(max(1, dark_repeats)):
            pds = np.asarray(pic.measure_raw(vectors[0]), float)
            dark.append(pds)
            log("dark", 0, s, 0.0, 0.0, bfm_off, vectors[0], pds)
        summary["dark_mean_live"] = np.mean(dark, 0)[list(LIVE_PDS)].tolist()

        # 2. laser ON at power, watchdog armed from this instant
        print(f"laser ON -> {power_dbm:+.1f} dBm ...")
        with laser_session(
            laser, duration_s=duration_s, power_dbm=power_dbm,
            bfm_off=bfm_off, emit_eps=emit_eps,
        ) as ls:
            mA, sp = ls.mA, ls.sp
            print(f"[watchdog] armed: laser forced OFF at t+{duration_s:.0f}s")
            summary.update(
                bfm_off=ls.bfm_off, bfm_on=ls.bfm_on, emitted=bool(ls.emitted),
                laser_mA=mA, laser_setpoint=sp,
            )
            edbm, emW = laser.setpoint_to_dbm(sp)
            print(
                f"  setpoint {sp:.1f} (~{emW:.2f} mW), {mA:.0f} mA, "
                f"monitor {ls.bfm_off:.3g}->{ls.bfm_on:.3g}"
            )
            if not ls.emitted:
                print(
                    "  ** WARNING: monitor PD did not rise -- laser may be armed (READY) "
                    "but NOT lasing. Treat outputs as suspect (see task #9). **"
                )

            # 3. signal phase -- apply each heater vector, sample the PDs
            for vec_id, vec in enumerate(vectors):
                if ls.expired():
                    print("  duration reached before all vectors ran -- stopping early.")
                    break
                for s, pds in enumerate(sample_pds(vec)):
                    log("signal", vec_id, s, power_dbm, mA, ls.bfm_on, vec, pds)
                    summary["n_rows"] += 1
                ls.keepalive(keepalive_s)  # hold the laser lit against its idle timeout
    finally:
        f.close()

    print(f"  wrote {summary['n_rows']} signal rows -> {out_path}")
    return summary


# ------------------------------------------------------------------- step / settling probe
def _report_step_edge(label, res, t):
    """One edge's per-live-PD settling table; returns ([(swing, settle)...], read rate)
    for live PDs with a finite settle. The swing rides along so the digest can drop
    PDs whose swing is at the noise floor (where the band crossing is meaningless)."""
    ts, fin, sw = res["t_settle"], res["final"], res["swing"]
    n = len(t)
    span = float(t[-1] - t[0]) if n > 1 else 0.0
    rate = (n - 1) / span if span > 0 else float("nan")
    print(
        f"\n  [{label}]  {n} reads over {span:.2f}s  "
        f"(~{1000 * span / max(n - 1, 1):.0f} ms/read, {rate:.1f} reads/s)"
    )
    print("    per live PD:   final V   swing mV   settle s")
    pairs = []
    for k in LIVE_PDS:
        st = ts[k]
        if np.isfinite(st) and st > t[0]:
            pairs.append((float(sw[k]), float(st)))
        ststr = f"{st:6.2f}" if np.isfinite(st) else " n/a "
        print(f"      pd{k:<2}   {fin[k]:7.3f}   {sw[k] * 1000:8.1f}   {ststr}")
    return pairs, rate


def _report_step(edges, channels, step_v, power_dbm, window, floor=0.005):
    """Digest across the captured edges (heat-up / cool-down) + a recommended settle_s.
    ``edges`` is a list of ``(label, settling_dict, t)``. Only PDs whose swing clears a
    noise gate (2x the band ``floor``) feed the recommendation -- a swing ~= the floor
    makes the band crossing land on noise, not on real settling. Returns the number."""
    print(
        f"\n[step] settling @ {power_dbm:+.1f} dBm, channels {list(channels)} "
        f"-> {step_v:g} V and back, window={window}"
    )
    all_pairs, rate = [], float("nan")
    for label, res, t in edges:
        pairs, r = _report_step_edge(label, res, t)
        all_pairs += pairs
        if r == r:  # keep a valid rate for the note below
            rate = r
    gate = 2 * floor
    strong = [st for sw, st in all_pairs if sw > gate]
    max_sw = max((sw for sw, _ in all_pairs), default=0.0)
    if strong:
        rec = max(strong) * 1.2
        print(
            f"\n  -> recommended host settle_s: {rec:.2f}s "
            f"(slowest edge/PD with swing > {gate * 1000:.0f} mV, +20%)"
        )
    elif max_sw > 0:
        rec = 0.0
        print(
            f"\n  -> swings <= {gate * 1000:.0f} mV (~2x the band) on every live PD -- "
            f"settling not reliably resolvable at this SNR (strongest {max_sw * 1000:.1f} "
            "mV). Drive a heater with a stronger PD response, or fit the exponential edge."
        )
    else:
        rec = 0.0
        print("\n  -> no resolvable transient (settled within the first read).")
    if rate == rate:  # not NaN
        print(
            f"     firmware already averages {ADC_AVG_MS} ms internally; a rolling window "
            f"of {window} reads (~{window / rate:.1f}s @ {rate:.1f}/s) smooths the rest."
        )
    return rec


def run_step_response(
    laser,
    pic,
    *,
    step_channels,
    step_v: float,
    seconds: float,
    power_dbm: float,
    out_path: str,
    window: int = 10,
    base=None,
    presettle_s: float = 1.0,
    dark_repeats: int = 5,
    rel_band: float = 0.05,
    floor: float = 0.005,
    tick_s: float = 0.5,
    emit_eps: float = 0.05,
) -> dict:
    """Light the chip, then capture BOTH thermal edges of a heater: step
    ``base`` -> ``step_channels`` at ``step_v`` (heat-up) and back to ``base`` (cool-down),
    polling every PD as fast as the firmware replies for ``seconds`` per edge -- i.e. how
    long the PD voltages take to stabilise heating up vs. cooling down. Per-PD settling is
    read off a rolling mean over ``window`` reads (the smoothing). Same watchdog/lock safety
    as :func:`run_experiment` via :func:`laser_session`; the poll keepalives the laser so it
    can't self-disable mid-capture. Every read is one CSV row (``phase`` = up / down).
    Returns a summary (per-edge ``settling_time`` dicts, raw traces, recommended settle_s)."""
    nd = pic.cfg.num_dac
    base = np.zeros(nd) if base is None else np.asarray(base, float).copy()
    v_step = base.copy()
    for c in step_channels:
        v_step[int(c)] = float(step_v)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    started = datetime.now()
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([f"# PIC step response  {started.isoformat(timespec='seconds')}"])
    w.writerow(
        [
            f"# channels={list(step_channels)} step_v={step_v} power_dbm={power_dbm} "
            f"window={window} seconds={seconds}"
        ]
    )
    w.writerow(["t_s", "phase"] + [f"pd{i}" for i in range(NUM_ADC_RAW)])

    def logrow(t, phase, pds):
        w.writerow([f"{t:.4f}", phase] + [f"{p:.4f}" for p in pds])
        f.flush()

    summary = {"out_path": out_path}
    t_up = raw_up = t_dn = raw_dn = None
    try:
        # 1. dark baseline -- laser OFF
        print("dark baseline (laser off) ...")
        bfm_off = float(laser.dev.measure("bfm_optical_power"))
        t0d = time.perf_counter()
        dark = []
        for _ in range(max(1, dark_repeats)):
            pds = np.asarray(pic.measure_raw(base), float)
            dark.append(pds)
            logrow(time.perf_counter() - t0d, "dark", pds)
        summary["dark_mean_live"] = np.mean(dark, 0)[list(LIVE_PDS)].tolist()

        # 2. laser ON, presettle cold, then heat-up edge + cool-down edge
        print(f"laser ON -> {power_dbm:+.1f} dBm ...")
        with laser_session(
            laser, duration_s=2 * seconds + presettle_s + 15.0, power_dbm=power_dbm,
            bfm_off=bfm_off, emit_eps=emit_eps,
        ) as ls:
            summary.update(
                bfm_off=ls.bfm_off, bfm_on=ls.bfm_on, emitted=bool(ls.emitted),
                laser_mA=ls.mA, laser_setpoint=ls.sp,
            )
            print(
                f"  monitor {ls.bfm_off:.3g}->{ls.bfm_on:.3g}"
                + ("" if ls.emitted else "  ** no rise -- treat as suspect (task #9) **")
            )
            pic.measure_raw(base)  # hold base cold, laser lit
            ls.keepalive(presettle_s)
            print(
                f"heat-up: channels {list(step_channels)} -> {step_v:g} V, "
                f"polling {seconds:g}s ..."
            )
            t_up, raw_up = poll_pds(pic, v_step, seconds, on_tick=ls.keepalive, tick_s=tick_s)
            for ti, pr in zip(t_up, raw_up):
                logrow(float(ti), "up", pr)
            print(f"cool-down: -> 0 V, polling {seconds:g}s ...")
            t_dn, raw_dn = poll_pds(pic, base, seconds, on_tick=ls.keepalive, tick_s=tick_s)
            for ti, pr in zip(t_dn, raw_dn):
                logrow(float(ti), "down", pr)
    finally:
        f.close()

    if t_up is None or len(t_up) < 2:
        print("  step aborted before enough reads -- no settling estimate.")
        return summary
    res_up = settling_time(t_up, raw_up, window=window, rel_band=rel_band, floor=floor)
    edges = [("heat-up", res_up, t_up)]
    summary.update(settling_up=res_up, t_up=t_up, raw_up=raw_up)
    if t_dn is not None and len(t_dn) >= 2:
        res_dn = settling_time(t_dn, raw_dn, window=window, rel_band=rel_band, floor=floor)
        edges.append(("cool-down", res_dn, t_dn))
        summary.update(settling_down=res_dn, t_down=t_dn, raw_down=raw_dn)
    rec = _report_step(edges, step_channels, step_v, power_dbm, window, floor=floor)
    summary["recommended_settle_s"] = rec
    n_rows = len(t_up) + (len(t_dn) if t_dn is not None else 0)
    print(f"  wrote {n_rows} poll rows -> {out_path}")
    return summary


# --------------------------------------------------------------------- no-hardware mock


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
    g.add_argument(
        "--sweep",
        action="store_true",
        help="one-at-a-time sweep: each channel over --sweep-levels, others at 0",
    )
    g.add_argument(
        "--step",
        action="store_true",
        help="step response: set --step-channels to --step-v, poll PDs, report settling",
    )
    ap.add_argument("--sweep-channels", default="all", help="e.g. all / 0-63 / 0,1,5")
    ap.add_argument(
        "--sweep-levels", default="0.25:2.0:0.25", help="list or start:stop:step [V]"
    )
    ap.add_argument("--step-channels", default="0-7", help="channels to step, e.g. 0-7")
    ap.add_argument("--step-v", type=float, default=VPI_NOMINAL, help="step voltage [V]")
    ap.add_argument(
        "--step-seconds", type=float, default=8.0, help="seconds to poll after the step"
    )
    ap.add_argument(
        "--window", type=int, default=10, help="rolling-average window [reads]"
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

    if a.step:
        try:
            step_ch = parse_channels(a.step_channels)
        except ValueError as e:
            print(f"bad step channels: {e}")
            return 2
        if not step_ch:
            print("no step channels")
            return 2
        if a.step_v > FIRMWARE_VMAX:
            print(
                f"note: step-v {a.step_v} exceeds the {FIRMWARE_VMAX} V firmware clamp -- "
                "it will be silently clamped on the device."
            )
        out = a.out or os.path.join(
            ROOT, "runs", f"step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        try:
            laser, pic = open_devices(a.mock, a.laser_port, a.pic_port)
        except (LaserError, PICError, PDMv5Error) as e:
            print(f"device open failed: {e}")
            return 1
        try:
            s = run_step_response(
                laser, pic, step_channels=step_ch, step_v=a.step_v,
                seconds=a.step_seconds, power_dbm=a.power, out_path=out,
                window=a.window, dark_repeats=a.dark_repeats,
            )
        except (LaserError, PICError, PDMv5Error) as e:
            print(f"step error: {e}")
            return 1
        finally:
            laser.close()
            pic.close()
        verdict = (
            "monitor PD rose (emission likely)"
            if s.get("emitted")
            else "monitor PD FLAT -- no light out"
        )
        print(f"done: step response, {len(step_ch)} channel(s); {verdict}.")
        return 0

    prefix = "exp"
    if a.sweep:
        try:
            chans = parse_channels(a.sweep_channels)
            levels = parse_levels(a.sweep_levels)
        except ValueError as e:
            print(f"bad sweep spec: {e}")
            return 2
        if not levels:
            print("no sweep levels")
            return 2
        over = [v for v in levels if v > FIRMWARE_VMAX]
        if over:
            print(
                f"note: levels {over} exceed the {FIRMWARE_VMAX} V firmware clamp -- "
                "they will be silently clamped on the device."
            )
        vectors, _meta = sweep_vectors(chans, levels)
        est = estimate_sweep_seconds(len(vectors), a.settle, a.repeats)
        a.duration = max(a.duration, est)
        prefix = "sweep"
        print(
            f"sweep: {len(chans)} channels x {len(levels)} levels = {len(vectors)} "
            f"vectors; ~{est / 60:.1f} min laser-on, watchdog set to {a.duration:.0f}s."
        )
    else:
        try:
            vectors = load_vectors(a.volts, a.volts_file, a.zero)
        except ValueError as e:
            print(f"bad heater vectors: {e}")
            return 2

    out = a.out or os.path.join(
        ROOT, "runs", f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
