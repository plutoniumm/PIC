"""Browser front end for the 4x4 rig: run a unitary matvec, read the calibration the run
stands on, and check the instruments before either.

Everything real happens in `pic.apply` and in the files under `pic_data/`; this file is
transport, a gate and markup. Run it, open the URL it prints, and the page talks to `/run`,
`/calib` and `/recal` over JSON.

    python ui.py            # against the bench
    python ui.py --mock     # against the physical mock, no hardware
    python ui.py --selftest # the gate below, on the mock, no server and no port

`--mock` is the page's own toggle too, so one server serves both. A run opens the rig, does
its work and closes it, which costs a couple of seconds and keeps no instrument held between
clicks -- the laser watchdog and the serial ports both prefer that to a long-lived handle.

**x is a batch.** `apply_unitary` has always taken X as (4, n) and `pic.matvec` applies the
columns independently under one heater program, so several vectors cost photodiode reads and
no extra thermal settle. The page therefore edits columns of X, and the per-column `rel_err`
and `sign_acc` come back as a distribution and are shown as one -- a histogram of the
per-element deviation and the quantiles of the per-x error, never a mean standing in for
them. `pic.apply` has the run where a mean of four vectors read 0.0002 while the vectors
were 1.1 out.

**The target is checked here, before an instrument opens.** `pic.apply._hostable` refuses a
complex target because the readout has no phase reference; what it cannot refuse is a real
matrix that is not orthogonal, which has no Clements phases and would be hosted as whatever
the tiling made of it. `||U^T U - I||_F` against `TOL_ORTH` is that check, server side so it
cannot be skipped and mirrored live on the page so it is visible before the click.

**Runs accumulate.** Every run is appended to a history in `localStorage`, newest first,
each entry carrying the U and the X that produced it so it can be put back into the inputs.
Two runs of one target are the measurement; one run is an anecdote.

**The calibration tab is read-only.** `/calib` reads `pic_data/calib.json` and
`pic_data/char_results.json` and serves them joined onto the heater map imported from
`theory.layout` at request time -- the map is never copied here, because a stale copy of it
is how role labels ended up on the wrong heaters once already. The per-heater reload button
posts to `/recal`, which composes `python -m pic char --channels <dac> --write` and refuses to run it
while any `python -m pic` already holds the board.

**The diagnostics tab is the pre-flight.** Eight rows, each an icon, a measured value and
its own reload button, so a dead instrument is found in seconds rather than after a
twelve-minute sweep. A wrong DAC chip-select pin cost a whole day because nothing in the
stack surfaced it -- SPI has no readback, the clamp table read back correctly, and every
sweep completed and fitted thermal drift. The rows are chosen so that each one is the
cheapest evidence that a particular link of the chain is alive, and the ones software
cannot decide say `unknown` rather than `pass`. `/diag` never runs itself -- it has a run-all
button and a per-row button and no poll -- because two of these rows move the rig: the
switch row walks the mirror, and the rail row puts every heater at its clamp.

**The mesh picture is derived, not drawn.** `mesh_diagram` places every element and every
heater from `clements.MESH`/`COLUMN` and `layout.THETA_DAC`/`PHI_DAC`/`AUX_RAIL`; the page
draws what it is handed and invents no coordinate of its own. `_check_diagram` reads the
element index back out of those maps for every heater on the picture, and `placement_text`
prints the whole placement so it can be checked without looking at it. A mesh figure drawn
from memory agrees with the chip right up until the map moves, and the map has moved twice.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import signal
import threading
from collections import deque
from contextlib import contextmanager
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from pic import Rig
from pic.log import ev
from pic.apply import ReadoutError, apply_unitary, box_for, table_for

TOL_ORTH = 1e-5  # Frobenius, on U^T U - I
MAX_COLS = 8  # columns of X the page will accept

CALIB_PATH = Path("pic_data/calib.json")
CHAR_PATH = Path("pic_data/char_results.json")
SKETCH_PATH = Path("Arduino/pic4x4/pic4x4.ino")


def _is_pic(argv) -> bool:
    """`<python> -m pic ...`: any pic CLI, which is what holds the board. Matched on argv, not a
    joined command line, so a shell that merely mentions it is not a match."""
    return (
        len(argv) >= 3
        and "python" in Path(argv[0]).name.lower()
        and any(
            argv[i] == "-m" and argv[i + 1].split(".")[0] in ("pic", "learn")
            for i in range(1, len(argv) - 1)
        )
        and "--mock" not in argv  # a mock run holds no port
    )


# A recharacterisation drives current through a heater. Off unless `--arm` says otherwise,
# so a misdirected click cannot reach the bench; `/recal` then reports the command instead
# of running it.
ARMED = False


class NotOrthogonal(ValueError):
    """Refused before any instrument opens, and a sibling of `ReadoutError` rather than a
    special case of it: the readout bars complex targets, this bars real ones that are not
    unitary. Both end on the page's Refused card with the number that decided it."""


class Busy(RuntimeError):
    """Something already holds the board, or we cannot prove nothing does."""


_BOARD_LOCK = threading.Lock()  # one board, and the page can fire several requests at once


@contextmanager
def _board_held(mock: bool):
    """Hold the board for this request, or refuse at once rather than queue behind a sweep."""
    if mock:
        yield
        return
    if not _BOARD_LOCK.acquire(blocking=False):
        raise Busy(
            "the board is in use by another request from this page (a run, a diagnostic or "
            "the rail test). Wait for it to finish, then try again."
        )
    try:
        yield
    finally:
        _BOARD_LOCK.release()


def _finite(v, name, n):
    """A flat array of n finite floats, or a ValueError that names the field."""
    try:
        a = np.asarray(v, float).ravel()
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numbers") from None
    if a.size != n:
        raise ValueError(f"{name} needs {n} numbers, got {a.size}")
    if not np.all(np.isfinite(a)) or np.abs(a).max() > 1e6:
        raise ValueError(f"{name} has a NaN, an infinity or an absurd magnitude")
    return a


def _tec_port():
    from pic.config import usb_ports

    ports = [d for d, r, _ in usb_ports() if r == "tec"]
    if not ports:
        raise ValueError("TEC controller not found on USB (pic.config.USB_SERIAL['tec'])")
    return ports[0]


_LASER_HELD = []  # what currently owns the laser's port inside this server, if anything


@contextmanager
def _laser_held(what, mock):
    if not mock:
        _LASER_HELD.append(what)
    try:
        yield
    finally:
        if not mock:
            _LASER_HELD.remove(what)


def orth_dev(U) -> float:
    U = np.asarray(U, float)
    return float(np.linalg.norm(U.T @ U - np.eye(len(U))))


# One page per tab, so a reload lands where it was. The markup, the stylesheet and the script
# are files under static/; this module only fills in which page is showing and serves data.
STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": "mv", "/calibration": "cal", "/diagnostics": "diag", "/logs": "logs"}
MIME = {".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".html": "text/html"}


_LASER_LOCK = threading.Lock()  # the page polls; two opens of one FTDI must never overlap


_LASER_READ = {}  # a few seconds' cache of the plain read: the page and the state strip both poll


def laser(b=None):
    if b is None and _LASER_READ.get("t", 0) > time.time() - 3:
        return _LASER_READ["v"]
    out = _laser_call(b)
    _LASER_READ.update(t=time.time(), v=out) if b is None else _LASER_READ.clear()
    return out


def _laser_call(b=None):
    # validated first: a bad request is refused whatever holds the port, and must never
    # half-switch the diode
    if b is not None and b.get("on") is True:
        dbm = _finite(b.get("dbm", 5), "dbm", 1)[0]
        if not -10.0 <= dbm <= 12.0:  # 12 dBm cap agreed at the bench
            raise ValueError(f"dbm must be between -10 and 12, got {dbm:g}")
    if JOB and JOB["p"].poll() is None and not JOB["mock"]:
        # the job owns the laser's port until it ends; its events say whether it is lit
        return {"held": JOB["what"], "on": _job_laser(_job_events())}
    if _LASER_HELD:
        return {"held": _LASER_HELD[0], "on": None}
    ext = sweeps_running()
    if ext:  # a terminal's pic process may hold the laser's port: do not open it, do not guess
        return {"held": f"pid {', '.join(map(str, ext))}", "on": None}
    with _LASER_LOCK:
        return _laser(b)


def _laser(b):
    from pic.devices.laser import Laser

    L = Laser().open()
    try:
        if b is not None:
            if b.get("on") is not True:  # anything but an explicit true is off
                L.hw_off()
            else:
                L.is_on() or L.hw_on()
                L.set(float(b.get("dbm", 5)))
        pre, d = L.preflight(), L.dev
        return {
            "on": bool(L.is_on()),
            "mA": round(float(L.measured_mA()), 1),
            "dbm": round(L.setpoint_to_dbm(float(d.read_setting("cw_current")))[0], 1),
            "key": pre["key switch ON"],
            "bnc": pre["BNC interlock closed"],
            "ext": pre["EXT interlock closed"],
            "drv": bool(d.measure("driver_enable")),
        }
    finally:
        L.close()


def ago(t: float) -> str:
    """How long ago, in three levels only: minutes, hours, days."""
    sec = max(0.0, time.time() - t)
    if sec < 3600:
        return f"{int(sec // 60)} min ago"
    if sec < 86400:
        return f"{int(sec // 3600)} hrs ago"
    return f"{int(sec // 86400)} days ago"


from events import EventLog

EVENTS = EventLog(Path("pic_data/logs/events.jsonl"))  # every printed line, as events


class _Tee:
    """Every print this server makes also lands in pic_data/logs/server.log, for the Logs tab."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        from pic.log import human

        # the terminal gets sentences; the log file and the event stream get the JSON
        self.stream.write("\n".join(human(ln) for ln in s.split("\n")))
        self.fh.write(s)
        self.fh.flush()
        EVENTS.feed("server", s)
        return len(s)

    def isatty(self):
        return False  # so the server's own events are emitted as JSON, then shown readable

    def flush(self):
        self.stream.flush()


def logs(name=None, lines: int = 400):
    """Every log file, newest first; with `name`, the last `lines` lines of that one."""
    files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = {"files": [{"name": p.name, "ago": ago(p.stat().st_mtime)} for p in files]}
    if name:
        p = LOG_DIR / Path(name).name  # a bare file name: nothing outside the log directory
        if not p.is_file() or p.suffix != ".log":
            raise ValueError(f"no such log: {name}")
        out["name"], out["text"] = p.name, "\n".join(
            p.read_text(errors="replace").splitlines()[-lines:]
        )
    return out


def render(page: str) -> str:
    html = (STATIC / "page.html").read_text(encoding="utf-8")
    html = html.replace("{{page}}", page)
    for k in PAGES.values():
        html = html.replace(f"{{{{hide_{k}}}}}", "" if k == page else "hide")
    return html


def run_once(payload, default_mock):
    mock = payload.get("mock", default_mock) is not False
    if "U" not in payload:
        raise ValueError("no target U in the request")
    U = _finite(payload["U"], "U", 16).reshape(4, 4)
    dev = orth_dev(U)
    if dev > TOL_ORTH:
        raise NotOrthogonal(
            f"target is not orthogonal: ||U^T U - I|| = {dev:.3e}, tolerance {TOL_ORTH:.0e}.\n\n"
            "This bench reads intensity, so it hosts real orthogonal targets only. A matrix "
            "this far off the manifold has no Clements phases, and running it would report "
            "whatever the tiling made of it as though you had asked for it. Press Random "
            "unitary, or orthogonalise your own U, and run again."
        )

    cols = payload.get("X") or ([payload["x"]] if "x" in payload else [])
    if not 1 <= len(cols) <= MAX_COLS:
        raise ValueError(f"give 1 to {MAX_COLS} input vectors x, got {len(cols)}")
    if any(not isinstance(c, list) or len(c) != 4 for c in cols):
        raise ValueError("each x needs 4 entries, one per input port")
    X = _finite(cols, "x", 4 * len(cols)).reshape(len(cols), 4).T
    zero = np.flatnonzero(~X.any(axis=0))
    if zero.size:
        raise ValueError(f"x{zero[0] + 1} is all zeros: there is nothing to multiply")

    kind = "mock" if mock else "hw"
    # One press runs both hostings on the same rig, same U, same x, back to back, so the two
    # measurements differ by the method and not by the minute they were taken in.
    #
    # They approximate different things, and each is scored against its own ideal. `tile`
    # recovers signed Ux through the shift and differential passes. `full` has the mesh hold U
    # outright, and one lit port at a time returns intensity, so what it can measure is
    # |U|^2 x -- non-negative by construction. Scoring it against Ux would mark it wrong for
    # a sign it never had a way to see.
    #
    why = _board_guard(mock)
    if why:
        raise Busy(f"refusing to run on the bench: {why}")
    dbm = _finite(payload.get("dbm", 5), "dbm", 1)[0]
    if not -10.0 <= dbm <= 12.0:
        raise ValueError(f"dbm must be between -10 and 12, got {dbm:g}")
    # Bench is the whole real experiment and mock the whole fake one, never a mixture: the
    # real laser, board, switch and TEC, inside the same guarded session every CLI command
    # uses -- TEC settled, laser lit, emission verified, laser off on every exit.
    tec = "mock" if mock else TEC.view()
    with _board_held(mock), _laser_held("matvec run", mock):
        with Rig(laser=kind, board=kind, switch=kind, tec=tec) as rig:
            # A run never waits on the TEC: the settle gate used to sit silent for up to five
            # minutes whenever the die was a hair outside the band. Check once and say so.
            if not mock and not rig.tec.stable:
                raise Busy(
                    f"TEC: die at {rig.tec.temperature():.2f} C, outside "
                    f"{rig.tec.target:.2f} ± {rig.tec.tolerance:.2f} C. The loop is pulling it "
                    "back; run again once the header turns green."
                )
            with rig.session(
                # the hard laser-off cutoff, not the run time: the run ends and switches the
                # laser off as soon as it is done. Sized to a few times a bench run's reads.
                duration_s=30 + 10 * X.shape[1],
                power_dbm=dbm,
                require_stable=False,  # checked above, once
            ) as s:
                if not s.emitted:
                    raise ReadoutError(
                        "no light: the laser was switched on but neither its monitor diode nor "
                        "the photodiodes rose. Check the key, the fibre and the coupling."
                    )
                box = box_for(rig, mock=mock)
                tab = table_for(rig, box, mock=mock)
                rn = apply_unitary(rig, U, X, transfers=tab, hosting="full", box=box)
                rt = apply_unitary(rig, U, X, transfers=tab, hosting="tile", box=box)

    def series(res, c):
        yd, yc = res.Y_device[:, c], res.Y_cpu[:, c]
        return {
            "device": [float(v) for v in yd],
            "ideal": [float(v) for v in yc],
            "kept": [bool(v) for v in np.sign(yd) == np.sign(yc)],
            "rel": float(res.rel_err[c]),
            "sign": float(res.sign_acc[c]),
            "fid": _overlap(yd, yc),
        }

    Td, dmeta = _dpnn_transfer(rn.raw["volts"])
    vecs = []
    for c in range(X.shape[1]):
        ux = rt.Y_cpu[:, c]
        vecs.append(
            {
                "x": [float(v) for v in X[:, c]],
                "normal": series(rn, c),
                "tiled": series(rt, c),
                "computed": [float(v) for v in ux],
                "dpnn": None if Td is None else [float(v) for v in Td @ X[:, c]],
                "null": bool(np.linalg.norm(ux) == 0),
            }
        )
    out = {
        "rails_out": list(range(4)),
        "orth_dev": dev,
        "vectors": vecs,
        "programs": {"normal": rn.programs, "tiled": rt.programs},
        "spread": {"normal": rn.spread, "tiled": rt.spread},
        "reach": float(rn.raw.get("reachable", float("nan"))),
        "dpnn": dmeta,
        # the transfer the chip measured and the one the DPNN predicts for the same volts
        "T": np.asarray(rn.raw["T"]).tolist(),
        "dpnn_T": None if Td is None else np.asarray(Td).tolist(),
    }
    if not mock:
        _record_errors(vecs, rn.raw["T"], Td)
    return out


ERRORS_PATH = Path("pic_data/errors.jsonl")


def _record_errors(vecs, T=None, Td=None):
    """One line per bench run: the live error of each predictor, measured on the chip. The
    full hosting leans only on the heater calibration, the tiled one on the transfer table,
    and the DPNN is judged on the transfer matrix the chip measured for the full hosting --
    not on T x, which with a signed x can nearly cancel and make a close T look far off."""
    med = lambda xs: float(np.median(xs)) if xs else None
    full = [v["normal"]["fid"] for v in vecs if np.isfinite(v["normal"]["fid"])]
    tiled = [v["tiled"]["fid"] for v in vecs if np.isfinite(v["tiled"]["fid"])]
    dpnn = (
        [float(np.linalg.norm(np.subtract(Td, T)) / max(np.linalg.norm(T), 1e-12))]
        if T is not None and Td is not None
        else []
    )
    try:
        chip = TEC.mean()[0]
    except Exception:
        chip = None
    rec = {
        "t": time.time(),
        "heaters": med(full),
        "table": med(tiled),
        "dpnn": -med(dpnn) if dpnn else None,
        "chip_c": chip,
    }  # every column: higher is better
    with open(ERRORS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


RECENT = 5  # runs judged against everything since the last refresh; see `_live`


def _live(pid, since):
    """(now, best, n, declined) for one predictor from the runs since it was last refreshed.

    Declined is a one-sided rank test of the last RECENT runs against the ones before them,
    at theory.stat.ALPHA -- no threshold of its own. With five on each side the smallest
    p a rank test can reach is 1/252, so it needs 2*RECENT runs before it can say anything."""
    from theory.stat import ALPHA, p_worse

    vals = []
    try:
        for ln in ERRORS_PATH.read_text().splitlines():
            r = json.loads(ln)
            if r["t"] > since and r.get(pid) is not None:
                vals.append(r[pid])
    except (OSError, ValueError):
        pass
    if not vals:
        return None, None, 0, False
    now = float(np.median(vals[-RECENT:]))
    best = max(
        float(np.median(vals[i : i + RECENT])) for i in range(max(1, len(vals) - RECENT + 1))
    )
    declined = len(vals) >= 2 * RECENT and p_worse(vals[-RECENT:], vals[:-RECENT]) < ALPHA
    return now, best, len(vals), declined


def _live_text(pid, since, label, sign=1):
    now, best, n, declined = _live(pid, since)
    if n == 0:
        return "", False
    f = lambda x: f"{sign * x:.3f}"
    return f" · {label} now: {f(now)} · best: {f(best)} · runs since refresh: {n}", declined


DPNN_CKPT = Path("runs/hw")
_DPNN = {}  # checkpoint mtime -> model, so a retrain is picked up without a restart


def _dpnn_transfer(volts):
    """The direct 4x4's transfer as the DPNN predicts it -> (T, meta), or (None, why).

    The same readout path as `pic.apply._direct_4x4`: one port lit at a time, the stored dark
    subtracted, then Sinkhorn, so the DPNN bar and the Full bar differ only by the model."""
    ck = DPNN_CKPT / "ckpt.pt"
    if not ck.exists():
        return None, {"error": f"no DPNN checkpoint at {ck}: python -m learn.train_hw"}
    try:
        import torch

        from pic.model import DpnnModel
        from theory.calib import Calibration
        from theory.drift import sinkhorn

        key = ck.stat().st_mtime
        if key not in _DPNN:
            _DPNN.clear()
            _DPNN[key] = DpnnModel(str(DPNN_CKPT))
        m = _DPNN[key]
        dark = np.asarray(Calibration.load_or_nominal().pd_offset, float)[:4]
        T = np.stack([m.predict(volts, port=j)[:4] for j in range(4)], axis=1)
        T = np.clip(T - dark[:, None], 1e-12, None)
        T = sinkhorn(torch.as_tensor(T, dtype=torch.float64)).numpy()
        meta = json.loads((DPNN_CKPT / "meta.json").read_text())
        return T, {"source": meta.get("source"), "r2": meta.get("r2_dpnn")}
    except Exception as e:  # the run stands without it; the page says why the bar is missing
        return None, {"error": f"{type(e).__name__}: {e}"}


def _overlap(a, b) -> float:
    """|<a,b>| / (|a||b|): 1 when the chip points where the ideal does, whatever the scale.
    Every term is a reading, so unlike an amplitude fidelity the bench measures it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(a @ b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else float("nan")


def _read_json(path):
    p = Path(path)
    try:
        return json.loads(p.read_text()), p.stat().st_mtime, None
    except FileNotFoundError:
        return None, None, f"{p} is not there"
    except (OSError, ValueError) as e:
        return None, None, f"{p}: {type(e).__name__}: {e}"


def _reject_reason(fit, gates) -> str:
    """Which clause of `characterize.passes` this stored fit failed. The fit file records
    `ok` but not why, and "red" with no reason is a colour, not a diagnosis."""
    why = []
    if fit.get("amplitude", 0.0) < gates["min_amplitude"]:
        why.append(
            f"amplitude {1e3 * fit.get('amplitude', 0.0):.1f} mV under "
            f"{1e3 * gates['min_amplitude']:.0f} mV"
        )
    if (fit.get("snr") or 0.0) < gates["min_snr"]:
        why.append(f"SNR {fit.get('snr') or 0.0:.1f} under {gates['min_snr']:.0f}")
    if (fit.get("r2") or 0.0) < gates["min_r2"]:
        why.append(f"r2 {fit.get('r2') or 0.0:+.3f} under {gates['min_r2']:.2f}")
    if (fit.get("visibility") or 0.0) < gates["min_visibility"]:
        why.append(
            f"visibility {fit.get('visibility') or 0.0:.3f} under " f"{gates['min_visibility']:.2f}"
        )
    return "rejected by characterize.passes: " + ("; ".join(why) or "no clause recorded")


def _blind():
    from theory.clements import COLUMN
    from theory.layout import PHI_DAC

    return {d for k, d in PHI_DAC.items() if COLUMN[k] == 0}


_BLIND = _blind()  # the column-0 input phases, invisible behind a 1x4 switch


def _pd_noise(j):
    res, _, _ = _read_json(CHAR_PATH)
    n = [f["noise_v"] for f in (res or {}).values() if f.get("pd") == j and f.get("noise_v")]
    return float(np.median(n)) if n else None


def heater_status(fit, nominal, gates):
    """Colour for one heater, and the sentence behind it.

    Every threshold is `pic.characterize`'s own -- `passes.__defaults__` for the gate,
    `STRONG_SNR`/`STRONG_R2` for its override branch, `VPI_MIN`/`VPI_MAX` and
    `resolvable_span` for the band. Nothing here is a number this file invented."""
    if fit is None:
        return (
            ("grey", "not calibrated: no fringe fit on record")
            if nominal
            else ("yellow", "Vpi on file with no fringe fit behind it")
        )
    if not fit.get("ok"):
        return "red", _reject_reason(fit, gates)
    vpi = float(fit.get("vpi") or 0.0)
    if not gates["vpi_min"] <= vpi <= gates["vpi_max"]:
        return "red", (
            f"fitted Vpi {vpi:.2f} V outside the physical band "
            f"{gates['vpi_min']}-{gates['vpi_max']} V"
        )
    n = len(fit.get("levels") or ())
    span, res = fit.get("span_pi") or 0.0, gates["resolvable"](n) if n else None
    if res is not None and span > res:
        return "red", (
            f"span {span:.1f} pi over the {res:.1f} pi that {n} levels resolve: "
            f"an alias, however good the residual"
        )
    snr, r2 = fit.get("snr") or 0.0, fit.get("r2") or 0.0
    if snr >= gates["strong_snr"] and r2 >= gates["strong_r2"]:
        return "green", f"SNR {snr:.0f}, r2 {r2:+.4f}, span {span:.2f} pi"
    short = []
    if snr < gates["strong_snr"]:
        short.append(f"SNR {snr:.1f} under {gates['strong_snr']:.0f}")
    if r2 < gates["strong_r2"]:
        short.append(f"r2 {r2:+.3f} under {gates['strong_r2']:.2f}")
    return "yellow", (
        f"marginal: passes on visibility "
        f"{fit.get('visibility') or 0.0:.3f}, but " + " and ".join(short)
    )


def pd_status(dark, full, contrast_db, dead, gates):
    """Colour for one photodiode. `dead` and the 1e-6 V floor are `pic.normalise`'s;
    the fringe floor and the SNR gate are `pic.characterize`'s, turned into the dB a
    detector needs before a fringe on it could clear them."""
    if dark is None or full is None:
        return "grey", "no normalisation on record for this detector"
    span = full - dark
    if dead or span <= 1e-6:
        return "red", f"dead: full scale minus dark is {1e3 * span:.4f} mV"
    if span < gates["min_amplitude"]:
        return "yellow", (
            f"full scale {1e3 * span:.1f} mV under the "
            f"{1e3 * gates['min_amplitude']:.0f} mV fringe floor: no fit on "
            f"this detector could clear the amplitude gate"
        )
    floor_db = 10.0 * math.log10(gates["min_snr"])
    if contrast_db is not None and contrast_db < floor_db:
        return "yellow", (
            f"contrast {contrast_db:.1f} dB under {floor_db:.1f} dB, which is "
            f"the SNR gate of {gates['min_snr']:.0f} expressed in dB"
        )
    return "green", (
        f"full {1e3 * full:.1f} mV over {1e3 * dark:.3f} mV dark, contrast " f"{contrast_db:.1f} dB"
        if contrast_db is not None
        else f"full {1e3 * full:.1f} mV over {1e3 * dark:.3f} mV dark"
    )


def mesh_diagram():
    """Where every element and every heater sits, derived from the two maps that own it.

    `clements.MESH[k]` is the rail PAIR element k couples and `clements.COLUMN[k]` the
    rectangular column it sits in, so (column, rail pair) places an element completely.
    Heaters attach to ELEMENTS, never to a rail on their own: `layout.THETA_DAC[k]` drives
    element k's internal arm phase and belongs inside its box, and `layout.PHI_DAC[k]`
    drives the external phase on that element's UPPER input arm, `MESH[k][0]`, so it
    belongs on that rail just before the box. The two exceptions are not mesh elements at
    all -- a spare sits on a bare rail in the gap column (`layout.AUX_RAIL`), and an output
    trimmer sits past the last column on its own rail -- and are placed as such.

    Every index here is read out of `THETA_DAC`/`PHI_DAC`, so the picture cannot drift from
    the wiring the way a hand-placed one does; `_check_diagram` asserts the round trip."""
    from theory.clements import COLUMN, MESH, NMODE
    from theory.layout import AUX_GAP_COLUMN, AUX_RAIL, HEATERS, PHI_DAC, THETA_DAC

    ncol = len(set(COLUMN))
    elements = [
        {
            "k": k,
            "col": COLUMN[k],
            "rails": [m, n],
            "theta_dac": THETA_DAC[k],
            "phi_dac": PHI_DAC[k],
        }
        for k, (m, n) in enumerate(MESH)
    ]

    place = {}
    for h in HEATERS:
        if h.role == "theta":
            place[h.h] = {
                "kind": "theta",
                "elem": h.index,
                "col": COLUMN[h.index],
                "rails": list(MESH[h.index]),
            }
        elif h.role == "phi":
            place[h.h] = {
                "kind": "phi",
                "elem": h.index,
                "col": COLUMN[h.index],
                "rails": [MESH[h.index][0]],
            }
        elif h.role == "aux" and h.h in AUX_RAIL:
            place[h.h] = {
                "kind": "aux",
                "elem": -1,
                "col": AUX_GAP_COLUMN,
                "rails": [AUX_RAIL[h.h]],
            }
        elif h.role == "out_phase":
            # one column past the mesh. The old figure gave the screen `COLUMN`'s own count
            # as a column index and then multiplied by the column pitch, which lands on top
            # of the photodiodes; nothing draws there today only because `ALPHA_DAC` is
            # empty, and the page should not depend on that staying true.
            place[h.h] = {"kind": "out", "elem": -1, "col": ncol, "rails": [h.index]}
    return {"nmode": NMODE, "ncol": ncol, "elements": elements, "place": place}


def placement_text() -> str:
    """The derived placement as text, so the picture can be checked without looking at it."""
    from theory.layout import DAC_HEATER, MIRROR_OF

    def wires(d):
        ds = [d] + sorted(p for p, q in MIRROR_OF.items() if q == d)
        return f"{DAC_HEATER[d]:<4} DAC {'+'.join(map(str, ds)):<5}"

    d = mesh_diagram()
    by_dac = d["place"]
    out = [f"{d['nmode']} rails, {len(d['elements'])} elements, {d['ncol']} columns"]
    for e in sorted(d["elements"], key=lambda e: (e["col"], e["rails"])):
        out.append(
            f"  col {e['col']}  MZI{e['k']}  rails {e['rails'][0]}-{e['rails'][1]}"
            f"   theta {wires(e['theta_dac'])} in the box"
            f"   phi {wires(e['phi_dac'])} on rail {e['rails'][0]}"
        )
    for dac, p in sorted(by_dac.items()):
        if p["kind"] in ("aux", "out"):
            out.append(
                f"  col {p['col']}  {p['kind']:<5}           "
                f"       {wires(dac)} on rail {p['rails'][0]}"
            )
    return "\n".join(out)


def calib_state():
    """`pic_data/calib.json` and `char_results.json` joined onto the live heater map.

    The map is imported here rather than held at module scope so a sweep that edits
    `theory.layout` shows up on the next poll, and so this file never keeps a copy of the
    DAC-to-heater wiring -- a stale copy of that map is what once put `out_phase` labels on
    mid-mesh phi heaters."""
    from theory.clements import COLUMN, MESH, NMODE
    from theory.layout import HEATERS, MIRROR_OF
    from pic.characterize import (
        MIN_SNR,
        STRONG_R2,
        STRONG_SNR,
        VPI_MAX,
        VPI_MIN,
        passes,
        resolvable_span,
    )
    from pic.config import VOLTAGE_MAX_CH, VPI_NOMINAL

    min_vis, min_amp, min_r2 = passes.__defaults__
    gates = {
        "min_visibility": min_vis,
        "min_amplitude": min_amp,
        "min_r2": min_r2,
        "min_snr": MIN_SNR,
        "strong_snr": STRONG_SNR,
        "strong_r2": STRONG_R2,
        "vpi_min": VPI_MIN,
        "vpi_max": VPI_MAX,
        "resolvable": resolvable_span,
    }

    cal, cal_t, cal_err = _read_json(CALIB_PATH)
    char, char_t, _ = _read_json(CHAR_PATH)
    warnings = [cal_err] if cal_err else []
    cal = cal or {}
    meta = cal.get("meta") or {}
    norm = meta.get("normalisation") or {}
    vpi_v = list(cal.get("vpi") or [])
    phi0_v = list(cal.get("phi0") or [])

    pad_of = {h.h: h.pad for h in HEATERS}
    fits, remapped = {}, False
    if char:
        old_pad = {int(d): str(f.get("label", "")).split(":")[0] for d, f in char.items()}
        if all(p == pad_of.get(d) for d, p in old_pad.items() if p):
            fits = {int(d): dict(f) for d, f in char.items()}
        else:
            # The file predates the 2026-09-22 rewire, so its keys are the OLD DAC channels.
            # Rejoin it by heater pad, which is what `calib.json`'s own remap_note says was
            # done to vpi/phi0 -- keying it by channel would label DAC0 with H15's fringe.
            remapped = True
            by_pad = {}
            for d, fit in char.items():
                pad = str(fit.get("label", "")).split(":")[0]
                if pad:
                    by_pad.setdefault(pad, []).append((int(d), fit))
            for h in HEATERS:
                got = by_pad.get(h.pad) or []
                pick = next((x for x in got if x[1].get("ok")), got[0] if got else None)
                if pick:
                    fits[h.h] = dict(pick[1], from_dac=pick[0])

    dia = mesh_diagram()
    rows = []
    for h in HEATERS:
        if h.h in MIRROR_OF:  # one heater on two wires: drawn and listed once
            continue
        dacs = [h.h] + sorted(p for p, q in MIRROR_OF.items() if q == h.h)
        vpi = vpi_v[h.h] if h.h < len(vpi_v) else None
        phi0 = phi0_v[h.h] if h.h < len(phi0_v) else None
        nominal = vpi is not None and abs(vpi - VPI_NOMINAL) < 1e-9 and not phi0
        fit = fits.get(h.h)
        status, why = heater_status(fit, nominal, gates)
        if h.h in _BLIND:
            status, why = "grey", (
                "not observable: an input phase before a first-column MZI is a global phase "
                "when one port is lit at a time, so no detector sees it and no measurement "
                "this rig makes depends on it"
            )
        vmax = VOLTAGE_MAX_CH[h.h]
        # Span at the ceiling the rig clamps this channel to today, not the one the old
        # sweep happened to reach: a bonded pair sources twice the current into one heater,
        # so the stored span_pi understates what the same Vpi now buys.
        span = (vmax / vpi) ** 2 if (vpi and not nominal) else None
        # Worth a warning only when a number was actually adopted. A rejected fit sitting
        # beside a nominal Vpi is the system working; a rejected fit beside a fitted Vpi
        # means the calibration in force came from a sweep this file no longer holds.
        disagree = bool(
            fit
            and vpi
            and not (nominal and not fit.get("ok"))
            and abs(float(fit.get("vpi") or 0.0) - vpi) > 1e-6
        )
        note = (
            f"calibration holds {vpi:.3f} V; the stored fit says "
            f"{float(fit.get('vpi') or 0.0):.3f} V"
            if disagree
            else ("nominal placeholder, never fitted" if nominal else "")
        )
        rows.append(
            {
                "pad": h.pad,
                "dacs": dacs,
                "role": h.role,
                "index": h.index,
                "label": h.label,
                "column": h.column,
                "vmax": vmax,
                "vpi": vpi,
                "phi0": phi0,
                "span_pi": span,
                # the fit's one-sigma on Vpi, carried to today's span: d span / span = 2 dVpi/Vpi
                "span_sd": (
                    2 * span * fit["vpi_sd"] / vpi
                    if span is not None and vpi and (fit or {}).get("vpi_sd") is not None
                    else None
                ),
                "phi0_sd": (fit or {}).get("phi0_sd"),
                "nominal": nominal,
                "fit_span": (fit or {}).get("span_pi"),
                "fit_vpi": (fit or {}).get("vpi"),
                "r2": (fit or {}).get("r2"),
                "snr": (fit or {}).get("snr"),
                "visibility": (fit or {}).get("visibility"),
                "amplitude": (fit or {}).get("amplitude"),
                "pd": (fit or {}).get("pd"),
                "port": (fit or {}).get("port"),
                "fit_label": (fit or {}).get("label"),
                "from_dac": (fit or {}).get("from_dac"),
                "fitted_this_run": h.h in (meta.get("fitted_this_run") or []),
                "status": status,
                "why": why,
                "disagree": disagree,
                "vpi_note": note,
                "place": dia["place"].get(h.h),
            }
        )
    rows.sort(
        key=lambda r: (
            r["column"],
            0 if r["role"] == "phi" else 1,
            r["index"] if r["index"] >= 0 else 99,
        )
    )

    dead = set(norm.get("dead_pds") or [])
    pds = []
    for j in range(NMODE):
        g = lambda k: (norm.get(k) or [None] * NMODE)[j] if j < len(norm.get(k) or []) else None
        dark, full = g("dark_v"), g("full_v")
        st, why = pd_status(dark, full, g("contrast_db"), j in dead, gates)
        pds.append(
            {
                "i": j,
                "dark_v": dark,
                "full_v": full,
                "contrast_db": g("contrast_db"),
                "port_full_v": g("port_full_v"),
                # read noise as the last sweep measured it, from the fits judged against it
                "noise_v": _pd_noise(j),
                "input_scale": g("input_scale"),
                "gain": (cal.get("pd_gain") or [None] * NMODE)[j],
                "offset": (cal.get("pd_offset") or [None] * NMODE)[j],
                "status": st,
                "why": why,
            }
        )

    counts = {k: 0 for k in ("ok", "weak", "broken", "grey")}
    key = {"green": "ok", "yellow": "weak", "red": "broken", "grey": "grey"}
    for r in rows + pds:
        counts[key[r["status"]]] += 1
    busy = sweeps_running()  # a sweep may be mid-write: the page locks rather than warns

    stamp = "  ".join(
        x
        for x in (f"calib {ago(cal_t)}" if cal_t else "", f"fits {ago(char_t)}" if char_t else "")
        if x
    )
    return {
        "heaters": rows,
        "pds": pds,
        "counts": counts,
        "warnings": warnings,
        "mesh": [list(p) for p in MESH],
        "column": list(COLUMN),
        "elements": dia["elements"],
        "ncol": dia["ncol"],
        "nmode": NMODE,
        "remapped": remapped,
        "meta": meta,
        "running": bool(busy),
        "stamp": stamp,
        "rev": f"{cal_t or 0:.0f}/{char_t or 0:.0f}/{bool(busy)}",
        "locked": bool(busy),
    }


def sweeps_running():
    """PIDs of every `python -m pic ...`.

    None when the process table cannot be read, and `/recal` treats that as busy: not knowing
    whether the board is held is not the same as knowing it is free."""
    import psutil

    me, out = os.getpid(), []
    try:
        for p in psutil.process_iter(["cmdline"]):
            if p.pid != me and _is_pic(p.info["cmdline"] or []):
                out.append(p.pid)
    except psutil.Error:
        return None
    return out


def _int(v, name):
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be whole numbers, got {v!r}") from None


JOB = {}  # the one calibration subprocess this page started: p, cmd, what, t0, log, mock
LOG_DIR = Path("pic_data/logs")


_FINISHED = {"fit", "recal"}  # a heater's result
_LIVE = {"prescan", "sweeping"}  # a heater being measured right now


def _job_events():
    """This job's own events, out of the shared stream every source feeds."""
    return EVENTS.since(0, None, job=JOB["id"]) if JOB else []


def _row_states(targets, evs, running):
    """pad -> queued | running | done, from the job's heater events: a fit or recal is done,
    and only the latest heater event can name the live one, and only while it measures."""
    hs = [e for e in evs if e.get("c") == "heater" and e.get("pad")]
    done = {e["pad"] for e in hs if e["s"] in _FINISHED}
    live = hs[-1]["pad"] if running and hs and hs[-1]["s"] in _LIVE else None
    return {p: "done" if p in done else "running" if p == live else "queued" for p in targets}


def _job_laser(evs):
    """Whether the job has the laser lit, from its own laser events. While a job holds the
    port this is the only honest answer, and "not yet" is not "on"."""
    sw = [e["s"] for e in evs if e.get("c") == "laser" and e.get("s") in ("on", "off")]
    return bool(sw) and sw[-1] == "on"


HISTORY = LOG_DIR / "history.json"  # how long each finished job took, for the next estimate


def _expected(kind, mock):
    """Median of this job's successful past durations on the same rig, or None."""
    h, _, _ = _read_json(HISTORY)
    s = [e["secs"] for e in (h or []) if e["kind"] == kind and e["mock"] == mock and e["rc"] == 0]
    return float(np.median(s)) if s else None


def _record(rc):
    h, _, _ = _read_json(HISTORY)
    h = (h or []) + [
        {
            "kind": _kind(JOB["cmd"]),
            "mock": JOB["mock"],
            "secs": round(time.time() - JOB["t0"]),
            "rc": rc,
            "log": JOB["log"].name,
        }
    ]
    HISTORY.write_text(json.dumps(h, indent=1))
    JOB["recorded"], JOB["t1"] = True, time.time()
    if not JOB["mock"]:
        # However the job ended -- finished, stopped, crashed, or its laser link dropped and
        # it could not switch the diode off itself -- the laser goes off here too.
        try:
            laser({"on": False})
        except Exception as e:
            ev(
                "server",
                "laser",
                f"job end: laser off failed ({_short(e)}); switch it off by hand",
                "error",
            )


def _row_fits(evs):
    """pad -> the fit a heater event reports, so the table can show it before the file is
    written. fastchar/char report vpi, phi0 and span; recal reports phi0 alone."""
    out = {}
    for e in evs:
        if e.get("c") != "heater" or e.get("s") not in _FINISHED or not e.get("pad"):
            continue
        if e["s"] == "recal":
            out[e["pad"]] = {"phi0_pi": e.get("phi0_pi"), "ok": True}
        else:
            out[e["pad"]] = {k: e.get(k) for k in ("vpi", "phi0_pi", "span")}
            out[e["pad"]]["ok"] = bool(e.get("ok"))
    return out


def _progress(kind, evs, rows):
    """0..1 from the job's own events, or None when it reports nothing countable."""

    def frac(c, *states):
        e = next(
            (e for e in reversed(evs) if e.get("c") == c and e.get("s") in states and e.get("n")),
            None,
        )
        return min(1.0, e["k"] / e["n"]) if e else 0.0

    done = sum(v == "done" for v in rows.values())
    if kind == "capture":
        return frac("table", "progress")
    if kind == "fastchar":  # prescan is the first half, the per-heater fits the second
        return 0.5 * frac("heater", "prescan") + 0.5 * (done / len(rows) if rows else 0.0)
    if kind in ("char", "recal"):
        return done / len(rows) if rows else None
    if kind == "learn.train_hw":
        # rounds finished (k of n on the round event) plus how far into this round it is
        r = next((e for e in reversed(evs) if e.get("c") == "dpnn" and e.get("s") == "round"), None)
        if not r or not r.get("n"):
            return 0.0
        i = evs.index(r)
        smp = [e for e in evs[i:] if e.get("c") == "dpnn" and e.get("s") == "sample" and e.get("n")]
        within = min(1.0, smp[-1]["k"] / smp[-1]["n"]) if smp else 0.0
        return min(1.0, (r["k"] + within) / r["n"])
    return None


def _stage(kind, evs, rows):
    """Where the job is, in its own units: the counts that make a percentage mean something."""
    last = lambda c, *s: next(
        (e for e in reversed(evs) if e.get("c") == c and e.get("s") in s and e.get("n")), None
    )
    if kind == "learn.train_hw":
        r = next((e for e in reversed(evs) if e.get("c") == "dpnn" and e.get("s") == "round"), None)
        if not r:
            return None
        smp = [e for e in evs[evs.index(r) :] if e.get("s") == "sample" and e.get("n")]
        pts = f" · {smp[-1]['k']}/{smp[-1]['n']} pts" if smp else ""
        return f"round {r.get('round', r['k'] + 1)}/{r['n']}{pts}"
    if kind == "capture":
        e = last("table", "progress")
        return e and f"{e['k']}/{e['n']} states"
    if kind in ("fastchar", "char", "recal"):
        done = sum(v == "done" for v in rows.values())
        e = last("heater", "prescan")
        if e and not done:
            return f"prescan {e['k']}/{e['n']}"
        return rows and f"fits {done}/{len(rows)}"
    return None


def _waiting(evs):
    """Blocked at the TEC gate, not working: the session's latest state says so."""
    s = [e["s"] for e in evs if e.get("c") == "session" and e.get("s") in _SESSION]
    return bool(s) and s[-1] in ("waiting", "paused")


_SESSION = {"waiting", "lit", "paused", "resumed", "closed", "tripped"}


def job_state():
    """What the calibration job is doing. Every page polls this, and Run, the diagnostics and
    the laser box key off it, so a sweep that holds the board or the laser is visible
    everywhere rather than discovered as a port error."""
    out = {"running": False}
    if JOB:
        rc = JOB["p"].poll()
        if rc is not None and not JOB.get("recorded"):
            _record(rc)
        evs = _job_events()
        run = rc is None
        out = {
            "running": rc is None,
            "rc": rc,
            "cmd": JOB["cmd"],
            "what": JOB["what"],
            "level": JOB.get("level"),
            "mock": JOB["mock"],
            "elapsed": round((JOB.get("t1") or time.time()) - JOB["t0"], 1),
            "expected": _expected(_kind(JOB["cmd"]), JOB["mock"]),
            "tail": [e["text"] for e in evs[-12:]],
            "log": str(JOB["log"]),
            "rows": _row_states(JOB["targets"], evs, run),
            "progress": _progress(_kind(JOB["cmd"]), evs, _row_states(JOB["targets"], evs, False)),
            "stage": _stage(_kind(JOB["cmd"]), evs, _row_states(JOB["targets"], evs, False)),
            # the job is blocked at the TEC gate, not working: the page says so, not a timer
            "waiting": run and _waiting(evs),
            "fits": _row_fits(evs) if run else {},
        }
    if CHAIN:
        out["chain"] = {k: CHAIN.get(k) for k in ("running", "at", "steps", "failed")}
    others = [p for p in (sweeps_running() or []) if not JOB or p != JOB["p"].pid]
    if others:
        out["external"] = others  # a pic process this page did not start, e.g. a terminal
    return out


def job_stop():
    """Stop the job, then make the bench safe: the CLI's own cleanup may not have run."""
    CHAIN["stop"] = True  # Stop ends the whole sequence, not just this step
    if not JOB or JOB["p"].poll() is not None:
        return job_state()
    # an interrupt first, like Ctrl-C, so the CLI's own cleanup runs (laser off, DACs to 0);
    # SIGTERM would skip every `finally`. Windows has no SIGINT for a child: terminate.
    p = JOB["p"]
    p.send_signal(signal.SIGINT) if os.name != "nt" else p.terminate()
    for step in (p.terminate, p.kill):
        try:
            p.wait(10)
            break
        except subprocess.TimeoutExpired:
            step()
    p.wait(5)
    if not JOB["mock"]:
        try:
            laser({"on": False})
        except Exception:
            pass  # reported by the laser box's next poll
    return job_state()


def _targets(cmd):
    """The rows a calibration command will touch, so the tables can show them queued."""
    from pic.layout import LABEL_OF_DAC
    from theory.layout import MIRROR_OF

    pad = lambda d: LABEL_OF_DAC[d].split(":")[0]
    m = re.search(r"--channels (\S+)", cmd)
    if m:
        return sorted({pad(int(c)) for c in m.group(1).split(",")})
    m = re.search(r"--pd (\d)", cmd)
    if m:
        return [f"PD{m.group(1)}"]
    if " fastchar" in cmd:
        return sorted({pad(d) for d in LABEL_OF_DAC if d not in MIRROR_OF})
    if " recal" in cmd:
        res, _, _ = _read_json(CHAR_PATH)
        return sorted({pad(int(k)) for k, f in (res or {}).items() if f.get("ok")})
    return []  # sync touches the transfer table, not a row


def _tec_ready(what):
    """Refuse a bench job the TEC would only make wait: every one of them opens a laser
    session, and the session will not light until the die is in band."""
    from pic.config import TEC_SETPOINT_C, TEC_TOLERANCE_C

    try:
        t, TEC_SETPOINT_C = TEC.mean()[:2]
    except Exception:
        return  # the job's own gate reports a TEC it cannot read
    if abs(t - TEC_SETPOINT_C) > TEC_TOLERANCE_C:
        raise Busy(
            f"not starting {what}: TEC: die at {t:.2f} C, outside {TEC_SETPOINT_C:.2f} ± "
            f"{TEC_TOLERANCE_C:.2f} C, and the job would only wait for it. Start it once the "
            "header turns green."
        )


def _kind(cmd):
    """`python -m pic fastchar ...` -> fastchar; `python -m learn.train_hw ...` -> learn.train_hw."""
    p = cmd.split()
    return p[3] if p[2] == "pic" else p[2]


def _start_job(cmd, what, mock):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_kind(cmd)}.log"
    fh = open(log, "w")
    p = subprocess.Popen(
        # UTF-8 mode: logs and JSON carry ±, °, → and φ; Windows' cp1252 default mangles them
        [sys.executable, "-X", "utf8", "-u", *cmd.split()[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # a server started in the background passes SIGINT down as ignored, and then
        # Stop could never interrupt the job cleanly; give the child the default back
        preexec_fn=(
            (lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)) if os.name != "nt" else None
        ),
    )
    src, jid = _kind(cmd), f"{log.stem}-{p.pid}"

    def pump():  # the raw log for the file view, and every line as an event for the Logs tab
        with fh:
            for ln in p.stdout:
                fh.write(ln)
                fh.flush()
                EVENTS.feed(src, ln, job=jid)

    threading.Thread(target=pump, daemon=True, name=f"log-{src}").start()
    if JOB and not JOB.get("recorded") and JOB["p"].poll() is not None:
        _record(JOB["p"].poll())  # a chain can start the next job before anyone polled
    JOB.clear()
    JOB.update(
        p=p, id=jid, cmd=cmd, what=what, t0=time.time(), log=log, mock=mock, targets=_targets(cmd)
    )
    return {"ok": True, "cmd": cmd, "pid": p.pid}


CHAIN_LEVELS = ("sweep", "pd", "dpnn", "table")
CHAIN = {}  # the running sequence: steps, the step it is on, and whether Stop was pressed


def _chain(mock):
    """Every predictor refreshed in order, each step only after the last one succeeded."""
    if CHAIN.get("running"):
        raise Busy("a calibration sequence is already running")
    first = recal_request({"level": CHAIN_LEVELS[0], "mock": mock})  # all the gates, once

    def run():
        try:
            for i, lv in enumerate(CHAIN_LEVELS):
                CHAIN["at"] = i
                if i:
                    recal_request({"level": lv, "mock": mock})
                rc = JOB["p"].wait()
                if rc != 0 or CHAIN.get("stop"):
                    CHAIN["failed"] = f"stopped at {lv} (exit {rc})"
                    return
        except Exception as e:  # a gate that refused mid-sequence ends it, and says why
            CHAIN["failed"] = f"{type(e).__name__}: {e}"
        finally:
            CHAIN["running"] = False

    CHAIN.clear()
    CHAIN.update(running=True, at=0, steps=list(CHAIN_LEVELS))
    threading.Thread(target=run, daemon=True).start()
    return {**first, "chain": list(CHAIN_LEVELS)}


def recal_request(payload):
    """Compose the recharacterisation this button asks for, and refuse it.

    Two gates, in this order. A sweep already holding the board is the one that matters --
    two processes on one serial port is the failure `pic.session` exists to prevent, and a
    reload button is exactly how a second one gets started. `ARMED` is the second: the
    endpoint drives current through a heater, so it stays inert until `--arm` says the bench
    is fit to be driven."""
    level = payload.get("level")
    if level == "all":
        return _chain(payload.get("mock") is True)
    if level is not None:
        cmd, what = {
            "sweep": ("python -m pic fastchar --write", "full sweep"),
            "recenter": ("python -m pic recal --write", "recenter"),
            "table": ("python -m pic sync", "table sync"),
            "capture": ("python -m pic capture --states 200", "table recapture"),
            # the detectors with every heater at 0 V: dark level and full scale per PD
            "pd": ("python -m pic calibrate --write", "PD sweep"),
            # physics with input gain and rank-2 crosstalk: held-out R^2 0.95 at 200 points,
            # 0.965 at 400 (bench, 2026-09-25); a checkpoint at 200, done at 400
            "dpnn": ("python -m learn.train_hw --rounds 2 --n-per-round 200", "DPNN training"),
        }.get(level, (None, None))
        if cmd is None:
            raise ValueError(f"no such calibration level: {level!r}")
    elif "pd" in payload:
        j = _int(payload["pd"], "pd")
        if not 0 <= j < 4:
            raise ValueError(f"no such detector: PD{j}")
        cmd = f"python -m pic char --pd {j} --write"
        what = f"PD{j}"
    else:
        chans = [
            _int(c, "channels") for c in str(payload.get("channels", "")).split(",") if c.strip()
        ]
        if not chans or any(not 0 <= c < 16 for c in chans):
            raise ValueError(f"channels must be 0-15, got {payload.get('channels')!r}")
        cmd = f"python -m pic char --channels {','.join(map(str, chans))} --write"
        what = f"DAC {'+'.join(map(str, chans))}"

    if JOB and JOB["p"].poll() is None:
        raise Busy(f"refusing to start {what}: {JOB['what']} is still running ({JOB['cmd']})")
    if payload.get("mock") is True:
        # a mock run must never overwrite the bench's calibration files
        cmd = cmd.replace(" --write", "") + " --mock"
        cmd += " --out runs/mock" if "train_hw" in cmd else ""  # never over the bench model
        cmd += " --dry-run" if " sync" in cmd else ""
        r = _start_job(cmd, what, True)
        JOB["level"] = level
        return r
    busy = sweeps_running()
    if busy is None:
        raise Busy(
            f"refusing to re-characterize {what}: cannot read the process table, so nothing here "
            f"can prove the board is free. Check by hand, then run:\n  {cmd}"
        )
    if busy:
        raise Busy(
            f"refusing to re-characterize {what}: a sweep already holds the board "
            f"(pid {', '.join(map(str, busy))}). Two processes on one serial port is "
            f"the failure pic.session exists to prevent. Wait for it, then run:\n"
            f"  {cmd}"
        )
    if not ARMED:
        raise Busy(
            f"refusing to re-characterize {what}: this server was started without "
            f"--arm, so it will not drive a heater. Run it yourself:\n  {cmd}"
        )
    _tec_ready(what)
    r = _start_job(cmd, what, False)
    JOB["level"] = level
    return r


PD_READS = 24  # frames behind each detector's dark level and its read noise
HOLD_S = 30  # how long the rail test holds the heaters at their clamp
HOLD_MAX_S = 60
TEC_RAIL_V = 4.99  # |drive| at or above this is the LT8722 at DAC_MAX in tec_pid.ino, 5.0
TEC_BASE_S = 4.0  # TEC telemetry read before the heaters are railed

# id -> row. Order here is the order on the page. `auto` is whether "run all" includes it:
# the TEC row resets the controller holding the die, and the DAC row is a question put to
# the operator, so neither may be swept up by a single click.
CHECKS = {
    "ports": {
        "name": "Serial ports",
        "auto": True,
        "tip": "USB devices by chip id (pic.config.USB_SERIAL); pass an id to any --*-port flag. The rig has three -- the "
        "board, the TEC controller and the laser's FTDI",
    },
    "board": {
        "name": "Board firmware",
        "auto": True,
        "tip": "boot banner plus the `C` capability line, against NUM_DAC, "
        "NUM_ADC_RAW and ADC_REF_V in pic/config.py. Opening the board leaves "
        "every DAC channel at 0 V",
    },
    "switch": {
        "name": "Optical switch",
        "auto": True,
        "tip": "`Q` passes POS through from the Sercalo; set a port with P<n> and "
        "re-read it. Moves the mirror, and puts it back",
    },
    "rail": {
        "name": "Power test",
        "auto": False,
        "tip": f"every heater at its clamp for {HOLD_S} s while the TEC is watched: "
        "maximum heater and TEC power at once. Fails if the die climbs out of "
        "the calibration band",
    },
    "laser": {
        "name": "Laser",
        "auto": True,
        "tip": "key, interlocks, driver enable, setpoint and measured diode current. "
        "Read-only: nothing here enables the diode, and laser_status == 1 "
        "does not prove emission -- only a photodiode rise does",
    },
    "dpnn": {
        "name": "DPNN",
        "auto": True,
        "tip": f"the surrogate at {DPNN_CKPT}: where it was trained, its R², its age against "
        "the calibration, and one inference (all four ports at 0 V) that must come back finite",
    },
    "table": {
        "name": "Transfer table",
        "auto": True,
        "tip": "the measured transfer table the tiled matvec and ES plan from: how many states, "
        "how old, and one plan of a random 2x2 rotation from it, scored against its target",
    },
    "pds": {
        "name": "Photodiodes",
        "auto": True,
        "tip": "dark level and read noise per detector with the switch dumped, against "
        "the stored full scale; a detector whose noise is comparable to its "
        "signal can carry no fringe that clears pic.characterize's SNR gate",
    },
}

# Every one of these needs the board open, so they share one open rather than resetting the
# Arduino once per row.
BOARD_CHECKS = ("board", "switch", "pds")
STATUSES = ("pass", "warn", "fail", "unknown")


def _row(i, status, value, tip=None):
    c = CHECKS[i]
    return {
        "id": i,
        "name": c["name"],
        "status": status,
        "value": str(value),
        "tip": c["tip"] if tip is None else tip,
        "auto": bool(c.get("auto")),
    }


def _short(e) -> str:
    return f"{type(e).__name__}: {e}".splitlines()[0][:220]


def _board_guard(mock: bool):
    """Why a hardware check must not run right now, or None.

    The same rule and the same anchored pattern as `/recal`: two processes on one serial
    port is the failure `pic.session` exists to prevent, and a page with a reload button on
    every row is exactly how a second one gets started. Not knowing whether the board is
    held counts as held."""
    if mock:
        return None
    if _BOARD_LOCK.locked():
        return "another request from this page is using the board"
    busy = sweeps_running()
    if busy is None:
        return "cannot read the process table, so nothing here can prove the board is free"
    if busy:
        return f"a pic process holds the board (pid {', '.join(map(str, busy))})"
    return None


def _open_board(mock: bool):
    """The board, open, plus the boot banner its own `open()` throws away.

    `PIC.open` sleeps out the reset and flushes, which is right for a measurement and wrong
    here: the banner is the only evidence of *which* sketch answered, and the TEC controller
    is the same Mega on the same glob. It has to be read on this open, because macOS pulses
    DTR only on the first open after the device enumerates -- a second look to fetch it
    comes back silent and reads exactly like a dead board."""
    from pic.interface import MockPIC
    from pic.config import READY_BANNER, RESET_WAIT_S

    if mock:
        from pic.devices.switch import MockSwitch

        sw = MockSwitch().open()
        return MockPIC(switch=sw).open(), READY_BANNER

    import serial

    from pic.devices.laser import Laser
    from pic.interface import PIC
    from pic.session import _resolve_pic_port

    try:
        laser_port = Laser.autodetect()  # a glob, not an open: it must not take the port
    except Exception:
        laser_port = None
    port = _resolve_pic_port(None, laser_port)
    b = PIC(port=port)
    b.cfg.port = port
    ser = serial.Serial(port, b.cfg.baud, timeout=b.cfg.timeout_s)
    banner, deadline = "", time.time() + 3 * RESET_WAIT_S
    while time.time() < deadline and READY_BANNER not in banner:
        time.sleep(0.2)
        banner += ser.read(ser.in_waiting or 0).decode("ascii", "replace")
    ser.reset_input_buffer()
    b.ser = ser
    return b, banner


def _line(board, want: str, timeout_s: float = None):
    """Read replies until one starts with `want`. None on timeout, never an exception."""
    ser = board.ser
    deadline = time.time() + (board.cfg.timeout_s if timeout_s is None else timeout_s)
    while time.time() < deadline:
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if raw.startswith(want):
            return raw
    return None


def _ck_ports(mock: bool):
    """Which serial devices are on the bus. A glob, so it opens nothing and holds nothing.

    The one row the mock toggle does not reach: whether the cables are in is a fact about
    the bench, and a mock run that reported them present would be reporting nothing."""
    from pic.config import usb_ports

    found = usb_ports()
    from pic.config import _serial_of

    # the chip id, not the device path: the id survives a replug, the path does not
    val = " · ".join(f"{r}: {_serial_of(d) or Path(d).name}" for d, r, _ in found)
    if not found:
        return _row("ports", "fail", "nothing plugged in")
    missing = sorted({"board", "tec", "laser"} - {r for _, r, _ in found})
    if missing:
        return _row("ports", "warn", (val + " · " if val else "") + "missing " + ", ".join(missing))
    return _row("ports", "pass", val)


def _ck_board(board, banner, mock):
    """Banner plus `C`, against the widths the host commands against.

    The three capability numbers are checked and not merely printed: CLAUDE.md's invariant
    is that NUM_DAC, NUM_ADC_RAW and ADC_REF_V match the firmware, and a host reading four
    ADC values off a board sending eight hangs rather than errors."""
    from pic.config import ADC_REF_V, NUM_ADC_RAW, NUM_DAC, READY_BANNER

    caps = board.capabilities(refresh=True)
    ready = READY_BANNER in (banner or "")
    val = "ready" if ready else "no boot banner"
    # the row shows the five the host commands against; the tip keeps the whole CAP line,
    # including the sweep limits `pic.acquisition` sizes a batched read from
    tip = CHECKS["board"]["tip"] + " · CAP " + " ".join(f"{k}={v}" for k, v in caps.items())
    row = lambda st, v: _row("board", st, v, tip)
    bad = []
    if caps.get("dac") not in (None, NUM_DAC):
        bad.append(f"dac {caps['dac']} vs NUM_DAC {NUM_DAC}")
    if caps.get("pins") not in (None, NUM_ADC_RAW):
        bad.append(f"pins {caps['pins']} vs NUM_ADC_RAW {NUM_ADC_RAW}")
    try:
        if "adcref" in caps and abs(float(caps["adcref"]) - ADC_REF_V) > 0.01:
            bad.append(f"adcref {caps['adcref']} vs ADC_REF_V {ADC_REF_V}")
    except (TypeError, ValueError):
        bad.append(f"adcref {caps['adcref']!r} unparseable")
    if bad:
        return row("fail", val + " · " + "; ".join(bad))
    if not ready:
        return row("fail", val)
    if not caps:
        return row("warn", val + " · firmware too old to report capabilities")
    if not caps.get("sweep"):
        return row("warn", val + " · no batched sweep")
    return row("pass", val)


def _sketch_vmax():
    """The firmware's clamp table read out of the sketch source -> (table, error).

    On the bench the board answers for itself. With nothing plugged in there is still a
    second copy to disagree with, and it is the copy that gets flashed -- so the invariant
    is checkable with no hardware, which is where a stale table should be caught."""
    try:
        src = SKETCH_PATH.read_text()
    except OSError as e:
        return None, f"{SKETCH_PATH}: {type(e).__name__}"
    m = re.search(r"VMAX\s*\[\s*NUM_DAC\s*\]\s*=\s*\{([^}]*)\}", src)
    if not m:
        return None, f"no `const float VMAX[NUM_DAC] = {{...}}` in {SKETCH_PATH}"
    try:
        return [float(x) for x in m.group(1).split(",") if x.strip()], None
    except ValueError as e:
        return None, f"{SKETCH_PATH}: {e}"


def _select(board, port: int):
    """Move the mirror. `port < 0` is the Sercalo's open channel, which is optically dark."""
    if not hasattr(board.ser, "write"):  # mock: no firmware to relay through
        sw = getattr(board, "switch", None)
        if sw is None:
            raise RuntimeError("mock board has no switch")
        return sw.dark() if port < 0 else sw.select(port)
    return board.select_port(port)


def _pos(board):
    """The mirror's OWN answer to POS -> (port or None, raw text, answered).

    `P<n>` is echoed by the firmware whether or not the Sercalo heard it -- `selectPort`
    discards the unit's replies -- so the echo proves nothing. `Q` relays POS verbatim, and
    silence there is itself the diagnosis: unpowered, unplugged and parked open all look
    identical from a `PORT` echo."""
    if not hasattr(board.ser, "write"):
        sw = getattr(board, "switch", None)
        p = None if sw is None else sw.position()
        return p, f"POS {0 if p is None else p + 1}", True
    board.ser.reset_input_buffer()
    board.ser.write(b"Q\n")
    raw = _line(board, "SWITCH", timeout_s=board.cfg.timeout_s + 2.0)
    if raw is None:
        return None, "(no SWITCH line)", False
    body = raw[len("SWITCH") :].strip()
    if not body or "no reply" in body:
        return None, body or "(empty)", False
    m = re.findall(r"-?\d+", body)
    if not m:
        return None, body, False
    n = int(m[-1])
    return (None if n == 0 else n - 1), body, True


def _ck_switch(board, banner, mock):
    """Ask where the mirror is, move it, and ask again. A port that does not track is a
    whole sweep silently taken at one position."""
    from theory.clements import NMODE

    p0, raw0, ok0 = _pos(board)
    if not ok0:
        return _row("switch", "fail", f"Q -> {raw0}: the Sercalo did not answer POS")
    tgt = (0 if p0 != 0 else 1) % NMODE
    try:
        _select(board, tgt)
        p1, raw1, ok1 = _pos(board)
    finally:
        try:
            _select(board, p0 if p0 is not None else -1)
        except Exception:
            pass
    val = f"path: P{p0} → P{tgt} → P{p0}"
    if not ok1:
        return _row("switch", "fail", f"no position after moving (Q -> {raw1})")
    if p1 != tgt:
        return _row("switch", "fail", f"asked P{tgt}, mirror at P{p1} · path: P{p0} → P{p1}")
    return _row("switch", "pass", val)


def _ck_pds(board, banner, mock):
    """Dark level and read noise per detector, with the optical path dumped.

    The gate is not a number invented here: a detector can only carry a fringe that clears
    `characterize.MIN_SNR` if its own full scale stands that far above its own read noise,
    so full-scale-over-noise is the ceiling on every fit that detector will ever produce."""
    from pic.characterize import MIN_SNR, STRONG_SNR
    from pic.config import ADC_BITS, ADC_REF_V, ADC_SAT_V, NUM_DAC, out_mask

    was, _, _ = _pos(board)
    m = out_mask()
    try:
        _select(board, -1)  # the Sercalo's open channel: dark without moving the laser
        y = np.array([board.measure_raw(np.zeros(NUM_DAC)) for _ in range(PD_READS)])[:, m]
    finally:
        try:
            _select(board, was if was is not None else -1)
        except Exception:
            pass
    dark, sd = y.mean(0), y.std(0, ddof=1)
    val = (
        "mV: "
        + "|".join(f"PD{i}" for i in range(len(dark)))
        + " · dark: "
        + "|".join(f"{1e3 * v:.2f}" for v in dark)
        + " · noise: "
        + "|".join(f"{1e3 * v:.2f}" for v in sd)
    )
    # a constant read at 0 is a dark level under one ADC step, not a dead detector
    lsb = ADC_REF_V / (2**ADC_BITS - 1)
    flat = np.flatnonzero((sd <= 0) & (np.abs(dark) >= lsb))
    if flat.size:
        return _row("pds", "fail", f"{val} · PD{flat[0]} stuck at one value")
    hot = np.flatnonzero(dark >= ADC_SAT_V)
    if hot.size:
        return _row("pds", "fail", f"{val} · PD{hot[0]} clips in the dark")

    cal, _, _ = _read_json(CALIB_PATH)
    norm = ((cal or {}).get("meta") or {}).get("normalisation") or {}
    full = np.asarray(norm.get("full_v") or [], float)
    ref = np.asarray(norm.get("dark_v") or [], float)
    if full.size != m.sum() or ref.size != full.size:
        return _row("pds", "warn", val + " · no full scale on file to compare it against")
    snr = (full - ref) / np.maximum(sd, lsb)  # under one ADC step reads as 0, not as silence
    j = int(np.argmin(snr))
    if snr[j] < MIN_SNR:
        return _row("pds", "fail", val + f" · PD{j} signal/noise {snr[j]:.0f}, gate {MIN_SNR:.0f}")
    if snr[j] < STRONG_SNR:
        return _row("pds", "warn", val + f" · PD{j} signal/noise {snr[j]:.0f}, weak fits")
    return _row("pds", "pass", val)


def _ck_laser(mock):
    with _LASER_LOCK:
        return _ck_laser_open(mock)


def _ck_laser_open(mock):
    """Everything the diode will tell us without being enabled.

    Read-only by construction: `Laser.open()` opens a serial port and `close()` leaves the
    laser state exactly where it was. An enabled diode is never a pass -- `laser_status == 1`
    is reported while the front panel sits at READY and no light leaves the fibre, so the
    only witness that counts is a photodiode rise, which is `pic.session.laser_session`'s
    job and not this row's."""
    from pic.devices.laser import Laser
    from pic.devices.mock import MockLaser

    try:
        L = MockLaser() if mock else Laser()
        L.open()
    except Exception as e:  # no FTDI, a held port, a refused open
        return _row("laser", "fail", _short(e))
    try:
        pre = L.preflight()
        d = L.dev
        # both enables, separately: `is_on` ORs them, and which one is latched is the
        # difference between a diode that is armed and one the master has released
        mst = int(d.read_setting("laser_status"))
        cw = int(d.read_setting("cw_laser_status"))
        mA = float(L.measured_mA())
    except Exception as e:
        return _row("laser", "fail", _short(e))
    finally:
        try:
            L.close()
        except Exception:
            pass
    key = int(pre["key switch ON"])
    bnc = int(pre["BNC interlock closed"])
    ext = int(pre["EXT interlock closed"])
    on = bool(mst or cw)
    val = f"{'on' if on else 'off'} · key: {'on' if key else 'off'} · current: {mA:.1f} mA"
    if not key:
        return _row("laser", "fail", val + " · key switch off")
    if not (bnc and ext):
        return _row("laser", "fail", val + " · interlock open")
    # driver_enable reads 1 whenever the key is on, so only master/CW mean light is wanted
    if on:
        return _row("laser", "warn", val + " · emission unproven from here")
    return _row("laser", "pass", val)


def _ck_heaters():
    """The heater calibration as a predictor: how much of it is fitted, how well, how old."""
    from theory.layout import ACTIVE_IDX

    cal, cal_t, err = _read_json(CALIB_PATH)
    if cal is None:
        return "fail", err
    res, _, _ = _read_json(CHAR_PATH)
    fits = [(res or {}).get(str(int(d))) or {} for d in ACTIVE_IDX]
    ok = [f for f in fits if f.get("ok")]
    score = float(np.median([f["score"] for f in ok])) if ok else float("nan")
    val = (
        f"{len(ok)}/{len(ACTIVE_IDX)} heaters fitted · last sweep: {ago(cal_t)} · "
        f"fit residual: {score:.2f} of the fringe"
    )
    chip = (cal.get("meta") or {}).get("chip_c")
    if chip is not None:
        val += f" · calibrated at: {float(chip):.1f} C"
    live, declined = _live_text("heaters", cal_t, "fidelity")
    return ("warn" if declined else "pass"), val + live


# the button a predictor's state calls for; the page highlights it instead of saying so
_REC = {
    # a measured decline: recenter first, it is the cheap fix for drift; a failure: sweep
    "heaters": lambda st, v: "recenter" if st == "warn" else "sweep" if st == "fail" else None,
    "table": lambda st, v: (
        "capture" if "worse than answering zero" in v else "table" if st != "pass" else None
    ),
    "dpnn": lambda st, v: "dpnn" if st != "pass" else None,
    "pds": lambda st, v: "pd" if st != "pass" else None,
}


def _ck_pds_pred():
    """The detectors as a predictor: how many pass their own check, and how fresh."""
    cal, _, err = _read_json(CALIB_PATH)
    if cal is None:
        return "fail", err
    pds = calib_state()["pds"]
    ok = sum(p["status"] == "green" for p in pds)
    when = (cal.get("meta") or {}).get("calibrated_at")
    val = f"{ok}/{len(pds)} detectors healthy · last PD sweep: {when or 'never'}"
    if ok < len(pds):
        return "fail", val
    return ("warn" if not when or when[:10] != time.strftime("%Y-%m-%d") else "pass"), val


def predictors():
    """Each thing a run predicts from, with its age, best known error and what to do next.
    The same checks the diagnostics run, so the two tabs cannot disagree."""
    st, val = _ck_heaters()
    out = [{"id": "heaters", "name": "Heaters", "status": st, "value": val}]
    pst, pval = _ck_pds_pred()
    out.append({"id": "pds", "name": "Photodiodes", "status": pst, "value": pval})
    for i in ("table", "dpnn"):
        r = _SOLO_FN[i](False)
        out.append({"id": i, "name": r["name"], "status": r["status"], "value": r["value"]})
    for p in out:
        p["rec"] = _REC[p["id"]](p["status"], p["value"])
    return out


# level -> the job it runs, for "how long will this take": the median of that job's past
# successful bench runs, so a planner (or a demo) knows before pressing
_LEVEL_KIND = {
    "sweep": "fastchar",
    "recenter": "recal",
    "pd": "calibrate",
    "table": "sync",
    "capture": "capture",
    "dpnn": "learn.train_hw",
}


def estimates():
    est = {lv: _expected(k, False) for lv, k in _LEVEL_KIND.items()}
    known = [est[lv] for lv in CHAIN_LEVELS]
    est["all"] = sum(known) if all(x is not None for x in known) else None
    return est


def _ck_dpnn(mock):
    """Is there a DPNN worth believing, and does it answer?"""

    t0 = time.time()
    T, meta = _dpnn_transfer(np.zeros(16))
    if T is None:
        return _row("dpnn", "fail", meta["error"] + "")
    ms = 1e3 * (time.time() - t0)
    if not np.all(np.isfinite(T)):
        return _row("dpnn", "fail", "inference returned NaN")
    ck = (DPNN_CKPT / "ckpt.pt").stat().st_mtime
    _, cal_t, _ = _read_json(CALIB_PATH)
    r2 = meta.get("r2") or 0.0
    val = f"R² {r2:.2f} · age: {ago(ck)}"
    if meta.get("source") != "hw":
        return _row(
            "dpnn",
            "fail",
            val + " · trained on the simulator",
        )
    live, declined = _live_text("dpnn", ck, "error vs chip", sign=-1)
    return _row("dpnn", "warn" if declined else "pass", val + live)


def _ck_table(mock):
    """Is there a transfer table, how stale is it, and does a plan from it land?"""
    from scipy.stats import special_ortho_group

    from pic.matvec import bench_box, load_transfers, plan_from_table

    tr = load_transfers()
    if tr is None or not len(tr):
        return _row("table", "fail", "no transfer table in pic_data/sessions")
    age = tr.path.stat().st_mtime if tr.path else None
    B = special_ortho_group.rvs(2, random_state=0)
    t0 = time.time()
    plan = plan_from_table(B, tr, box=bench_box())
    err = float(np.linalg.norm(plan.predict() - B) / np.linalg.norm(B))
    ms = 1e3 * (time.time() - t0)
    val = f"{len(tr)} states · age: {ago(age) if age else '?'} · " f"one 2x2 plan: error {err:.3f}"
    if not np.isfinite(err) or err >= 1.0:  # 1.0 is what answering zero scores
        return _row("table", "fail", val + " · worse than answering zero")
    live, declined = _live_text("table", age or 0, "tiled fidelity")
    return _row("table", "warn" if declined else "pass", val + live)


_BOARD_FN = {"board": _ck_board, "switch": _ck_switch, "pds": _ck_pds}
_SOLO_FN = {
    "ports": _ck_ports,
    "laser": _ck_laser,
    "dpnn": _ck_dpnn,
    "table": _ck_table,
    "rail": lambda mock: power_test(mock),  # defined below
}


DIAG_LAST = {}  # check id -> its last row, so leaving the tab does not throw a result away


def diag_catalog():
    """Every row at `unknown`, so the page can draw the list without touching anything.

    The tab must be openable with no consequence: two of these rows move the rig."""
    return [DIAG_LAST.get(i) or _row(i, "unknown", "not run") for i in CHECKS]


def diag_run(ids=None, mock: bool = False):
    """Run the named checks and return one row each, in page order.

    Nothing raises out of here. A missing port, a refused open, a board somebody else is
    holding and a calibration file that is not there are all statuses -- a traceback on this
    page would be one more thing to diagnose."""
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or not set(ids) <= set(CHECKS):
            raise ValueError(f"checks must be a list drawn from {', '.join(CHECKS)}")
    want = [i for i in CHECKS if ids is None or i in set(ids)]
    out = {}
    board_ids = [i for i in want if i in BOARD_CHECKS]
    if board_ids:
        why = _board_guard(mock)
        if why:
            for i in board_ids:
                out[i] = _row(i, "unknown", why)
        else:
            try:
                with _board_held(mock):
                    board = None
                    try:
                        board, banner = _open_board(mock)
                        for i in board_ids:
                            try:
                                out[i] = _BOARD_FN[i](board, banner, mock)
                            except Exception as e:
                                out[i] = _row(i, "fail", _short(e))
                    except Exception as e:
                        for i in board_ids:
                            out.setdefault(i, _row(i, "fail", _short(e)))
                    finally:
                        if board is not None:
                            try:
                                board.close()  # leaves every channel at 0 V
                            except Exception:
                                pass
            except Busy as e:
                for i in board_ids:
                    out[i] = _row(i, "unknown", str(e))
    for i in want:
        if i in out:
            continue
        try:
            out[i] = _SOLO_FN[i](mock)
        except Exception as e:
            out[i] = _row(i, "fail", _short(e))
    if not mock:  # the bench's results are what the tab shows on its next visit
        for i, r in out.items():
            DIAG_LAST[i] = {**r, "value": r["value"] + f" · run: {time.strftime('%H:%M')}"}
    return [out[i] for i in CHECKS if i in out]


def power_test(mock):
    """Every heater at a quarter of its power while the TEC is watched. A quarter is enough
    to see whether the loop holds, without dumping the full load into the chip and the
    TEC's heatsink, which then takes many minutes to recover."""
    return diag_hold({"mock": mock, "fraction": 0.25}, mock)["row"]


def diag_hold(payload, default_mock: bool = False):
    """Rail every heater and watch the chip temperature: heaters and TEC at full power.

    ~3 W lands on the die in one step. A regulating loop absorbs it and the temperature
    stays inside the calibration band; a loop with no power stage, a dead bus or no headroom
    lets it climb, so the temperature rise under load is the whole test. The TEC port is opened once with DTR held low and kept open across the
    hold, so reading it does not reset the controller mid-test.

    Same two gates as `/recal`, in the same order, and the DACs are zeroed on every exit."""
    from pic.config import NUM_DAC, VOLTAGE_MAX_CH

    mock = bool(payload.get("mock", default_mock))
    secs = max(1.0, min(float(payload.get("seconds", HOLD_S)), HOLD_MAX_S))
    # power goes as V^2, so a fraction of the power is sqrt(fraction) of the clamp voltage
    frac = min(max(float(payload.get("fraction", 1.0)), 0.0), 1.0)
    what = f"all {NUM_DAC} channels at {frac:.0%} power for {secs:.0f} s"
    why = _board_guard(mock)
    if why:
        raise Busy(f"refusing to drive {what}: {why}")
    if not ARMED and not mock:
        raise Busy(f"refusing to drive {what}: this server was started without --arm")

    with _board_held(mock):
        samples, marks = [], {}

        def pull(tag):
            marks.setdefault(tag, time.time())

        v = np.sqrt(frac) * np.asarray(VOLTAGE_MAX_CH, float)
        board = None
        try:
            tb = time.time()
            while not mock and time.time() - tb < TEC_BASE_S:
                pull("base")
                time.sleep(0.2)
            board, _ = _open_board(mock)
            t0 = time.time()
            while time.time() - t0 < secs:
                board.measure_raw(v)  # re-asserted: one write holds, this also proves it
                pull("load")
                time.sleep(0.5)
            held = time.time() - t0
        finally:
            if board is not None:
                try:
                    board.measure_raw(np.zeros(NUM_DAC))
                    board.close()
                except Exception:
                    pass

    if not mock:  # the shared feed, split at the moment the heaters were railed
        t_load = marks.get("load", time.time())
        samples = [("base" if ts < t_load else "load", t, d) for ts, t, _, d in TEC.rows(since=tb)]
    status, verdict, short = _tec_verdict(samples)
    row = _row("rail", status, verdict)
    row["short"] = short
    return {"ok": True, "held_s": round(held, 1), "row": row, "cmd": f"held {what}"}


class _TecOwner:
    """The one connection to the TEC. Opened once, kept open; a thread keeps the recent
    readings, and everything in this server -- the header, a bench run, the power test --
    reads them. Jobs reach the same readings over /api/tec (`pic.devices.tec.auto_tec`), so
    nothing reopens the port, which could also reset the controller."""

    def __init__(self):
        self.tec, self.buf, self.lock, self.err = None, deque(maxlen=600), threading.Lock(), None

    def ensure(self):
        with self.lock:
            if self.tec is not None:
                return
            # a terminal's pic process opened the port itself: do not become a second reader
            ext = [p for p in (sweeps_running() or []) if not JOB or p != JOB["p"].pid]
            if ext:
                raise Busy(f"the TEC port is held by pid {', '.join(map(str, ext))}")
            # another UI server (a second tab's `make run`, a test instance) may own it already
            import psutil

            lock = Path("pic_data/.tec_owner")
            try:
                other = int(lock.read_text())
            except (OSError, ValueError):
                other = None
            if other and other != os.getpid() and psutil.pid_exists(other):
                raise Busy(f"the TEC port is held by another UI server (pid {other})")
            lock.write_text(str(os.getpid()))
            from pic.devices.tec import SerialTEC

            self.tec = SerialTEC(_tec_port()).open()  # open() writes TEC_SETPOINT_C
            self.err = None
            threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            try:
                row = self.tec._read_line()
                if row is not None:
                    self.buf.append((time.time(), *row))
                time.sleep(0.2)
            except Exception as e:  # unplugged or reset: drop it, the next read reopens
                with self.lock:
                    self.err, tec, self.tec = _short(e), self.tec, None
                try:
                    tec.close()
                except Exception:
                    pass
                return

    def rows(self, since=None, n=None):
        self.ensure()
        r = [x for x in self.buf if since is None or x[0] >= since]
        return r[-n:] if n else r

    def mean(self, n=5):
        r = self.rows(n=n)
        if not r or time.time() - r[-1][0] > 3.0:
            raise ValueError(f"no fresh TEC telemetry{f' ({self.err})' if self.err else ''}")
        t, _, d = np.mean([x[1:] for x in r], axis=0)
        # the band everything gates on is the operating temperature, not the controller's
        # target: the controller always aims TEC_SETPOINT_C, and may not get there
        return float(t), op_temp(), float(d), len(r)

    def view(self):
        """A TEC object for a Rig, fed from this connection; its close() leaves the port open."""
        from pic.devices.tec import FeedTEC

        return FeedTEC(lambda: self.mean()[:3])


TEC = _TecOwner()


OP_PATH = Path("pic_data/op_temp.json")


def op_temp():
    from pic.config import TEC_SETPOINT_C

    saved, _, _ = _read_json(OP_PATH)
    return float(saved["c"]) if saved else TEC_SETPOINT_C


def op_set(b):
    """Set the operating temperature runs gate and pause on (±TEC_TOLERANCE_C). The TEC is
    not retargeted: it keeps aiming TEC_SETPOINT_C, so a die stuck warm can still be run at."""
    from pic.devices.tec import T_MAX_C, T_MIN_C

    c = _finite(b.get("set"), "set", 1)[0]
    if not T_MIN_C <= c <= T_MAX_C:
        raise ValueError(f"operating temperature must be {T_MIN_C:.0f}..{T_MAX_C:.0f} C, got {c:g}")
    OP_PATH.write_text(json.dumps({"c": c, "at": time.strftime("%Y-%m-%d %H:%M")}))
    ev("tec", "op", f"operating temperature -> {c:.2f} C", set_c=c)
    return {"ok": True, "set": c}


def tec_now(n: int = 5):
    """Die temperature for the header: the mean of the last n readings on the shared port."""
    from pic.config import TEC_SETPOINT_C, TEC_TOLERANCE_C

    try:
        t, sp, d, k = TEC.mean(n)
    except Busy:
        return {"held": True}
    TEC_SETPOINT_C = sp  # the operating temperature
    return {
        "c": round(t, 2),
        "drive": round(d, 2),
        "n": k,
        "set": TEC_SETPOINT_C,
        "tol": TEC_TOLERANCE_C,
        "status": getattr(TEC.tec, "status_line", None),
    }


def _tec_verdict(samples):
    """(status, text, one-line summary) from the temperature before and under load."""
    from pic.rig import CALIB_TEMP_TOL_C

    base = [(t, d) for g, t, d in samples if g == "base"]
    load = [(t, d) for g, t, d in samples if g == "load"]
    if not base or len(load) < 3:
        return "unknown", "no TEC telemetry", "no telemetry"
    tail = load[-max(3, len(load) // 3) :]  # the settled end of the hold
    t0, d0 = np.mean(base, axis=0)
    t1, d1 = np.mean(tail, axis=0)
    txt = f"temperature: {t0:.2f} C → {t1:.2f} C · TEC drive: {d0:+.2f} V → {d1:+.2f} V"
    short = f"{t0:.2f} C → {t1:.2f} C, drive {d1:+.2f} V"
    if abs(d0) >= TEC_RAIL_V:
        return (
            "unknown",
            "no verdict · " + txt + " · drive was already at its limit before the load",
            short,
        )
    if t1 - t0 > CALIB_TEMP_TOL_C:
        why = (
            "drive at its limit, so the TEC is out of authority under this heater load; "
            "check the TEC supply and current limit"
            if abs(d1) >= TEC_RAIL_V
            else "drive stayed below its limit yet the die heated; check the TEC wiring and polarity"
        )
        return "fail", f"not holding · {txt} · {why}", short
    return "pass", "holding · " + txt, short


def _raise(e):
    raise e


def _nan_to_none(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    return o


# deepest frame in these files names the part that failed, so an error says where to look
_WHERE = (
    ("devices/laser", "laser"),
    ("devices/pdmv5", "laser"),
    ("devices/tec", "TEC"),
    ("devices/switch", "optical switch"),
    ("pic/interface", "board"),
    ("pic/session", "serial ports"),
    ("theory/calib", "calibration"),
    ("pic/characterize", "calibration"),
    ("pic/apply", "matvec"),
    ("pic/matvec", "matvec"),
)


def _where(e) -> str:
    if isinstance(e, NotOrthogonal):
        return "target U"
    if isinstance(e, Busy):
        return "board busy"
    for f in reversed(traceback.extract_tb(e.__traceback__)):
        p = f.filename.replace("\\", "/")
        for key, name in _WHERE:
            if key in p:
                return name
    return "input" if isinstance(e, ValueError) else "server"


# The rig as a set of explicit states. `classify` is pure -- probes in, states out -- so every
# transition can be driven in the self-test without hardware; `rig_state` does the probing.
# Each state names who acts: "auto" (the rig already handles it; nothing to do) or "user"
# (the one thing to do). "ok" states have no fix.
def classify(p):
    from pic.config import TEC_SETPOINT_C, TEC_TOLERANCE_C

    S = []

    def add(comp, state, level, detail="", who=None, fix=""):
        S.append(
            {
                "component": comp,
                "state": state,
                "level": level,
                "detail": detail,
                "who": who,
                "fix": fix,
            }
        )

    missing = sorted({"board", "tec", "laser"} - set(p.get("ports", [])))
    if missing:
        add(
            "ports",
            "missing",
            "fail",
            ", ".join(missing),
            "user",
            f"plug in / power the {', '.join(missing)}; they are named by chip id",
        )
    else:
        add("ports", "ok", "pass")

    L = p.get("laser") or {}
    if L.get("error"):
        add("laser", "unreachable", "fail", L["error"], "user", "check the laser's USB and power")
    elif L.get("held"):
        add(
            "laser",
            "held",
            "pass",
            f"{L['held']} ({'on' if L.get('on') else 'off'})",
            "auto",
            "the job switches it off when it ends; the server does too",
        )
    elif not L.get("key", True):
        add("laser", "key off", "fail", "", "user", "turn the laser key on")
    elif not (L.get("bnc", True) and L.get("ext", True)):
        add("laser", "interlock open", "fail", "", "user", "close the laser interlocks")
    elif L.get("on"):
        add(
            "laser",
            "on, idle",
            "warn",
            f"{L.get('mA')} mA with no job",
            "user",
            "switch it off unless you are aligning by hand",
        )
    else:
        add("laser", "off", "pass")

    T = p.get("tec") or {}
    st = T.get("status") or ""
    if T.get("error") or T.get("held"):
        add(
            "tec",
            "unreachable" if T.get("error") else "held",
            "fail" if T.get("error") else "warn",
            T.get("error") or "another process has the port",
            "user",
            "check the TEC's USB" if T.get("error") else "wait for that process",
        )
    elif "FAULT" in st:
        add(
            "tec",
            "fault cleared",
            "warn",
            st,
            "auto",
            "the firmware clears latched faults every 5 s",
        )
    elif abs(T["c"] - T.get("set", TEC_SETPOINT_C)) > T.get("tol", TEC_TOLERANCE_C):
        railed = abs(T.get("drive", 0)) >= TEC_RAIL_V
        if railed and not p.get("job_running"):
            add(
                "tec",
                "at its limit",
                "fail",
                f"{T['c']:.2f} C, drive railed with no load",
                "user",
                "the heatsink is not shedding heat: cool the TEC's hot side, or wait",
            )
        else:
            add(
                "tec",
                "settling",
                "warn",
                f"{T['c']:.2f} C",
                "auto",
                "bench actions stay disabled until the die is back in band",
            )
    else:
        add("tec", "in band", "pass", f"{T['c']:.2f} C")

    ext = p.get("external") or []
    if ext:
        add(
            "board",
            "held elsewhere",
            "warn",
            f"pid {', '.join(map(str, ext))}",
            "user",
            "a pic process outside this page holds the rig: let it finish or stop it",
        )
    elif p.get("board_busy"):
        add("board", "busy", "pass", "a run or check from this page", "auto", "it frees itself")
    else:
        add("board", "free", "pass")

    J = p.get("job") or {}
    if J.get("running"):
        add(
            "job",
            "waiting for TEC" if J.get("waiting") else "running",
            "warn" if J.get("waiting") else "pass",
            J.get("what", ""),
            "auto" if J.get("waiting") else None,
            "it resumes when the die is in band" if J.get("waiting") else "",
        )
    elif J.get("rc") not in (None, 0):
        add(
            "job",
            "failed",
            "fail",
            f"{J.get('what')}: {J.get('reason') or 'exit ' + str(J.get('rc'))}",
            "user",
            "see the Logs tab; the laser was switched off",
        )
    else:
        add("job", "idle" if not J else "finished", "pass", J.get("what", ""))
    return S


def _fail_reason(evs):
    """What a failed job died of, in one line: its last error, before the bare `exit N` the
    CLI ends on; failing that, the last line it printed raw, which is where a SystemExit's
    message lands."""
    errs = [e for e in evs if e.get("lvl") == "error"]
    real = [e for e in errs if not (e.get("c") == "job" and e.get("s") == "failed")]
    raw = [e for e in evs if e.get("lvl") == "raw"]
    e = (real or raw or errs or [None])[-1]
    return e and e["text"][:160]


def rig_state():
    """Probe everything cheaply -- no board open, no laser switched -- and classify."""
    from pic.config import usb_ports

    p = {"ports": [r for _, r, _ in usb_ports()]}
    try:
        p["laser"] = laser()
    except Exception as e:
        p["laser"] = {"error": _short(e)}
    try:
        p["tec"] = tec_now()
    except Exception as e:
        p["tec"] = {"error": _short(e)}
    j = job_state()
    if j.get("rc") not in (None, 0):
        try:
            j["reason"] = _fail_reason(_job_events())
        except Exception:
            pass
    p["job"], p["job_running"] = j, j.get("running", False)
    p["external"] = j.get("external") or []
    p["board_busy"] = _BOARD_LOCK.locked()
    return {"states": classify(p), "probes": p}


# A browser that closes a tab or navigates away mid-poll hangs up on a reply in flight. That is
# normal, not an error, and each one used to print a 30-line traceback.
_HANGUP = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class Server(ThreadingHTTPServer):
    daemon_threads = True  # Ctrl-C does not wait on a poll that is mid-flight

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], _HANGUP):
            super().handle_error(request, client_address)


_LAST_REFUSAL = {}  # route -> the last message logged for it: a poll repeats itself


def _log_refusal(path, e, lvl, tb=None):
    """Every refusal or error the page is sent also goes to the Logs tab -- once per change,
    not once per 5 s poll that hits the same one."""
    route = path.split("?")[0]
    msg = f"{type(e).__name__}: {e}".splitlines()[0][:200]
    if _LAST_REFUSAL.get(route) == msg:
        return
    _LAST_REFUSAL[route] = msg
    ev(
        "server",
        "refused" if lvl == "warn" else "error",
        msg,
        lvl,
        route=route,
        where=_where(e),
        **({"trace": tb[-1500:]} if tb else {}),
    )


class Handler(BaseHTTPRequestHandler):
    default_mock = True

    def _send(self, code, body, ctype):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, fn, code=200):
        try:
            out = fn()
        except (ReadoutError, NotOrthogonal, Busy, ValueError) as e:  # actionable refusals
            out = {"error": str(e), "ok": False, "where": _where(e)}
            _log_refusal(self.path, e, "warn")
        except Exception as e:  # the page must survive a bad run
            out = {
                "error": f"{_where(e)}: {type(e).__name__}: {e}\n\n{traceback.format_exc()}",
                "ok": False,
                "where": _where(e),
            }
            _log_refusal(self.path, e, "error", traceback.format_exc())
        try:
            txt = json.dumps(out, allow_nan=False)
        except ValueError:  # NaN/inf: invalid JSON to a browser, so it would kill the page
            txt = json.dumps(_nan_to_none(out), allow_nan=False)
        self._send(code, txt, "application/json")

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in PAGES:
            self._send(200, render(PAGES[route]), "text/html; charset=utf-8")
        elif route == "/api/calib":
            self._json(calib_state)
        elif route == "/api/laser":
            self._json(laser)
        elif route == "/api/predictors":
            self._json(lambda: {"preds": predictors(), "est": estimates()})
        elif route == "/api/events":
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(self.path).query)
            self._json(lambda: {"events": EVENTS.since(float((qs.get("after") or [0])[0]))})
        elif route == "/api/logs":
            from urllib.parse import parse_qs, urlparse

            name = (parse_qs(urlparse(self.path).query).get("name") or [None])[0]
            self._json(lambda: logs(name))
        elif route == "/api/state":
            self._json(rig_state)
        elif route == "/api/tec":
            self._json(tec_now)
        elif route == "/api/job":
            self._json(job_state)
        elif route == "/api/diag":
            # the catalogue only: opening the tab must touch no instrument
            self._json(lambda: {"rows": diag_catalog(), "armed": ARMED})
        else:
            f = (STATIC / route.lstrip("/")).resolve()
            if f.is_file() and STATIC in f.parents:  # never serve outside static/
                self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
            else:
                self._send(404, "not found", "text/plain")

    def _body(self):
        try:
            n = int(self.headers.get("content-length", 0))
        except ValueError:
            n = -1
        if not 0 <= n <= 1 << 20:
            raise ValueError("request body missing or larger than 1 MB")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("request body is not JSON") from None
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def do_POST(self):
        route = self.path.split("?")[0]
        if route not in (
            "/api/laser",
            "/api/recal",
            "/api/diag",
            "/api/run",
            "/api/job/stop",
            "/api/tec",
        ):
            return self._json(lambda: _raise(ValueError(f"no such endpoint: {route}")), 404)
        try:
            body = self._body()
        except ValueError as e:
            return self._json(lambda: _raise(e), 400)
        if route == "/api/laser":
            self._json(lambda: laser(body))
        elif route == "/api/recal":
            self._json(lambda: recal_request(body))
        elif route == "/api/tec":
            self._json(lambda: op_set(body))
        elif route == "/api/job/stop":
            self._json(job_stop)
        elif route == "/api/diag":
            self._json(
                lambda: {
                    "rows": diag_run(
                        body.get("checks"), body.get("mock", self.default_mock) is True
                    )
                }
            )
        else:
            self._json(lambda: run_once(body, self.default_mock))

    def log_message(self, *a):
        pass


def _selftest(cols: int = 3):
    """The gate on this file: the refusal, the batch, that nothing is averaged away, and
    that the calibration view says what the calibration files say.

    Mock only. The bench has an optical fault and `--mock` is the run, not a rehearsal.
    What is checked here is the transport -- `pic.apply._selftest` owns the seam and
    `pic.matvec._selftest` the physics. The one claim worth making twice is that the
    per-element sign flags this page plots are the same numbers `score` averaged into
    `sign_acc`, because a histogram drawn from a second opinion would be decoration."""
    rng = np.random.default_rng(0)
    Uo, _ = np.linalg.qr(rng.normal(size=(4, 4)))

    assert orth_dev(np.eye(4)) == 0.0
    assert orth_dev(Uo) < 1e-12, orth_dev(Uo)
    # 4 decimals is what the page used to write and it fails the gate; 6 passes
    assert orth_dev(np.round(Uo, 4)) > TOL_ORTH
    assert orth_dev(np.round(Uo, 6)) <= TOL_ORTH

    for bad in (1.01 * Uo, Uo + 1e-4):
        try:
            run_once({"U": bad.ravel().tolist(), "X": [[1, 0, 0, 0]]}, True)
            raise AssertionError("a non-orthogonal target was accepted")
        except NotOrthogonal as e:
            assert "tolerance" in str(e) and f"{orth_dev(bad):.3e}" in str(e), str(e)

    Xc = rng.normal(size=(cols, 4)).tolist()
    out = run_once({"U": Uo.ravel().tolist(), "X": Xc}, True)
    legacy = run_once({"U": Uo.ravel().tolist(), "x": [1, 0, 0, 0]}, True)
    assert len(legacy["vectors"]) == 1

    # one press runs both: the mesh holding U (one program) and U tiled (four), same x
    assert out["programs"] == {"normal": 1, "tiled": 4}, out["programs"]
    assert out["rails_out"] == [0, 1, 2, 3], out["rails_out"]
    V = out["vectors"]
    assert len(V) == cols
    for k in ("normal", "tiled"):
        sp = out["spread"][k]
        assert sp["n"] == cols and sp["min"] <= sp["median"] <= sp["max"], (k, sp)
    for i, v in enumerate(V):
        assert v["x"] == [float(q) for q in Xc[i]], i
        for k in ("normal", "tiled"):
            m = v[k]
            assert len(m["device"]) == len(m["ideal"]) == len(m["kept"]) == 4, (k, i)
            assert abs(np.mean(m["kept"]) - m["sign"]) < 1e-12, (k, i)
        # each measured series is scored against its OWN ideal: the computed bar is Ux,
        # tiled approximates that, and normal can only approximate |U|^2 x
        assert np.allclose(v["tiled"]["ideal"], v["computed"]), i
        assert np.allclose(v["normal"]["ideal"], np.abs(Uo) ** 2 @ np.asarray(Xc[i])), i

    s = calib_state()
    _check_calib(s)
    _check_diagram(s)
    for t in ("gate_text",):  # the page carries the colours, not a paragraph about them
        assert t not in s, t
    assert not any(
        "predates the rewire" in w or "the stored fit says" in w for w in s["warnings"]
    ), s["warnings"]

    for cmd, hit in (
        ("/opt/envs/pic/bin/python -m pic char --channels 6,7 --write", True),
        ("/opt/envs/pic/bin/python3.14 -m pic", True),
        ("/opt/envs/pic/bin/python -u -m learn.train_hw --rounds 4", True),
        ("/opt/envs/pic/bin/python -u -m pic fastchar --write", True),
        ("C:\\venv\\Scripts\\python.exe -m pic fastchar", True),
        ("/bin/zsh -c python -m pic char", False),
        ("/opt/envs/pic/bin/python ui.py --mock --port 8745", False),
        ("/opt/envs/pic/bin/python -m pic char --mock --channels 6", False),
    ):
        assert _is_pic(cmd.split()) == hit, cmd
    busy = sweeps_running()
    assert busy is None or all(isinstance(x, int) for x in busy), busy

    for bad in ({"channels": "99"}, {"channels": ""}, {"pd": 9}):
        try:
            recal_request(bad)
            raise AssertionError(f"/recal accepted {bad}")
        except ValueError:
            pass
    try:  # the bench is faulty: only the refusal path is exercised
        recal_request({"channels": "6,7"})
        raise AssertionError("/recal ran a sweep from a disarmed server")
    except Busy as e:
        assert "python -m pic char --channels 6,7 --write" in str(e), str(e)

    # the stress-test findings: every bad request is a named refusal, never a crash or NaN
    for bad, msg in (
        ({}, "no target U"),
        ({"U": [1, 2, 3], "X": [[1, 0, 0, 0]]}, "16 numbers"),
        ({"U": [float("nan")] * 16, "X": [[1, 0, 0, 0]]}, "NaN"),
        ({"U": Uo.ravel().tolist(), "X": [[float("nan"), 0, 0, 0]]}, "NaN"),
        ({"U": Uo.ravel().tolist(), "X": [[0, 0, 0, 0]]}, "all zeros"),
        ({"U": Uo.ravel().tolist(), "X": []}, "give 1 to"),
    ):
        try:
            run_once(bad, True)
            raise AssertionError(f"accepted {bad}")
        except ValueError as e:
            assert msg in str(e), (bad, str(e))
    for b in ({"on": True, "dbm": float("nan")}, {"on": True, "dbm": 50}):
        try:
            laser(b)  # refused before the laser's port is opened
            raise AssertionError(f"laser accepted {b}")
        except ValueError:
            pass
    assert (
        json.dumps(_nan_to_none({"a": [float("nan"), 1.0]}), allow_nan=False)
        == '{"a": [null, 1.0]}'
    )

    _check_job_events()

    _check_states()
    _check_live()
    import tempfile

    import events

    with tempfile.TemporaryDirectory() as d:
        events._selftest(d)
    diag = _check_diag()

    print(
        f"orth gate {TOL_ORTH:.0e}: identity 0, haar {orth_dev(Uo):.1e}, "
        f"4 dp {orth_dev(np.round(Uo, 4)):.1e} refused, "
        f"6 dp {orth_dev(np.round(Uo, 6)):.1e} accepted"
    )
    for h in ("normal", "tiled"):
        sp = out["spread"][h]
        print(
            f"{h}: {len(out['vectors'])} vectors, {out['programs'][h]} program(s), rel err "
            f"min {sp['min']:.3f} median {sp['median']:.3f} max {sp['max']:.3f}, sign "
            f"{sp['sign_mean']:.0%} mean / {sp['sign_min']:.0%} worst"
        )
    print(placement_text())
    c = s["counts"]
    print(
        f"calib: {len(s['heaters'])} heaters + {len(s['pds'])} detectors -> "
        f"{c['ok']} good, {c['weak']} weak, {c['broken']} broken, "
        f"{c['grey']} uncalibrated; remapped={s['remapped']}, "
        f"{len(s['warnings'])} warning(s)"
    )
    print("recal refuses a bad channel, and refuses a good one while disarmed")
    for r in diag:
        print(f"  {r['status']:<7} {r['name']:<15} {r['value']}")
    print(
        f"diag: {len(diag)} checks, {sum(c['auto'] for c in CHECKS.values())} in run-all; "
        f"hardware rows refuse while the board is held; sketch VMAX matches the host"
    )
    return out


def _check_diagram(s):
    """The mesh picture, against the two maps it claims to be drawn from.

    Three claims. Every element box spans exactly the rails `MESH[k]` names, in column
    `COLUMN[k]`. Every heater on that box is the one `THETA_DAC`/`PHI_DAC` says drives it,
    read back out of the map rather than compared to a literal written here. And a phi is
    on the UPPER of its element's two rails, `MESH[k][0]`, which is the arm the `e^{i phi}`
    in `clements.T` multiplies -- putting it on the lower arm draws a phase the mesh does
    not contain."""
    from theory.clements import COLUMN, MESH, NMODE
    from theory.layout import AUX_GAP_COLUMN, AUX_RAIL, PHI_DAC, THETA_DAC

    el = {e["k"]: e for e in s["elements"]}
    assert sorted(el) == list(range(len(MESH))), sorted(el)
    for k, (m, n) in enumerate(MESH):
        e = el[k]
        assert e["rails"] == [m, n], (k, e)
        assert e["col"] == COLUMN[k], (k, e)
        assert e["theta_dac"] == THETA_DAC[k] and e["phi_dac"] == PHI_DAC[k], (k, e)
    # Two elements in one column act on disjoint pairs -- that is what a column IS, and it
    # is why a column can be swept in one pass. A drawing that violated it would overlap
    # two boxes on one rail, which is also how you would notice.
    for c in range(s["ncol"]):
        rails = [r for e in el.values() if e["col"] == c for r in e["rails"]]
        assert len(rails) == len(set(rails)), (c, rails)
        assert all(0 <= r < NMODE for r in rails), (c, rails)

    drawn = {}
    for r in s["heaters"]:
        p, d = r["place"], r["dacs"][0]
        assert p, r
        drawn.setdefault(p["kind"], []).append(d)
        if p["kind"] == "theta":
            assert THETA_DAC[p["elem"]] == d, r
            assert p["rails"] == list(MESH[p["elem"]]), r
            assert p["col"] == COLUMN[p["elem"]] == el[p["elem"]]["col"], r
        elif p["kind"] == "phi":
            assert PHI_DAC[p["elem"]] == d, r
            assert p["rails"] == [MESH[p["elem"]][0]], r
            assert p["col"] == COLUMN[p["elem"]] == el[p["elem"]]["col"], r
        elif p["kind"] == "aux":
            assert p["rails"] == [AUX_RAIL[d]] and p["col"] == AUX_GAP_COLUMN, r
            # a spare sits in the gap its column leaves, so no element there holds its rail
            assert all(
                p["rails"][0] not in e["rails"] for e in el.values() if e["col"] == p["col"]
            ), r
        elif p["kind"] == "out":
            assert p["col"] == s["ncol"], r  # past the mesh, never inside a column
        else:
            raise AssertionError(r)
    assert sorted(drawn.get("theta", [])) == sorted(THETA_DAC.values()), drawn
    assert sorted(drawn.get("phi", [])) == sorted(PHI_DAC.values()), drawn


def _check_live():
    """Advice comes from measured decline only: steady runs never trigger it, a real drop does,
    and too few runs say nothing rather than guess."""
    import tempfile

    global ERRORS_PATH
    keep = ERRORS_PATH
    try:
        with tempfile.TemporaryDirectory() as d:
            ERRORS_PATH = Path(d) / "e.jsonl"
            rows = [0.95, 0.94, 0.96, 0.95, 0.94] * 2
            ERRORS_PATH.write_text(
                "".join(json.dumps({"t": i + 1, "heaters": v}) + "\n" for i, v in enumerate(rows))
            )
            assert _live("heaters", 0)[3] is False  # steady: no advice
            ERRORS_PATH.write_text(
                "".join(
                    json.dumps({"t": i + 1, "heaters": v}) + "\n"
                    for i, v in enumerate(rows[:5] + [0.80, 0.81, 0.79, 0.82, 0.80])
                )
            )
            now, best, n, declined = _live("heaters", 0)
            assert declined and n == 10 and now < best, (now, best, n)
            assert _live("heaters", 6)[3] is False  # only runs since the refresh count
    finally:
        ERRORS_PATH = keep


def _check_states():
    """Every component through its transitions: each lands in the right state, and the fix
    goes to the right party -- the rig when it recovers by itself, the user when only they can."""
    ok = {
        "ports": ["board", "tec", "laser"],
        "laser": {"on": False, "key": True, "bnc": True, "ext": True},
        "tec": {"c": 25.1, "drive": 1.0, "status": "STATUS 0x3"},
        "job": {},
    }
    cases = [
        ({}, {"ports": "ok", "laser": "off", "tec": "in band", "board": "free", "job": "idle"}),
        ({"ports": ["board", "laser"]}, {"ports": "missing"}),
        ({"laser": {"error": "no FTDI"}}, {"laser": "unreachable"}),
        ({"laser": {"key": False, "bnc": True, "ext": True}}, {"laser": "key off"}),
        ({"laser": {"key": True, "bnc": True, "ext": False}}, {"laser": "interlock open"}),
        ({"laser": {"on": True, "mA": 50, "key": True, "bnc": 1, "ext": 1}}, {"laser": "on, idle"}),
        ({"laser": {"held": "table recapture", "on": True}}, {"laser": "held"}),
        ({"tec": {"error": "TEC sent no telemetry"}}, {"tec": "unreachable"}),
        ({"tec": {"held": True}}, {"tec": "held"}),
        (
            {"tec": {"c": 25.2, "drive": 1, "status": "FAULT 0x20 cleared"}},
            {"tec": "fault cleared"},
        ),
        ({"tec": {"c": 26.4, "drive": 2.0}}, {"tec": "settling"}),
        ({"tec": {"c": 26.4, "drive": 5.0}}, {"tec": "at its limit"}),
        (
            {
                "tec": {"c": 26.4, "drive": 5.0},
                "job_running": True,
                "job": {"running": True, "what": "sweep"},
            },
            {"tec": "settling", "job": "running"},
        ),
        ({"job": {"running": True, "waiting": True, "what": "DPNN"}}, {"job": "waiting for TEC"}),
        (
            {"job": {"running": False, "rc": 1, "what": "capture", "reason": "TECError: x"}},
            {"job": "failed"},
        ),
        ({"job": {"running": False, "rc": 0, "what": "sync"}}, {"job": "finished"}),
        ({"external": [4242]}, {"board": "held elsewhere"}),
        ({"board_busy": True}, {"board": "busy"}),
    ]
    for over, want in cases:
        got = {s["component"]: s for s in classify({**ok, **over})}
        for comp, state in want.items():
            assert got[comp]["state"] == state, (over, comp, got[comp])
        for s in got.values():  # a problem always says who acts, and a fine state never nags
            assert (s["level"] == "pass") or s["who"] in ("auto", "user"), s
            assert s["who"] != "user" or s["fix"], s
    who = {s["component"]: s["who"] for s in classify({**ok, "tec": {"c": 26.4, "drive": 5.0}})}
    assert who["tec"] == "user"  # railed at rest is the heatsink: only a person can fix that
    return len(cases)


def _check_job_events():
    """The job's view -- rows, fits, progress, laser, waiting, failure -- read off its events
    exactly as `EventLog` parses them from the printed `@ev` lines."""
    import contextlib
    import io
    import tempfile

    import events

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev("session", "waiting", "waiting for the TEC")
        ev("laser", "on", "LASER ON")
        ev("session", "lit", "emission verified", "ok")
        ev("heater", "prescan", "prescan 1/2 H1", k=1, n=2, pad="H1")
        ev("heater", "prescan", "prescan 2/2 H10", k=2, n=2, pad="H10")
        ev("heater", "fit", "H10 ok", "ok", pad="H10", vpi=5.0, phi0_pi=0.25, span=0.4, ok=True)
        ev("heater", "sweeping", "sweeping H1", pad="H1")
        ev("session", "paused", "paused: die at 26", "warn")
        print("an unconverted line")
        ev("job", "failed", "exit 2", "error", rc=2)
    with tempfile.TemporaryDirectory() as d:
        log = events.EventLog(Path(d) / "e.jsonl")
        log.feed("fastchar", buf.getvalue(), job="j")
        E = log.since(0, None, job="j")
    rows = _row_states(["H1", "H10"], E, True)
    assert rows == {"H1": "running", "H10": "done"}, rows
    assert _row_states(["H1", "H10"], E, False)["H1"] == "queued"
    assert _row_fits(E) == {"H10": {"vpi": 5.0, "phi0_pi": 0.25, "span": 0.4, "ok": True}}
    assert _progress("fastchar", E, _row_states(["H1", "H10"], E, False)) == 0.75
    assert _job_laser(E) and _waiting(E) and not _waiting(E[:3])
    assert _fail_reason(E) == "an unconverted line"  # the raw line beats a bare exit code
    E.append({"lvl": "error", "kind": "error", "text": "pic.devices.tec.TECError: chip"})
    assert _fail_reason(E).startswith("pic.devices.tec.TECError")
    E.append({"c": "laser", "s": "off", "lvl": "ok", "text": "LASER OFF"})
    assert not _job_laser(E)
    rec = [{"c": "heater", "s": "recal", "pad": "H4", "phi0_pi": -0.5, "k": 1, "n": 2}]
    assert _row_fits(rec) == {"H4": {"phi0_pi": -0.5, "ok": True}}
    cap = [{"c": "table", "s": "progress", "k": 3, "n": 12}]
    assert _progress("capture", cap, {}) == 0.25 and _progress("capture", [], {}) == 0.0


def _check_diag():
    """The diagnostics tab: the catalogue, the refusals, and the one gate that needs no rig.

    Three claims worth making here. The catalogue touches nothing, because opening the tab
    must not move the mirror or reset the cooler. Every hardware row refuses -- as `unknown`
    and not as `fail` -- when something already holds the board, and refuses the same way
    when it cannot find out; this is checked by standing in for `sweeps_running`, which also keeps
    this function from ever opening a serial port. And the firmware's clamp table is
    compared to `pic.config.VOLTAGE_MAX_CH` from the SKETCH SOURCE, which is the whole
    invariant with no board present: a table edited on one side only is caught here rather
    than at the next open."""
    from pic.config import VOLTAGE_MAX_CH
    from pic.rig import FIRMWARE_VMAX_TOL_V

    cat = diag_catalog()
    assert [r["id"] for r in cat] == list(CHECKS), cat
    assert all(r["status"] == "unknown" for r in cat), cat
    assert "tec" not in CHECKS and not CHECKS["rail"]["auto"], CHECKS

    table, err = _sketch_vmax()
    assert table is not None, err
    assert len(table) == len(VOLTAGE_MAX_CH), (len(table), len(VOLTAGE_MAX_CH))
    worst = max(abs(a - b) for a, b in zip(table, VOLTAGE_MAX_CH))
    assert worst <= FIRMWARE_VMAX_TOL_V, (
        f"Arduino/pic4x4/pic4x4.ino VMAX[] and pic.config.VOLTAGE_MAX_CH differ by "
        f"{worst:.3f} V; the host would command volts the DAC clamps to something else"
    )

    # Standing in for the process table does two jobs: it exercises both refusal branches, and it
    # guarantees this gate never reaches `_open_board` -- the bench is not ours to open.
    real = globals()["sweeps_running"]
    try:
        for stub, mark in ((lambda: [999999], "pid 999999"), (lambda: None, "process table")):
            globals()["sweeps_running"] = stub
            held = diag_run(BOARD_CHECKS, mock=False)
            assert [r["id"] for r in held] == list(BOARD_CHECKS), held
            assert all(r["status"] == "unknown" and mark in r["value"] for r in held), held
            try:
                diag_hold({"mock": False}, False)
                raise AssertionError("/diag/hold drove the board while it was held")
            except Busy as e:
                assert mark in str(e), str(e)
    finally:
        globals()["sweeps_running"] = real

    quick = [i for i in CHECKS if i != "rail"]
    rows = {r["id"]: r for r in diag_run(quick, mock=True)}
    assert list(rows) == quick, list(rows)
    for r in rows.values():
        assert r["status"] in STATUSES, r
        assert r["value"] and r["tip"], r
    # the mock board carries no clamp table on the wire, so the row falls back to the sketch
    assert rows["switch"]["status"] == "pass", rows["switch"]

    # The mock laser carries the refusals the real unit has, so the two branches that stop
    # a run before it starts are exercised rather than assumed.
    import pic.devices.mock as _m

    real_laser = _m.MockLaser
    try:
        for kw, mark in (({"key": 0}, "key switch off"), ({"interlock": 0}, "interlock open")):
            _m.MockLaser = lambda kw=kw: real_laser(**kw)
            r = _ck_laser(True)
            assert r["status"] == "fail" and mark in r["value"], (kw, r)
    finally:
        _m.MockLaser = real_laser

    h = diag_hold({"mock": True, "seconds": 1}, True)
    assert h["ok"] and h["held_s"] >= 1.0, h
    try:
        diag_hold({"mock": False}, False)  # disarmed, and nothing holds the board
        raise AssertionError("/diag/hold drove the board from a disarmed server")
    except Busy as e:
        assert "refusing to drive" in str(e), str(e)
    return [rows[i] for i in quick]


def _check_calib(s):
    """Every claim the calibration tab makes, against the map it claims to be drawn from."""
    from theory.layout import HEATERS, MIRROR_OF, N_PHYSICAL_HEATERS, DAC_HEATER

    # a bonded pair is one heater, so it is one row and one dot
    assert len(s["heaters"]) == N_PHYSICAL_HEATERS, len(s["heaters"])
    assert all(DAC_HEATER[p] == DAC_HEATER[q] for p, q in MIRROR_OF.items())
    seen = sorted(d for r in s["heaters"] for d in r["dacs"])
    assert seen == list(range(len(HEATERS))), seen
    for r in s["heaters"]:
        assert r["status"] in ("green", "yellow", "red", "grey"), r
        assert r["why"], r
        # a mirror never carries its own row, and every row's first DAC is the primary
        assert r["dacs"][0] not in MIRROR_OF, r
        assert all(d in MIRROR_OF for d in r["dacs"][1:]), r
        # a nominal Vpi has no span worth printing, and its colour is grey only
        # when nothing was ever swept -- a swept-and-rejected channel is red
        if r["nominal"]:
            assert r["span_pi"] is None and r["status"] in ("grey", "red"), r
    assert len(s["pds"]) == s["nmode"]
    assert len(s["mesh"]) == len(s["column"])
    assert s["counts"]["ok"] + s["counts"]["weak"] + s["counts"]["broken"] + s["counts"][
        "grey"
    ] == len(s["heaters"]) + len(s["pds"])


def main():
    os.chdir(Path(__file__).resolve().parent)  # data paths are relative to this directory
    # Windows: the server itself must be in UTF-8 mode too (files, console, the events it
    # writes). A plain `python ui.py` there re-launches itself once with -X utf8.
    if os.name == "nt" and not sys.flags.utf8_mode:
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true", help="default the page to the mock chip")
    p.add_argument("--selftest", action="store_true", help="run the gate, no server")
    p.add_argument(
        "--arm",
        action="store_true",
        help="let the reload buttons actually start `python -m pic char`. Off by default: "
        "the endpoint drives current through a heater.",
    )
    p.add_argument("--port", type=int, default=8744)
    a = p.parse_args()
    if not a.selftest:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(LOG_DIR / "server.log", "a")
        sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
        ev("server", "started", "started", at=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid())
    if a.selftest:
        _selftest()
        return
    global ARMED
    ARMED = a.arm
    Handler.default_mock = a.mock
    # a crash leaves the laser in whatever state it was; a restart must not inherit it lit
    # A second UI server (a test instance) must not touch a rig another one is running: the
    # laser-off below would cut a beam that server's job has lit.
    import psutil

    try:
        owner = int(Path("pic_data/.tec_owner").read_text())
    except (OSError, ValueError):
        owner = None
    if owner and owner != os.getpid() and psutil.pid_exists(owner):
        ev(
            "server",
            "note",
            f"another UI server (pid {owner}) runs this rig: laser and TEC left alone",
            "warn",
            pid=owner,
        )
    else:
        try:
            st = laser({"on": False})
            ev("server", "laser", "laser off at start", "ok", **st)
        except Exception as e:
            ev("server", "laser", "laser not reached", "error", error=_short(e))
        try:  # the one TEC connection, opened before any job so every job reads through it
            TEC.ensure()
            ev("server", "tec", "holding the TEC port; jobs read it through the server", "ok")
        except Exception as e:
            ev("server", "tec", "TEC not opened", "error", error=_short(e))
    srv = Server(("127.0.0.1", a.port), Handler)
    ev(
        "server",
        "listening",
        f"UI at http://127.0.0.1:{a.port} (ctrl-c to stop)",
        "ok",
        armed=bool(a.arm),
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        # Ctrl-C is how this server is stopped, not an error: leave the rig safe, say one line
        if JOB and JOB["p"].poll() is None:
            job_stop()  # the job's own cleanup runs; it switches its laser off
        try:
            laser({"on": False})
        except Exception:
            pass
        if TEC.tec is not None:
            TEC.tec.close()
        ev("server", "stopped", "stopped: laser off, TEC released", "ok")


if __name__ == "__main__":
    main()
