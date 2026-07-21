"""Interactive browser console for the PIC -- a live view of the chip itself.

    python ui.py               # open http://localhost:8787
    python ui.py --mock        # start with both devices mocked
    python ui.py --port N      # serve on a different HTTP port
    python ui.py --no-open     # don't auto-open the browser

The page draws the real 6x6P schematic (from `src.pic.layout`, the same scene graph the
static figure uses) and paints live state onto it: the laser feeding the input coupler on
the left, every heater tinted by its supplied DAC voltage, every photodiode showing its
measured value where it physically sits.

Two independent serial devices, each independently mockable: the laser is an FTDI, the
PIC's Arduino is a numeric CH340. They get separate locks so a slow photodiode read can
never delay the laser watchdog.

SAFETY -- the laser is real hardware driven from a web page. Do not weaken:
  * every laser-on arms a backend `threading.Timer` that hard-offs after `duration_s`.
    It lives in the server, so closing the browser, losing the network, or a hung front
    end cannot keep the laser lit;
  * the countdown shown in the page is advisory only; the deadline is server-authoritative
    and re-synced every poll;
  * all laser-serial access is under one lock, so the watchdog can never collide with the
    keepalive or telemetry;
  * the driver self-disables after a few idle seconds, so a background thread re-pokes the
    setpoint while the laser is on;
  * `laser_status == 1` does NOT prove emission -- we baseline the built-in monitor
    photodiode off-vs-on and report `emitted` separately (see CLAUDE.md, task #9);
  * server shutdown drives the laser off and zeroes the DACs.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pic import (
    PIC,
    MockPIC,
    PICError,
    find_port,
    NUM_DAC,
    NUM_ADC_RAW,
    VOLTAGE_MIN,
    DAMAGED_PDS,
)
from src.pic.layout import build_scene
from src.pic.wiring import load_map
from laser.laser import Laser, LaserError
from laser.mock import MockLaser
from laser.pdmv5 import PDMv5Error

VMAX = 5.0
HERE = os.path.dirname(os.path.abspath(__file__))
UIDIR = os.path.join(HERE, "ui")
STATIC = {
    "/app.js": "text/javascript",
    "/style.css": "text/css",
    "/vue.global.prod.js": "text/javascript",
}

DEFAULT_DURATION_S = 60.0
MAX_DURATION_S = 900.0
SAFE_DBM = 5.0  # PD-safe policy for this rig; above this the UI demands an unlock
EMIT_EPS = 0.05  # monitor-PD rise that counts as "light is out"
TELEMETRY_HZ = 2.0
KEEPALIVE_S = 0.4

LOCK = threading.Lock()  # serialises PIC hardware access

SCENE = build_scene()
WIRING = load_map(root=HERE)


def _clip(v):
    return float(np.clip(v, VOLTAGE_MIN, VMAX))


# --------------------------------------------------------------------------- laser side
class LaserSession:
    """Owns the laser device, its watchdog, and its cached telemetry."""

    def __init__(self):
        self.dev = None
        self.mock = True
        self.port = None
        self.info = {}
        self.lock = threading.Lock()
        self._wd = None  # threading.Timer
        self.deadline = None  # time.time() when the watchdog fires
        self.duration_s = DEFAULT_DURATION_S
        self.bfm_dark = None  # monitor-PD baseline, taken with the laser off
        self.tele = {}  # cached readings, refreshed off-thread
        self.last_error = None

    @property
    def connected(self):
        return self.dev is not None

    # -- lifecycle ------------------------------------------------------------------
    def connect(self, mock: bool, port: str | None):
        self.disconnect()
        dev = MockLaser(port) if mock else Laser(port=port)
        dev.open()
        self.dev = dev
        self.mock = mock
        self.port = dev.dev.port
        with self.lock:
            self.info = {
                "port": self.port,
                "mock": mock,
                "address": _safe(lambda: dev.dev.read_address()),
                "version": _safe(lambda: dev.dev.version()),
                "preflight": _safe(lambda: dev.preflight(), {}),
                "sp_max": dev.sp_max,
                "pmax_dbm": dev.PMAX_DBM,
                "cal": {"a": dev.CAL_A, "b": dev.CAL_B},
            }
            if not dev.is_on():  # only meaningful while dark
                self.bfm_dark = _safe(lambda: float(dev.dev.measure("bfm_optical_power")))
        self.refresh_telemetry()
        return self.info

    def disconnect(self):
        dev, self.dev = self.dev, None
        self._cancel_wd()
        if dev is None:
            return
        with self.lock:
            try:
                dev.hw_off()  # never leave a laser lit behind a closed socket
            except Exception as e:  # noqa: BLE001 -- report, still release the port
                self.last_error = f"hw_off on disconnect: {e}"
            try:
                dev.close()
            except Exception:
                pass
        self.deadline = None
        self.tele = {}
        self.info = {}

    # -- watchdog -------------------------------------------------------------------
    def _cancel_wd(self):
        if self._wd is not None:
            self._wd.cancel()
            self._wd = None

    def _arm(self, duration_s: float):
        self._cancel_wd()
        d = max(1.0, min(float(duration_s), MAX_DURATION_S))
        self.duration_s = d
        self.deadline = time.time() + d
        self._wd = threading.Timer(d, self._force_off)
        self._wd.daemon = True
        self._wd.start()

    def _force_off(self):
        """Fires in the timer thread. Must never raise."""
        try:
            with self.lock:
                if self.dev is not None:
                    self.dev.hw_off()
            self.last_error = None
        except Exception as e:  # noqa: BLE001 -- a watchdog must not die silently
            self.last_error = f"watchdog hw_off failed: {e}"
        finally:
            self.deadline = None
            self._wd = None
        self.refresh_telemetry()

    def remaining(self):
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    # -- control --------------------------------------------------------------------
    def on(self, duration_s: float | None):
        if not self.connected:
            raise LaserError("laser not connected")
        with self.lock:
            if not self.dev.is_on() and self.bfm_dark is None:
                self.bfm_dark = _safe(
                    lambda: float(self.dev.dev.measure("bfm_optical_power")))
            self.dev.hw_on()
        self._arm(duration_s or self.duration_s)
        self.refresh_telemetry()

    def off(self):
        if not self.connected:
            raise LaserError("laser not connected")
        self._cancel_wd()
        with self.lock:
            self.dev.hw_off()
        self.deadline = None
        self.refresh_telemetry()

    def power(self, dbm: float):
        if not self.connected:
            raise LaserError("laser not connected")
        with self.lock:
            self.dev.set(float(dbm))
        self.refresh_telemetry()

    def extend(self, duration_s: float | None):
        if not self.connected or self.deadline is None:
            raise LaserError("laser is not on -- nothing to extend")
        self._arm(duration_s or self.duration_s)

    # -- telemetry ------------------------------------------------------------------
    def refresh_telemetry(self):
        if not self.connected:
            self.tele = {}
            return self.tele
        dev = self.dev
        with self.lock:
            if dev is not self.dev or self.dev is None:
                return self.tele  # disconnected while we waited
            on = bool(_safe(dev.is_on, False))
            sp = _safe(lambda: float(dev.dev.read_setting("cw_current")), 0.0)
            mA = _safe(dev.measured_mA, 0.0)
            bfm = _safe(lambda: float(dev.dev.measure("bfm_optical_power")))
            pre = _safe(lambda: dev.preflight(), {})
            temp = _safe(lambda: float(dev.dev.measure("diode_temperature")))
            if on:  # keepalive: the driver self-disables when idle
                _safe(lambda: dev.dev.write_setting("cw_current", sp, verify=False))
        dbm, mW = dev.setpoint_to_dbm(sp) if sp > 0 else (None, 0.0)
        emitted = None
        if bfm is not None and self.bfm_dark is not None:
            emitted = bool(on and (bfm - self.bfm_dark) > EMIT_EPS)
        self.tele = {
            "on": on, "setpoint": sp, "mA": mA, "bfm": bfm, "bfm_dark": self.bfm_dark,
            "dbm": dbm, "mW": mW, "emitted": emitted, "preflight": pre,
            "diode_temp": temp,
        }
        return self.tele

    def optical_mW(self) -> float:
        """What the mock PIC should see. 0 unless the diode is really lasing."""
        if not self.connected:
            return 0.0
        if self.mock and hasattr(self.dev, "emitting"):
            return float(self.tele.get("mW") or 0.0) if self.dev.emitting() else 0.0
        return float(self.tele.get("mW") or 0.0) if self.tele.get("emitted") else 0.0

    def snapshot(self):
        return {
            "connected": self.connected, "mock": self.mock, "port": self.port,
            "info": self.info, "tele": self.tele, "remaining": self.remaining(),
            "duration_s": self.duration_s, "deadline": self.deadline,
            "safe_dbm": SAFE_DBM, "max_duration_s": MAX_DURATION_S,
            "last_error": self.last_error,
        }


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 -- telemetry must never take the server down
        return default


LASER = LaserSession()


# ----------------------------------------------------------------------------- PIC side
def _mock_forward():
    """Smooth bounded 64->14 map, scaled by however much light the laser is actually
    putting in -- so a mocked chain reads dark until the mocked laser really lases."""
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.15, (NUM_ADC_RAW, NUM_DAC))
    b = rng.uniform(0.0, 2 * np.pi, NUM_ADC_RAW)
    dead = np.array([i in DAMAGED_PDS for i in range(NUM_ADC_RAW)])
    floor = 0.004  # ADC dark floor

    def fwd(v):
        y = 0.015 + 0.05 * (1 + np.sin(W @ (np.asarray(v) / VMAX) + b))
        if LASER.connected:
            scale = min(1.0, LASER.optical_mW() / 3.16)  # 3.16 mW == +5 dBm
        else:
            scale = 1.0  # no laser attached: exercise the chip UI on its own
        y = floor + (y - floor) * scale
        y[dead] = 0.001
        return y

    return fwd


def _looks_like_pic(port: str) -> bool:
    """The Arduino's CH340 enumerates with a purely numeric serial (…-1120); the laser's
    FTDI has an alphanumeric one (…-AU05XLI8). find_port's glob matches both."""
    return bool(re.fullmatch(r"\d+", port.rsplit("-", 1)[-1]))


def _pic_port(explicit):
    """PIC (CH340) port, never the laser's FTDI -- whether or not the laser is attached."""
    if explicit:
        return explicit, [explicit]
    env = os.environ.get("PIC_PORT")
    if env:
        return env, [env]
    _, cands = find_port(None)
    cands = [c for c in cands if c != (LASER.port or "")]
    preferred = [c for c in cands if _looks_like_pic(c)]
    pool = preferred or cands  # fall back only if nothing looks like a CH340
    return (pool[0] if pool else None), cands


class Session:
    def __init__(self):
        self.pic = None
        self.mock = True
        self.port = None
        self.vec = np.zeros(NUM_DAC)
        self.adc = np.zeros(NUM_ADC_RAW)
        self.hold = False
        self.info = {}

    @property
    def connected(self):
        return self.pic is not None

    def connect(self, mock, port):
        self.disconnect(zero=False)
        if mock:
            self.pic = MockPIC(_mock_forward(), noise=3e-4, voltage_max=VMAX).open()
            self.port = "mock"
        else:
            p, cands = _pic_port(port)
            if not p:
                raise PICError(f"no PIC serial port found (looked at {cands or 'none'}; "
                               "the laser's FTDI is excluded)")
            self.pic = PIC(port=p, voltage_max=VMAX).open()
            self.port = p
        self.mock = mock
        self.info = {"port": self.port, "mock": mock, "baud": 115200,
                     "num_dac": NUM_DAC, "num_adc": NUM_ADC_RAW,
                     "dead_pds": list(DAMAGED_PDS)}
        self.vec = np.zeros(NUM_DAC)
        self.adc = np.asarray(self.pic.measure_raw(self.vec), float)

    def disconnect(self, zero):
        pic, self.pic = self.pic, None
        self.info = {}
        if pic is None:
            return
        if self.mock or zero or not self.hold:
            pic.close()  # PIC.close() zeros the DACs
            return
        ser = getattr(pic, "ser", None)  # hold: drop the handle without zeroing
        if ser not in (None, "mock"):
            try:
                ser.close()
            except Exception:
                pass
        pic.ser = None

    def apply(self):
        self.adc = np.asarray(self.pic.measure_raw(self.vec), float)
        return self.adc

    def snapshot(self):
        p, cands = _pic_port(None)
        return {
            "connected": self.connected, "mock": self.mock, "port": self.port,
            "hold": self.hold, "ports": cands,
            "auto_port": p, "info": self.info,
            "vec": [round(float(x), 3) for x in self.vec],
            "adc": [round(float(x), 4) for x in self.adc],
            "dead": list(DAMAGED_PDS), "num_dac": NUM_DAC, "num_adc": NUM_ADC_RAW,
            "vmax": VMAX, "vmin": VOLTAGE_MIN,
        }


SESSION = Session()


def _state():
    return {"pic": SESSION.snapshot(), "laser": LASER.snapshot(),
            "wiring": WIRING.as_json()}


def _measure_payload():
    return {"vec": [round(float(x), 3) for x in SESSION.vec],
            "adc": [round(float(x), 4) for x in SESSION.adc]}


# ------------------------------------------------------------------------- API handlers
def api_pic_connect(body):
    with LOCK:
        SESSION.connect(bool(body.get("mock", True)), body.get("port") or None)
    return _state()


def api_pic_disconnect(body):
    with LOCK:
        SESSION.disconnect(zero=bool(body.get("zero", True)))
    return _state()


def api_laser_connect(body):
    LASER.connect(bool(body.get("mock", True)), body.get("port") or None)
    return _state()


def api_laser_disconnect(_body):
    LASER.disconnect()
    return _state()


def api_laser_on(body):
    LASER.on(body.get("duration_s"))
    return _state()


def api_laser_off(_body):
    LASER.off()
    return _state()


def api_laser_power(body):
    dbm = float(body.get("dbm", 0.0))
    if dbm > SAFE_DBM and not body.get("unlock"):
        return {"error": f"{dbm:+.1f} dBm exceeds the {SAFE_DBM:+.0f} dBm PD-safe "
                         "policy -- pass unlock to override"}
    LASER.power(dbm)
    return _state()


def api_laser_extend(body):
    LASER.extend(body.get("duration_s"))
    return _state()


def api_set(body):
    if not SESSION.connected:
        return {"error": "PIC not connected"}
    with LOCK:
        if "vec" in body:
            v = np.asarray(body["vec"], float).ravel()
            if v.size != NUM_DAC:
                return {"error": f"expected {NUM_DAC} values, got {v.size}"}
            SESSION.vec = np.array([_clip(x) for x in v])
        for k, val in (body.get("channels") or {}).items():
            SESSION.vec[int(k)] = _clip(float(val))
        SESSION.apply()
    return _measure_payload()


def api_measure(_body):
    if not SESSION.connected:
        return {"error": "PIC not connected"}
    with LOCK:
        SESSION.apply()
    return _measure_payload()


def api_zero(_body):
    if not SESSION.connected:
        return {"error": "PIC not connected"}
    with LOCK:
        SESSION.vec = np.zeros(NUM_DAC)
        SESSION.apply()
    return _measure_payload()


def api_geometry(_body):
    return {"scene": SCENE, "wiring": WIRING.as_json()}


ROUTES = {
    "/api/state": lambda b: _state(),
    "/api/geometry": api_geometry,
    "/api/pic/connect": api_pic_connect,
    "/api/pic/disconnect": api_pic_disconnect,
    "/api/laser/connect": api_laser_connect,
    "/api/laser/disconnect": api_laser_disconnect,
    "/api/laser/on": api_laser_on,
    "/api/laser/off": api_laser_off,
    "/api/laser/power": api_laser_power,
    "/api/laser/extend": api_laser_extend,
    "/api/set": api_set,
    "/api/measure": api_measure,
    "/api/zero": api_zero,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, "application/json", json.dumps(obj).encode())

    def _static(self, path):
        fn = "index.html" if path == "/" else path.lstrip("/")
        full = os.path.join(UIDIR, fn)
        if not os.path.isfile(full):
            self._send(404, "text/plain", b"not found")
            return
        ctype = "text/html" if fn == "index.html" else STATIC.get(path, "text/plain")
        with open(full, "rb") as f:
            self._send(200, ctype, f.read())

    def _route(self, body):
        try:
            self._json(ROUTES[self.path](body))
        except (LaserError, PICError, PDMv5Error) as e:
            self._json({"error": str(e)}, 200)  # expected device refusals, not crashes
        except Exception as e:  # noqa: BLE001 -- report failures, don't 500-crash
            self._json({"error": str(e)}, 500)

    def do_GET(self):
        if self.path in ROUTES:
            self._route(None)
        elif self.path == "/favicon.ico":
            self._send(204, "text/plain", b"")
        else:
            self._static(self.path)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if self.path in ROUTES:
            self._route(body)
        else:
            self._send(404, "text/plain", b"not found")


def _telemetry_loop(stop: threading.Event):
    """Keeps laser readings fresh (and the laser lit) without blocking /api/state."""
    period = 1.0 / TELEMETRY_HZ
    while not stop.wait(period):
        if LASER.connected:
            _safe(LASER.refresh_telemetry)


def main(argv):
    port = 8787
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    if "--mock" in argv:
        for name, fn in (("laser", lambda: LASER.connect(True, None)),
                         ("PIC", lambda: SESSION.connect(True, None))):
            try:
                fn()
                print(f"  mock {name} attached")
            except Exception as e:  # noqa: BLE001
                print(f"  mock {name} failed: {e}")

    stop = threading.Event()
    tele = threading.Thread(target=_telemetry_loop, args=(stop,), daemon=True)
    tele.start()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"PIC console at {url}   (Ctrl-C to quit)")
    if not WIRING.verified:
        print(f"  note: DAC->heater map is provisional ({WIRING.source}); "
              "heater positions on the diagram are a guess, voltages are real.")
    if "--no-open" not in argv:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    # A lit laser must not survive the server. Ctrl-C raises KeyboardInterrupt, but a plain
    # `kill` or a closed terminal would otherwise skip cleanup entirely, so catch those too.
    # shutdown() has to run off the serving thread; it makes serve_forever return normally.
    def _bye(signum, _frame):
        print(f"\nsignal {signum}: shutting down (laser off, DACs zeroed).")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError, AttributeError):
            pass  # not the main thread, or the platform lacks it

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down: laser off, DACs zeroed.")
    finally:
        stop.set()
        LASER.disconnect()  # drives the laser safe
        with LOCK:
            SESSION.disconnect(zero=True)
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
