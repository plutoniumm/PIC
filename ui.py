"""Interactive browser console for the PIC.

    python ui.py            # open http://localhost:8787 in your browser
    python ui.py --port N   # serve on a different HTTP port
    python ui.py --no-open  # don't auto-open the browser

The page lets you pick a serial port (or Mock), set the 64 DAC heater voltages by
hand (sliders / numbers), watch the 14 photodiodes live, and run `.pic` scripts with
streamed output. It is a thin shell over `src.pic.PIC` and the `picscript` parser --
no build step, no node_modules; the front end is plain Vue 3 served from ./ui.
"""

from __future__ import annotations
import json
import os
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
    find_port,
    NUM_DAC,
    NUM_ADC_RAW,
    VOLTAGE_MIN,
    DAMAGED_PDS,
)
from picscript import parse

VMAX = 5.0
HERE = os.path.dirname(os.path.abspath(__file__))
UIDIR = os.path.join(HERE, "ui")
STATIC = {
    "/app.js": "text/javascript",
    "/style.css": "text/css",
    "/vue.global.prod.js": "text/javascript",
}

LOCK = threading.Lock()  # serialises all hardware access
STOP = threading.Event()  # aborts a running script
RUNNING = threading.Event()  # a script is streaming


def _clip(v):
    return float(np.clip(v, VOLTAGE_MIN, VMAX))


def _mock_forward():
    """A smooth, bounded 64->14 map so the mock board's PD bars respond to edits."""
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.15, (NUM_ADC_RAW, NUM_DAC))
    b = rng.uniform(0.0, 2 * np.pi, NUM_ADC_RAW)
    dead = np.array([i in DAMAGED_PDS for i in range(NUM_ADC_RAW)])

    def fwd(v):
        y = 0.015 + 0.05 * (1 + np.sin(W @ (np.asarray(v) / VMAX) + b))
        y[dead] = 0.001
        return y

    return fwd


class Session:
    def __init__(self):
        self.pic = None
        self.mock = True
        self.port = None
        self.vec = np.zeros(NUM_DAC)
        self.adc = np.zeros(NUM_ADC_RAW)
        self.hold = False  # hold DACs (don't zero) on disconnect

    @property
    def connected(self):
        return self.pic is not None

    def connect(self, mock, port):
        self.disconnect(zero=False)
        if mock:
            self.pic = MockPIC(_mock_forward(), noise=3e-4, voltage_max=VMAX).open()
            self.port = "mock"
        else:
            p, cands = find_port(port)
            if not p:
                raise RuntimeError(
                    f"no serial port found (looked at {cands or 'none'})"
                )
            self.pic = PIC(port=p, voltage_max=VMAX).open()
            self.port = p
        self.mock = mock
        self.vec = np.zeros(NUM_DAC)
        self.adc = np.asarray(self.pic.measure_raw(self.vec), float)

    def disconnect(self, zero):
        pic, self.pic = self.pic, None
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
        p, cands = find_port(None)
        return {
            "connected": self.connected,
            "mock": self.mock,
            "port": self.port,
            "hold": self.hold,
            "running": RUNNING.is_set(),
            "ports": cands,
            "auto_port": p,
            "vec": [round(float(x), 3) for x in self.vec],
            "adc": [round(float(x), 4) for x in self.adc],
            "dead": list(DAMAGED_PDS),
            "num_dac": NUM_DAC,
            "num_adc": NUM_ADC_RAW,
            "vmax": VMAX,
            "vmin": VOLTAGE_MIN,
        }


SESSION = Session()


def _measure_payload():
    return {
        "vec": [round(float(x), 3) for x in SESSION.vec],
        "adc": [round(float(x), 4) for x in SESSION.adc],
    }


def api_connect(body):
    with LOCK:
        SESSION.connect(bool(body.get("mock", True)), body.get("port") or None)
    return SESSION.snapshot()


def api_disconnect(body):
    with LOCK:
        SESSION.disconnect(zero=bool(body.get("zero", False)))
    return SESSION.snapshot()


def api_poll(body):
    """Scan for a board and connect to it in one round trip. With ``mock`` set,
    attaches the simulator; otherwise auto-detects the serial device (honouring an
    optional ``port`` override) and opens it. A no-op if already connected; if no
    device is found, returns the snapshot with the ports seen under ``detected``."""
    with LOCK:
        if SESSION.connected:
            return SESSION.snapshot()
        if bool(body.get("mock", False)):
            SESSION.connect(True, None)
            return SESSION.snapshot()
        port, cands = find_port(body.get("port") or None)
        if port:
            SESSION.connect(False, port)
        return {**SESSION.snapshot(), "detected": cands}


def api_set(body):
    if not SESSION.connected:
        return {"error": "not connected"}
    if RUNNING.is_set():
        return {"error": "script running"}
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
        return {"error": "not connected"}
    if RUNNING.is_set():
        return {**_measure_payload(), "running": True}
    with LOCK:
        SESSION.apply()
    return _measure_payload()


def api_zero(_body):
    if not SESSION.connected:
        return {"error": "not connected"}
    if RUNNING.is_set():
        STOP.set()
        return {"stopping": True}
    with LOCK:
        SESSION.vec = np.zeros(NUM_DAC)
        SESSION.apply()
    return _measure_payload()


def api_stop(_body):
    STOP.set()
    return {"stopping": True}


def api_examples(_body):
    seen, out = set(), []
    for d in ("examples", "scripts", "."):  # wherever .pic files live
        base = os.path.join(HERE, d)
        for fn in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if fn.endswith(".pic") and fn not in seen:
                seen.add(fn)
                with open(os.path.join(base, fn)) as f:
                    out.append({"name": fn, "text": f.read()})
    return {"examples": out}


def _nonzero(vec):
    return {i: round(float(x), 3) for i, x in enumerate(vec) if x}


def run_script_stream(text, emit):
    """Parse and execute a `.pic` script on the live connection, emitting one JSON
    event per pulse. Holds the hardware lock for the whole run so manual sets queue."""
    if not SESSION.connected:
        emit({"type": "error", "msg": "not connected -- connect a board first"})
        return
    try:
        blocks = parse(text)
    except Exception as e:  # noqa: BLE001 -- surface the parse message to the UI
        emit({"type": "error", "msg": f"parse error: {e}"})
        return
    runs = [r for b in blocks for r in b.runs()]
    if not runs:
        emit({"type": "error", "msg": "no runnable blocks found"})
        return
    total = sum(
        s.get("loop", 0) * (max(1, int(s.get("iters", 1))) - 1) for _, _, s in runs
    )
    emit(
        {
            "type": "plan",
            "blocks": len(blocks),
            "runs": len(runs),
            "eta": round(total, 1),
            "items": [
                {
                    "name": n,
                    "channels": len(_nonzero(v)),
                    "iters": int(s.get("iters", 1)),
                    "loop": s.get("loop", 0),
                }
                for n, v, s in runs
            ],
        }
    )
    STOP.clear()
    with LOCK:
        RUNNING.set()
        try:
            for name, vec, settings in runs:
                if STOP.is_set():
                    break
                loop = settings.get("loop", 0.0)
                iters = max(1, int(settings.get("iters", 1)))
                settle = settings.get("settle", 0.0)
                SESSION.vec = vec.copy()
                emit(
                    {
                        "type": "run",
                        "name": name,
                        "iters": iters,
                        "loop": loop,
                        "vec": [round(float(x), 3) for x in vec],
                    }
                )
                t0 = time.time()
                for i in range(iters):
                    if STOP.is_set():
                        break
                    if settle:
                        SESSION.pic.measure_raw(vec)
                        if STOP.wait(settle):
                            break
                    adc = np.asarray(SESSION.pic.measure_raw(vec), float)
                    SESSION.adc = adc
                    emit(
                        {
                            "type": "sample",
                            "name": name,
                            "i": i,
                            "iters": iters,
                            "t": round(time.time() - t0, 2),
                            "adc": [round(float(x), 4) for x in adc],
                        }
                    )
                    if loop and i < iters - 1 and STOP.wait(loop):
                        break
        except (
            Exception
        ) as e:  # noqa: BLE001 -- surface a serial/hardware fault to the UI
            emit({"type": "error", "msg": f"hardware fault mid-run: {e}"})
        finally:
            RUNNING.clear()
    emit({"type": "done", "aborted": STOP.is_set()})


ROUTES = {
    "/api/state": lambda b: SESSION.snapshot(),
    "/api/connect": api_connect,
    "/api/poll": api_poll,
    "/api/disconnect": api_disconnect,
    "/api/set": api_set,
    "/api/measure": api_measure,
    "/api/zero": api_zero,
    "/api/stop": api_stop,
    "/api/examples": api_examples,
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
        except (
            Exception
        ) as e:  # noqa: BLE001 -- report failures to the UI, don't 500-crash
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
        if self.path == "/api/script/run":
            self._stream_script(body)
        elif self.path in ROUTES:
            self._route(body)
        else:
            self._send(404, "text/plain", b"not found")

    def _stream_script(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(ev):
            try:
                self.wfile.write((json.dumps(ev) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                STOP.set()  # client hung up -> abort the run

        run_script_stream(body.get("text", ""), emit)


def main(argv):
    port = 8787
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"PIC console at {url}   (Ctrl-C to quit)")
    if "--no-open" not in argv:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down; zeroing DACs if connected.")
        with LOCK:
            SESSION.disconnect(zero=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
