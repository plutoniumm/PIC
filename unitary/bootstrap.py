"""Set up and run the rig on any OS, with nothing but a Python 3.11+ on PATH.

    python bootstrap.py          # .venv with the requirements (uv if installed, else venv+pip)
    python bootstrap.py run      # the UI on http://127.0.0.1:8744, armed
    python bootstrap.py test     # every self-test, no hardware
    python bootstrap.py ports    # each USB instrument, its chip id and its port

The Makefile calls this, so macOS and Linux keep `make run`; Windows has no make, touch or
rm, and this needs none of them. Everything runs in Python's UTF-8 mode (-X utf8): our logs
and JSON carry ±, °, → and φ, and Windows' default cp1252 would mangle files and crash the
console on them.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ = HERE / "requirements.txt"


def works() -> bool:
    """A .venv copied from another machine or OS is dead weight: its interpreter points at a
    Python that is not here. Try it rather than trust that it exists."""
    try:
        return subprocess.call([str(PY), "-c", "import numpy, serial, torch"], cwd=HERE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def venv():
    stamp = VENV / ".req"
    if PY.exists() and stamp.exists() and stamp.read_bytes() == REQ.read_bytes() and works():
        return
    if VENV.exists() and not works():
        print("rebuilding .venv: the one here was made on another machine or is broken")
        shutil.rmtree(VENV, ignore_errors=True)
    uv = shutil.which("uv")
    if uv:
        subprocess.check_call([uv, "venv", "-q", "--allow-existing", str(VENV)])
        subprocess.check_call([uv, "pip", "install", "-q", "--python", str(PY), "-r", str(REQ)])
    else:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        subprocess.check_call([str(PY), "-m", "pip", "install", "-q", "-r", str(REQ)])
    stamp.write_bytes(REQ.read_bytes())  # rebuilt only when the requirements change


def run(*args):
    p = subprocess.Popen([str(PY), "-X", "utf8", *args], cwd=HERE)
    while True:
        try:
            return p.wait()
        except KeyboardInterrupt:
            # The child got the same Ctrl-C and is shutting the rig down (laser off, TEC
            # released). subprocess.call would kill it here, cutting that short; wait instead.
            continue


def main(argv):
    venv()
    cmd = argv[0] if argv else None
    if cmd == "run":
        return run("ui.py", "--arm", *argv[1:])
    if cmd == "ports":
        return run("-m", "pic", "ports")
    if cmd == "test":
        return run("-m", "pic", "selftest") or run("ui.py", "--selftest")
    if cmd is None:
        print(f"ready: {PY}")
        return 0
    raise SystemExit(f"unknown command {cmd!r}: use run, test or ports")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
