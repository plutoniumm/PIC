"""Thin shim: the thermal-settling + loop-speed characterization now lives in `pic.bringup`.

    python scripts/settling.py step --selftest   # or: python -m pic bringup step --selftest
"""
import sys
from pic.bringup import settling_main

if __name__ == "__main__":
    sys.exit(settling_main())
