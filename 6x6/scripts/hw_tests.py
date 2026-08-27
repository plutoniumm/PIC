"""Thin shim: the PIC hardware bring-up tests now live in `pic.bringup`.

    python scripts/hw_tests.py --mock dark --samples 300   # or: python -m pic bringup ...
"""
import sys
from pic.bringup import hwtests_main

if __name__ == "__main__":
    sys.exit(hwtests_main() or 0)
