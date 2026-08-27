"""Thin shim: the signed intensity-only matvec runner now lives in `pic.compute.matvec`.

    python scripts/intensity_matvec_hw.py --mock --n 4   # or: python -m pic matvec --mock --n 4
"""
from pic.compute.matvec import main

if __name__ == "__main__":
    raise SystemExit(main())
