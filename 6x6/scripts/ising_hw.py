"""Thin shim: the hardware Ising runner now lives in `pic.compute.ising`.

    python scripts/ising_hw.py --mock              # or: python -m pic ising --mock
"""
from pic.compute.ising import main

if __name__ == "__main__":
    raise SystemExit(main())
