"""Thin shim: the paired baseline-vs-corrected Ising A/B runner now lives in
`pic.compute.ising` (as `main_ab`).

    python scripts/ising_ab.py --n 4 --instances 6 --config-base <frozen>.json ...
"""
from pic.compute.ising import main_ab

if __name__ == "__main__":
    raise SystemExit(main_ab())
