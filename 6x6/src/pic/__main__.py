"""Back-compat shim -- `python -m src.pic ...` now dispatches to `python -m pic ...`."""
from pic.__main__ import main

if __name__ == "__main__":
    main()
