"""pic.compute -- hardware optical-compute runners (Ising + signed matvec).

These are opt-in submodules: they pull heavy deps (`torch`, `theory`, `scipy`) and
are deliberately NOT imported by `pic/__init__.py`, so `import pic` / `python -m pic
measure` stay lightweight. Import the runner you want directly:

    from pic.compute import ising, matvec
    ising.main(["--mock", "--n", "3"])      # offline pipeline check

The math kernels live in the pure-math `theory` package; these modules only wire
those kernels to the real rig (laser + PIC) through `pic.session`.
"""
