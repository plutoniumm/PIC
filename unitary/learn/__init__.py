"""The physics surrogate for the 4x4 chip.

    from learn import unitary_fit

`unitary_fit` is the physics: 52 parameters, unitary by construction, invertible in closed
form. It needs torch, so nothing imports it at module scope -- `import pic` stays light.

`python -m learn.train_hw` fits it from hardware.
"""

__all__ = ["train_hw", "unitary_fit"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
