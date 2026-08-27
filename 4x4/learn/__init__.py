"""Surrogate models for the 4x4 chip, in both flavours.

    from learn import unitary_fit, dpnn

`unitary_fit` is the physics: 56 parameters, unitary by construction, invertible in closed
form. `dpnn` is the pruned network carried over from the 6x6 rig, reduced to this chip's
width. `prune` is the pruning core both share. Everything here needs torch, so nothing
imports it at module scope -- `import pic` stays light.

`python -m learn.train_hw` trains both from hardware in the same loop.
"""

__all__ = ["dpnn", "prune", "train_hw", "unitary_fit"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
