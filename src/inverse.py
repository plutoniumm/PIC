"""Inverse solvers: find DAC voltages that produce a desired photodiode output.

`MonteCarloInverse` reproduces the original heuristic (seed-perturb-keep-if-better,
annealed step) but drives any ``measure(v) -> output`` callable (PIC/MockPIC/surrogate).
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


class MonteCarloInverse:
    def __init__(self, vmin=0.0, vmax=4.0, step=0.5, perturb_prob=0.3, seed=0):
        self.vmin, self.vmax, self.step = vmin, vmax, step
        self.perturb_prob = perturb_prob
        self.rng = np.random.default_rng(seed)

    def seed_from_dataset(self, dataset_outputs, desired, dataset_inputs, k: int = 10):
        """Return the k dataset inputs whose outputs are nearest the target."""
        d = cdist(np.atleast_2d(desired), np.asarray(dataset_outputs))[0]
        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]
        return [np.asarray(dataset_inputs)[i] for i in idx]

    def perturb(self, v, strength: float = 1.0):
        out = np.asarray(v, float).copy()
        for i in range(out.size):
            if self.rng.random() < self.perturb_prob:
                change = self.rng.choice([-1.0, -0.5, 0.5, 1.0]) * strength
                x = np.clip(out[i] + change, self.vmin, self.vmax)
                out[i] = round(x / self.step) * self.step
        return out

    def run(
        self,
        measure,
        desired,
        seeds,
        iters: int = 50,
        per_seed: int = 3,
        early_stop: float = 0.01,
        verbose: bool = False,
    ):
        """Optimise. ``measure(v)->output``; ``desired`` and outputs share a space."""
        desired = np.asarray(desired, float)
        best = {"v": None, "y": None, "loss": np.inf, "iter": 0}
        current = [np.asarray(s, float) for s in seeds]
        for it in range(iters):
            strength = 1.0 / np.sqrt(it + 1)
            cands = [
                self.perturb(s, strength) for s in current for _ in range(per_seed)
            ]
            for v in cands:
                y = np.asarray(measure(v), float)
                loss = float(np.linalg.norm(y - desired))
                if loss < best["loss"]:
                    best.update(v=v.copy(), y=y.copy(), loss=loss, iter=it + 1)
            if verbose:
                print(f"  iter {it + 1}/{iters}: best loss {best['loss']:.4f}")
            if best["loss"] < early_stop:
                break
            current = [best["v"]] + [
                self.perturb(best["v"]) for _ in range(max(0, len(seeds) - 1))
            ]
        return best


class GradientInverse:
    """Gradient-based inverse on a differentiable forward surrogate (grey-box or NN). Stub."""

    def __init__(self, forward):
        self.forward = forward

    def run(self, *args, **kw):
        raise NotImplementedError(
            "gradient inverse pending a differentiable forward model"
        )
