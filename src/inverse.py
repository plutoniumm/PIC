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
    """Projected-gradient optimiser over a differentiable torch objective.

    ``objective(v)`` maps a leaf tensor of decision variables to a scalar tensor to
    MAXIMISE; gradients come from autograd (backprop through a frozen surrogate). After
    each Adam step ``v`` is projected back onto the box ``[vmin, vmax]``. This is the
    gradient counterpart to :class:`MonteCarloInverse`; ``scripts/pic_gd.py`` builds the
    surrogate-DPNN objective and drives this. torch is imported lazily so importing this
    module stays torch-free."""

    def __init__(self, objective, vmin: float = 0.0, vmax: float = 4.0):
        self.objective = objective
        self.vmin, self.vmax = vmin, vmax

    def run(self, v0, iters: int = 400, lr: float = 0.05, log_every: int = 50,
            verbose: bool = False):
        import torch

        v = torch.tensor(np.asarray(v0, float), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr)
        hist = []
        for it in range(iters):
            opt.zero_grad()
            obj = self.objective(v)
            (-obj).backward()
            opt.step()
            with torch.no_grad():
                v.clamp_(self.vmin, self.vmax)
            hist.append(float(obj.detach()))
            if verbose and log_every and (it % log_every == 0 or it == iters - 1):
                print(f"  iter {it + 1:4d}/{iters}: objective {hist[-1]:+.5f}")
        with torch.no_grad():
            final = float(self.objective(v))
        return {"v": v.detach().numpy(), "obj": final,
                "obj0": hist[0] if hist else final, "history": hist}
