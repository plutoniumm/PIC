"""DAC-channel -> photodiode influence map (eta^2, main effect only -- a coarse
routing map that under-counts purely interferometric channels)."""

from __future__ import annotations
import numpy as np


def eta2_matrix(X, Y) -> np.ndarray:
    """Per (channel, output) fraction of output variance explained by that channel
    alone (ANOVA eta^2). Returns E[n_channels, n_outputs] in [0, 1]."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n_ch, n_out = X.shape[1], Y.shape[1]
    E = np.zeros((n_ch, n_out))
    for k in range(n_out):
        y = Y[:, k]
        tot = y.var()
        if tot == 0:
            continue
        gm = y.mean()
        for c in range(n_ch):
            xc = X[:, c]
            between = 0.0
            for L in np.unique(xc):
                sel = xc == L
                between += sel.mean() * (y[sel].mean() - gm) ** 2
            E[c, k] = between / tot
    return E


def top_channels(E, output_idx: int, n: int = 10, thresh: float = 1e-4):
    """Indices of the ``n`` most-influential channels for one output (eta^2 > thresh)."""
    order = np.argsort(-E[:, output_idx])
    return [int(c) for c in order if E[c, output_idx] > thresh][:n]


def dead_channels(E, thresh: float = 0.01):
    """Channels whose total influence across all outputs is below ``thresh``."""
    return list(np.where(E.sum(axis=1) < thresh)[0])
