"""Dataset loading for the 100k characterization set, with an npz cache.

Layout of `100k_data_original.xlsx` (single sheet, no header):
    col 0       : 64 comma-separated input voltages (string)
    cols 1..14  : 14 raw photodiode voltages (PD0..PD13)
"""
from __future__ import annotations
import os
import numpy as np

DAMAGED_PDS = (0, 2, 7, 11)
LIVE_PDS = [i for i in range(14) if i not in DAMAGED_PDS]


def load_xlsx(path: str, cache: str | None = "/tmp/pic_100k.npz"):
    """Load (X[N,64], Y[N,14]) from the xlsx, caching to npz for fast reloads."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["Y"]
    import pandas as pd
    df = pd.read_excel(path, header=None)
    X = np.stack(df[0].map(lambda s: np.fromstring(s, sep=",")).values)
    Y = df.iloc[:, 1:15].to_numpy(dtype=float)
    if cache:
        np.savez_compressed(cache, X=X, Y=Y)
    return X, Y


def live(Y) -> np.ndarray:
    """Drop the 4 damaged photodiodes: Y[:,14] -> Y[:,10]."""
    return np.asarray(Y)[:, LIVE_PDS]
