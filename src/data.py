"""Dataset loading for the 100k characterization set, with an npz cache.
Source layout: col 0 = 64 comma-separated input voltages, cols 1..14 = 14 raw PDs."""

from __future__ import annotations
import os
import numpy as np

DAMAGED_PDS = (0, 2, 7, 11)
LIVE_PDS = [i for i in range(14) if i not in DAMAGED_PDS]


def load_xlsx(path: str, cache: str | None = "/tmp/pic_100k.npz"):
    """Load (X[N,64], Y[N,14]) from the xlsx, caching to npz for fast reloads.

    If `100k_data_original.xlsx` is absent it is reconstructed from the four
    22-July `pic_data` sessions (the canonical source it deduped to)."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["Y"]
    if not os.path.exists(path) and "100k_data_original" in os.path.basename(path):
        import glob

        base = os.path.dirname(path) or "."
        sess = sorted(glob.glob(os.path.join(base, "pic_data", "*22july*.xlsx")))
        if not sess:
            raise FileNotFoundError(
                f"{path} removed and no pic_data 22-July sessions found to rebuild it"
            )
        XY = [load_session(f) for f in sess]
        X = np.vstack([x for x, _ in XY])
        Y = np.vstack([y for _, y in XY])
        if cache:
            np.savez_compressed(cache, X=X, Y=Y)
        return X, Y
    import pandas as pd

    df = pd.read_excel(path, header=None)
    X = np.stack(df[0].map(lambda s: np.fromstring(s, sep=",")).values)
    Y = df.iloc[:, 1:15].to_numpy(dtype=float)
    if cache:
        np.savez_compressed(cache, X=X, Y=Y)
    return X, Y


def load_session(path: str):
    """Load one acquisition xlsx -> (X[N,64], Y[N,14]), auto-detecting and dropping
    a header row if present."""
    import pandas as pd

    df = pd.read_excel(path, header=None)
    first = str(df.iloc[0, 0])
    if (
        not first.replace(".", "")
        .replace(",", "")
        .replace("-", "")
        .replace(" ", "")
        .isdigit()
    ):
        df = df.iloc[1:].reset_index(drop=True)
    X = np.stack(df[0].map(lambda s: np.fromstring(str(s), sep=",")).values)
    Y = df.iloc[:, 1:15].to_numpy(dtype=float)
    return X, Y


def load_sessions(paths):
    """Load many sessions -> {name: (X, Y)} keyed by basename."""
    import os

    return {os.path.basename(p): load_session(p) for p in paths}


def live(Y) -> np.ndarray:
    """Drop the 4 damaged photodiodes: Y[:,14] -> Y[:,10]."""
    return np.asarray(Y)[:, LIVE_PDS]
