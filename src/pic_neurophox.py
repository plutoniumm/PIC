"""Neurophox replica of the full PIC — a runnable optical SVD  M = U · Σ · V.

Pipeline, matching the chip / the schematic: an input vector is launched through the
splitting tree, transformed by the 6×6 Clements mesh **U**, attenuated by the diagonal
**Σ** bank, transformed by the second 6×6 Clements mesh **V**, and detected as
photodiode intensities |field|².  This module builds that whole chain in neurophox and
runs it end to end — a structural, full replica of the packaged chip.

    from src.pic_neurophox import PICReplica
    pic = PICReplica(seed=0)              # random (uncalibrated) mesh phases
    y   = pic.readout([1, 0, 0, 0, 0, 0]) # 6 photodiode intensities for an input
    pic.diagram("pic_replica.png")        # draw the structure

Calibration caveat: the heater→phase map is not characterised yet (NUS R²≈78%, being
redone), so the mesh phases are free parameters — random by default, or pass your own
via `PICReplica(seed=...)` / set them on `.U`, `.V`.  `from_voltages` is the stub a
real calibration will fill in so 64 DAC volts can drive the replica directly.

neurophox needs tensorflow (installed in the `pic` env).  The import is lazy — the
first mesh build pays a one-off ~6 s tf load — so plain `import src` stays fast.
"""

from __future__ import annotations
import numpy as np

UNITS = 6
_RMNumpy = None


def _rmnumpy():
    """neurophox's rectangular (Clements) numpy mesh class, imported lazily."""
    global _RMNumpy
    if _RMNumpy is None:
        try:
            from neurophox.numpy import RMNumpy
        except ImportError as e:  # neurophox/tensorflow absent
            raise ImportError(
                "the PIC replica needs neurophox + tensorflow; install them into the "
                "`pic` conda env"
            ) from e
        _RMNumpy = RMNumpy
    return _RMNumpy


def build_mesh(seed: int = 0, units: int = UNITS):
    """One `units`×`units` Clements (rectangular) MZI mesh — a unitary half (U or V),
    with phases seeded for reproducibility."""
    np.random.seed(seed)
    return _rmnumpy()(units=units)


def sigma_from_matrix(A: np.ndarray) -> np.ndarray:
    """Σ for a target matrix A: its singular values, normalised to ≤1 (passive
    attenuators, so divided by the max — the chip can attenuate, never amplify)."""
    sv = np.linalg.svd(np.asarray(A), compute_uv=False)
    return sv / sv.max() if sv.max() else sv


class PICReplica:
    """The full PIC: two Clements meshes around a diagonal attenuator bank.

    ``field_out = V · Σ · U · x`` — each mesh applied as a neurophox ``transform``
    (i.e. ``mesh.matrix.T @ ·``); photodiodes report ``|field_out|²``.
    """

    def __init__(self, units: int = UNITS, seed: int = 0, sigma=None):
        self.units = units
        self.U = build_mesh(seed, units)  # "beginning" mesh
        self.V = build_mesh(seed + 1, units)  # "end" mesh
        self.sigma = (
            np.ones(units) if sigma is None else sigma_from_matrix(np.diag(sigma))
        )

    def encode_sigma(self, A: np.ndarray) -> np.ndarray:
        """Program Σ to the (normalised) singular-value spectrum of a target matrix A."""
        self.sigma = sigma_from_matrix(A)
        return self.sigma

    @property
    def matrix(self) -> np.ndarray:
        """The 6×6 linear operator the configured chip implements (``field = matrix · x``)."""
        return self.V.matrix.T @ np.diag(self.sigma) @ self.U.matrix.T

    def field(self, x) -> np.ndarray:
        """Propagate the complex input field x through U → Σ → V; return the output field."""
        x = np.atleast_2d(np.asarray(x, complex))
        f = self.U.transform(x)[0]
        f = self.sigma * f
        return self.V.transform(np.atleast_2d(f))[0]

    def readout(self, x) -> np.ndarray:
        """Photodiode intensities |field|² for input x — what the chip actually reports."""
        return np.abs(self.field(x)) ** 2

    def check(self) -> dict:
        """Self-consistency: mesh unitarity, Σ reproduced by the realised matrix, and
        power conservation (Σ≤1 ⇒ sub-unitary, never gain)."""
        eye = np.eye(self.units)
        sv = np.linalg.svd(self.matrix, compute_uv=False)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self.units) + 1j * rng.standard_normal(self.units)
        x /= np.linalg.norm(x)
        y = self.readout(x)
        return {
            "U_unitary_err": float(
                np.abs(self.U.matrix.conj().T @ self.U.matrix - eye).max()
            ),
            "V_unitary_err": float(
                np.abs(self.V.matrix.conj().T @ self.V.matrix - eye).max()
            ),
            "sigma_reproduced_err": float(
                np.abs(np.sort(sv)[::-1] - np.sort(self.sigma)[::-1]).max()
            ),
            "field_vs_matrix_err": float(np.abs(self.field(x) - self.matrix @ x).max()),
            "power_in": float(np.linalg.norm(x) ** 2),
            "power_out": float(y.sum()),
        }

    @classmethod
    def from_voltages(cls, v, **kw):
        """(stub) Build a replica configured by the 64 DAC voltages, once the
        heater→phase calibration exists.  Blocked on re-characterization."""
        raise NotImplementedError(
            "heater→phase calibration not available yet (NUS R²≈78%, being redone); "
            "drive the replica with explicit mesh phases for now"
        )

    def diagram(self, path: str = "pic_replica.png", dpi: int = 150) -> str:
        """Render the replica's structure — splitting-tree input, the two Clements
        meshes, the Σ bank, and the photodiodes — to `path`."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        n = self.units
        fig, ax = plt.subplots(figsize=(12, 1.1 + 0.7 * n))
        ink, mzi_c, sig_c, dim = "#0a0a0a", "#2563eb", "#dc2626", "#9ca3af"

        def mesh(x0, label):
            """Draw one rectangular mesh of `n` layers starting at x0; return its width."""
            for l in range(n):  # rail segments through this mesh
                pass
            for l in range(n):  # MZIs at Clements positions
                cx = x0 + l + 0.5
                for i in range(l % 2, n - 1, 2):
                    yc = i + 0.5
                    ax.add_patch(
                        FancyBboxPatch(
                            (cx - 0.28, yc - 0.34),
                            0.56,
                            0.68,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=1.2,
                            edgecolor=mzi_c,
                            facecolor="white",
                            zorder=3,
                        )
                    )
                    ax.plot(
                        [cx - 0.28, cx + 0.28], [i, i + 1], color=mzi_c, lw=1, zorder=2
                    )
                    ax.plot(
                        [cx - 0.28, cx + 0.28], [i + 1, i], color=mzi_c, lw=1, zorder=2
                    )
            ax.text(
                x0 + n / 2,
                n - 0.35,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
                color=ink,
                weight="bold",
            )
            return n

        # rails span the whole device
        wU, wV = n, n
        xU, xSig, xV = 1.4, 1.4 + wU + 0.5, 1.4 + wU + 1.4
        x_end = xV + wV + 0.6
        for i in range(n):
            ax.plot([0.2, x_end], [i, i], color=dim, lw=1, zorder=1)
            ax.text(0.05, i, f"x{i}", ha="right", va="center", fontsize=10, color=ink)
            ax.text(
                x_end + 0.1,
                i,
                f"PD{i}",
                ha="left",
                va="center",
                fontsize=10,
                color=ink,
                family="monospace",
            )

        ax.text(
            0.55,
            n - 0.35,
            "split\ntree",
            ha="center",
            va="bottom",
            fontsize=8,
            color=dim,
        )
        mesh(xU, "U  (6×6 Clements)")
        for i in range(n):  # Σ attenuator bank
            ax.add_patch(
                FancyBboxPatch(
                    (xSig - 0.22, i - 0.22),
                    0.44,
                    0.44,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=1.2,
                    edgecolor=sig_c,
                    facecolor="white",
                    zorder=3,
                )
            )
            ax.text(
                xSig,
                i,
                "σ",
                ha="center",
                va="center",
                fontsize=9,
                color=sig_c,
                zorder=4,
            )
        ax.text(
            xSig,
            n - 0.35,
            "Σ",
            ha="center",
            va="bottom",
            fontsize=11,
            color=ink,
            weight="bold",
        )
        mesh(xV, "V  (6×6 Clements)")
        for i in range(n):  # photodiodes
            ax.plot(
                [x_end - 0.05, x_end + 0.05, x_end + 0.05, x_end - 0.05],
                [i - 0.14, i - 0.14, i + 0.14, i + 0.14],
                color=ink,
                lw=1.2,
                zorder=3,
            )

        ax.set_xlim(-0.6, x_end + 1.0)
        ax.set_ylim(-0.7, n + 0.2)
        ax.axis("off")
        ax.set_title(
            "PIC replica  —  M = U · Σ · V   (neurophox, run with .readout(x))",
            fontsize=11,
            color=ink,
            pad=8,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path


def demo():
    """Build the replica, encode a target's Σ-spectrum, run an input, verify SVD + power."""
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    pic = PICReplica(seed=1)
    print("PIC replica: two 6×6 Clements meshes (U, V) + Σ bank, built in neurophox.")

    rng = np.random.default_rng(0)
    A = rng.standard_normal((UNITS, UNITS))
    print(
        f"\nEncoded Σ to a target matrix's normalised singular values:\n  σ = {pic.encode_sigma(A)}"
    )

    c = pic.check()
    print("\nSelf-consistency:")
    print(f"  U, V unitary err   = {c['U_unitary_err']:.1e}, {c['V_unitary_err']:.1e}")
    print(f"  sv(M) reproduces σ = {c['sigma_reproduced_err']:.1e} max err")
    print(f"  field == M · x     = {c['field_vs_matrix_err']:.1e} max err")
    print(
        f"  power in / out     = {c['power_in']:.4f} / {c['power_out']:.4f}  (out ≤ in ⇒ passive)"
    )

    x = np.zeros(UNITS, complex)
    x[0] = 1.0
    print(f"\nReadout for a single-mode input (x = e0):\n  PDs = {pic.readout(x)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Neurophox full-PIC replica")
    ap.add_argument(
        "--diagram",
        nargs="?",
        const="slides/images/pic_replica.png",
        default=None,
        metavar="PATH",
        help="render the structure diagram to PATH and exit",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.diagram:
        print("wrote", PICReplica(seed=args.seed).diagram(args.diagram))
    else:
        demo()
