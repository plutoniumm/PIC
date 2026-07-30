"""Run the sandbox experiments and cache results.

    python -m theory.study            # everything
    python -m theory.study reach      # one section

Sections:
  reach    what 6x6 maps the mesh can realise (all heaters vs only the characterized ones)
  homo     homodyne field recovery: exactness, LO-calibration dependence, ADC floor
  matvec   signed matrix-vector product end to end
  ising    6-spin Ising energy, power route vs homodyne route
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from . import ising, matvec
from .hw import Hardware, adc_noise
from .program import fit_matrix
from .readout import lo_health, open_reference, recovery_error
from .twin import NH, NSIG, Twin

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"  -> results/{name}.json")


def sec_spectrum(nsample=400):
    """What the mesh's own singular spectrum looks like — the shape of its reachable set.
    The edge drops bleed the outer rails, so the chip is effectively lower rank than 6."""
    print("\n== reachable-set shape ==")
    tw, rng = Twin(), np.random.default_rng(0)
    S = np.array([np.linalg.svd(
        tw.matrix(torch.rand(120, generator=torch.Generator().manual_seed(t)) * 2 * torch.pi
                  ).detach().numpy(), compute_uv=False) for t in range(nsample)])
    chip = np.median(S / S[:, :1], axis=0)
    R = np.array([np.linalg.svd(matvec.random_signed_matrix(rng), compute_uv=False)
                  for _ in range(nsample)])
    gauss = np.median(R / R[:, :1], axis=0)
    eff = float((S.sum(1) ** 2 / (S ** 2).sum(1)).mean())  # participation ratio
    print(f"  chip   s_k/s_1  {np.round(chip, 4)}")
    print(f"  gauss  s_k/s_1  {np.round(gauss, 4)}")
    print(f"  effective rank (participation ratio): {eff:.2f} of 6")
    res = {"chip": chip.tolist(), "gauss": gauss.tolist(), "eff_rank": eff,
           "throughput_median": float(np.median(S[:, 0]))}
    save("spectrum", res)
    return res


def sec_reach(steps=2500, restarts=16, ntrial=4):
    """Which targets the non-unitary mesh can hit, and what losing 38 heaters costs.

    Scored both strictly and up to a left diagonal output phase — the mesh has no output
    phase shifters, so the diagonal is not a programming failure (see program.diag_match_loss).
    """
    print("\n== reachability ==")
    tw, hwd, rng = Twin(), Hardware(), np.random.default_rng(0)
    rows = []
    self_targets = [tw.matrix(torch.rand(NH, generator=torch.Generator().manual_seed(900 + t))
                              * 2 * torch.pi).detach().numpy() for t in range(ntrial)]
    cases = [("all 120 heaters", None), ("characterized only", hwd.known),
             ("drivable (incl. uncal)", hwd.driveable)]
    for label, mask in cases:
        for kind in ("self", "gauss", "pm1"):
            Ts = (self_targets if kind == "self" else
                  [matvec.random_signed_matrix(rng, kind=kind) for _ in range(ntrial)])
            fits = [fit_matrix(M, twin=tw, trainable=mask, seed=t, steps=steps,
                               restarts=restarts) for t, M in enumerate(Ts)]
            row = {"heaters": label, "target": kind,
                   "err_diag": float(np.mean([f["err_diag"] for f in fits])),
                   "err_strict": float(np.mean([f["err_strict"] for f in fits])),
                   "n": int(mask.sum()) if mask is not None else NH}
            rows.append(row)
            print(f"  {label:<24} {kind:<6} up-to-diag {row['err_diag']:.4f}   "
                  f"strict {row['err_strict']:.4f}")
    save("reach", rows)
    return rows


def sec_homo(ntrial=40):
    """Homodyne recovery quality: ideal, without LO calibration, with the ADC floor."""
    print("\n== homodyne readout ==")
    tw = Twin()
    rng = np.random.default_rng(1)
    ideal, nolo, noisy, lo_pow = [], [], [], []
    for _ in range(ntrial):
        ph = open_reference(torch.rand(NH) * 2 * torch.pi)
        ideal.append(recovery_error(tw, ph)[0])
        nolo.append(recovery_error(tw, ph, derotate=False)[0])
        noisy.append(recovery_error(tw, ph, rng, adc_noise)[0])
        lo_pow.append(lo_health(tw, ph).mean())
    # how much laser headroom would fix the ADC-limited case
    scaled = []
    for g in (1, 3, 10, 30, 100):
        tw_g = Twin()
        errs = []
        for _ in range(12):
            ph = open_reference(torch.rand(NH) * 2 * torch.pi)
            E = tw_g.forward(ph)
            errs.append(recovery_error(tw_g, ph, rng,
                                       lambda y, r, g=g: adc_noise(y * g, r) / g)[0])
        scaled.append({"gain": g, "err": float(np.mean(errs))})
        print(f"  laser/gain x{g:<4} rel err {np.mean(errs):.4f}")
    res = {"ideal": float(np.mean(ideal)), "no_lo_cal": float(np.mean(nolo)),
           "adc_noise": float(np.mean(noisy)), "lo_power": float(np.mean(lo_pow)),
           "gain_sweep": scaled}
    print(f"  noiseless {res['ideal']:.2e}   no LO cal {res['no_lo_cal']:.3f}   "
          f"ADC floor {res['adc_noise']:.3f}")
    save("homo", res)
    return res


def sec_matvec(ntrial=6, nvec=12, steps=2000, restarts=12):
    print("\n== signed matrix-vector ==")
    tw, hwd = Twin(), Hardware()
    rng = np.random.default_rng(2)
    rows = []
    for label, mask in [("all 120", None), ("characterized only", hwd.known)]:
        for nl, noise in [("noiseless", None), ("ADC floor", adc_noise)]:
            r = matvec.run(n_trials=ntrial, n_vecs=nvec, trainable=mask, twin=tw,
                           seed=3, noise=noise, steps=steps, restarts=restarts)
            rows.append(matvec.digest(r, f"{label} / {nl}"))
    save("matvec", rows)
    return rows


def sec_ising(nprob=6, steps=2000, restarts=12):
    print("\n== 6-spin Ising ==")
    tw, hwd = Twin(), Hardware()
    rows = []
    for route in ("power", "homodyne"):
        for label, mask in [("all 120", None), ("characterized only", hwd.known)]:
            for nl, noise in [("noiseless", None), ("ADC floor", adc_noise)]:
                r = ising.run(route=route, n_problems=nprob, trainable=mask, twin=tw,
                              seed=4, noise=noise, steps=steps, restarts=restarts)
                rows.append(ising.digest(r, f"{route} / {label} / {nl}"))
    save("ising", rows)
    return rows


def sec_size(steps=1500, restarts=12, ntrial=3):
    """Signed-matvec error vs target size, placed on the most central rails.
    The edge rails are the ones the odd-column drops bleed, so placement matters."""
    print("\n== matvec capacity vs problem size ==")
    from .program import fit_submatrix, inner_rails
    tw, hwd, rng = Twin(), Hardware(), np.random.default_rng(0)
    rows = []
    for n in (2, 3, 4, 5, 6):
        Ts = []
        for _ in range(ntrial):
            M = rng.normal(size=(n, n))
            Ts.append(M / np.linalg.norm(M, 2))
        row = {"n": n, "rails": inner_rails(n)}
        for key, mask in (("all120", None), ("char82", hwd.known)):
            row[key] = float(np.mean([
                fit_submatrix(M, twin=tw, trainable=mask, seed=i, steps=steps,
                              restarts=restarts)["err"] for i, M in enumerate(Ts)]))
        rows.append(row)
        print(f"  n={n} rails {str(row['rails']):<18} all120 {row['all120']:.3f}   "
              f"char82 {row['char82']:.3f}")
    save("size", rows)
    return rows


def sec_gram(nprob=8, steps=1500, restarts=12, weights=(0.0, 0.02, 0.1, 0.3)):
    """Ising by Gram matching, and how much coupling amplitude buys under the ADC floor.

    Matching Re(M^H M) instead of M frees every left-unitary, which is why an arbitrary
    6-spin instance programmes exactly. The leftover slack is then spent on amplitude —
    the thing that decides whether config-to-config power clears the 4.9 mV ADC floor.
    """
    print("\n== Ising via Gram matching ==")
    from .gram import fit_gram
    tw, hwd = Twin(), Hardware()
    rows = []
    print(f"  {'w':>5}{'gram err':>10}{'ampl':>9}{'E-corr':>9}{'GS hit':>8}{'excess':>9}")
    for w in weights:
        rng = np.random.default_rng(7)
        ge, amp, cor, hit, exc = [], [], [], [], []
        for p in range(nprob):
            J = ising.random_ising(rng)
            f = fit_gram(J, twin=tw, trainable=hwd.known, seed=50 + p, steps=steps,
                         restarts=restarts, throughput=w)
            ge.append(f["err"])
            amp.append(f["scale"])
            cfgs, Et = ising.brute_force(J)
            Ech = []
            for s in cfgs:
                ph = f["phases"].clone()
                ph[:24] = torch.as_tensor(ising.spin_phases(s), dtype=torch.float32)
                mon = adc_noise(tw.forward(ph)["mon"].detach().numpy(), rng)
                Ech.append(-0.5 * (mon / tw.tap).sum())
            Ech = np.array(Ech)
            k, b = np.polyfit(Ech, Et, 1)
            pred = k * Ech + b
            gs = int(np.argmin(pred))
            hit.append(bool(np.isclose(Et[gs], Et.min())))
            cor.append(float(np.corrcoef(pred, Et)[0, 1]))
            exc.append(float((Et[gs] - Et.min()) / (Et.max() - Et.min())))
        rows.append({"w": w, "gram_err": float(np.mean(ge)), "ampl": float(np.mean(amp)),
                     "pearson": float(np.mean(cor)), "found_gs": float(np.mean(hit)),
                     "excess": float(np.mean(exc))})
        print(f"  {w:>5.2f}{np.mean(ge):>10.4f}{np.mean(amp):>9.3f}{np.mean(cor):>9.3f}"
              f"{np.mean(hit):>8.0%}{np.mean(exc):>9.3f}")
    save("gram", rows)
    return rows


SECTIONS = {"spectrum": sec_spectrum, "reach": sec_reach, "size": sec_size,
            "homo": sec_homo, "matvec": sec_matvec, "ising": sec_ising, "gram": sec_gram}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    t0 = time.time()
    for name in want:
        SECTIONS[name]()
    print(f"\n[{time.time() - t0:.0f}s]")
