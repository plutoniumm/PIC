"""When input-referred drift correction works, and why the window is narrow.

Paper section C. A surrogate g (the hardware DPNN) stands in for the chip f. The chip
drifts; model the drift as input-referred, f_t(V) = f(V + d), and correct by inferring d
from probes through g and commanding V - d_hat. `scripts/dpnn_drift_hw.py` (deleted, see
hw.md) did exactly this on hardware and lost. This module says analytically why, and
verifies it against the real DPNN checkpoint.

WHAT IS BEING RECOVERED. d itself is NOT identifiable: 112 heaters against a rank-14
Jacobian, so most of d lies in a null space the photodiodes cannot see. Measured d-space
recovery peaks at **0.098** (median 0.090) even with a perfect model and zero noise --
i.e. ~90% of the drift vector is simply invisible. That is fine, and it is the point: the
correction does not need d, it needs the OUTPUT back, and every d_hat in the same fibre
restores it equally. Output recovery over the same runs reaches 85%. So everything below
scores

    recovery = 1 - ||y(V + d - d_hat) - y(V)|| / ||y(V + d) - y(V)||          (0)

on HELD-OUT probes, and rho = 1 - recovery is the relative correction error. rho < 1
means the correction helped; rho > 1 means it made things worse.

THE RESULT. Write J = dg/dV, J* = df/dV, dJ = J* - J, and D = f - g the static model
offset. The fit solves g(V + d_hat) = f(V + d), so d_hat absorbs BOTH the drift and
whatever part of the static mismatch D lies in the range of J:

    d_hat  =  d  +  J^+ D  +  J^+ n  +  J^+ dJ d  +  (1/2) J^+ H*[d,d]

The middle two terms do not scale with d -- they are a fixed spurious offset, so their
RELATIVE contribution blows up as d -> 0. Projected into the observable subspace:

                    rho(d) = A/d + eps + kap*d                                 (1)

    A   = |J^+ D| + |J^+ n|/sqrt(K)   static model offset AND probe noise, together
    eps = |J^+ dJ|                    Jacobian mismatch -- scale-free, a true floor
    kap = |J^+ H*| / 2                curvature -- the linearisation's own error

This is the correction that matters: static model error is NOT a constant floor, it acts
like extra noise. A surrogate that is merely biased is as damaging at small drift as a
noisy measurement, which is why r^2 alone never predicted whether correction would work.

Correction helps iff rho < 1, i.e. only inside a window of drift magnitudes:

    d_min, d_max = [(1-eps) -/+ sqrt((1-eps)^2 - 4 A kap)] / (2 kap)           (2)
    best at d* = sqrt(A/kap),  rho_min = eps + 2 sqrt(A kap)                   (3)

Read off (2)-(3):

  * A sets the LOWER edge. Drift below the combined noise-plus-model-offset floor is not
    observable, so the fit is fitting bias and injects more error than it removes. This
    is the reanchor failure quantitatively -- 18 deg injected against 13 deg of drift.
  * CURVATURE sets the UPPER edge. d_hat is a linearisation; the chip's fringes are
    cosines in V^2, so once the drift is big enough to turn J, V - d_hat overshoots.
  * The window CLOSES when eps + 2 sqrt(A kap) >= 1. Since A carries the model offset,
    improving the surrogate widens the window from the LEFT -- it buys sensitivity to
    small drift, not tolerance of large drift.

Measured here against the real DPNN (see --sweep): with a good surrogate the window is
wide and correction recovers 80-85% of the output error; as the surrogate degrades the
lower edge marches right until it meets the curvature edge and the window shuts. PIC B's
own numbers put it past that point, which is why every hardware attempt in hw.md failed --
not because the method is wrong, but because this chip sits outside its window.

    PYTHONPATH=. python theory/drift_window.py --calibrate --scan --out runs/drift_window.json
    PYTHONPATH=. python theory/drift_window.py --predict          # PIC B's window
"""
from __future__ import annotations

import json

import numpy as np

# Measured on PIC B; sources in hw.md / Readme.md. Used by predict_window().
PICB = {
    "dpnn_r2": 0.556,        # fresh hardware-trained DPNN, held-out (hw.md dpnn_drift_hw)
    "repeat_sigma": 0.008,   # back-to-back repeat noise, PD volts (range 0.005-0.011)
    "order_sigma": 0.034,    # same config after 60 others -- thermal history, the real floor
    "drift_13h": 0.080,      # 13-hour drift, same units (SNR 1.9 against order_sigma)
    "n_pd": 8,               # responsive PDs
}

CKPT = "runs/dpnn_hw"


def rho(d, A, eps, kap):
    """Relative correction error, eq (1). rho < 1 means the correction helps."""
    d = np.asarray(d, float)
    return A / d + eps + kap * d


def window(A, eps, kap):
    """Drift magnitudes where correction helps, eq (2)-(3).

    Returns (d_min, d_max, d_star, rho_min); the edges are NaN when the window is empty,
    which happens as soon as eps + 2 sqrt(A kap) exceeds 1.
    """
    A, kap = max(A, 1e-15), max(kap, 1e-15)
    d_star = float(np.sqrt(A / kap))
    rho_min = float(eps + 2.0 * np.sqrt(A * kap))
    disc = (1.0 - eps) ** 2 - 4.0 * A * kap
    if disc < 0:
        return float("nan"), float("nan"), d_star, rho_min
    r = np.sqrt(disc)
    return float((1 - eps - r) / (2 * kap)), float((1 - eps + r) / (2 * kap)), d_star, rho_min


def fit_constants(drifts, rhos):
    """Fit eq (1) to a measured rho(d) scan -> (A, eps, kap).

    Fitted in LOG space. rho spans decades across a scan (tens at the noise-dominated
    end, ~0.2 at the optimum), so a plain least-squares on rho is set entirely by the
    largest few points and returns nonsense for kap. Weighting each point equally in
    log rho is what makes the three regimes contribute comparably. Non-negativity is
    imposed by optimising log of each coefficient.
    """
    from scipy.optimize import least_squares

    d = np.asarray(drifts, float)
    y = np.log(np.asarray(rhos, float))

    def resid(p):
        return np.log(rho(d, *np.exp(p))) - y

    seed = np.log([max(np.min(rhos) * np.min(d), 1e-6), max(np.min(rhos), 1e-3), 1.0])
    out = least_squares(resid, seed, method="lm", max_nfev=20000)
    A, eps, kap = (float(v) for v in np.exp(out.x))
    pred = rho(d, A, eps, kap)
    ss = 1.0 - np.sum((np.log(pred) - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return {"A": A, "eps": eps, "kap": kap, "log_r2": float(ss)}


class _Rig:
    """The DPNN as surrogate g, plus a `chip` f that differs from it by a tunable eps."""

    def __init__(self, ckpt=CKPT, seed=0, v0=1.0):
        import copy

        import torch

        import scripts.pic_gd as G

        self.torch = torch
        model, norm, buf, meta, feat, cfg = G.load_surrogate(ckpt, "relu", 8)
        op = float(np.median(buf["dbm"]))
        tel = G.operating_telemetry(buf, op)
        self.g = G.make_predict(model, norm, feat, op, tel)
        self.nh = len(feat)
        self.v0 = np.full(self.nh, float(v0))
        self.rng = np.random.default_rng(seed)

        # A second copy of the net, kicked in weight space. Blending g with it gives a
        # chip whose mismatch is a knob rather than an accident of architecture.
        alt = copy.deepcopy(model)
        with torch.no_grad():
            for p in alt.parameters():
                p.add_(torch.randn_like(p) * p.std() * 5e-2)
        self._alt = G.make_predict(alt, norm, feat, op, tel)

    def chip(self, blend):
        """f = (1-blend) g + blend g_alt. blend=0 is a perfect model."""
        if blend <= 0:
            return self.g
        return lambda v: (1 - blend) * self.g(v) + blend * self._alt(v)

    def fwd(self, f, V):
        t = self.torch
        return np.stack([f(t.tensor(v, dtype=t.float32)).detach().numpy()
                         for v in np.asarray(V, float)])

    def probes(self, k, scale=0.15):
        return self.v0 + self.rng.normal(scale=scale, size=(k, self.nh))

    def drift(self, rms):
        """A random input-referred drift with the requested per-heater RMS."""
        u = self.rng.normal(size=self.nh)
        return rms * u / np.linalg.norm(u) * np.sqrt(self.nh)

    def infer(self, probes, Y, *, lr, iters=600, l2=1e-4):
        """Fit d_hat so g(V + d_hat) matches the drifted readings, jointly over probes.

        Relative loss -- absolute MSE on ~0.05 V outputs is small enough that any useful
        l2 would dominate it (the bug that made this look dead the first time). Adam with
        cosine decay and lr scaled to the drift being looked for; a fixed lr cannot
        resolve a drift smaller than one step.
        """
        t = self.torch
        d = t.zeros(self.nh, requires_grad=True)
        opt = t.optim.Adam([d], lr=lr)
        sch = t.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        P = t.tensor(np.asarray(probes), dtype=t.float32)
        Yt = t.tensor(np.asarray(Y), dtype=t.float32)
        den = (Yt ** 2).mean()
        for _ in range(iters):
            opt.zero_grad()
            pred = t.stack([self.g(P[k] + d) for k in range(P.shape[0])])
            (((pred - Yt) ** 2).mean() / den + l2 * (d ** 2).mean()).backward()
            opt.step()
            sch.step()
        return d.detach().numpy()


def recover(rig, drift_rms, *, sigma=0.0, blend=0.0, k_train=24, k_hold=16, iters=600,
            l2=1e-4):
    """One correction experiment -> rho, eq (0). Lower is better; rho<1 means it helped.

    sigma  per-PD probe noise (0 = noiseless)
    blend  chip/surrogate mismatch (0 = perfect model)
    """
    f = rig.chip(blend)
    tr, ho = rig.probes(k_train), rig.probes(k_hold)
    d = rig.drift(drift_rms)

    Y = rig.fwd(f, tr + d)
    if sigma > 0:
        Y = Y + rig.rng.normal(scale=sigma, size=Y.shape)
    d_hat = rig.infer(tr, Y, lr=max(drift_rms, 1e-3) / 4, iters=iters, l2=l2)

    y0, yd, yc = rig.fwd(f, ho), rig.fwd(f, ho + d), rig.fwd(f, ho + d - d_hat)
    before = np.linalg.norm(yd - y0)
    return {"drift_rms": float(drift_rms),
            "rho": float(np.linalg.norm(yc - y0) / before) if before > 0 else float("nan"),
            "d_recovery": float(1 - np.linalg.norm(d_hat - d) / np.linalg.norm(d)),
            "sigma": float(sigma), "blend": float(blend)}


def calibrate(rig, *, iters=600, verbose=True):
    """Isolate each constant of eq (1) by switching the other two off.

    kap : no noise, perfect model -> rho should rise linearly in d      (upper edge)
    nu  : perfect model, noise on -> rho should fall as 1/d             (lower edge)
    eps : no noise, small drift, chip != model -> rho flat at eps       (the floor)
    """
    out = {}

    if verbose:
        print("  [kap] curvature only  (sigma=0, perfect model)")
    ds = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    r = [recover(rig, d, sigma=0.0, blend=0.0, iters=iters) for d in ds]
    for x in r:
        if verbose:
            print(f"        d={x['drift_rms']:6.3f}  rho={x['rho']:.3f}")
    A = np.stack([np.ones(len(ds)), ds], axis=1)
    c0, kap = np.linalg.lstsq(A, [x["rho"] for x in r], rcond=None)[0]
    out["kap"] = {"value": float(max(kap, 1e-9)), "intercept": float(c0), "rows": r}

    if verbose:
        print("  [nu] noise only  (perfect model, sigma = PIC B repeat noise)")
    r = [recover(rig, d, sigma=PICB["repeat_sigma"], blend=0.0, iters=iters)
         for d in [0.005, 0.01, 0.02, 0.05]]
    for x in r:
        if verbose:
            print(f"        d={x['drift_rms']:6.3f}  rho={x['rho']:.3f}")
    # rho - (c0 + kap d) = nu/d  ->  one-parameter fit on the excess
    ex = [x["rho"] - (c0 + out["kap"]["value"] * x["drift_rms"]) for x in r]
    inv = np.array([1.0 / x["drift_rms"] for x in r])
    nu = float(max(np.dot(inv, ex) / np.dot(inv, inv), 1e-12))
    out["nu"] = {"value": nu, "sigma": PICB["repeat_sigma"], "rows": r}

    if verbose:
        print("  [eps] model error only  (sigma=0, chip != surrogate)")
    rows = []
    for b in (0.02, 0.05, 0.1, 0.2):
        x = recover(rig, 0.05, sigma=0.0, blend=b, iters=iters)
        x["excess"] = x["rho"] - (c0 + out["kap"]["value"] * 0.05)
        rows.append(x)
        if verbose:
            print(f"        blend={b:5.2f}  rho={x['rho']:.3f}  excess={x['excess']:+.3f}")
    out["eps"] = {"rows": rows,
                  "note": "eps grows with chip/surrogate mismatch; see scan for the fit"}
    return out


def bootstrap(rows, n_boot=1000, seed=0):
    """Seed-cluster bootstrap for the fitted constants and the window edges.

    A seed fixes the drift DIRECTION and the probe set, and `scan` rebuilds the rig with
    the same seed at every drift magnitude, so one seed traces a coherent curve across the
    whole scan. Resampling rows independently would therefore break the correlation the
    data actually has. Each replicate instead resamples SEEDS with replacement and reuses
    that same set at every d, which is the cluster bootstrap appropriate to this design.

    Returns percentile intervals (2.5, 50, 97.5) for A, eps, kap, d_min, d_max, rho_min.
    Replicates whose window is empty contribute NaN edges and are dropped from the edge
    intervals only, with the fraction that closed reported as `p_closed`.
    """
    if not rows or len(rows[0].get("rho_all", [])) < 3:
        return None
    d = np.array([r["drift_rms"] for r in rows], float)
    Y = np.array([r["rho_all"] for r in rows], float)      # (n_drift, n_seed)
    n = Y.shape[1]
    rng = np.random.default_rng(seed)
    keys = ("A", "eps", "kap", "d_min", "d_max", "rho_min")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        try:
            f = fit_constants(d, Y[:, pick].mean(axis=1))
        except Exception:
            continue
        lo, hi, _, rmin = window(f["A"], f["eps"], f["kap"])
        for k, v in zip(keys, (f["A"], f["eps"], f["kap"], lo, hi, rmin)):
            draws[k].append(v)
    out = {}
    for k in keys:
        v = np.array(draws[k], float)
        v = v[np.isfinite(v)]
        out[k] = [float(x) for x in np.percentile(v, [2.5, 50, 97.5])] if v.size else None
    out["n_boot"] = n_boot
    out["p_closed"] = float(np.mean(~np.isfinite(np.array(draws["d_min"], float))))
    return out


def scan(drifts, *, sigma, blend, seeds=(0, 1), iters=300, k_train=16, k_hold=12,
         verbose=True, ckpt=CKPT):
    """Sweep drift magnitude -> measured rho(d) + the fitted window.

    Averaged over `seeds`: one realisation per point is far too noisy to fit three
    constants through, because both the drift direction and the probe set are random.
    """
    rows = []
    n = len(seeds)
    for d in drifts:
        vals = [recover(_Rig(seed=s, ckpt=ckpt), d, sigma=sigma, blend=blend,
                        k_train=k_train, k_hold=k_hold, iters=iters) for s in seeds]
        per_seed = [float(v["rho"]) for v in vals]
        r = float(np.mean(per_seed))
        rows.append({"drift_rms": float(d), "rho": r,
                     "rho_sd": float(np.std(per_seed)),
                     "rho_sem": float(np.std(per_seed, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                     "rho_all": per_seed,
                     "d_recovery": float(np.mean([v["d_recovery"] for v in vals]))})
        if verbose:
            print(f"    d={d:7.4f}  rho={r:7.3f} +-{rows[-1]['rho_sem']:6.3f}  "
                  f"{'HELPS' if r < 1 else 'hurts '}  (d-recov {rows[-1]['d_recovery']:+.2f})")
    fit = fit_constants([x["drift_rms"] for x in rows], [x["rho"] for x in rows])
    lo, hi, ds, rmin = window(fit["A"], fit["eps"], fit["kap"])
    helped = [x["drift_rms"] for x in rows if x["rho"] < 1]
    return {"rows": rows, "fit": fit, "n_seeds": n, "boot": bootstrap(rows),
            "window": {"d_min": lo, "d_max": hi, "d_star": ds, "rho_min": rmin},
            "observed_window": [min(helped), max(helped)] if helped else None,
            "sigma": sigma, "blend": blend}


def sweep_quality(drifts, blends, *, sigma=0.0, seeds=(0, 1), iters=300, verbose=True,
                  ckpt=CKPT):
    """THE HEADLINE RESULT: how the window moves as the surrogate degrades.

    For each surrogate quality `blend` (0 = the model is the chip), scan drift and fit
    eq (1). Shows the lower edge marching right as A grows with model offset, until it
    meets the fixed curvature edge and the window shuts.
    """
    out = []
    for b in blends:
        if verbose:
            print(f"\n  surrogate mismatch blend={b:.4g} (sigma={sigma})")
        s = scan(drifts, sigma=sigma, blend=b, seeds=seeds, iters=iters,
                 verbose=verbose, ckpt=ckpt)
        f, w = s["fit"], s["window"]
        if verbose:
            edge = (f"{w['d_min']:.4g} .. {w['d_max']:.4g}"
                    if np.isfinite(w["d_min"]) else "EMPTY")
            print(f"    -> A={f['A']:.4g} eps={f['eps']:.4g} kap={f['kap']:.4g}"
                  f"  window {edge}  rho_min {w['rho_min']:.3f}")
        out.append(s)
    return out


def picb_A(A_ref, sigma_ref, sigma):
    """Rescale the calibrated A from the noise it was measured at to a target sigma.

    A is a sum of a model-offset part and a noise part; only the noise part scales, but
    on PIC B the thermal-ordering floor (0.034) dominates the repeat floor (0.008), so
    scaling the whole thing is the conservative reading and is what is reported.
    """
    return float(A_ref * (sigma / sigma_ref))


def predict_window(A, kap, *, eps=None):
    """PIC B's window from measured chip constants.

    eps from the DPNN's held-out r^2: a surrogate explaining r^2 of the variance leaves
    relative amplitude sqrt(1-r^2), the scale-free stand-in for the projected ||J^+ dJ||.
    """
    eps = float(np.sqrt(1.0 - PICB["dpnn_r2"])) if eps is None else float(eps)
    lo, hi, ds, rmin = window(A, eps, kap)
    return {"eps": eps, "A": float(A), "kap": float(kap),
            "sigma": PICB["order_sigma"], "d_min": lo, "d_max": hi,
            "d_star": ds, "rho_min": rmin, "window_empty": bool(not np.isfinite(lo)),
            "drift_13h": PICB["drift_13h"]}


def requirements(A_ref, sigma_ref, kap, *, eps_now=None, targets=(0.0, 0.3, 0.666)):
    """What would it take to REOPEN PIC B's window? Inverts eq (3) for sigma.

    rho_min = eps + 2 sqrt(A kap) < 1  =>  A < ((1-eps)/2)^2 / kap, and A scales with
    the probe noise, so this converts straight into a required sigma and, via sqrt(N)
    averaging, a required number of repeats per probe. Reported at several surrogate
    qualities because the two levers trade off.
    """
    eps_now = float(np.sqrt(1.0 - PICB["dpnn_r2"])) if eps_now is None else eps_now
    rows = []
    for eps in targets:
        if eps >= 1.0:
            rows.append({"eps": eps, "feasible": False})
            continue
        A_max = ((1.0 - eps) / 2.0) ** 2 / kap
        sigma_max = sigma_ref * A_max / A_ref
        rows.append({
            "eps": float(eps),
            "r2_needed": float(1.0 - eps ** 2),
            "A_max": float(A_max),
            "sigma_max": float(sigma_max),
            "n_avg_from_repeat": float(max((PICB["repeat_sigma"] / sigma_max) ** 2, 1.0)),
            "n_avg_from_order": float(max((PICB["order_sigma"] / sigma_max) ** 2, 1.0)),
            "feasible": True,
        })
    return {"eps_now": float(eps_now), "kap": float(kap), "rows": rows}


def scenarios(A_ref, sigma_ref, kap, rows=None):
    """The constructive result: which (surrogate quality, averaging) pairs reopen a window.

    Each row is (label, R^2, sigma, N). eps = sqrt(1-R^2); A scales as sigma/sqrt(N).

    CAVEAT, load-bearing -- sqrt(N) assumes INDEPENDENT repeats. PIC B's dominant floor is
    the thermal-ordering noise (sigma 0.034), which is history-correlated, not white, so
    averaging against it saturates rather than following sqrt(N). Rows below that assume
    the repeat floor (0.008) are therefore conditional on TEC converting ordering noise
    into something averageable; rows at 0.034 are conditional on nothing and are the
    pessimistic bound. This is the single biggest assumption in the TEC case.
    """
    rows = rows if rows is not None else [
        ("PIC B now (no TEC)",           PICB["dpnn_r2"], PICB["order_sigma"], 1),
        ("PIC B, N=100 averaging",       PICB["dpnn_r2"], PICB["order_sigma"], 100),
        ("TEC (R2=0.86), N=1",           0.86, PICB["repeat_sigma"], 1),
        ("TEC (R2=0.86), N=25",          0.86, PICB["repeat_sigma"], 25),
        ("TEC (R2=0.86), N=100",         0.86, PICB["repeat_sigma"], 100),
        ("TEC (R2=0.95), N=25",          0.95, PICB["repeat_sigma"], 25),
        ("TEC (R2=0.95), N=100",         0.95, PICB["repeat_sigma"], 100),
    ]
    out = []
    for label, r2, sigma, n in rows:
        eps = float(np.sqrt(1.0 - r2))
        A = float(A_ref * (sigma / np.sqrt(n)) / sigma_ref)
        lo, hi, ds, rmin = window(A, eps, kap)
        out.append({"label": label, "r2": r2, "eps": eps, "sigma": sigma, "n_avg": n,
                    "A": A, "d_min": lo, "d_max": hi, "d_star": ds, "rho_min": rmin,
                    "open": bool(np.isfinite(lo))})
    return {"rows": out, "kap": float(kap),
            "sqrtN_assumes_independent_repeats": True,
            "caveat": ("sigma_ord (0.034) is history-correlated, not white; sqrt(N) "
                       "averaging against it saturates. Rows at the repeat floor assume "
                       "TEC makes the residual noise averageable.")}


P_FIT = 112   # the calibration sweeps infer a full per-heater offset field


def regimes(A_ref, sigma_ref, kap, rows=None):
    """Why the window is OPEN for the Ising correction and SHUT for per-setting correction.

    A is a noise gain through the pseudo-inverse, so it scales as sqrt(p / (K*P)) in the
    number of parameters p the correction infers. The simulation infers a full per-heater
    offset field (p = 112). The Ising correction infers ONE global phase offset, which is
    a ~10x smaller noise gain for the same probes, and it runs immediately after a fresh
    calibration, i.e. against the back-to-back repeat floor rather than the thermal-
    ordering floor that governs an aged session.

    Stack those and the window opens -- but only just (rho_min ~ 0.95-1.07, i.e. right at
    the boundary), which is exactly a correction that works briefly and then stops. This
    is the regime the n=4 hardware runs sit in; it is a prediction of eq (1), not an
    exception to it.
    """
    # Fresh rows pair the fresh noise floor with the fresh model (r^2=0.67 right after
    # training, decaying to 0.556 over ~13 h -- hw.md); aged rows pair the drifted floor
    # with the aged model. A session is fresh or aged as a whole.
    eps_fresh, eps_aged = float(np.sqrt(1 - 0.67)), float(np.sqrt(1 - PICB["dpnn_r2"]))
    rows = rows if rows is not None else [
        ("per-heater field, aged session",   P_FIT, PICB["order_sigma"],  eps_aged),
        ("per-heater field, fresh session",  P_FIT, PICB["repeat_sigma"], eps_fresh),
        ("global offset, aged session",          1, PICB["order_sigma"],  eps_aged),
        ("global offset, fresh session",         1, PICB["repeat_sigma"], eps_fresh),
    ]
    out = []
    for label, p, sigma, eps in rows:
        A = float(A_ref * (sigma / sigma_ref) * np.sqrt(p / P_FIT))
        lo, hi, ds, rmin = window(A, eps, kap)
        out.append({"label": label, "n_params": p, "sigma": sigma, "eps": eps, "A": A,
                    "d_min": lo, "d_max": hi, "rho_min": rmin,
                    "open": bool(np.isfinite(lo))})
    return {"rows": out, "kap": float(kap), "p_fit": P_FIT,
            "note": ("A scales as sqrt(p/(K*P)); the Ising correction infers one global "
                     "offset, not a 112-dim field, and runs on the fresh-calibration noise "
                     "floor. The window it opens is marginal and short-lived.")}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="store_true",
                    help="window vs surrogate quality (the headline result)")
    ap.add_argument("--noise", action="store_true",
                    help="window vs probe noise, at a perfect surrogate")
    ap.add_argument("--predict", action="store_true", help="locate PIC B's window")
    ap.add_argument("--scenarios", action="store_true",
                    help="TEC / averaging scenarios that reopen the window")
    ap.add_argument("--blends", default="0,0.005,0.02,0.05,0.1")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    # theory/results/ is tracked; runs/ is gitignored, so results default to the former.
    ap.add_argument("--out", default="theory/results/drift_window.json")
    a = ap.parse_args(argv)
    if not (a.sweep or a.noise or a.predict or a.scenarios):
        a.sweep = a.noise = a.predict = a.scenarios = True

    seeds = tuple(range(a.seeds))
    drifts = np.geomspace(0.01, 0.8, 8)
    out = {"config": {"iters": a.iters, "seeds": list(seeds),
                      "drifts": [float(d) for d in drifts]}}
    kap = A_ref = None

    if a.sweep:
        blends = [float(x) for x in a.blends.split(",")]
        print("window vs surrogate quality (noiseless -- isolates the model-offset term)")
        out["quality"] = sweep_quality(drifts, blends, sigma=0.0, seeds=seeds,
                                       iters=a.iters)
        kap = float(np.median([s["fit"]["kap"] for s in out["quality"]]))
        print(f"\n  curvature is a property of the chip, not the model: "
              f"kap = {kap:.4g} (median over blends)")

    if a.noise:
        print(f"\nwindow vs probe noise (perfect surrogate -- isolates the noise term)")
        out["noise"] = sweep_quality(drifts, [0.0], sigma=PICB["repeat_sigma"],
                                     seeds=seeds, iters=a.iters)
        A_ref = out["noise"][0]["fit"]["A"]
        kap = kap or out["noise"][0]["fit"]["kap"]
        print(f"\n  A(sigma={PICB['repeat_sigma']}) = {A_ref:.4g}")

    if (a.predict or a.scenarios) and not (kap and A_ref):
        # --predict on its own: reuse the constants from a previous calibrated run
        # rather than silently doing nothing.
        try:
            prev = json.load(open(a.out or "runs/drift_window.json"))
            kap = kap or float(np.median([s["fit"]["kap"] for s in prev["quality"]]))
            A_ref = A_ref or prev["noise"][0]["fit"]["A"]
            print(f"(constants loaded from a previous run: kap={kap:.4g}, "
                  f"A_ref={A_ref:.4g})")
        except (OSError, KeyError, ValueError):
            print("--predict needs calibrated constants: run --sweep --noise first "
                  "(or point --out at a previous results JSON).")

    if a.predict and kap and A_ref:
        A = picb_A(A_ref, PICB["repeat_sigma"], PICB["order_sigma"])
        out["picb"] = predict_window(A, kap)
        p = out["picb"]
        print(f"\nPIC B: eps={p['eps']:.3f} (DPNN r2={PICB['dpnn_r2']}), "
              f"A={p['A']:.4g} (sigma={p['sigma']} thermal-ordering), kap={p['kap']:.4g}")
        if p["window_empty"]:
            print(f"  WINDOW EMPTY -- rho_min={p['rho_min']:.2f} > 1 at every drift "
                  "magnitude.\n  No input-referred correction can help this chip as "
                  "instrumented.")
        else:
            print(f"  window {p['d_min']:.4g} .. {p['d_max']:.4g} V rms "
                  f"(best {p['d_star']:.4g}, rho_min {p['rho_min']:.3f})")

        out["requirements"] = requirements(A_ref, PICB["repeat_sigma"], kap)
        print("\n  what would reopen it (eq 3 inverted for sigma):")
        print(f"    {'eps':>6} {'r2 needed':>10} {'sigma max':>10} "
              f"{'x avg (repeat)':>15} {'x avg (order)':>14}")
        for r in out["requirements"]["rows"]:
            if not r["feasible"]:
                continue
            print(f"    {r['eps']:6.3f} {r['r2_needed']:10.3f} {r['sigma_max']:10.2e} "
                  f"{r['n_avg_from_repeat']:15.0f} {r['n_avg_from_order']:14.0f}")

    if a.scenarios and kap and A_ref:
        out["scenarios"] = scenarios(A_ref, PICB["repeat_sigma"], kap)
        print("\nwhat reopens the window (eps=sqrt(1-R2); A scales as sigma/sqrt(N)):")
        print(f"  {'scenario':<26}{'R2':>6}{'eps':>7}{'sigma':>8}{'N':>5}{'A':>7}"
              f"{'rho_min':>9}   window (V rms)")
        for r in out["scenarios"]["rows"]:
            w = (f"{r['d_min']:.3f} .. {r['d_max']:.3f}" if r["open"] else "CLOSED")
            print(f"  {r['label']:<26}{r['r2']:6.3f}{r['eps']:7.3f}{r['sigma']:8.4f}"
                  f"{r['n_avg']:5d}{r['A']:7.3f}{r['rho_min']:9.2f}   {w}")
        print(f"  CAVEAT: {out['scenarios']['caveat']}")

    if a.out:
        # Merge, never clobber: a bare --predict run holds only `picb`/`requirements`,
        # and blindly dumping it would destroy the expensive `quality`/`noise` sweeps
        # already in the file. Only keys this run actually produced are replaced.
        try:
            with open(a.out) as fh:
                merged = json.load(fh)
        except (OSError, ValueError):
            merged = {}
        merged.update(out)
        with open(a.out, "w") as fh:
            json.dump(merged, fh, indent=1, default=float)
        print(f"\nwrote {a.out} ({', '.join(sorted(merged))})")
    return out


if __name__ == "__main__":
    main()
