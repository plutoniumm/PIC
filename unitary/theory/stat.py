"""Is this trace a signal, or is it noise that happens to fit?

Every gate here is a p-value against a null hypothesis, never a threshold in volts. A
constant like `MIN_AMPLITUDE_V = 0.011` is 2 sigma of the read noise on one particular day
frozen into the source, and it goes stale silently: the same number is a strict gate on a
quiet detector and a rubber stamp on a noisy one. What the gate should compare against is
what the same run measured -- this detector's noise, this trace's residual.

The only constant is a significance level, which is what a test is allowed to have.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.01


def p_amplitude(n: int, amplitude: float, sigma: float) -> float:
    """P(a flat trace plus noise produces a fringe at least this tall).

    Fitting a sinusoid of unknown frequency to `n` Gaussian samples is a periodogram peak
    search: the tail for one frequency is exp(-n B^2 / 4 sigma^2), and searching over the
    n/2 resolvable frequencies costs that factor in the bound. Replaces both
    `MIN_AMPLITUDE_V` and `MIN_SNR`, which were the same statement made twice with two
    different frozen numbers."""
    sigma = max(float(sigma), 1e-12)
    B = abs(float(amplitude))
    if not np.isfinite(B) or n < 2:
        return 1.0
    return float(min(1.0, 0.5 * n * np.exp(-n * B * B / (4.0 * sigma * sigma))))


def p_shape(y, model, n_params: int = 4) -> float:
    """P(the model explains no more of `y` than its mean does) -- nested-model F test.

    Replaces `min_r2` and `min_visibility`. Neither was a statement about evidence: r2 rises
    with amplitude on a trace that never turns over, which is precisely the case `better`
    had to special-case by hand."""
    from scipy.stats import f as fdist

    y = np.asarray(y, float).ravel()
    m = np.asarray(model, float).ravel()
    n = y.size
    dof = n - n_params
    ss_fit = float(np.sum((y - m) ** 2))
    ss_const = float(np.sum((y - y.mean()) ** 2))
    if dof <= 0 or ss_fit <= 0 or ss_const <= ss_fit:
        return 1.0
    fstat = ((ss_const - ss_fit) / (n_params - 1)) / (ss_fit / dof)
    return float(fdist.sf(fstat, n_params - 1, dof))


def p_worse(recent, earlier) -> float:
    """One-sided rank test: how likely a sample at least this much lower than `earlier` would be
    if `recent` came from the same distribution. Ranks, not means, so one outlier run cannot
    call a decline on its own. 1.0 when either side is empty."""
    from scipy.stats import mannwhitneyu

    if len(recent) == 0 or len(earlier) == 0:
        return 1.0
    return float(mannwhitneyu(recent, earlier, alternative="less").pvalue)


def accepts(p_amp: float, p_shp: float, alpha: float = ALPHA) -> bool:
    """A fit is real only if it clears BOTH tests: tall enough, and the right shape.

    Both, because they fail independently. A big fringe fitted badly is two heaters moving
    at once; a clean fit with no amplitude is noise the optimiser flattered."""
    return bool(p_amp < alpha and p_shp < alpha)


def _selftest(n: int = 21, seed: int = 0):
    assert p_worse([0.80, 0.81, 0.79, 0.80, 0.82], [0.95, 0.94, 0.96, 0.95, 0.93]) < ALPHA
    assert p_worse([0.95, 0.94, 0.96, 0.93, 0.95], [0.95, 0.94, 0.96, 0.95, 0.93]) > ALPHA
    """Noise must be rejected and a real fringe accepted, with no threshold in volts."""
    rng = np.random.default_rng(seed)
    v = np.sqrt(np.linspace(0, 4.5**2, n))
    sigma = 0.006

    noise = rng.normal(0, sigma, n)
    assert not accepts(
        p_amplitude(n, np.ptp(noise) / 2, sigma), p_shape(noise, np.full(n, noise.mean()))
    )

    true = 0.2 + 0.15 * np.cos(np.pi * (v / 3.5) ** 2 + 0.7)
    obs = true + rng.normal(0, sigma, n)
    assert accepts(p_amplitude(n, 0.15, sigma), p_shape(obs, true))

    # a fringe buried under noise a hundred times its size must NOT pass on shape alone
    buried = true + rng.normal(0, 0.15 * 100, n)
    assert not accepts(p_amplitude(n, 0.15, 0.15 * 100), p_shape(buried, true))
    return "noise rejected, fringe accepted, buried fringe rejected"


if __name__ == "__main__":
    print(_selftest())
