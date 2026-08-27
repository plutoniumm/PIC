# What unitary projection can and cannot fix

The chip is unitary by construction, so every matrix it can produce lies on the orthogonal
manifold O(k). That gives a target-free error bar, and it also sets a trap. Both halves are
measured below; an earlier version of this note got the second half wrong and the correction
is the interesting part.

## Two kinds of error

**Normal error** takes the matrix *off* the manifold — read noise, gain errors, a stale
per-port scale. None of these produce a valid orthogonal matrix, so they are visible without
knowing the answer: the distance from the measurement to the nearest orthogonal matrix
measures them directly.

**Tangential error** slides the matrix *along* the manifold. A mesh whose phases have moved
still produces a perfectly valid orthogonal matrix — a different one. No residual, no
inconsistency, nothing to detect.

Polar projection maps to the nearest point on O(k), removing the normal component and
leaving the tangential one untouched. Measured with known perturbations: a 5% gain error is
pure normal and polar removes it **6.0x**; a 0.15 rad rotation is pure tangential and polar
returns **1.01x**, with the residual floored at exactly the rotation angle. `manifold_dist`
reads 0.019 in **both** cases — it is blind to tangential error by construction.

## The correction: ordering cannot be argued on accuracy

The earlier claim here was that drift must be corrected before projecting, or the projection
locks in the error. The accuracy half of that is **false**, and the reason is a one-line
identity: for orthogonal `R`,

    polar(M R) = polar(M) R

exactly. So correcting a tangential error before projecting and projecting before correcting
return **the same matrix**. There is no accuracy argument to make.

What survives is an **evidence** argument, and it is the one that matters in practice.
Across 12 measured composed products, `manifold_dist` correlates **+0.995** with the true
error while the measurement is unprojected, and is **identically zero** afterwards.
Projection does not merely fail to reveal what is wrong — it deletes the only number that
could have told you.

**So: correct first, project last, and report `manifold_dist` of the *unprojected*
measurement.** Right order, right reason.

## What the error on this rig actually is

Measured on 12 pairs planned off the 25 C table and read off the 20 C table — a real 5 C
excursion with the truth still known:

| | error vs truth | factor |
|---|---|---|
| raw (`manifold_dist` 0.1994) | 0.2127 | — |
| **polar onto O(2)** | **0.0735** | **2.90x** |
| QR | 0.1178 | 1.81x |
| clip sigma <= 1 | 0.1804 | 1.18x |
| clip sigma <= fitted s | 0.2050 | 1.04x |

The error decomposes as 13% hosting residual and 16% drift-on-top against 25% for a purely
isotropic error, so **the temperature change on this rig is predominantly NORMAL, not
tangential** — it enters by making a different table state the best host at a different
brightness, not by rotating the realised matrix. That is why polar still returns 2.9x on
drifted data, and why re-anchoring the plan on the fresh table buys only 12% of the raw
error (0.2127 -> 0.1868) and nothing at all after projection.

## Two projections that sound right and are not

**Singular-value clipping.** A k x k sub-block of a larger unitary is a contraction, so
`sigma -> min(sigma, 1)` is the correct Frobenius projection onto that set — convex, unique,
nonexpansive, verified to 2e-16. It is still useless here, twice over. We never hold an
amplitude sub-block: entrywise absolute value is not norm-decreasing, and over 3600 measured
blocks `sqrt(T)` breaks `sigma <= 1` in **37%** of cases (12% even on the best sign class),
so a contraction test on unsigned intensity measures that artefact. And in the signed domain
it loses anyway: the measured singular pairs look like (1.02, 0.80), so most of the error is
a *small* singular value being too small — which clipping cannot touch and polar can.

**Substochastic projection.** Column sums are 1 by construction after `to_transfer` and only
3.2% of rows exceed 1. Projecting made agreement with the same heater state re-measured at
20 C **worse** (0.0475 -> 0.0551, helping in 0.8% of blocks). The row-sum excess correlates
at **r = +0.993** across the two temperatures: it is systematic — column-only normalisation
plus real per-row loss — not noise a projection is entitled to remove.

## Scope, which is narrow

Polar belongs in the composition / group-property check and nowhere else, because that is the
only place we host factors we *chose* to be orthogonal. On an arbitrary hosted target there
is no manifold at all, and projecting one measured **0.1350 -> 1.6998, thirteen times
worse**.

Two further limits. The polar map's Lipschitz constant is `2/(sigma_min(X)+sigma_min(Y))`
(Li & Sun, SIAM JMAA 23), so it *amplifies* perturbations when the singular values are
small — check `sigma_min` before trusting it. And the manifold is **O(k), not U(k)**:
intensity gives moduli, the differential passes recover signs, no phase is measured anywhere.

## What is genuinely worth keeping

`manifold_dist` needs no target, is a rigorous lower bound on the true error, and correlated
+0.995 with it here. It is the only honest self-check available mid-experiment — provided
nothing has projected the evidence away first.

`contraction_dist` is a target-free **refusal**: `sigma_max > 1` on the best sign class
proves a measured block is not a sub-block of any unitary. 12% of measured blocks fail it.

And unistochasticity is not the problem. Per-entry distance of the measured 4x4 matrices:
Birkhoff 0.0660, unistochastic 0.0670 (+1.5%), orthostochastic 0.0722 (+9%). The 0.066 is
loss and readout, not polytope geometry.
