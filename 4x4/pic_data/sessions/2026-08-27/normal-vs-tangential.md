# Why unitary projection cannot fix drift

The chip is unitary by construction, so every matrix it can possibly produce lies on the
orthogonal manifold O(k). That single fact is worth a lot — it gives a target-free error
bar — and it is also the source of a trap that is easy to walk into and hard to notice.

## Two kinds of error, and only one of them is projectable

A measured matrix `M` differs from the true one `B` in two geometrically distinct ways.

**Normal error** takes the matrix *off* the manifold. Read noise, gain errors, a stale
per-port scale: none of these produce a valid unitary, so they push `M` into the space
around O(k). This error is visible without knowing `B` at all — the distance from `M` to
the nearest orthogonal matrix is a direct measurement of it.

**Tangential error** slides the matrix *along* the manifold. Drift is the clean example: a
mesh whose phases have moved still produces a perfectly valid unitary. It is simply a
different one. There is no residual, no inconsistency, nothing to detect.

Polar projection maps `M` to the nearest point on O(k). By construction it removes the
normal component and leaves the tangential component untouched. It cannot fix drift — not
because the implementation is weak, but because drift never left the manifold.

## Why the order matters, and why projecting first is worse than useless

Project first and two things happen. The tangential error survives, because projection has
no purchase on it. And the normal error — the only externally visible symptom — is deleted.
What comes out is a mathematically valid unitary, sitting exactly on the manifold, that is
the wrong answer. `manifold_dist` reads small. Every check passes. The result is confidently
incorrect and there is nothing left in the data to say so.

**Projection does not just fail to help a drifted measurement. It destroys the evidence that
the measurement was drifted.**

Correct the tangential part first — re-anchor against a freshly measured transfer table —
and what remains is predominantly normal, which is exactly the regime projection is good at.

## The measurement, and how far it can be trusted

Measured on the twin, composed product of two hosted orthogonal matrices:

| noise | distance to O(2) | raw error | polar | QR |
|---|---|---|---|---|
| none | 0.0077 | 0.0098 | **0.0047** | 0.0066 |
| 1.2% read noise | 0.0712 | 0.0814 | **0.0293** | 0.0407 |

Two things in that table are load-bearing.

Polar beats QR, and it is not a matter of taste. `W @ Vh` from the SVD is the *nearest*
orthogonal matrix in Frobenius norm; QR's `Q` is merely *an* orthogonal matrix and depends
on column ordering. Permuting the columns moved QR's answer by 0.0508 — more than the 0.0407
of error it removed.

The 2.2x improvement was measured under **iid Gaussian noise**, which is isotropic and
therefore mostly normal to the manifold. That is the best case for projection, and it is not
the case drift presents. The gain against a systematic tangential error is unmeasured and
should be assumed smaller.

The general statement is the one to remember: **polar provably reduces distance to the
manifold, never necessarily distance to the truth.** Those two coincide only when the error
is predominantly normal, and establishing that it is, is a prerequisite for trusting the
projection rather than a consequence of applying it.

## What it is genuinely good for

`manifold_dist` needs no target. On a bench measuring a matrix nobody knows the answer to,
it is a real error bar — a rigorous lower bound on the true error, since the nearest
orthogonal matrix is at least as close as the correct one. Measured, it read 0.8–0.9 of the
true error. That makes it the only honest self-check available mid-experiment, provided
nothing has projected the evidence away first.

One caveat on scope: a k x k sub-block of a larger unitary is not itself orthogonal.
Projection applies to a composed product of hosted orthogonal matrices, not to an arbitrary
2x2 target picked out of a 4x4 mesh.

The manifold here is **O(k), not U(k)**: intensity gives moduli, the differential passes
recover signs, and no phase is measured anywhere in this pipeline.
