# Diagnostic discussion of the pilot benchmark

These observations describe the deterministic one-replicate pilot in this
folder. They are useful for choosing the next experiments, but uncertainty
bands from the `full` profile are required before treating small differences as
stable rankings.

## Headline comparison

There is no single winner. At the paper-like default conditions, two-round
MrGap and Manifold Fitting are nearly tied on Cassini and the full torus. MrGap
is better on the half-torus, while Manifold Fitting is better on pilot RP(3).
Manifold Fitting is generally 2--6 times faster through `n=1000` in this Python
port.

The comparison is asymmetric in an important way: Manifold Fitting receives
the true simulation noise standard deviation. MrGap keeps the paper's GP and
neighborhood parameters fixed even when simulation noise changes. Thus the
noise scan measures published-setting robustness, not each method after oracle
retuning.

## Noise failure modes

Both methods can make lightly noisy data worse. Paper parameters were selected
at nonzero reference noise, and their smoothing bias dominates when noise is
only `0.01--0.02`, especially on RP(3), torus, and half-torus. For Manifold
Fitting, shrinking `sigma` also shrinks its cylinder. The implementation then
often has fewer than 11 cylinder points and falls back to a five-neighbor local
mean. Consequently, its error need not decrease monotonically as noise tends to
zero. At exactly zero the official bandwidth formula is undefined; the plotted
zero endpoint is the benchmark's explicitly labelled identity convention.

At high noise, fixed-parameter MrGap degrades sharply on Cassini and especially
RP(3). Manifold Fitting is more robust in those pilot scans, helped by its use
of the true `sigma`. This is not evidence that its bandwidth can be selected as
well when noise is unknown.

## Is tangent estimation the bottleneck?

Sometimes. On RP(3), replacing PCA tangents by true tangents reduces two-round
error by about `0.087` at `sigma=0.08` and `0.093` at `sigma=0.12`. Cassini also
shows a substantial gain (`0.017`) at `sigma=0.08`. In those regimes, local
chart estimation is a major bottleneck.

Oracle tangents barely help the ordinary torus and are slightly worse on the
half-torus. The half-torus result is especially informative: boundary-neighborhood
asymmetry and GP smoothing bias remain even with the exact tangent plane. A
better tangent estimator alone will not solve the boundary problem.

## Iteration count

At each manifold's default condition:

| manifold | 1 round | 2 rounds | 3 rounds | 4 rounds | 5 rounds |
|---|---:|---:|---:|---:|---:|
| Cassini | 0.03006 | 0.02817 | **0.02813** | 0.02820 | 0.02833 |
| RP(3) | 0.09041 | **0.08861** | 0.09325 | 0.10184 | 0.11264 |
| Torus | 0.08681 | 0.07110 | **0.06748** | 0.06917 | 0.07419 |
| Half-torus | 0.06901 | 0.05870 | **0.05684** | 0.05720 | 0.06030 |

The useful stopping point is two or three rounds, not universally two.
Cassini's two- versus three-round difference is negligible, while the third
round gives a clearer gain on the torus examples. RP(3) deteriorates immediately
after round two. Rounds four and five show the general risk of accumulated
smoothing and neighborhood-drift bias, so iteration should not be treated as a
monotone optimizer.

This scan does not establish convergence under MrGap's published stopping
rule. The paper re-estimates `(A,rho,sigma)` by summed local marginal likelihood
and stops when the change in `sigma` is small. The public `Mfit1.m` takes those
parameters as inputs, and the current benchmark does not implement the missing
optimizer. Instead, round 1 uses the stored first-round tuple and rounds 2--5
reuse the stored final-round tuple. The curve is therefore a fixed-parameter
repeated-map diagnostic; its GRMSE-optimal round is an oracle simulation result.

## Sample size and effective local size

The observation sets for every `n` were generated independently. MrGap is
usually better at `n=50--100`, where Manifold Fitting's cylinder often contains
too few points and invokes its fallback. Differences narrow as `n` grows. On
RP(3), Manifold Fitting overtakes MrGap around `n=500`; on Cassini, torus, and
half-torus, MrGap remains slightly better through the exact pilot endpoint
`n=1000`.

The local-neighbor plot is more diagnostic than global `n`. For example, at
`n=5000`, the median MrGap epsilon-neighborhood sizes would be about 354
(Cassini), 165 (RP3), 161 (torus), and 334 (half-torus). Dense GP solves at every
center then dominate runtime and memory. Pilot mode records these cases as
`skipped_resource_guard` rather than silently changing the algorithm;
Manifold Fitting results are still recorded through `n=5000`. Full mode removes
that guard.

## Neighborhood sensitivity

Small MrGap epsilon values fail through insufficient local PCA samples. In the
pilot, the best scanned epsilon is `0.6` for Cassini, `0.5` for RP(3), `0.8` for
torus, and `1.0` for half-torus. RP(3) and half-torus attain their best result at
the upper scan boundary, suggesting that their small-sample pilot is still
variance-limited.

Manifold Fitting is flat across bandwidth multipliers on RP(3) because its
median cylinder count remains one: the cylinder branch is mostly inactive and
the five-neighbor fallback controls the result. On torus and half-torus,
doubling both cylinder radii causes a large error increase, a clear curvature
bias signal. Cassini has a shallower optimum near the official multiplier one.

## Recommended next runs

1. Run `--profile full --repeats 3` first on Cassini and torus to attach
   uncertainty to the current curves.
2. Add a noise-estimation stage for Manifold Fitting; otherwise it retains
   oracle information unavailable in real data.
3. Compare official manifold-informed `sample_init` against the paired `Y`
   initialization used here. This isolates initialization advantage.
4. For the `n=5000` MrGap scalability limit, benchmark an explicitly named
   approximation (fixed-k GP neighborhoods or inducing points) rather than
   silently calling it the original method.
