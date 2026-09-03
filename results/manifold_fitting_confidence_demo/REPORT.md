# Manifold Fitting confidence-band demo: first diagnostic results

This report summarizes the first CI runs of `experiments/manifold_fitting_confidence_demo.py` on the draft branch `demo/manifold-fitting-confidence-band`. The experiment is diagnostic rather than a coverage theorem.

## Setup

The final estimator is a closed image curve

\[
\widehat M_h=\widehat G_h(\widetilde M),
\]

not a collection of independently denoised anchors. The refinement update is normal-only and uses smooth tangent/normal localization. The bootstrap is simultaneous over the full curve and is conditional on the preliminary manifold.

Two modes are distinguished:

- `oracle`: the true circle/ellipse is used only as the preliminary parameter manifold, isolating the refinement/UQ layer;
- `data`: an independent pilot split is passed through the repository's Manifold Fitting port and converted to a closed radial pilot curve.

In oracle mode, a large independent Monte Carlo sample also approximates the population normal displacement. This is used only for simulation diagnostics (`oracle bias-aware` and `oracle bias-corrected`), not as a practical bias estimator.

## Quick preflight

Settings: `n=600`, `sigma=0.05`, `h/sigma=1.5`, 8 Monte Carlo repetitions, 80 bootstrap repetitions.

| manifold | pilot | stochastic-only constant tube | spatially varying tube | oracle bias-aware | oracle bias-corrected | mean directed error | mean stochastic radius | population-bias envelope | bias / stochastic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | data | 0.875 | 0.625 | -- | -- | 0.03244 | 0.04704 | -- | -- |
| circle | oracle | 1.000 | 0.625 | 1.000 | 1.000 | 0.03208 | 0.04585 | 0.00644 | 0.141 |
| ellipse | data | 1.000 | 0.375 | -- | -- | 0.03604 | 0.05105 | -- | -- |
| ellipse | oracle | 1.000 | 0.625 | 1.000 | 1.000 | 0.03233 | 0.05334 | 0.00885 | 0.167 |

At this sample size the experiment is variance dominated. The population bias is only around 14--17% of the simultaneous stochastic radius, so this run is not expected to expose the true-manifold undercoverage problem strongly. The low coverage of the current spatially varying/studentized tube, despite adequate constant-radius coverage, indicates that the local studentization/normal-tube construction should not yet be trusted.

## Targeted high-n bias-dominated scan

Settings: oracle pilot only, `n=8000`, `sigma=0.08`, `h/sigma=1.5`, 20 Monte Carlo repetitions, 150 bootstrap repetitions.

| manifold | stochastic-only constant tube | spatially varying tube | oracle bias-aware | oracle bias-corrected | mean directed error | mean stochastic radius | population-bias envelope | bias / stochastic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | 0.000 | 0.000 | 1.000 | 0.800 | 0.01742 | 0.01226 | 0.01202 | 0.982 |
| ellipse | 0.000 | 0.000 | 1.000 | 0.900 | 0.02434 | 0.01325 | 0.02105 | 1.590 |

This run shows the intended failure mode very clearly: once the stochastic radius falls to the same scale as, or below, the population displacement, the variance-only tube misses the true curve in every replicate, even with the true preliminary manifold. Adding the oracle population-bias envelope restores coverage in this small scan, though the resulting 100% should not be interpreted as calibrated 95% coverage with only 20 repetitions.

The result is also quantitatively consistent with the local curvature calculation. For the unit circle,

\[
\frac{h^2+\sigma^2}{2R}
=0.0104,
\]

while the Monte Carlo population-bias envelope is `0.0120`. For the ellipse with semiaxes `(a,b)=(1.4,0.8)`, the minimum radius of curvature is

\[
\tau_{\min}=b^2/a\approx0.4571,
\]

so the corresponding maximal local-curvature benchmark is

\[
\frac{h^2+\sigma^2}{2\tau_{\min}}
\approx0.02275,
\]

close to the observed population-bias envelope `0.02105`.

## Interpretation

The first diagnostic therefore supports the proposed decomposition

\[
\text{manifold error}
=
\text{population curvature/EIV displacement}
+
\text{sampling fluctuation}.
\]

More specifically:

1. The refinement-stage bootstrap itself is not the main obstacle in the high-n oracle-pilot experiment; the true curve is missed because the population center is displaced.
2. The effect is stronger for the ellipse, consistent with its larger maximum curvature.
3. A variance-only confidence tube can look satisfactory in a variance-dominated regime and then fail completely as `n` grows.
4. The oracle bias-corrected coverage of 0.8/0.9 is not yet fully satisfactory. Possible causes include finite Monte Carlo error in the estimated population displacement, residual nonlinear geometry not removed by a normal pointwise correction, and finite-bootstrap/grid effects. This should be diagnosed before claiming that an explicit bias correction works.
5. The current spatially varying/studentized tube is substantially less reliable than the constant Hausdorff-style radius and should be treated as experimental.

## Next experiment

The most informative next scan is not a generic bandwidth sweep. It is a regime-transition experiment over `(n,sigma)` using oracle pilot first:

- record `population bias / stochastic radius`;
- record simultaneous true-curve coverage;
- check whether coverage collapses when this ratio crosses order one;
- compare circle with ellipse high-curvature locations;
- only after this works, add data-pilot uncertainty.

A second useful comparison is to replace the smooth refinement weights by the exact hard-cylinder rule used in the Manifold Fitting port, while keeping the same image-curve and bootstrap machinery. That will show whether the phenomenon is intrinsic to the original cylinder estimator rather than an artifact of the smooth demo weights.
