# Notes-faithful GP contraction diagnostic

This experiment compares the original cylinder average with a scalar GP
prediction at projected coordinate `q=0`. Both estimators use the same
Yao ball-step direction and exactly the same cylinder observations. The
independent first split supplies only the closed query scaffold.

## Frozen setup

- `n=3000`, `sigma=0.06`, `20` Monte Carlo replicates per manifold;
- query offset `c_offset*sigma=1.0*sigma`;
- GP amplitude `A=1.0*sigma^2`, length scale `ell=1.0*r`;
- constant unknown GP mean handled by universal kriging;
- finite-grid Bonferroni multiplier for UQ visualization.

## Point-estimation results

| manifold | mean H_avg | mean H_GP | median H_avg | median H_GP | fraction H_GP < H_avg |
|---|---:|---:|---:|---:|---:|
| circle | 0.02652 | 0.03032 | 0.02762 | 0.03017 | 0.30 |
| ellipse | 0.03027 | 0.03172 | 0.02909 | 0.03055 | 0.35 |

With these frozen hyperparameters, the GP does not improve the primary
Hausdorff criterion on average. It beats the shared-cylinder average in
30% of circle replicates and
35% of ellipse replicates. The
ellipse top-curvature quartile has only a small positive mean local-error
difference (0.00083)
in favor of GP. This is weak local evidence and does not overturn the
whole-curve Hausdorff comparison.

The comparison estimates the empirical difference between
`E[s | Y in V_z]` and a GP estimate of `E[s | q=0]`. The ellipse curvature
figure is post-hoc: curvature never enters either estimator.

For a local quadratic graph, the motivating heuristic is that cylinder
averaging contains both an EIV term and a transverse-window term, whereas
the GP target at `q=0` may remove the extra transverse-window contribution.
The remaining population bias can still be of order `kappa*sigma^2/2`;
the experiment does not subtract it and does not claim that GP eliminates bias.

## Conditional UQ diagnostics

| manifold | posterior coverage | frequentist-mean coverage | mean s_post/s_F | max s_post/s_F | posterior tube inside noise band |
|---|---:|---:|---:|---:|---:|
| circle | 0.95 | 0.95 | 1.113 | 1.201 | 1.00 |
| ellipse | 0.95 | 0.85 | 1.119 | 1.219 | 1.00 |

These are finite-grid conditional GP simultaneous bands. The posterior SD
and `sigma*||a_z||` quantify different uncertainties and are reported
separately. Empirical inclusion here is a simulation diagnostic, not an
honest true-manifold confidence theorem. A maximum half-width below
`1.96*sigma` also does not imply geometric containment; containment is
checked from the dense band boundaries.

## Neighborhood diagnostics

| manifold | mean median cylinder n | minimum cylinder n | ball fallback fraction | cylinder fallback fraction | high-curvature GP improvement |
|---|---:|---:|---:|---:|---:|
| circle | 46.8 | 26 | 0.0000 | 0.0000 | -0.00107 |
| ellipse | 41.9 | 24 | 0.0000 | 0.0000 | 0.00083 |

The last column uses the top curvature quartile. It is most meaningful for
the ellipse; the circle has constant curvature and acts as a control.
