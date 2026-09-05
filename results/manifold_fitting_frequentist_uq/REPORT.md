# Exact Manifold Fitting + frequentist UQ diagnostic

This experiment keeps the point-estimator center equal to the final, smoothed output of the repository's faithful `manfit_ours.m` port. No GP is used in any proposed MF tube. The run used n=3000, sigma=0.06, 100 Monte Carlo replicates per geometry, a 4800-point truth grid, and B=200 full-algorithm resamples on each of 2 selected datasets per geometry.

## Main results

| geometry | method | coverage | mean half-width / sigma | max half-width / sigma | inside noise | reps |
|---|---|---:|---:|---:|---:|---:|
| circle | full_algorithm_bootstrap | 1.00 | 0.59 | 0.61 | 1.00 | 2 |
| circle | sampling_only | 0.95 | 0.48 | 0.79 | 1.00 | 100 |
| circle | sampling_plus_direction_additive | 1.00 | 0.65 | 13.12 | 0.61 | 100 |
| circle | sampling_plus_direction_rss | 1.00 | 0.52 | 12.61 | 0.73 | 100 |
| ellipse | full_algorithm_bootstrap | 1.00 | 0.76 | 0.80 | 1.00 | 2 |
| ellipse | sampling_only | 0.95 | 0.51 | 0.84 | 1.00 | 100 |
| ellipse | sampling_plus_direction_additive | 1.00 | 0.70 | 10.10 | 0.29 | 100 |
| ellipse | sampling_plus_direction_rss | 0.99 | 0.55 | 9.52 | 0.58 | 100 |

- **circle.** Sampling-only coverage was 0.95; the analytic direction term changed it to 1.00. The selected-dataset full bootstrap coverage was 1.00.
- **ellipse.** Sampling-only coverage was 0.95; the analytic direction term changed it to 1.00. The selected-dataset full bootstrap coverage was 1.00.

1. **Conditional averaging uncertainty was sufficient at 95% in this run:** it attained 0.95 coverage for the circle and 0.95 for the ellipse. This is an empirical result for the stated setting, not a general guarantee.
2. **Adding the analytic direction term raised 95% coverage to 1.00 and 1.00.** The gain came with unstable maxima: the largest additive width was 13.12 sigma for the circle and 10.10 sigma for the ellipse, caused by very small direction signals.
3. **Mean analytic-additive and bootstrap radii were fairly close:** 0.65 versus 0.59 sigma for circle, and 0.70 versus 0.76 sigma for ellipse. Bootstrap values use only 2 selected dataset(s) per geometry.
4. **The residual bias has the expected geometric order and inward sign.** Its circle magnitude lies between the sigma^2/(2R) and sigma^2/R references. On the ellipse, curvature correlates -0.953 with signed residual and 0.953 with its magnitude, supporting an O(sigma^2 kappa) pattern while rejecting a universal C=1/2 assumption.
5. **The stable MF-centered bands were materially narrower than the raw noise band.** Sampling-only mean widths were 0.48 and 0.51 sigma and had strict-containment fraction 1.00. The selected-dataset bootstrap means were 0.59 and 0.76 sigma and were also strictly contained. The additive analytic band's rare direction singularities reduced strict containment.

The analytic direction term is computed from the ball-mean covariance, the normalization Jacobian, and the normal-projected contraction Jacobian. The actual repository estimator outputs the complete cylinder mean. Conditional on fixed cylinder membership that complete mean is constant in the direction: the derivatives of its normal and tangent components cancel. Consequently the analytic direction scale is a diagnostic for the projected contraction component, while only the full bootstrap includes discrete cylinder-membership changes. This is a material limitation, not a confidence theorem.

Before smoothing, the recorded sampling scale is exactly `sample_variance(s_i) / m` for the unweighted axial cylinder statistic; its noise-only reference is `sigma^2 / m`. For the main final-MF tube, the code explicitly composes the cylinder and smoothing averages into observation weights, then uses their empirical weighted covariance and its largest directional variance. The corresponding Gaussian reference is `sigma^2 sum(w_i^2)`. This preserves the final smoothing and accounts for observation overlap within each final point's linear representation.

The bootstrap intervals use the empirical quantile of the Hausdorff deviation between a full bootstrap rerun and the original final MF cloud. They include the ball mean, direction, cylinder selection, contraction average, and final smoothing. Bootstrap coverage in the table uses only the selected datasets (`replicates` records that denominator), so it is less stable than the 100-replicate analytic results.

## Oracle-direction ablation (95%)

| geometry | estimated H | oracle H | estimated sampling coverage | oracle sampling coverage | estimated bootstrap radius | oracle bootstrap radius |
|---|---:|---:|---:|---:|---:|---:|
| circle | 0.0229 | 0.0231 | 0.95 | 0.96 | 0.0356 | 0.0318 |
| ellipse | 0.0257 | 0.0249 | 0.95 | 0.95 | 0.0459 | 0.0375 |

The oracle changes only the cylinder direction and is never used by the proposed estimator. For the ellipse it changed mean geometric error from 0.0257 to 0.0249, while sampling-band coverage stayed at 0.95. Thus direction uncertainty does **not** explain an ellipse failure mode for the exact MF estimator in this experiment. The strong inverse relation between direction signal and direction error remains visible, but it mainly makes the first-order analytic direction correction unstable at a small number of points.

## Population/geometric bias

After averaging over Monte Carlo sampling and replacing the direction by its oracle value, the remaining signed normal displacement is used as the residual population-bias diagnostic. The circle mean absolute residual was 0.00292; the references are sigma^2/R=0.00360 and sigma^2/(2R)=0.00180. For the ellipse, correlation between signed residual bias/sigma^2 and curvature across bins was -0.953; correlation with bias magnitude was 0.953. The negative signed relation means higher-curvature regions move farther inward. This comparison tests an O(sigma^2 kappa) pattern; it does not estimate or apply a truth-based correction, and C=1/2 is shown only as a circle-inspired reference.

The strict-containment column evaluates both constructed normal boundaries geometrically against the dense true curve and the 1.96 sigma reference width. Width ratios below 1.96 indicate narrower average radii; strict containment additionally accounts for displacement of the MF center.

## Files

- `uq_summary.csv`: requested coverage and width table.
- `direction_diagnostics.csv`: direction signal, error, delta scale, and curvature.
- `bias_diagnostics.csv`: curvature-binned estimated and oracle signed residuals.
- `oracle_ablation.csv`: estimator, analytic, and bootstrap oracle comparison.
- `bootstrap_diagnostics.csv`: full-rerun radii and coverage on selected datasets.
- `mf_trace_circle.csv` and `mf_trace_ellipse.csv`: every local set size, ball mean, direction, contraction statistic, contracted point, final smoothed point, and analytic scale for the representative datasets.

All coverage statements are finite-simulation diagnostics for these settings. No confidence theorem is claimed.
