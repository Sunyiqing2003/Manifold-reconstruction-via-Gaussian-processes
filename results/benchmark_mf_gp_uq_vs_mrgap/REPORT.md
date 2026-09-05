# MF-level reconstruction with GP uncertainty versus MrGap

This benchmark uses shared noisy replicates. MrGap is the repository's
one-round Mfit1-equivalent with frozen published example parameters; MF is
the faithful Yao cylinder estimator; Ours uses a split query scaffold and
the estimated Yao ball direction to regress axial displacement on projected
ambient coordinates. Truth enters evaluation and the labeled oracle-direction
ablation only.

## Default reconstruction (`n=3000`, `sigma=0.06`)

| geometry | method | repeats | mean Hausdorff | median Hausdorff | mean symmetric distance |
|---|---|---:|---:|---:|---:|
| circle | Manifold Fitting | 100 | 0.01463 | 0.01430 | 0.00591 |
| circle | MrGap | 100 | 0.01818 | 0.01810 | 0.00590 |
| circle | Ours | 100 | 0.03093 | 0.03071 | 0.00894 |
| ellipse | Manifold Fitting | 100 | 0.04729 | 0.04686 | 0.01583 |
| ellipse | MrGap | 100 | 0.02001 | 0.02020 | 0.00639 |
| ellipse | Ours | 100 | 0.03379 | 0.03301 | 0.00943 |

## Default 95% finite-grid UQ

| geometry | method / scale | coverage | mean width / sigma | max width / sigma | inside noise fraction |
|---|---|---:|---:|---:|---:|
| circle | MrGap posterior | 0.620 | 0.319 | 0.393 | 1.000 |
| circle | Ours frequentist_gp_mean | 0.860 | 0.619 | 0.833 | 1.000 |
| circle | Ours posterior | 0.950 | 0.690 | 0.990 | 1.000 |
| ellipse | MrGap posterior | 0.460 | 0.336 | 0.424 | 1.000 |
| ellipse | Ours frequentist_gp_mean | 0.640 | 0.644 | 0.873 | 1.000 |
| ellipse | Ours posterior | 0.780 | 0.722 | 1.129 | 1.000 |

## Answers to the four benchmark questions

1. **Reconstruction preservation.** On circle, Ours has 2.11 times MF's mean Hausdorff error, so MF-level accuracy is not preserved there. On ellipse the ratio is 0.71, so Ours improves on this MF implementation but remains worse than MrGap. No tuning used truth or attempted to reverse the earlier negative GP-versus-average result.
2. **Calibration.** At nominal 95%, Ours posterior coverage is 0.95 on circle and 0.78 on ellipse. The corresponding same-GP-mean values are 0.86 and 0.64. Calibration is therefore geometry-dependent; the ellipse remains undercovered.
3. **Efficiency.** Ours posterior mean half-width is 0.69 sigma on circle and 0.72 sigma on ellipse. Strict containment in the truth-centered `1.96 sigma` reference band is 1.00 and 1.00, checked from dense boundaries rather than inferred from width alone.
4. **Failure modes.** The robustness grid shows larger reconstruction error at small n and large sigma. Sparse settings can under-cover even when normalized widths are larger. The oracle-direction ablation changes ellipse coverage much more than circle coverage, identifying direction estimation as an important ellipse failure mechanism. The curvature plot also shows error spikes that are not matched by comparably large posterior SD changes.

The broad grid uses 3 replicates per non-primary setting; the base circle/ellipse report setting uses 100. Grid endpoints are diagnostics rather than precise coverage estimates.

The earlier frozen mechanism result remains part of the repository:
circle `H_avg=0.02652`, `H_GP=0.03032`; ellipse `H_avg=0.03027`,
`H_GP=0.03172`. Thus the GP layer is not introduced as a universal fitting
improvement. Its role is to attach explicit probabilistic uncertainty while
keeping reconstruction in the same practical range.

## Torus local diagnostic

- Manifold Fitting: mean Hausdorff `1.076`, mean symmetric distance `0.177`.
- MrGap: mean Hausdorff `0.828`, mean symmetric distance `0.166`.
- Ours: mean Hausdorff `0.822`, mean symmetric distance `0.176`.
Ours has mean posterior-to-frequentist SD ratio `1.71`. These are local scale checks; the sparse common query budget creates a large surface-discretization component, and no global torus coverage is reported.

## Geometric SNR diagnostic

- `R/sigma=8.3`: Ours mean Hausdorff `0.0296` over `3` replicates.
- `R/sigma=33.3`: Ours mean Hausdorff `0.0384` over `3` replicates.
With only three replicates at each added radius, this check does not support a monotone reach/noise conclusion.

Paired RMSE is left missing because all three reported objects use a common
query/image representation rather than retaining observation-to-latent pairing.

## MrGap comparability limitation

The public MrGap release omits the empirical-Bayes optimizer. Planar curves
therefore use the frozen first-round Cassini tuple; torus uses its published
first-round tuple. Posterior uncertainty is the latent covariance from that
same local GP fit. It is labeled posterior credible uncertainty and is not
truth-calibrated or claimed to be a frequentist interval.
