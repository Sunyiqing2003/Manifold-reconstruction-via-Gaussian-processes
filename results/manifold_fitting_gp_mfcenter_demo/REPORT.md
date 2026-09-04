# MF-centered GP-UQ diagnostic

GitHub Actions run: 33853100425  
Branch: `demo/manifold-fitting-confidence-band`

Configuration:
- data-driven Manifold Fitting pilot;
- `n=3000`, `sigma=0.06`, `h=1.5 sigma`;
- 60 pilot-grid points and dense geometric evaluation;
- 20 Monte Carlo replicates per manifold;
- GP simultaneous critical value: finite-grid Bonferroni;
- reference observation-noise band: `+-1.96 sigma`.

The point estimator is kept at the Manifold Fitting pilot.  The local GP is used to estimate a residual normal field and posterior uncertainty.  The MF-centered GP tube has half-width

`abs(GP residual posterior mean) + q * GP posterior sd`.

## Results

| manifold | mean MF Hausdorff | mean GP-refined Hausdorff | fraction MF better | MF-centered GP tube coverage | tube strictly inside reference noise band | mean max tube width / noise half-width |
|---|---:|---:|---:|---:|---:|---:|
| circle | 0.0165 | 0.0257 | 1.00 | 1.00 | 1.00 | 0.572 |
| ellipse | 0.0498 | 0.0269 | 0.00 | 1.00 | 0.00 | 0.775 |

## Interpretation

The earlier visual impression that the GP-refined center is worse than the MF pilot is correct for the circle under this configuration, but it is not a general conclusion.  On the ellipse, the current MF pilot has a systematic geometric error near high-curvature regions and the GP normal correction substantially improves Hausdorff error.

The MF-centered residual-aware GP tube covers the true manifold in all 20 replicates for both shapes.  For the circle it also stays strictly inside the `+-1.96 sigma` reference observation band in all replicates.  For the ellipse it does not: although its own maximal half-width is on average only about 77.5% of the reference noise half-width, the MF center is displaced enough that the outer tube boundary crosses the reference band.  Thus `tube width < noise width` is not sufficient for strict set containment when the center itself is biased.

This suggests that a fixed policy of either always keeping the MF center or always replacing it by the GP-refined center is too crude.  The next design should treat the GP residual mean as a geometric correction when it is large relative to its uncertainty, while shrinking or suppressing the correction when it is statistically weak.  A natural candidate is a shrinkage update of the form

`M_center(z) = M_MF(z) + lambda(z) * m_GP(z) * n(z)`

with `lambda(z)` determined by residual signal-to-uncertainty.  The UQ layer can then be built around this stabilized center.
