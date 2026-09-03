# Manifold Fitting + GP confidence-band diagnostic

This report records the first CI runs of `experiments/manifold_fitting_gp_confidence_demo.py`.
The final estimator is a closed image curve

\[
\widehat M_{GP}=\{z+\widehat f_z(0)n_z:z\in\widetilde M\},
\]

where `M_tilde` is either the oracle true curve or an independent data-driven
preliminary curve obtained from the repository's Manifold Fitting port.

The local GP uses a squared-exponential kernel at the tangent coordinate and smooth
localization through a heteroskedastic nugget.  Two uncertainty scales are compared:

1. latent GP posterior standard deviation;
2. frequentist sampling standard deviation of the same GP posterior mean,
   `sigma * ||a||`, where `a` is the exact kriging linear-smoother weight vector.

The finite-grid simultaneous bands use a Bonferroni critical value.  This is
intentionally conservative and avoids assuming that independently fitted local GPs
form a coherent joint posterior process across chart centers.

## Quick run

Configuration: `n=600`, `sigma=0.05`, `h/sigma=1.5`, 36 curve centers, five Monte
Carlo repeats.

| manifold | pilot | GP posterior coverage | freq GP-mean coverage | mean max GP radius | mean max freq radius | GP/freq radius ratio |
|---|---|---:|---:|---:|---:|---:|
| circle | data | 1.00 | 1.00 | 0.0997 | 0.0693 | 1.44 |
| circle | oracle | 1.00 | 1.00 | 0.0972 | 0.0747 | 1.30 |
| ellipse | data | 1.00 | 1.00 | 0.1182 | 0.0778 | 1.52 |
| ellipse | oracle | 1.00 | 1.00 | 0.1108 | 0.0733 | 1.51 |

For the oracle pilot, the simulation-only population-bias envelopes were about
`0.0129` for the circle and `0.0145` for the ellipse.  They are much smaller than
the Bonferroni simultaneous radii in this small-sample run.

## Moderate oracle run

Configuration: `n=2500`, `sigma=0.06`, `h/sigma=1.5`, 48 curve centers, ten Monte
Carlo repeats, with the local GP capped at 120 highest-localization-weight points.

| manifold | GP posterior coverage | freq GP-mean coverage | mean max GP radius | mean max freq radius | pop. bias envelope | bias / GP radius | bias / freq radius |
|---|---:|---:|---:|---:|---:|---:|---:|
| circle | 1.00 | 1.00 | 0.0493 | 0.0419 | 0.0132 | 0.268 | 0.315 |
| ellipse | 1.00 | 1.00 | 0.0533 | 0.0446 | 0.0113 | 0.213 | 0.254 |

The posterior scale remains larger than the directly computed repeated-sampling
scale: the ratio of the simultaneous radii is approximately 1.18--1.20 in this run.
Thus the first GP-based simultaneous bands are conservative rather than
under-covering.

## Interpretation

These numbers should not yet be interpreted as evidence that GP posterior variance is
frequentist-valid.  The current simultaneous critical value is Bonferroni and is
therefore deliberately conservative.  The more informative next diagnostic is
pointwise calibration of

\[
\frac{\widehat f_z(0)-f_z^{\rm target}(0)}{s_{post}(z)}
\quad\text{versus}\quad
\frac{\widehat f_z(0)-f_z^{\rm target}(0)}{\sigma\|a_z\|},
\]

followed by a simultaneous Gaussian-process calibration that accounts for cross-chart
correlation.

The current runs nevertheless show an important numerical fact: for this hybrid
Manifold-Fitting + GP estimator, the usual latent GP posterior standard deviation is
not the same as the frequentist sampling standard deviation of the GP mean.  In these
runs it is systematically larger, by roughly 18--50 percent depending on sample size
and pilot mode.

The population-bias issue also remains separate.  At sufficiently large effective
sample size, either variance scale can become smaller than the deterministic
curvature/EIV displacement.  A true-manifold confidence region then needs an honest
bias allowance or inversion/correction of the population forward operator; shrinking
GP variance alone cannot resolve that target mismatch.

## Caveats

- GP hyperparameters are fixed in this first diagnostic (`A = sigma^2`,
  `ell = h`) so that UQ calibration is not confounded by MLE instability.
- Smooth localization is implemented through a heteroskedastic nugget and a
  computational cap on local points; it is not a completed EIV correction.
- Data-pilot UQ is conditional on the preliminary curve and currently omits pilot-stage
  uncertainty.
- Oracle population-bias quantities use simulation truth and are diagnostics only.
- The finite-grid Bonferroni band is intentionally conservative; it is not the final
  confidence-tube construction proposed for theory.
