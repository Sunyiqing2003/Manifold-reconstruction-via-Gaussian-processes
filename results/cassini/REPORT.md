# Cassini oval: first sensitivity study

This report uses 20 deterministic replicates (`seed=20260801`). The dense
reference contains 100,000 points generated from equation (19). In all scans,
the published settings are held fixed:

- `d=1`, `epsilon=0.3`, `delta=0.6`;
- round 1: `A=0.014`, `rho=0.2`, `noise_variance=0.002`;
- round 2: `A=0.048`, `rho=0.3`, `noise_variance=2e-5`.

The scans therefore measure robustness of the paper's setting, not performance
after retuning parameters for every condition.

## Stored-data reproduction

| Output | This reproduction | Paper |
|---|---:|---:|
| Observed | 0.05905 | 0.0590 |
| Denoised after round 2 | 0.02242 | 0.0224 |
| Interpolated | 0.02153 | 0.0216 |

## Noise sensitivity (`n=102`)

| Noise SD | Observed | Round 1 | Round 2 |
|---:|---:|---:|---:|
| 0.00 | 0.00002 | 0.00570 | 0.00502 |
| 0.01 | 0.01393 | 0.00817 | 0.00773 |
| 0.02 | 0.02785 | 0.01261 | 0.01226 |
| 0.04 | 0.05565 | 0.02475 | 0.02373 |
| 0.06 | 0.08340 | 0.04479 | 0.03958 |
| 0.08 | 0.11104 | 0.07191 | 0.06217 |
| 0.12 | 0.16594 | 0.12094 | 0.10994 |

At the paper setting (`sigma=0.04`), round 2 reduces GRMSE by about 57%. At
`sigma=0.12`, the reduction is only about 34% and replicate-to-replicate
variation is much larger. With no noise, fitting creates an error near 0.005;
this is the local regression's smoothing bias.

## Sample-size sensitivity (`sigma=0.04`)

| Samples | Observed | Round 1 | Round 2 | Median epsilon-neighbors |
|---:|---:|---:|---:|---:|
| 30 | 0.05445 | 0.04365 | 0.04319 | 4.0 |
| 50 | 0.05525 | 0.03642 | 0.03538 | 6.0 |
| 75 | 0.05527 | 0.03088 | 0.03048 | 8.0 |
| 102 | 0.05564 | 0.02675 | 0.02558 | 10.75 |
| 150 | 0.05644 | 0.02145 | 0.02009 | 14.0 |
| 250 | 0.05614 | 0.01785 | 0.01666 | 22.5 |

At 30 samples, the median minimum epsilon-neighborhood size across replicates
is only 1.5 (some local PCA problems see only the center itself). Consequently,
the estimated tangent direction can be arbitrary and round 2 reduces error by
only about 21%. At 250 samples, the reduction is about 70%.

## Practical takeaways

1. Performance is controlled more directly by the number of points in each
   local neighborhood than by the global sample count.
2. Fixed `epsilon` cannot be expected to work uniformly as sample size or noise
   changes. Too few neighbors destabilize local PCA; too large a neighborhood
   increases curvature bias.
3. A second denoising round is not the primary source of improvement. It is a
   modest refinement and cannot rescue poorly populated local neighborhoods.
4. Hyperparameter adaptation matters. These results deliberately reuse the
   paper's parameters, so part of the high-noise degradation is parameter
   mismatch rather than a definitive limit of MrGap.
5. The next useful experiment is a two-dimensional grid over `epsilon` and
   noise/sample size, followed by a comparison with an adaptive k-nearest-
   neighbor neighborhood. That would separate data scarcity from bandwidth
   choice without requiring theoretical analysis.
