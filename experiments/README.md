# Experiments

## Cassini oval sensitivity experiment

`cassini_sensitivity.py` is a direct Python translation of `Mfit1.m` and
`Mfit2.m`. It first runs the repository's stored Cassini data, then studies:

- noise standard deviation at fixed sample size 102;
- sample size at fixed noise standard deviation 0.04.

The GP and neighborhood parameters remain fixed at the paper's values. Thus,
the experiment measures sensitivity of the published setting rather than the
best result obtainable after retuning every simulation.

Run from the repository root with the Anaconda Python environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /opt/anaconda3/bin/python \
  experiments/cassini_sensitivity.py --repeats 10
```

Outputs are written to `results/cassini/`. The `*_raw.csv` files contain every
replicate; `*_summary.csv` and the PNG figures contain means and one-standard-
deviation bands. `metadata.json` records the seed and all fixed parameters.

The simulation uses uniformly sampled curve parameters. The paper describes
its 102 Cassini samples as non-uniform but does not provide their sampling law;
the stored `Cassini oval.mat` dataset is therefore kept as a separate baseline.

## Circle GP-UQ / MLE stability prototype

`circle_gp_uq.py` is a diagnostic prototype for the current theory work on
random local charts, errors-in-variables regression, GP reconstruction, and
frequentist uncertainty quantification.  It is intentionally separate from the
published MrGap benchmark.

The default experiment uses a noisy unit circle in `R^2` and:

- splits the observations 50/50 into a chart pool and a regression pool;
- estimates each local tangent/normal frame by PCA in an outer ambient ball;
- restricts GP training to an inner tangent-coordinate interval;
- fits a squared-exponential GP with a profiled constant mean;
- estimates `(A, ell, s2)` by bounded log-scale multistart marginal likelihood;
- records the exact GP/kriging linear-smoother weights `a`;
- reports the leading frequentist variance
  `Omega = sigma^2 ||a||^2` using the known simulation noise;
- reports the usual latent GP posterior variance only as a diagnostic, not as a
  frequentist confidence variance;
- checks local MLE identifiability by comparing near-optimal multistart
  solutions.

Run from the repository root:

```bash
python experiments/circle_gp_uq.py
```

A lighter diagnostic run is, for example,

```bash
python experiments/circle_gp_uq.py --centers 8 --random-starts 2
```

Outputs are written by default to `results/circle_gp_uq/`:

- `circle_gp_uq_results.csv`: per-center reconstruction, MLE, and UQ metrics;
- `circle_gp_mle_multistart.csv`: every optimizer start for every local GP;
- `summary.json`: aggregate reconstruction, coverage, and MLE-stability metrics;
- `metadata.json`: all simulation and diagnostic settings;
- `local_geometry.png`, `mle_multistart_nll.png`, and
  `sandwich_intervals.png`: diagnostic figures.

### Current theoretical caveat

The code still applies the observed outer ambient localization before the inner
tangent-coordinate restriction.  Therefore it should not yet be read as a
complete implementation of the proposed selection-clean theoretical estimator:
the localization/selection bridge remains a separate theoretical issue.  The
current purpose is narrower: test whether the GP/MLE layer is numerically
stable, identify low-signal patches where the length scale is effectively
unidentified, and compare GP posterior uncertainty with the directly
computable frequentist sampling scale.
