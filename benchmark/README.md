# Unified MrGap / Manifold Fitting benchmark

This benchmark compares the repository's MrGap implementation with the
official MATLAB behavior of Yao et al.'s `manfit_ours.m` from
[Manifold Fitting](https://github.com/zhigang-yao/manifold-fitting/tree/master/Manifold%20Fitting/Matlab).

## What is implemented

- independently generated Cassini oval, RP(3), torus, and half-torus samples;
- ambient isotropic Gaussian noise;
- MrGap after every round from 1 through 5 with paper parameters;
- an oracle-tangent MrGap replacing local PCA by the true tangent basis;
- a line-by-line behavioral port of `manfit_ours.m` (cylinder averaging and its
  optional final averaging pass);
- noise, independently sampled sample-size, and neighborhood scans;
- one-sided clean-manifold GRMSE, paired RMSE, runtime, iteration count,
  local-neighborhood size, and Python-traced peak memory.

Observation samples for different `n` are generated independently. The dense
reference cloud is a separate controlled Monte Carlo approximation used only
for distance evaluation; observations are never subsampled from it. Each
trial's exact clean points are added to the evaluation reference so that the
noise-free distance is exactly zero and RP(3) does not inherit an avoidable
finite-reference error floor.

## Fairness conventions and unavoidable asymmetries

Manifold Fitting requires the true noise standard deviation and defines

```text
r = 5 sigma / log10(n)
R = 10 sigma sqrt(log(1/sigma)) / log10(n).
```

It therefore receives the simulation `sigma`. MrGap instead receives the fixed
GP and neighborhood parameters reported in its paper. This is intentional and
is recorded in the report: Manifold Fitting has oracle noise knowledge, while
MrGap has possible parameter mismatch away from its default condition.

The MrGap paper defines an empirical-Bayes step that jointly maximizes the sum
of local log marginal likelihoods over `(A,rho,sigma)` in every round. The
public `Mfit1.m` only evaluates the denoising map conditional on supplied
`A,rho,sig`; the repository does not include that optimizer or its experiment
drivers. Consequently, this benchmark does not test the paper's stopping rule
based on the change in the MLE of `sigma`. It uses stored example-specific
tuples: Cassini and torus match the estimates reported in Sections 4.1--4.2 of
the paper, while RP(3) and half-torus are retained as reproduction settings.
Round 1 uses the first tuple and rounds 2--5 reuse the final-round tuple.

Official demos create a separate, manifold-informed `sample_init`. For the main
paired benchmark we set `sample_init=Y`, so both methods return one fitted point
per noisy observation and paired RMSE is meaningful. The official
`manfit_ours.m` is a single pass, not an iterative optimizer; its iteration
count is therefore 1. MrGap runtime is cumulative through the reported round.

At `sigma=0`, the official formulas contain `log(1/sigma)` and are undefined.
The benchmark uses and labels an identity convention for this single endpoint.

The half-torus paper example reports only its last-round GP parameters. Those
values are reused for all rounds and this limitation is recorded.

## Running

Pilot mode runs all requested noise values and sample sizes, but skips exact
MrGap at `n=5000`; this makes the scalability failure explicit instead of
silently replacing the dense local GP by an approximation:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /opt/anaconda3/bin/python \
  benchmark/manifold_benchmark.py --profile pilot
```

Full mode runs exact MrGap at every requested sample size and uses three
replicates plus a larger independent reference cloud. It can be very slow at
`n=5000` because every center solves a dense local GP system:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache /opt/anaconda3/bin/python \
  benchmark/manifold_benchmark.py --profile full
```

Use `--manifolds cassini torus` or `--repeats 1` for targeted diagnostics.
Outputs include `benchmark_rows.csv`, metadata, seven plots, `REPORT.md`, and a
diagnostic discussion.

## Output columns

The main CSV directly supports the requested table:

```text
method | manifold | n | sigma | reconstruction_error | runtime_sec
```

It additionally records raw clean-manifold distance, paired RMSE, iterations,
bandwidth, median/minimum local sample size, memory, oracle status, and endpoint
status. Peak memory is measured with Python's `tracemalloc`; native BLAS/SciPy
allocations may not be fully represented, so it should be treated as a lower
bound rather than process peak RSS.
