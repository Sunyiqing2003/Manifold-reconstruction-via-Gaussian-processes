# MrGap vs Manifold Fitting: benchmark summary

This is an empirical diagnostic benchmark; no theoretical claim is made.

## Default-condition means

| method | manifold | n | sigma | error | runtime (s) | median neighborhood |
|---|---|---:|---:|---:|---:|---:|
| manifold_fitting | cassini | 102 | 0.04 | 0.02848 | 0.012 | 6.0 |
| manifold_fitting | half_torus | 400 | 0.12 | 0.06699 | 0.045 | 4.0 |
| manifold_fitting | rp3 | 500 | 0.04 | 0.07973 | 0.056 | 1.0 |
| manifold_fitting | torus | 500 | 0.12 | 0.06911 | 0.056 | 3.5 |
| mrgap | cassini | 102 | 0.04 | 0.02817 | 0.023 | 9.0 |
| mrgap | half_torus | 400 | 0.12 | 0.05870 | 0.100 | 28.0 |
| mrgap | rp3 | 500 | 0.04 | 0.08861 | 0.129 | 21.0 |
| mrgap | torus | 500 | 0.12 | 0.07110 | 0.124 | 18.0 |

## Empirical diagnostic highlights

### cassini
- `mrgap` non-improving positive-noise values: none in positive-noise scan.
- `manifold_fitting` non-improving positive-noise values: 0.01.
- Largest absolute oracle-tangent gain: 0.01695 at sigma=0.08.

### half_torus
- `mrgap` non-improving positive-noise values: 0.01, 0.02.
- `manifold_fitting` non-improving positive-noise values: 0.01, 0.02.
- Largest absolute oracle-tangent gain: -0.00091 at sigma=0.08.

### rp3
- `mrgap` non-improving positive-noise values: 0.01, 0.02.
- `manifold_fitting` non-improving positive-noise values: 0.01, 0.02.
- Largest absolute oracle-tangent gain: 0.09301 at sigma=0.12.

### torus
- `mrgap` non-improving positive-noise values: 0.01, 0.02.
- `manifold_fitting` non-improving positive-noise values: 0.01, 0.02, 0.04.
- Largest absolute oracle-tangent gain: 0.00411 at sigma=0.12.

A non-improving endpoint means reconstruction GRMSE is no smaller than the noisy input; it is a diagnostic convention, not a universal failure definition.

## Interpretation constraints

- Observation samples are freshly generated for every `(manifold,n,sigma,repeat)`; they are not prefixes or subsets of a 100,000-point cloud.
- Dense reference samples are independent and used only for evaluation; each trial's exact clean observations are added to that reference to remove Monte Carlo distance floor.
- Manifold Fitting is given the true simulation noise `sigma`, as required by its official implementation. MrGap uses fixed paper hyperparameters.
- The official Manifold Fitting bandwidth is undefined at `sigma=0`; the benchmark records an explicit identity convention there.
- Manifold Fitting is one official pass, not an iterative optimizer. MrGap rows report cumulative runtime for every round from 1 through 5.
- For half-torus MrGap, the paper publishes only last-round GP parameters; the benchmark reuses them for every round.

## Plots

- `noise_sensitivity.png`: failure as ambient noise increases.
- `sample_size_sensitivity.png`: error against independently generated sample size.
- `error_vs_local_neighbors.png`: distinguishes global `n` from effective local sample size.
- `bandwidth_sensitivity.png`: method-specific neighborhood scans.
- `oracle_tangent.png`: local-PCA MrGap against true-tangent MrGap.
- `runtime_scaling.png`: cumulative method runtime against `n`.

- `iteration_comparison.png`: MrGap 1--5-round scan at each default condition.

See `benchmark_rows.csv` for the requested `method | manifold | n | sigma | error | runtime` data and all diagnostics.
