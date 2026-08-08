# Cassini oval sensitivity experiment

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
