#!/usr/bin/env python3
"""Sample-size and MLE-identifiability diagnostic for the circle GP-UQ prototype.

This script builds on ``circle_gp_uq.py`` and adds an explicit stability gate.
If the local GP marginal likelihood does not identify the length scale well, or
if the fitted signal-to-noise ratio A/s2 is too small, the point estimate falls
back to a fixed-bandwidth local-linear intercept.  Both the GP estimate and the
fallback remain linear smoothers, so the leading frequentist variance
sigma^2 ||a||^2 is available in either branch.

The main purpose is to answer two questions:

1. does increasing n improve the stability of the local GP fit?;
2. does a simple instability gate prevent obviously unidentified GP fits from
   degrading reconstruction or pointwise uncertainty calculations?

The default sweep uses n in {600, 1200, 2400, 4800}.  For a heavier run, raise
``--centers`` and ``--seeds``.  Outputs are written to
``results/circle_gp_uq_stability/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circle_gp_uq import (
    ROOT,
    Config,
    estimate_local_frame,
    fit_gp_multistart,
    gp_predict_and_weights,
    sample_circle,
    true_random_chart_value,
)


def local_linear_at_zero(w: np.ndarray, z: np.ndarray, bandwidth: float):
    """Gaussian-weighted local-linear intercept at w=0 and its smoother weights."""
    weights_kernel = np.exp(-(w * w) / (2.0 * bandwidth**2))
    design = np.column_stack((np.ones(len(w)), w))
    gram = design.T @ (weights_kernel[:, None] * design)
    gram += 1e-10 * np.eye(2)
    smoother = np.linalg.solve(gram, design.T * weights_kernel[None, :])[0]
    return float(smoother @ z), smoother


def classify_gp_fit(
    best: dict,
    diagnostics: pd.DataFrame,
    near_opt_nll_tol: float,
    ell_ratio_threshold: float,
    low_signal_ratio: float,
):
    near = diagnostics[
        diagnostics["nll"] <= diagnostics["nll"].min() + near_opt_nll_tol
    ]
    ell_ratio = float(near["ell"].max() / near["ell"].min())
    signal_ratio = float(best["A"] / best["s2"])
    ell_unidentified = ell_ratio > ell_ratio_threshold
    low_signal = signal_ratio < low_signal_ratio
    return {
        "near_optimal_starts": int(len(near)),
        "near_opt_ell_ratio": ell_ratio,
        "A_over_s2": signal_ratio,
        "ell_unidentified": bool(ell_unidentified),
        "low_signal": bool(low_signal),
        "unstable": bool(ell_unidentified or low_signal),
    }


def run_one(
    n: int,
    sigma: float,
    seed: int,
    centers: int,
    random_starts: int,
    chart_radius: float,
    regression_radius: float,
    fallback_bandwidth: float,
    near_opt_nll_tol: float,
    ell_ratio_threshold: float,
    low_signal_ratio: float,
):
    rng = np.random.default_rng(seed)
    theta, _, y = sample_circle(n, sigma, 1.0, rng)
    permutation = rng.permutation(n)
    chart_idx = permutation[: n // 2]
    regression_idx = permutation[n // 2 :]

    order = chart_idx[np.argsort(theta[chart_idx])]
    center_idx = order[np.linspace(0, len(order) - 1, centers, dtype=int)]

    rows = []
    for center_number, k in enumerate(center_idx):
        center = y[k]
        frame = estimate_local_frame(y[chart_idx], center, chart_radius, 10)
        if frame is None:
            continue
        tangent, normal, n_chart_local = frame

        displacement = y[regression_idx] - center
        w_all = displacement @ tangent
        z_all = displacement @ normal
        use = (
            (np.linalg.norm(displacement, axis=1) <= chart_radius)
            & (np.abs(w_all) <= regression_radius)
        )
        w = w_all[use]
        z = z_all[use]
        if len(w) < 20:
            continue

        best, diagnostics = fit_gp_multistart(
            w,
            z,
            regression_radius,
            sigma,
            rng,
            random_starts,
        )
        gp_prediction, gp_weights, gp_posterior_variance, _ = gp_predict_and_weights(
            w,
            z,
            best["A"],
            best["ell"],
            best["s2"],
        )
        status = classify_gp_fit(
            best,
            diagnostics,
            near_opt_nll_tol,
            ell_ratio_threshold,
            low_signal_ratio,
        )

        fallback_prediction, fallback_weights = local_linear_at_zero(
            w, z, fallback_bandwidth
        )
        if status["unstable"]:
            robust_prediction = fallback_prediction
            robust_weights = fallback_weights
            method = "local_linear_fallback"
        else:
            robust_prediction = gp_prediction
            robust_weights = gp_weights
            method = "gp_mle"

        f0_true = true_random_chart_value(center, normal, 1.0)
        gp_error = float(gp_prediction - f0_true)
        fallback_error = float(fallback_prediction - f0_true)
        robust_error = float(robust_prediction - f0_true)

        gp_se = sigma * float(np.linalg.norm(gp_weights))
        fallback_se = sigma * float(np.linalg.norm(fallback_weights))
        robust_se = sigma * float(np.linalg.norm(robust_weights))

        rows.append(
            {
                "n": n,
                "seed": seed,
                "center": center_number,
                "n_chart_local": n_chart_local,
                "n_reg": len(w),
                "F0_true": f0_true,
                "A_hat": best["A"],
                "ell_hat": best["ell"],
                "s2_hat": best["s2"],
                "posterior_variance": gp_posterior_variance,
                **status,
                "selected_method": method,
                "gp_signed_error": gp_error,
                "gp_abs_error": abs(gp_error),
                "fallback_signed_error": fallback_error,
                "fallback_abs_error": abs(fallback_error),
                "robust_signed_error": robust_error,
                "robust_abs_error": abs(robust_error),
                "gp_se": gp_se,
                "fallback_se": fallback_se,
                "robust_se": robust_se,
                "gp_cover_95": abs(gp_error) <= 1.96 * gp_se,
                "fallback_cover_95": abs(fallback_error) <= 1.96 * fallback_se,
                "robust_cover_95": abs(robust_error) <= 1.96 * robust_se,
            }
        )

    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame):
    return (
        raw.groupby("n")
        .agg(
            patches=("center", "size"),
            median_n_reg=("n_reg", "median"),
            unstable_fraction=("unstable", "mean"),
            low_signal_fraction=("low_signal", "mean"),
            ell_unidentified_fraction=("ell_unidentified", "mean"),
            median_ell=("ell_hat", "median"),
            ell_iqr=(
                "ell_hat",
                lambda x: float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
            ),
            median_A_over_s2=("A_over_s2", "median"),
            raw_gp_mae=("gp_abs_error", "median"),
            fallback_mae=("fallback_abs_error", "median"),
            robust_mae=("robust_abs_error", "median"),
            raw_gp_coverage=("gp_cover_95", "mean"),
            fallback_coverage=("fallback_cover_95", "mean"),
            robust_coverage=("robust_cover_95", "mean"),
        )
        .reset_index()
    )


def make_plots(summary: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(summary["n"], summary["unstable_fraction"], marker="o", label="unstable")
    ax.plot(
        summary["n"],
        summary["ell_unidentified_fraction"],
        marker="o",
        label="ell spread > threshold",
    )
    ax.plot(
        summary["n"],
        summary["low_signal_fraction"],
        marker="o",
        label="low A/s2",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Fraction of fitted local patches")
    ax.set_title("Local GP identifiability versus sample size")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "stability_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(summary["n"], summary["raw_gp_mae"], marker="o", label="raw MLE GP")
    ax.plot(
        summary["n"],
        summary["fallback_mae"],
        marker="o",
        label="local-linear baseline",
    )
    ax.plot(
        summary["n"],
        summary["robust_mae"],
        marker="o",
        label="gated estimator",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Median absolute error")
    ax.set_title("Circle reconstruction error versus sample size")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "mae_vs_n.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[600, 1200, 2400, 4800])
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260824, 20260825, 20260826])
    parser.add_argument("--centers", type=int, default=20)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--chart-radius", type=float, default=0.45)
    parser.add_argument("--regression-radius", type=float, default=0.30)
    parser.add_argument("--fallback-bandwidth", type=float, default=0.15)
    parser.add_argument("--near-opt-nll-tol", type=float, default=1e-3)
    parser.add_argument("--ell-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--low-signal-ratio", type=float, default=1e-2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "circle_gp_uq_stability",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames = []
    for n in args.sample_sizes:
        for seed in args.seeds:
            frames.append(
                run_one(
                    n=n,
                    sigma=args.sigma,
                    seed=seed,
                    centers=args.centers,
                    random_starts=args.random_starts,
                    chart_radius=args.chart_radius,
                    regression_radius=args.regression_radius,
                    fallback_bandwidth=args.fallback_bandwidth,
                    near_opt_nll_tol=args.near_opt_nll_tol,
                    ell_ratio_threshold=args.ell_ratio_threshold,
                    low_signal_ratio=args.low_signal_ratio,
                )
            )

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_scan.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"output_dir": str(args.output_dir)}, handle, indent=2)
    make_plots(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
