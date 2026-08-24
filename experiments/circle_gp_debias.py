#!/usr/bin/env python3
"""Leading-bias correction diagnostic for circle GP reconstruction.

This script reuses the circle GP machinery in ``circle_gp_uq.py`` and adds a
local-quadratic plug-in estimate of the leading smoothing/EIV bias.  The key
identity is

    mhat(0) = sum_i a_i Z_i,

so with mu1 = sum_i a_i W_i and mu2 = sum_i a_i W_i^2,

    Bias_smooth ~= m'(0) mu1 + 0.5 m''(0) mu2.

For the isotropic Gaussian oracle EIV expansion on the circle we also include
0.5 sigma^2 m''(0).  A local quadratic fit estimates m'(0) and m''(0).  Because
that fit is itself linear in Z, the debiased estimator remains a linear
smoother, so its frequentist variance is recomputed as sigma^2 ||a_deb||^2.

The oracle circle derivatives are reported only as a diagnostic to check that
the plug-in bias estimate targets the correct leading term.
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

import circle_gp_uq as base

ROOT = Path(__file__).resolve().parents[1]


def local_quadratic_rows(w: np.ndarray, bandwidth: float) -> np.ndarray:
    weights = np.exp(-(w * w) / (2.0 * bandwidth**2))
    design = np.column_stack((np.ones(len(w)), w, w * w))
    gram = design.T @ (weights[:, None] * design)
    ridge = 1e-10 * max(float(np.trace(gram)), 1.0) * np.eye(3)
    return np.linalg.solve(gram + ridge, design.T * weights[None, :])


def true_circle_derivatives(
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    f0: float,
) -> tuple[float, float]:
    alpha = float(center @ tangent)
    beta = float(center @ normal)
    denominator = f0 + beta
    first = -alpha / denominator
    second = -(1.0 + first * first) / denominator
    return first, second


def run(
    n: int,
    sigma: float,
    centers: int,
    chart_radius: float,
    regression_radius: float,
    derivative_bandwidth: float,
    random_starts: int,
    seed: int,
    output_dir: Path,
):
    rng = np.random.default_rng(seed)
    theta, x_true, y = base.sample_circle(n, sigma, 1.0, rng)
    permutation = rng.permutation(n)
    chart_idx = permutation[: n // 2]
    regression_idx = permutation[n // 2 :]
    ordered = chart_idx[np.argsort(theta[chart_idx])]
    center_idx = ordered[np.linspace(0, len(ordered) - 1, centers, dtype=int)]

    rows = []
    for center_number, k in enumerate(center_idx):
        center = y[k]
        frame = base.estimate_local_frame(y[chart_idx], center, chart_radius, 10)
        if frame is None:
            continue
        tangent, normal, _ = frame

        displacement = y[regression_idx] - center
        w_all = displacement @ tangent
        z_all = displacement @ normal
        use = (
            (np.linalg.norm(displacement, axis=1) <= chart_radius)
            & (np.abs(w_all) <= regression_radius)
        )
        w = w_all[use]
        z = z_all[use]
        if len(w) < 25:
            continue

        best, _ = base.fit_gp_multistart(
            w, z, regression_radius, sigma, rng, random_starts
        )
        prediction, a, posterior_variance, _ = base.gp_predict_and_weights(
            w, z, best["A"], best["ell"], best["s2"]
        )
        f0 = base.true_random_chart_value(center, normal, 1.0)

        mu1 = float(a @ w)
        mu2 = float(a @ (w * w))

        # beta = L z and beta[2] multiplies w^2, hence m''(0) = 2 beta[2].
        L = local_quadratic_rows(w, derivative_bandwidth)
        beta = L @ z
        m1_hat = float(beta[1])
        m2_hat = float(2.0 * beta[2])

        # Estimated leading bias:
        #   m1*mu1 + 0.5*m2*(mu2 + sigma^2)
        # = beta1*mu1 + beta2*(mu2 + sigma^2).
        correction_weights = mu1 * L[1] + (mu2 + sigma**2) * L[2]
        a_debiased = a - correction_weights
        debiased_prediction = float(a_debiased @ z)

        f1_true, f2_true = true_circle_derivatives(
            center, tangent, normal, f0
        )
        oracle_bias = f1_true * mu1 + 0.5 * f2_true * (mu2 + sigma**2)
        oracle_debiased = prediction - oracle_bias

        raw_error = prediction - f0
        debiased_error = debiased_prediction - f0
        oracle_error = oracle_debiased - f0
        raw_se = sigma * float(np.linalg.norm(a))
        debiased_se = sigma * float(np.linalg.norm(a_debiased))

        rows.append(
            {
                "center": center_number,
                "theta": float(theta[k]),
                "n_reg": len(w),
                "F0_true": f0,
                "raw_hat": prediction,
                "debiased_hat": debiased_prediction,
                "oracle_debiased_hat": oracle_debiased,
                "raw_error": raw_error,
                "debiased_error": debiased_error,
                "oracle_debiased_error": oracle_error,
                "raw_se": raw_se,
                "debiased_se": debiased_se,
                "raw_cover_95": abs(raw_error) <= 1.96 * raw_se,
                "debiased_cover_95": abs(debiased_error) <= 1.96 * debiased_se,
                "mu1": mu1,
                "mu2": mu2,
                "m1_hat": m1_hat,
                "m2_hat": m2_hat,
                "f1_true": f1_true,
                "f2_true": f2_true,
                "plugin_bias_hat": prediction - debiased_prediction,
                "oracle_leading_bias": oracle_bias,
                "A_hat": best["A"],
                "ell_hat": best["ell"],
                "s2_hat": best["s2"],
                "posterior_variance": posterior_variance,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No local patches were fitted.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "circle_gp_debias_results.csv", index=False)

    summary = {
        "patches": int(len(results)),
        "mean_signed_error_raw": float(results["raw_error"].mean()),
        "mean_signed_error_debiased": float(results["debiased_error"].mean()),
        "mean_signed_error_oracle_leading_debiased": float(
            results["oracle_debiased_error"].mean()
        ),
        "median_abs_error_raw": float(np.median(np.abs(results["raw_error"]))),
        "median_abs_error_debiased": float(
            np.median(np.abs(results["debiased_error"]))
        ),
        "median_abs_error_oracle_leading_debiased": float(
            np.median(np.abs(results["oracle_debiased_error"]))
        ),
        "pointwise_95_coverage_raw": float(results["raw_cover_95"].mean()),
        "pointwise_95_coverage_debiased": float(
            results["debiased_cover_95"].mean()
        ),
        "corr_plugin_vs_oracle_leading_bias": float(
            np.corrcoef(
                results["plugin_bias_hat"], results["oracle_leading_bias"]
            )[0, 1]
        ),
        "median_raw_se": float(results["raw_se"].median()),
        "median_debiased_se": float(results["debiased_se"].median()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    for prefix, error_column, se_column, coverage_column, title in [
        (
            "raw",
            "raw_error",
            "raw_se",
            "raw_cover_95",
            "Raw GP + sandwich pointwise tube",
        ),
        (
            "debiased",
            "debiased_error",
            "debiased_se",
            "debiased_cover_95",
            "Leading-bias-corrected GP + updated sandwich tube",
        ),
    ]:
        ordered_results = results.sort_values("theta")
        angle = ordered_results["theta"].to_numpy()
        error = ordered_results[error_column].to_numpy()
        se = ordered_results[se_column].to_numpy()
        angle_closed = np.r_[angle, angle[0] + 2.0 * np.pi]
        error_closed = np.r_[error, error[0]]
        se_closed = np.r_[se, se[0]]
        radius_hat = 1.0 + error_closed
        lower = radius_hat - 1.96 * se_closed
        upper = radius_hat + 1.96 * se_closed

        dense_angle = np.linspace(0.0, 2.0 * np.pi, 600)
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
        ax.plot(np.cos(dense_angle), np.sin(dense_angle), linewidth=2.0, label="True circle")
        ax.plot(
            radius_hat * np.cos(angle_closed),
            radius_hat * np.sin(angle_closed),
            marker="o",
            markersize=3,
            linewidth=1.3,
            label="Reconstruction",
        )
        ax.fill(
            np.r_[
                upper * np.cos(angle_closed),
                (lower * np.cos(angle_closed))[::-1],
            ],
            np.r_[
                upper * np.sin(angle_closed),
                (lower * np.sin(angle_closed))[::-1],
            ],
            alpha=0.22,
            label="Naive pointwise 95% tube",
        )
        coverage = float(ordered_results[coverage_column].mean())
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nempirical coverage = {coverage:.1%}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_circle_tube.png", dpi=190)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.scatter(results["oracle_leading_bias"], results["plugin_bias_hat"])
    limit = max(
        float(np.max(np.abs(results["oracle_leading_bias"]))),
        float(np.max(np.abs(results["plugin_bias_hat"]))),
    )
    ax.plot([-limit, limit], [-limit, limit], linewidth=1.0)
    ax.set_xlabel("Oracle leading bias on circle")
    ax.set_ylabel("Plug-in estimated bias")
    ax.set_title("Plug-in versus oracle leading GP bias")
    fig.tight_layout()
    fig.savefig(output_dir / "bias_plugin_vs_oracle.png", dpi=190)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--centers", type=int, default=20)
    parser.add_argument("--chart-radius", type=float, default=0.45)
    parser.add_argument("--regression-radius", type=float, default=0.30)
    parser.add_argument("--derivative-bandwidth", type=float, default=0.20)
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "circle_gp_debias",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        n=args.n,
        sigma=args.sigma,
        centers=args.centers,
        chart_radius=args.chart_radius,
        regression_radius=args.regression_radius,
        derivative_bandwidth=args.derivative_bandwidth,
        random_starts=args.random_starts,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
