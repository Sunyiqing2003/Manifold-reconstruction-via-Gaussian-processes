#!/usr/bin/env python3
"""Circle prototype for GP reconstruction, MLE stability, and frequentist UQ.

This experiment is deliberately diagnostic rather than a final benchmark.  It
implements a clean sample split between chart construction and regression, uses
an outer ambient ball to isolate a local manifold sheet, and then restricts the
GP training set to an inner tangent-coordinate interval.  Each local GP uses a
squared-exponential covariance with a profiled constant mean and bounded,
log-scale multistart marginal-likelihood optimization for (A, ell, s2).

For every fitted center the script records

* the random-chart reconstruction error at w=0;
* multistart MLE stability for A, ell, and s2;
* the leading frequentist variance Omega = sigma^2 ||a||^2, where a is the
  exact GP/kriging linear-smoother weight vector;
* the usual latent GP posterior variance, retained only as a diagnostic;
* nominal 95% pointwise coverage for both uncertainty scales.

The current implementation still uses the observed outer ambient localization
before the inner tangent restriction.  It therefore should not be interpreted
as closing the localization-selection issue in the theory; the purpose here is
to expose the numerical behavior of the GP/MLE/UQ layer on the simplest
manifold with known truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    n: int = 1200
    sigma: float = 0.03
    radius: float = 1.0
    chart_radius: float = 0.45
    regression_radius: float = 0.30
    centers: int = 30
    random_starts: int = 6
    min_chart_points: int = 10
    min_regression_points: int = 20
    seed: int = 20260824
    near_opt_nll_tol: float = 1e-3
    unidentified_ell_ratio: float = 2.0
    low_signal_ratio: float = 1e-2


def sample_circle(n: int, sigma: float, radius: float, rng: np.random.Generator):
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    y = x + rng.normal(scale=sigma, size=(n, 2))
    return theta, x, y


def estimate_local_frame(
    y_chart: np.ndarray,
    center: np.ndarray,
    chart_radius: float,
    min_points: int,
):
    centered = y_chart - center
    mask = np.linalg.norm(centered, axis=1) <= chart_radius
    local = centered[mask]
    if len(local) < min_points:
        return None

    covariance = (local.T @ local) / len(local)
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, np.argmax(values)]
    normal = np.array([-tangent[1], tangent[0]])

    if np.dot(normal, center) < 0:
        normal = -normal
    tangent = np.array([normal[1], -normal[0]])
    return tangent, normal, int(mask.sum())


def true_random_chart_value(center: np.ndarray, normal: np.ndarray, radius: float):
    """Return F_k(0) by intersecting center + t * normal with the true circle."""
    b = float(np.dot(center, normal))
    c = float(np.dot(center, center) - radius**2)
    discriminant = b * b - c
    if discriminant < 0:
        return np.nan
    roots = (-b + np.sqrt(discriminant), -b - np.sqrt(discriminant))
    return float(min(roots, key=abs))


def gp_profile_nll(log_parameters: np.ndarray, w: np.ndarray, z: np.ndarray):
    amplitude, length_scale, noise_variance = np.exp(log_parameters)
    pairwise = w[:, None] - w[None, :]
    correlation = np.exp(-(pairwise * pairwise) / (2.0 * length_scale**2))
    covariance = amplitude * correlation + noise_variance * np.eye(len(w))

    try:
        factor = cho_factor(
            covariance + 1e-10 * np.eye(len(w)), lower=True, check_finite=False
        )
        ones = np.ones(len(w))
        cinv_ones = cho_solve(factor, ones, check_finite=False)
        cinv_z = cho_solve(factor, z, check_finite=False)
        mean = float((ones @ cinv_z) / (ones @ cinv_ones))
        residual = z - mean
        cinv_residual = cho_solve(factor, residual, check_finite=False)
        logdet = 2.0 * np.log(np.diag(factor[0])).sum()
        return float(
            0.5
            * (
                residual @ cinv_residual
                + logdet
                + len(w) * np.log(2.0 * np.pi)
            )
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 1e100


def make_starts(
    w: np.ndarray,
    z: np.ndarray,
    regression_radius: float,
    sigma: float,
    rng: np.random.Generator,
    random_starts: int,
):
    variance_z = max(float(np.var(z)), 1e-6)
    amplitude_low, amplitude_high = 1e-8, max(1.0, 20.0 * variance_z)
    ell_low = max(regression_radius / 50.0, 1e-3)
    ell_high = max(2.0 * regression_radius, 0.25)
    noise_low = 1e-8
    noise_high = max(0.2, 20.0 * sigma**2, 10.0 * variance_z)

    starts = [
        (variance_z, regression_radius / 2.0, sigma**2, "default"),
        (variance_z, regression_radius / 4.0, sigma**2, "short_l"),
        (variance_z, regression_radius, sigma**2, "long_l"),
        (
            variance_z,
            regression_radius / 2.0,
            max(sigma**2 / 4.0, noise_low),
            "low_nugget",
        ),
        (
            variance_z,
            regression_radius / 2.0,
            min(4.0 * sigma**2, noise_high),
            "high_nugget",
        ),
    ]

    for index in range(random_starts):
        starts.append(
            (
                np.exp(rng.uniform(np.log(amplitude_low), np.log(amplitude_high))),
                np.exp(rng.uniform(np.log(ell_low), np.log(ell_high))),
                np.exp(rng.uniform(np.log(noise_low), np.log(noise_high))),
                f"random_{index + 1}",
            )
        )

    bounds = [
        (np.log(amplitude_low), np.log(amplitude_high)),
        (np.log(ell_low), np.log(ell_high)),
        (np.log(noise_low), np.log(noise_high)),
    ]
    return starts, bounds


def fit_gp_multistart(
    w: np.ndarray,
    z: np.ndarray,
    regression_radius: float,
    sigma: float,
    rng: np.random.Generator,
    random_starts: int,
):
    starts, bounds = make_starts(
        w, z, regression_radius, sigma, rng, random_starts
    )
    rows = []

    for start_id, (a0, ell0, s20, label) in enumerate(starts):
        result = minimize(
            gp_profile_nll,
            np.log([a0, ell0, s20]),
            args=(w, z),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        amplitude, length_scale, noise_variance = np.exp(result.x)
        rows.append(
            {
                "start_id": start_id,
                "start_label": label,
                "success": bool(result.success),
                "nll": float(result.fun),
                "A": float(amplitude),
                "ell": float(length_scale),
                "s2": float(noise_variance),
                "nit": int(result.nit),
            }
        )

    diagnostics = pd.DataFrame(rows)
    best = diagnostics.loc[diagnostics["nll"].idxmin()].to_dict()
    return best, diagnostics


def gp_predict_and_weights(
    w: np.ndarray,
    z: np.ndarray,
    amplitude: float,
    length_scale: float,
    noise_variance: float,
):
    pairwise = w[:, None] - w[None, :]
    correlation = np.exp(-(pairwise * pairwise) / (2.0 * length_scale**2))
    covariance = amplitude * correlation + noise_variance * np.eye(len(w))
    factor = cho_factor(
        covariance + 1e-10 * np.eye(len(w)), lower=True, check_finite=False
    )

    ones = np.ones(len(w))
    cinv_ones = cho_solve(factor, ones, check_finite=False)
    cinv_z = cho_solve(factor, z, check_finite=False)
    mean = float((ones @ cinv_z) / (ones @ cinv_ones))

    kernel_to_zero = amplitude * np.exp(-(w * w) / (2.0 * length_scale**2))
    cinv_kernel = cho_solve(factor, kernel_to_zero, check_finite=False)

    base = cinv_ones / (ones @ cinv_ones)
    weights = base + cinv_kernel - base * (ones @ cinv_kernel)
    prediction = float(weights @ z)

    posterior_base = amplitude - kernel_to_zero @ cinv_kernel
    posterior_mean_correction = (1.0 - ones @ cinv_kernel) ** 2 / (ones @ cinv_ones)
    posterior_variance = float(max(posterior_base + posterior_mean_correction, 0.0))
    return prediction, weights, posterior_variance, mean


def plot_geometry(
    path: Path,
    y: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    radius: float,
    chart_radius: float,
    regression_radius: float,
):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y[:, 0], y[:, 1], s=6, alpha=0.2, label="Noisy observations")
    ax.scatter(center[0], center[1], s=55, label="Query center")

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(radius * np.cos(angle), radius * np.sin(angle), label="True circle")
    ax.add_patch(plt.Circle(center, chart_radius, fill=False, linewidth=1.2))

    segment = np.linspace(-regression_radius, regression_radius, 100)
    line = center[None, :] + segment[:, None] * tangent[None, :]
    ax.plot(line[:, 0], line[:, 1], linewidth=2.0, label="Inner tangent range")
    ax.set_aspect("equal")
    ax.set_title("Circle prototype: outer chart ball and inner regression range")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_multistart(path: Path, diagnostics: pd.DataFrame):
    ordered = diagnostics.sort_values("nll").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(np.arange(len(ordered)), ordered["nll"])
    ax.set_xlabel("Multistart solution, sorted by final NLL")
    ax.set_ylabel("Final negative log marginal likelihood")
    ax.set_title("MLE multistart stability for one local GP")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_intervals(path: Path, results: pd.DataFrame):
    ordered = results.sort_values("theta").reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(
        x,
        ordered["F0_hat"],
        yerr=1.96 * ordered["se_sandwich"],
        fmt="o",
        markersize=4,
        capsize=2,
        label="GP estimate + sandwich 95% radius",
    )
    ax.scatter(x, ordered["F0_true"], marker="x", label="True random-chart F_k(0)")
    ax.set_xlabel("Query centers ordered around the circle")
    ax.set_ylabel("Normal coordinate")
    ax.set_title("Pointwise random-chart reconstruction and sandwich uncertainty")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: Config, output_dir: Path):
    rng = np.random.default_rng(config.seed)
    theta, x_true, y = sample_circle(config.n, config.sigma, config.radius, rng)
    permutation = rng.permutation(config.n)
    chart_idx = permutation[: config.n // 2]
    regression_idx = permutation[config.n // 2 :]

    chart_order = chart_idx[np.argsort(theta[chart_idx])]
    selected_positions = np.linspace(
        0, len(chart_order) - 1, config.centers, dtype=int
    )
    centers = chart_order[selected_positions]

    results_rows = []
    multistart_rows = []
    first_plot_payload = None

    for center_number, k in enumerate(centers):
        center = y[k]
        frame = estimate_local_frame(
            y[chart_idx], center, config.chart_radius, config.min_chart_points
        )
        if frame is None:
            continue
        tangent, normal, n_chart_local = frame

        displacement = y[regression_idx] - center
        w_all = displacement @ tangent
        z_all = displacement @ normal

        outer_mask = np.linalg.norm(displacement, axis=1) <= config.chart_radius
        inner_mask = np.abs(w_all) <= config.regression_radius
        use = outer_mask & inner_mask
        w = w_all[use]
        z = z_all[use]
        if len(w) < config.min_regression_points:
            continue

        best, diagnostics = fit_gp_multistart(
            w,
            z,
            config.regression_radius,
            config.sigma,
            rng,
            config.random_starts,
        )
        prediction, weights, posterior_variance, profiled_mean = gp_predict_and_weights(
            w, z, best["A"], best["ell"], best["s2"]
        )
        f0_true = true_random_chart_value(center, normal, config.radius)

        near = diagnostics[
            diagnostics["nll"] <= diagnostics["nll"].min() + config.near_opt_nll_tol
        ]
        ell_ratio = float(near["ell"].max() / near["ell"].min())
        amplitude_ratio = float(near["A"].max() / near["A"].min())
        noise_ratio = float(near["s2"].max() / near["s2"].min())

        omega = config.sigma**2 * float(weights @ weights)
        error = float(prediction - f0_true)
        signal_ratio = float(best["A"] / best["s2"])

        results_rows.append(
            {
                "center": center_number,
                "theta": float(theta[k]),
                "n_chart_local": n_chart_local,
                "n_reg": len(w),
                "F0_true": f0_true,
                "F0_hat": prediction,
                "signed_error": error,
                "abs_error": abs(error),
                "profiled_mean": profiled_mean,
                "A_hat": best["A"],
                "ell_hat": best["ell"],
                "s2_hat": best["s2"],
                "A_over_s2": signal_ratio,
                "best_nll": best["nll"],
                "omega_true_sigma": omega,
                "se_sandwich": np.sqrt(omega),
                "posterior_variance": posterior_variance,
                "se_posterior": np.sqrt(posterior_variance),
                "sandwich_cover_95": abs(error) <= 1.96 * np.sqrt(omega),
                "posterior_cover_95": abs(error) <= 1.96 * np.sqrt(posterior_variance),
                "near_optimal_starts": len(near),
                "near_opt_ell_ratio": ell_ratio,
                "near_opt_A_ratio": amplitude_ratio,
                "near_opt_s2_ratio": noise_ratio,
                "ell_unidentified_flag": ell_ratio > config.unidentified_ell_ratio,
                "low_signal_flag": signal_ratio < config.low_signal_ratio,
            }
        )

        diagnostics = diagnostics.copy()
        diagnostics["center"] = center_number
        multistart_rows.append(diagnostics)

        if first_plot_payload is None:
            first_plot_payload = (center, tangent, diagnostics.copy())

    if not results_rows:
        raise RuntimeError("No local GP fits were successful; relax the radii or point thresholds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results_rows)
    multistart = pd.concat(multistart_rows, ignore_index=True)
    results.to_csv(output_dir / "circle_gp_uq_results.csv", index=False)
    multistart.to_csv(output_dir / "circle_gp_mle_multistart.csv", index=False)

    summary = {
        "centers_requested": config.centers,
        "centers_fitted": int(len(results)),
        "median_local_regression_n": float(results["n_reg"].median()),
        "median_absolute_error": float(results["abs_error"].median()),
        "mean_signed_error": float(results["signed_error"].mean()),
        "sandwich_95_coverage_true_sigma": float(results["sandwich_cover_95"].mean()),
        "gp_posterior_95_coverage_diagnostic": float(results["posterior_cover_95"].mean()),
        "median_ell_hat": float(results["ell_hat"].median()),
        "median_s2_hat": float(results["s2_hat"].median()),
        "median_A_over_s2": float(results["A_over_s2"].median()),
        "fraction_ell_unidentified": float(results["ell_unidentified_flag"].mean()),
        "fraction_low_signal": float(results["low_signal_flag"].mean()),
        "median_posterior_se_over_sandwich_se": float(
            np.median(results["se_posterior"] / results["se_sandwich"])
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    metadata = {
        "config": asdict(config),
        "notes": {
            "selection": "outer ambient chart ball AND inner tangent-coordinate interval",
            "sandwich_variance": "Omega = sigma^2 ||a||^2 uses the known simulation sigma",
            "posterior_variance": "reported only as a GP-model diagnostic, not a frequentist confidence variance",
            "mle": "bounded log-scale multistart optimization of (A, ell, s2) with profiled constant mean",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    center, tangent, first_diagnostics = first_plot_payload
    plot_geometry(
        output_dir / "local_geometry.png",
        y,
        center,
        tangent,
        config.radius,
        config.chart_radius,
        config.regression_radius,
    )
    plot_multistart(output_dir / "mle_multistart_nll.png", first_diagnostics)
    plot_intervals(output_dir / "sandwich_intervals.png", results)

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=Config.n)
    parser.add_argument("--sigma", type=float, default=Config.sigma)
    parser.add_argument("--chart-radius", type=float, default=Config.chart_radius)
    parser.add_argument(
        "--regression-radius", type=float, default=Config.regression_radius
    )
    parser.add_argument("--centers", type=int, default=Config.centers)
    parser.add_argument("--random-starts", type=int, default=Config.random_starts)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "circle_gp_uq",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        n=args.n,
        sigma=args.sigma,
        chart_radius=args.chart_radius,
        regression_radius=args.regression_radius,
        centers=args.centers,
        random_starts=args.random_starts,
        seed=args.seed,
    )
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
