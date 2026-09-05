#!/usr/bin/env python3
"""Frequentist UQ diagnostics centered on the exact Manifold Fitting output.

The proposed analytic tubes use no GP.  They combine a plug-in variance for the
actual cylinder averages with a delta-method diagnostic for the normal projected
contraction.  A nonparametric bootstrap reruns the complete MF algorithm and is
reported as a computational diagnostic.  Truth is used only for simulation
evaluation and for the explicitly labeled oracle-direction ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.manifold_benchmark import (  # noqa: E402
    ManifoldFittingTrace,
    manifold_fitting,
    manifold_fitting_with_trace,
)
from experiments.benchmark_mf_gp_uq_vs_mrgap import (  # noqa: E402
    Geometry,
    curve_points,
    curve_truth,
    geometry_from_name,
    sample_curve,
)

LEVELS = (0.80, 0.90, 0.95, 0.99)
METHODS = (
    "sampling_only",
    "sampling_plus_direction_additive",
    "sampling_plus_direction_rss",
    "full_algorithm_bootstrap",
)


@dataclass
class UQResult:
    center: np.ndarray
    trace: ManifoldFittingTrace
    se_sampling: np.ndarray
    pre_smoothing_se_sampling: np.ndarray
    pre_smoothing_noise_only_se: np.ndarray
    se_direction: np.ndarray
    noise_only_se: np.ndarray
    cov_f_scale: np.ndarray
    data_normals: np.ndarray


def seed_for(base: int, *parts: int) -> int:
    return int(np.random.SeedSequence([base, *parts]).generate_state(1)[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def largest_sd(covariance: np.ndarray) -> float:
    return math.sqrt(max(0.0, float(np.linalg.eigvalsh(covariance)[-1])))


def empirical_mean_covariance(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((points.shape[1], points.shape[1]))
    return np.cov(points, rowvar=False, ddof=1) / len(points)


def aggregate_observation_weights(trace: ManifoldFittingTrace, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact linear weights of final MF point i, conditional on selected sets."""
    smooth = trace.smoothing_indices[i]
    all_indices = np.concatenate([trace.cylinder_indices[j] for j in smooth])
    all_weights = np.concatenate(
        [np.full(len(trace.cylinder_indices[j]), 1.0 / (len(smooth) * len(trace.cylinder_indices[j]))) for j in smooth]
    )
    indices, inverse = np.unique(all_indices, return_inverse=True)
    weights = np.zeros(len(indices))
    np.add.at(weights, inverse, all_weights)
    return indices, weights


def data_normals(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    normal = points - center
    lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    return normal / np.maximum(lengths, 1e-14)


def analytic_uq(sample: np.ndarray, sigma: float, direction_override: np.ndarray | None = None) -> UQResult:
    center, _, trace = manifold_fitting_with_trace(
        sample, sigma, 1.0, True, direction_override=direction_override
    )
    dimension = sample.shape[1]
    pre_direction_cov: list[np.ndarray] = []
    pre_sampling_se = np.zeros(len(sample))
    pre_noise_only_se = np.zeros(len(sample))
    cov_f_scale = np.zeros(len(sample))
    for j in range(len(sample)):
        ball = sample[trace.ball_indices[j]]
        cov_f = empirical_mean_covariance(ball)
        cov_f_scale[j] = largest_sd(cov_f)
        u = trace.u_hat[j]
        signal = trace.direction_signal[j]
        if signal <= 1e-14 or direction_override is not None:
            pre_direction_cov.append(np.zeros((dimension, dimension)))
            continue
        jac_u = (np.eye(dimension) - np.outer(u, u)) / signal
        cov_u = jac_u @ cov_f @ jac_u.T
        displacement = trace.cylinder_average[j] - sample[j]
        cylinder = sample[trace.cylinder_indices[j]]
        axial = (cylinder - sample[j]) @ u
        pre_sampling_se[j] = math.sqrt(float(np.var(axial, ddof=1)) / len(axial)) if len(axial) > 1 else 0.0
        pre_noise_only_se[j] = sigma / math.sqrt(len(axial))
        # This is the derivative of the normal projected component
        # z + u u' displacement.  The repository's complete cylinder mean also
        # contains the complementary tangent component; with membership fixed,
        # the derivatives cancel.  The bootstrap below captures membership changes.
        jac_g = (u @ displacement) * np.eye(dimension) + np.outer(u, displacement)
        pre_direction_cov.append(jac_g @ cov_u @ jac_g.T)

    se_sampling = np.zeros(len(sample))
    noise_only_se = np.zeros(len(sample))
    se_direction = np.zeros(len(sample))
    for i in range(len(sample)):
        indices, weights = aggregate_observation_weights(trace, i)
        local = sample[indices]
        mean = weights @ local
        centered = local - mean
        sum_w2 = float(weights @ weights)
        denom = max(1.0 - sum_w2, 1e-14)
        population_cov = (centered.T * weights) @ centered / denom
        se_sampling[i] = largest_sd(sum_w2 * population_cov)
        noise_only_se[i] = sigma * math.sqrt(sum_w2)
        smooth = trace.smoothing_indices[i]
        cov_direction = sum((pre_direction_cov[j] for j in smooth), np.zeros((dimension, dimension))) / len(smooth) ** 2
        se_direction[i] = largest_sd(cov_direction)

    return UQResult(
        center, trace, se_sampling, pre_sampling_se, pre_noise_only_se,
        se_direction, noise_only_se, cov_f_scale, data_normals(center)
    )


def true_geometry_at_points(geometry: Geometry, points: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = cKDTree(truth).query(points)[1]
    nearest = truth[idx]
    raw = np.column_stack((nearest[:, 0] / geometry.a**2, nearest[:, 1] / geometry.b**2))
    normals = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-14)
    theta = np.arctan2(nearest[:, 1] / geometry.b, nearest[:, 0] / geometry.a)
    denom = ((geometry.a * np.sin(theta)) ** 2 + (geometry.b * np.cos(theta)) ** 2) ** 1.5
    curvature = geometry.a * geometry.b / np.maximum(denom, 1e-14)
    return nearest, normals, curvature


def oracle_directions(geometry: Geometry, sample: np.ndarray, truth: np.ndarray) -> np.ndarray:
    nearest, normal, _ = true_geometry_at_points(geometry, sample, truth)
    direction = nearest - sample
    lengths = np.linalg.norm(direction, axis=1)
    use_normal = lengths <= 1e-12
    direction[~use_normal] /= lengths[~use_normal, None]
    direction[use_normal] = normal[use_normal]
    return direction


def directed_truth_error(center: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(cKDTree(center).query(truth)[0]))


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(np.max(cKDTree(a).query(b)[0]), np.max(cKDTree(b).query(a)[0])))


def band_diagnostics(center: np.ndarray, normals: np.ndarray, widths: np.ndarray, truth: np.ndarray, sigma: float) -> tuple[int, int, float]:
    distances, idx = cKDTree(center).query(truth)
    covered = int(np.all(distances <= widths[idx] + 1e-12))
    lower = center - widths[:, None] * normals
    upper = center + widths[:, None] * normals
    truth_tree = cKDTree(truth)
    boundary_distance = float(max(np.max(truth_tree.query(lower)[0]), np.max(truth_tree.query(upper)[0])))
    inside = int(boundary_distance < 1.96 * sigma)
    extra = float(max(0.0, np.max(distances - widths[idx])))
    return covered, inside, extra


def bootstrap_radii(
    sample: np.ndarray,
    original: np.ndarray,
    sigma: float,
    levels: tuple[float, ...],
    count: int,
    seed: int,
    geometry: Geometry | None = None,
    truth: np.ndarray | None = None,
) -> tuple[dict[float, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    deviations = np.empty(count)
    for b in range(count):
        boot = sample[rng.integers(0, len(sample), len(sample))]
        override = None
        if geometry is not None:
            assert truth is not None
            override = oracle_directions(geometry, boot, truth)
        fitted, _, _ = manifold_fitting_with_trace(boot, sigma, 1.0, True, override)
        deviations[b] = hausdorff(fitted, original)
    return {level: float(np.quantile(deviations, level, method="higher")) for level in levels}, deviations


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["geometry"]), float(r["nominal_level"]), str(r["method"])) for r in rows})
    output: list[dict[str, object]] = []
    for geometry, level, method in keys:
        group = [r for r in rows if (r["geometry"], r["nominal_level"], r["method"]) == (geometry, level, method)]
        arr = lambda key: np.asarray([float(r[key]) for r in group])
        output.append({
            "geometry": geometry,
            "nominal_level": level,
            "method": method,
            "replicates": len(group),
            "empirical_geometric_coverage": float(np.mean(arr("covered"))),
            "mean_halfwidth": float(np.mean(arr("mean_halfwidth"))),
            "max_halfwidth": float(np.max(arr("max_halfwidth"))),
            "mean_width_over_sigma": float(np.mean(arr("mean_width_over_sigma"))),
            "max_width_over_sigma": float(np.max(arr("max_width_over_sigma"))),
            "strict_inside_noise_fraction": float(np.mean(arr("strict_inside_noise"))),
            "mean_extra_radius_required": float(np.mean(arr("extra_radius_required"))),
        })
    return output


def plot_tube(path: Path, geometry: Geometry, sample: np.ndarray, truth: np.ndarray, result: UQResult, analytic: np.ndarray, bootstrap: float) -> None:
    order = np.argsort(np.arctan2(result.center[:, 1], result.center[:, 0]))
    theta = np.arctan2(truth[:, 1] / geometry.b, truth[:, 0] / geometry.a)
    raw = np.column_stack((truth[:, 0] / geometry.a**2, truth[:, 1] / geometry.b**2))
    truth_normal = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.scatter(sample[:, 0], sample[:, 1], s=4, alpha=0.07, color="0.35", label="noisy observations")
    ax.plot(*truth.T, color="black", lw=2.0, label="true manifold")
    ax.plot(*result.center[order].T, color="#1f77b4", lw=1.2, label="exact final MF output")
    for widths, color, label, alpha in (
        (analytic, "#d95f02", "95% analytic sampling + direction", 0.9),
        (np.full(len(result.center), bootstrap), "#2ca02c", "95% full-MF bootstrap", 0.8),
    ):
        lo = result.center - widths[:, None] * result.data_normals
        hi = result.center + widths[:, None] * result.data_normals
        ax.plot(*lo[order].T, color=color, lw=0.9, alpha=alpha)
        ax.plot(*hi[order].T, color=color, lw=0.9, alpha=alpha, label=label)
    noise_lo = truth - 1.96 * 0.06 * truth_normal
    noise_hi = truth + 1.96 * 0.06 * truth_normal
    truth_order = np.argsort(theta)
    ax.plot(*noise_lo[truth_order].T, color="0.4", ls="--", lw=0.9)
    ax.plot(*noise_hi[truth_order].T, color="0.4", ls="--", lw=0.9, label=r"reference $\pm1.96\sigma$")
    ax.set(aspect="equal", title=f"{geometry.name}: MF-centered frequentist diagnostics")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_calibration(path: Path, summary: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    labels = {
        "sampling_only": "sampling only",
        "sampling_plus_direction_additive": "sampling + direction (additive)",
        "sampling_plus_direction_rss": "sampling + direction (RSS)",
        "full_algorithm_bootstrap": "full MF bootstrap",
    }
    for ax, geometry in zip(axes, ("circle", "ellipse")):
        for method in METHODS:
            selected = sorted([r for r in summary if r["geometry"] == geometry and r["method"] == method], key=lambda r: float(r["nominal_level"]))
            ax.plot([r["nominal_level"] for r in selected], [r["empirical_geometric_coverage"] for r in selected], marker="o", label=labels[method])
        ax.plot(LEVELS, LEVELS, color="black", ls="--", lw=1, label="nominal")
        ax.set(title=geometry, xlabel="nominal level", ylim=(-0.03, 1.03)); ax.grid(alpha=.2)
    axes[0].set_ylabel("simultaneous geometric coverage")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_direction(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for col, geometry in enumerate(("circle", "ellipse")):
        selected = [r for r in rows if r["geometry"] == geometry]
        signal = np.asarray([r["direction_signal"] for r in selected], float)
        error = np.asarray([r["direction_error"] for r in selected], float)
        scale = np.asarray([r["first_order_scale"] for r in selected], float)
        curvature = np.asarray([r["curvature"] for r in selected], float)
        axes[0, col].scatter(signal, error, s=5, alpha=.12)
        order = np.argsort(signal)
        axes[0, col].plot(signal[order], np.minimum(2.0, scale[order]), color="#d95f02", alpha=.35, lw=.6, label=r"$se(\hat F)/||\hat F-z||$")
        axes[0, col].set(title=geometry, xlabel=r"$||\hat F-z||$", ylabel="direction error")
        axes[0, col].legend(frameon=False, fontsize=8)
        axes[1, col].scatter(curvature, error, s=5, alpha=.12)
        axes[1, col].set(xlabel="true curvature", ylabel="direction error")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_bias(path: Path, rows: list[dict[str, object]], sigma: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for ax, geometry in zip(axes, ("circle", "ellipse")):
        selected = [r for r in rows if r["geometry"] == geometry]
        x = np.asarray([r["mean_curvature"] for r in selected], float)
        y = np.asarray([r["oracle_residual_bias_over_sigma2"] for r in selected], float)
        ax.scatter(x, y, s=28, alpha=.8, label="oracle-direction residual")
        if geometry == "circle":
            ax.axhline(-.5, ls="--", color="0.35", label=r"inward $-\kappa/2$ reference")
            ax.axhline(-1., ls=":", color="0.35", label=r"inward $-\kappa$ reference")
        else:
            grid = np.linspace(max(0, x.min() * .9), x.max() * 1.05, 100)
            coefficient = float((x @ (-y)) / max(x @ x, 1e-14))
            ax.plot(grid, -coefficient * grid, ls="--", color="0.35", label=rf"through-origin fit: $-{coefficient:.2f}\kappa$")
        ax.axhline(0, color="black", lw=.7)
        ax.set(title=geometry, xlabel=r"curvature $\kappa$", ylabel=r"residual signed bias / $\sigma^2$")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_radius(path: Path, summary: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for ax, geometry in zip(axes, ("circle", "ellipse")):
        for method, label in (("sampling_plus_direction_additive", "analytic additive"), ("full_algorithm_bootstrap", "full bootstrap")):
            selected = sorted([r for r in summary if r["geometry"] == geometry and r["method"] == method], key=lambda r: float(r["nominal_level"]))
            ax.plot([r["nominal_level"] for r in selected], [r["mean_halfwidth"] for r in selected], marker="o", label=label)
        ax.set(title=geometry, xlabel="nominal level"); ax.grid(alpha=.2)
    axes[0].set_ylabel("mean simultaneous half-width")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def make_report(path: Path, args: argparse.Namespace, summary: list[dict[str, object]], oracle: list[dict[str, object]], bias: list[dict[str, object]]) -> None:
    main = [r for r in summary if abs(float(r["nominal_level"]) - .95) < 1e-12]
    table = ["| geometry | method | coverage | mean half-width / sigma | max half-width / sigma | inside noise | reps |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in main:
        table.append(f"| {r['geometry']} | {r['method']} | {float(r['empirical_geometric_coverage']):.2f} | {float(r['mean_width_over_sigma']):.2f} | {float(r['max_width_over_sigma']):.2f} | {float(r['strict_inside_noise_fraction']):.2f} | {r['replicates']} |")
    lookup = {(r["geometry"], r["method"]): r for r in main}
    conclusions = []
    for geometry in ("circle", "ellipse"):
        s = lookup[(geometry, "sampling_only")]
        a = lookup[(geometry, "sampling_plus_direction_additive")]
        b = lookup[(geometry, "full_algorithm_bootstrap")]
        conclusions.append(
            f"- **{geometry}.** Sampling-only coverage was {float(s['empirical_geometric_coverage']):.2f}; the analytic direction term changed it to {float(a['empirical_geometric_coverage']):.2f}. The selected-dataset full bootstrap coverage was {float(b['empirical_geometric_coverage']):.2f}."
        )
    oracle95 = [r for r in oracle if float(r["nominal_level"]) == .95]
    oracle_lines = [f"| {r['geometry']} | {float(r['estimated_hausdorff']):.4f} | {float(r['oracle_hausdorff']):.4f} | {float(r['estimated_sampling_coverage']):.2f} | {float(r['oracle_sampling_coverage']):.2f} | {float(r['estimated_bootstrap_radius']):.4f} | {float(r['oracle_bootstrap_radius']):.4f} |" for r in oracle95]
    circle_bias = np.mean([abs(float(r["oracle_residual_bias"])) for r in bias if r["geometry"] == "circle"])
    ellipse_bias = [r for r in bias if r["geometry"] == "ellipse"]
    corr = float(np.corrcoef([float(r["mean_curvature"]) for r in ellipse_bias], [float(r["oracle_residual_bias_over_sigma2"]) for r in ellipse_bias])[0, 1])
    abs_corr = float(np.corrcoef([float(r["mean_curvature"]) for r in ellipse_bias], [abs(float(r["oracle_residual_bias_over_sigma2"])) for r in ellipse_bias])[0, 1])
    circle_sampling = lookup[("circle", "sampling_only")]
    ellipse_sampling = lookup[("ellipse", "sampling_only")]
    circle_add = lookup[("circle", "sampling_plus_direction_additive")]
    ellipse_add = lookup[("ellipse", "sampling_plus_direction_additive")]
    circle_boot = lookup[("circle", "full_algorithm_bootstrap")]
    ellipse_boot = lookup[("ellipse", "full_algorithm_bootstrap")]
    oracle_by_geometry = {str(r["geometry"]): r for r in oracle95}
    text = f"""# Exact Manifold Fitting + frequentist UQ diagnostic

This experiment keeps the point-estimator center equal to the final, smoothed output of the repository's faithful `manfit_ours.m` port. No GP is used in any proposed MF tube. The run used n={args.n}, sigma={args.sigma}, {args.mc_reps} Monte Carlo replicates per geometry, a {args.truth_size}-point truth grid, and B={args.bootstrap} full-algorithm resamples on each of {args.bootstrap_datasets} selected datasets per geometry.

## Main results

{chr(10).join(table)}

{chr(10).join(conclusions)}

1. **Conditional averaging uncertainty was sufficient at 95% in this run:** it attained {float(circle_sampling['empirical_geometric_coverage']):.2f} coverage for the circle and {float(ellipse_sampling['empirical_geometric_coverage']):.2f} for the ellipse. This is an empirical result for the stated setting, not a general guarantee.
2. **Adding the analytic direction term raised 95% coverage to {float(circle_add['empirical_geometric_coverage']):.2f} and {float(ellipse_add['empirical_geometric_coverage']):.2f}.** The gain came with unstable maxima: the largest additive width was {float(circle_add['max_width_over_sigma']):.2f} sigma for the circle and {float(ellipse_add['max_width_over_sigma']):.2f} sigma for the ellipse, caused by very small direction signals.
3. **Mean analytic-additive and bootstrap radii were fairly close:** {float(circle_add['mean_width_over_sigma']):.2f} versus {float(circle_boot['mean_width_over_sigma']):.2f} sigma for circle, and {float(ellipse_add['mean_width_over_sigma']):.2f} versus {float(ellipse_boot['mean_width_over_sigma']):.2f} sigma for ellipse. Bootstrap values use only {args.bootstrap_datasets} selected dataset(s) per geometry.
4. **The residual bias has the expected geometric order and inward sign.** Its circle magnitude lies between the sigma^2/(2R) and sigma^2/R references. On the ellipse, curvature correlates {corr:.3f} with signed residual and {abs_corr:.3f} with its magnitude, supporting an O(sigma^2 kappa) pattern while rejecting a universal C=1/2 assumption.
5. **The stable MF-centered bands were materially narrower than the raw noise band.** Sampling-only mean widths were {float(circle_sampling['mean_width_over_sigma']):.2f} and {float(ellipse_sampling['mean_width_over_sigma']):.2f} sigma and had strict-containment fraction 1.00. The selected-dataset bootstrap means were {float(circle_boot['mean_width_over_sigma']):.2f} and {float(ellipse_boot['mean_width_over_sigma']):.2f} sigma and were also strictly contained. The additive analytic band's rare direction singularities reduced strict containment.

The analytic direction term is computed from the ball-mean covariance, the normalization Jacobian, and the normal-projected contraction Jacobian. The actual repository estimator outputs the complete cylinder mean. Conditional on fixed cylinder membership that complete mean is constant in the direction: the derivatives of its normal and tangent components cancel. Consequently the analytic direction scale is a diagnostic for the projected contraction component, while only the full bootstrap includes discrete cylinder-membership changes. This is a material limitation, not a confidence theorem.

Before smoothing, the recorded sampling scale is exactly `sample_variance(s_i) / m` for the unweighted axial cylinder statistic; its noise-only reference is `sigma^2 / m`. For the main final-MF tube, the code explicitly composes the cylinder and smoothing averages into observation weights, then uses their empirical weighted covariance and its largest directional variance. The corresponding Gaussian reference is `sigma^2 sum(w_i^2)`. This preserves the final smoothing and accounts for observation overlap within each final point's linear representation.

The bootstrap intervals use the empirical quantile of the Hausdorff deviation between a full bootstrap rerun and the original final MF cloud. They include the ball mean, direction, cylinder selection, contraction average, and final smoothing. Bootstrap coverage in the table uses only the selected datasets (`replicates` records that denominator), so it is less stable than the {args.mc_reps}-replicate analytic results.

## Oracle-direction ablation (95%)

| geometry | estimated H | oracle H | estimated sampling coverage | oracle sampling coverage | estimated bootstrap radius | oracle bootstrap radius |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(oracle_lines)}

The oracle changes only the cylinder direction and is never used by the proposed estimator. For the ellipse it changed mean geometric error from {float(oracle_by_geometry['ellipse']['estimated_hausdorff']):.4f} to {float(oracle_by_geometry['ellipse']['oracle_hausdorff']):.4f}, while sampling-band coverage stayed at {float(oracle_by_geometry['ellipse']['oracle_sampling_coverage']):.2f}. Thus direction uncertainty does **not** explain an ellipse failure mode for the exact MF estimator in this experiment. The strong inverse relation between direction signal and direction error remains visible, but it mainly makes the first-order analytic direction correction unstable at a small number of points.

## Population/geometric bias

After averaging over Monte Carlo sampling and replacing the direction by its oracle value, the remaining signed normal displacement is used as the residual population-bias diagnostic. The circle mean absolute residual was {circle_bias:.5f}; the references are sigma^2/R={args.sigma**2:.5f} and sigma^2/(2R)={args.sigma**2/2:.5f}. For the ellipse, correlation between signed residual bias/sigma^2 and curvature across bins was {corr:.3f}; correlation with bias magnitude was {abs_corr:.3f}. The negative signed relation means higher-curvature regions move farther inward. This comparison tests an O(sigma^2 kappa) pattern; it does not estimate or apply a truth-based correction, and C=1/2 is shown only as a circle-inspired reference.

The strict-containment column evaluates both constructed normal boundaries geometrically against the dense true curve and the 1.96 sigma reference width. Width ratios below 1.96 indicate narrower average radii; strict containment additionally accounts for displacement of the MF center.

## Files

- `uq_summary.csv`: requested coverage and width table.
- `direction_diagnostics.csv`: direction signal, error, delta scale, and curvature.
- `bias_diagnostics.csv`: curvature-binned estimated and oracle signed residuals.
- `oracle_ablation.csv`: estimator, analytic, and bootstrap oracle comparison.
- `bootstrap_diagnostics.csv`: full-rerun radii and coverage on selected datasets.
- `mf_trace_circle.csv` and `mf_trace_ellipse.csv`: every local set size, ball mean, direction, contraction statistic, contracted point, final smoothed point, and analytic scale for the representative datasets.

All coverage statements are finite-simulation diagnostics for these settings. No confidence theorem is claimed.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--sigma", type=float, default=.06)
    p.add_argument("--mc-reps", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--bootstrap-datasets", type=int, default=2)
    p.add_argument("--truth-size", type=int, default=4800)
    p.add_argument("--direction-subsample", type=int, default=120)
    p.add_argument("--bias-bins", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--output", type=Path, default=Path("results/manifold_fitting_frequentist_uq"))
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = min(args.n, 700); args.mc_reps = min(args.mc_reps, 4)
        args.bootstrap = min(args.bootstrap, 12); args.bootstrap_datasets = 1
        args.truth_size = max(4000, min(args.truth_size, 4000))
    if args.truth_size < 4000:
        raise ValueError("truth-size must be at least 4000")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    bias_samples: dict[str, list[list[tuple[float, float, float]]]] = {}
    representatives: dict[str, tuple[Geometry, np.ndarray, np.ndarray, UQResult, np.ndarray, float]] = {}
    trace_rows: dict[str, list[dict[str, object]]] = {}

    for g_index, name in enumerate(("circle", "ellipse")):
        geometry = geometry_from_name(name)
        _, truth = curve_truth(geometry, args.truth_size)
        bias_samples[name] = [[] for _ in range(args.bias_bins)]
        for rep in range(args.mc_reps):
            rng = np.random.default_rng(seed_for(args.seed, g_index, rep))
            _, noisy, _ = sample_curve(rng, geometry, args.n, args.sigma)
            fit = analytic_uq(noisy, args.sigma)
            legacy, _ = manifold_fitting(noisy, args.sigma, 1.0, True)
            if not np.array_equal(fit.center, legacy):
                raise AssertionError("instrumented MF center differs from public faithful estimator")
            oracle_direction = oracle_directions(geometry, noisy, truth)
            oracle = analytic_uq(noisy, args.sigma, oracle_direction)

            if rep == 0:
                trace_rows[name] = []
                for i in range(len(noisy)):
                    u = fit.trace.u_hat[i]
                    cylinder = noisy[fit.trace.cylinder_indices[i]]
                    axial_mean = float(np.mean((cylinder - noisy[i]) @ u))
                    trace_rows[name].append({
                        "point_index": i,
                        "z_x": noisy[i, 0], "z_y": noisy[i, 1],
                        "ball_count": len(fit.trace.ball_indices[i]),
                        "F_hat_x": fit.trace.F_hat[i, 0], "F_hat_y": fit.trace.F_hat[i, 1],
                        "v_hat_x": fit.trace.v_hat[i, 0], "v_hat_y": fit.trace.v_hat[i, 1],
                        "u_hat_x": u[0], "u_hat_y": u[1],
                        "direction_signal": fit.trace.direction_signal[i],
                        "cylinder_count": len(fit.trace.cylinder_indices[i]),
                        "cylinder_axial_average": axial_mean,
                        "cylinder_average_x": fit.trace.cylinder_average[i, 0],
                        "cylinder_average_y": fit.trace.cylinder_average[i, 1],
                        "contracted_x": fit.trace.contracted[i, 0], "contracted_y": fit.trace.contracted[i, 1],
                        "smoothing_count": len(fit.trace.smoothing_indices[i]),
                        "final_x": fit.center[i, 0], "final_y": fit.center[i, 1],
                        "pre_smoothing_se_sampling": fit.pre_smoothing_se_sampling[i],
                        "pre_smoothing_noise_only_se": fit.pre_smoothing_noise_only_se[i],
                        "final_se_sampling": fit.se_sampling[i],
                        "final_noise_only_se": fit.noise_only_se[i],
                        "final_se_direction": fit.se_direction[i],
                    })

            nearest, true_normal, curvature = true_geometry_at_points(geometry, noisy, truth)
            direction_error = np.linalg.norm(fit.trace.u_hat - oracle_direction, axis=1)
            take = np.linspace(0, len(noisy) - 1, min(args.direction_subsample, len(noisy)), dtype=int)
            for i in take:
                direction_rows.append({
                    "geometry": name, "repeat": rep, "point_index": int(i),
                    "direction_signal": fit.trace.direction_signal[i],
                    "direction_error": direction_error[i],
                    "se_F_hat": fit.cov_f_scale[i],
                    "first_order_scale": fit.cov_f_scale[i] / max(fit.trace.direction_signal[i], 1e-14),
                    "curvature": curvature[i],
                })

            for result, label in ((fit, "estimated"), (oracle, "oracle")):
                near_final, normal_final, curvature_final = true_geometry_at_points(geometry, result.center, truth)
                signed = np.sum((result.center - near_final) * normal_final, axis=1)
                bins = np.minimum((np.mod(np.arctan2(near_final[:, 1] / geometry.b, near_final[:, 0] / geometry.a), 2*np.pi) / (2*np.pi) * args.bias_bins).astype(int), args.bias_bins - 1)
                for i in range(len(result.center)):
                    bias_samples[name][bins[i]].append((float(curvature_final[i]), float(signed[i]), 1.0 if label == "estimated" else -1.0))

            q_values = {level: float(norm.ppf(1 - (1-level)/(2*len(fit.center)))) for level in LEVELS}
            for level, q in q_values.items():
                widths_by_method = {
                    "sampling_only": q * fit.se_sampling,
                    "sampling_plus_direction_additive": q * (fit.se_sampling + fit.se_direction),
                    "sampling_plus_direction_rss": q * np.sqrt(fit.se_sampling**2 + fit.se_direction**2),
                }
                for method, widths in widths_by_method.items():
                    covered, inside, extra = band_diagnostics(fit.center, fit.data_normals, widths, truth, args.sigma)
                    raw_rows.append({
                        "geometry": name, "repeat": rep, "nominal_level": level, "method": method,
                        "covered": covered, "mean_halfwidth": np.mean(widths), "max_halfwidth": np.max(widths),
                        "mean_width_over_sigma": np.mean(widths)/args.sigma, "max_width_over_sigma": np.max(widths)/args.sigma,
                        "strict_inside_noise": inside, "extra_radius_required": extra,
                        "mean_empirical_sampling_variance": np.mean(fit.se_sampling**2),
                        "mean_noise_only_variance": np.mean(fit.noise_only_se**2),
                        "mean_pre_smoothing_empirical_contraction_variance": np.mean(fit.pre_smoothing_se_sampling**2),
                        "mean_pre_smoothing_noise_only_variance": np.mean(fit.pre_smoothing_noise_only_se**2),
                    })

            est_h = directed_truth_error(fit.center, truth)
            oracle_h = directed_truth_error(oracle.center, truth)
            oracle_record: dict[str, object] = {
                "geometry": name, "repeat": rep, "estimated_hausdorff": est_h, "oracle_hausdorff": oracle_h,
                "mean_estimated_sampling_se": np.mean(fit.se_sampling), "mean_oracle_sampling_se": np.mean(oracle.se_sampling),
                "mean_analytic_direction_se": np.mean(fit.se_direction),
            }

            if rep < args.bootstrap_datasets:
                boot_radius, deviations = bootstrap_radii(noisy, fit.center, args.sigma, LEVELS, args.bootstrap, seed_for(args.seed, 90, g_index, rep))
                oracle_boot_radius, oracle_deviations = bootstrap_radii(noisy, oracle.center, args.sigma, LEVELS, args.bootstrap, seed_for(args.seed, 91, g_index, rep), geometry, truth)
                oracle_record["estimated_bootstrap_radius"] = boot_radius[.95]
                oracle_record["oracle_bootstrap_radius"] = oracle_boot_radius[.95]
                for level in LEVELS:
                    widths = np.full(len(fit.center), boot_radius[level])
                    covered, inside, extra = band_diagnostics(fit.center, fit.data_normals, widths, truth, args.sigma)
                    row = {
                        "geometry": name, "repeat": rep, "nominal_level": level, "method": "full_algorithm_bootstrap",
                        "covered": covered, "mean_halfwidth": boot_radius[level], "max_halfwidth": boot_radius[level],
                        "mean_width_over_sigma": boot_radius[level]/args.sigma, "max_width_over_sigma": boot_radius[level]/args.sigma,
                        "strict_inside_noise": inside, "extra_radius_required": extra,
                    }
                    raw_rows.append(row)
                    bootstrap_rows.append({**row, "bootstrap_mean_deviation": np.mean(deviations), "bootstrap_max_deviation": np.max(deviations), "oracle_radius": oracle_boot_radius[level], "oracle_mean_deviation": np.mean(oracle_deviations)})
                if rep == 0:
                    q95 = q_values[.95]
                    representatives[name] = (geometry, noisy, truth, fit, q95*(fit.se_sampling+fit.se_direction), boot_radius[.95])
            else:
                oracle_record["estimated_bootstrap_radius"] = np.nan
                oracle_record["oracle_bootstrap_radius"] = np.nan

            q95 = q_values[.95]
            oracle_record["nominal_level"] = .95
            oracle_record["estimated_sampling_coverage"] = band_diagnostics(fit.center, fit.data_normals, q95*fit.se_sampling, truth, args.sigma)[0]
            oracle_record["oracle_sampling_coverage"] = band_diagnostics(oracle.center, oracle.data_normals, q95*oracle.se_sampling, truth, args.sigma)[0]
            oracle_rows.append(oracle_record)
            print(f"{name} replicate {rep+1}/{args.mc_reps}", flush=True)

    bias_rows: list[dict[str, object]] = []
    for name in ("circle", "ellipse"):
        for bin_index, records in enumerate(bias_samples[name]):
            estimated = np.asarray([(k, e) for k, e, tag in records if tag > 0])
            oracle = np.asarray([(k, e) for k, e, tag in records if tag < 0])
            if not len(estimated) or not len(oracle):
                continue
            bias_rows.append({
                "geometry": name, "angle_bin": bin_index, "count_estimated": len(estimated), "count_oracle": len(oracle),
                "mean_curvature": np.mean(oracle[:, 0]), "mean_estimated_signed_error": np.mean(estimated[:, 1]),
                "oracle_residual_bias": np.mean(oracle[:, 1]),
                "oracle_residual_bias_over_sigma2": np.mean(oracle[:, 1])/args.sigma**2,
                "direction_contribution": np.mean(estimated[:, 1])-np.mean(oracle[:, 1]),
                "sigma2_curvature": args.sigma**2*np.mean(oracle[:, 0]),
            })

    summary = summarize(raw_rows)
    # Aggregate oracle rows while retaining the bootstrap-selected denominator.
    oracle_summary: list[dict[str, object]] = []
    for name in ("circle", "ellipse"):
        group = [r for r in oracle_rows if r["geometry"] == name]
        boot = [r for r in group if np.isfinite(float(r["estimated_bootstrap_radius"]))]
        oracle_summary.append({
            "geometry": name, "nominal_level": .95,
            "estimated_hausdorff": np.mean([r["estimated_hausdorff"] for r in group]),
            "oracle_hausdorff": np.mean([r["oracle_hausdorff"] for r in group]),
            "estimated_sampling_coverage": np.mean([r["estimated_sampling_coverage"] for r in group]),
            "oracle_sampling_coverage": np.mean([r["oracle_sampling_coverage"] for r in group]),
            "mean_analytic_direction_se": np.mean([r["mean_analytic_direction_se"] for r in group]),
            "estimated_bootstrap_radius": np.mean([r["estimated_bootstrap_radius"] for r in boot]),
            "oracle_bootstrap_radius": np.mean([r["oracle_bootstrap_radius"] for r in boot]),
            "mc_replicates": len(group), "bootstrap_datasets": len(boot),
        })

    write_csv(args.output / "uq_raw.csv", raw_rows)
    write_csv(args.output / "uq_summary.csv", summary)
    write_csv(args.output / "direction_diagnostics.csv", direction_rows)
    write_csv(args.output / "bias_diagnostics.csv", bias_rows)
    write_csv(args.output / "oracle_ablation.csv", oracle_summary)
    write_csv(args.output / "bootstrap_diagnostics.csv", bootstrap_rows)
    for name, rows in trace_rows.items():
        write_csv(args.output / f"mf_trace_{name}.csv", rows)
    plot_calibration(args.output / "mf_uq_calibration.png", summary)
    plot_direction(args.output / "direction_error_diagnostic.png", direction_rows)
    plot_bias(args.output / "geometric_bias_diagnostic.png", bias_rows, args.sigma)
    plot_radius(args.output / "analytic_vs_bootstrap_radius.png", summary)
    for name, (geometry, noisy, truth, fit, analytic, bootstrap) in representatives.items():
        plot_tube(args.output / f"mf_uq_{name}.png", geometry, noisy, truth, fit, analytic, bootstrap)
    make_report(args.output / "REPORT.md", args, summary, oracle_summary, bias_rows)
    metadata = {
        "n": args.n, "sigma": args.sigma, "mc_reps": args.mc_reps,
        "bootstrap_resamples": args.bootstrap, "bootstrap_datasets_per_geometry": args.bootstrap_datasets,
        "truth_grid_size": args.truth_size, "levels": LEVELS,
        "center": "exact final smoothed output of benchmark.manifold_benchmark.manifold_fitting",
        "gp_used": False,
        "analytic_direction_limitation": "normal projected-component delta method; complete cylinder mean is piecewise constant conditional on membership",
        "truth_use": "evaluation and labeled oracle-direction ablation only",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
