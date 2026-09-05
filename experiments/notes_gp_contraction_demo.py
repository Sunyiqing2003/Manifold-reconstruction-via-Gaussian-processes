#!/usr/bin/env python3
"""Section-10-style GP modification of the Yao contraction step.

The first data split constructs only a closed query scaffold.  On the independent
second split, every query uses the Yao ball mean to estimate its contraction
direction.  The original cylinder average and the GP prediction at projected
coordinate zero then use exactly the same observations and direction.

Truth enters only simulation evaluation and plotting.  The reported tubes are
finite-grid conditional GP diagnostics, not frequentist confidence theorems.
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
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manifold_fitting_confidence_demo import (
    CURVES,
    CurveSpec,
    angle_diff,
    densify_periodic,
    frames_from_closed_curve,
    geometric_errors,
    sample_noisy_curve,
    seed_for,
    true_curve_polar,
    variable_tube_contains_truth,
)


@dataclass(frozen=True)
class LocalFit:
    average_point: np.ndarray
    gp_point: np.ndarray
    direction: np.ndarray
    direction_signal: float
    posterior_sd: float
    frequentist_sd: float
    cylinder_size: int
    ball_fallback: bool
    cylinder_fallback: bool


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def radial_scaffold(
    sample: np.ndarray, phi: np.ndarray, angle_bandwidth: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build a smooth closed radial query curve without fitting local directions."""
    center = np.mean(sample, axis=0)
    relative = sample - center
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    radii = np.linalg.norm(relative, axis=1)
    delta = angle_diff(angles[None, :] - phi[:, None])
    weights = np.exp(-0.5 * (delta / angle_bandwidth) ** 2)
    radius = (weights @ radii) / np.maximum(weights.sum(axis=1), 1e-14)
    radial_direction = np.column_stack((np.cos(phi), np.sin(phi)))
    scaffold = center + radius[:, None] * radial_direction
    # This frame is used only to create the controlled query offset.  It is never
    # passed to the Yao cylinder or GP estimator.
    _, scaffold_normal = frames_from_closed_curve(scaffold, center=center)
    return scaffold, scaffold_normal, {
        "scaffold_center_x": float(center[0]),
        "scaffold_center_y": float(center[1]),
        "scaffold_min_radius": float(np.min(radius)),
        "scaffold_max_radius": float(np.max(radius)),
    }


def yao_bandwidths(n: int, sigma: float, multiplier: float) -> tuple[float, float, float]:
    """Bandwidth scaling used by the repository's port of Yao et al. (2023)."""
    r = multiplier * 5.0 * sigma / np.log10(n)
    R = multiplier * 10.0 * sigma * np.sqrt(np.log(1.0 / sigma)) / np.log10(n)
    return 2.0 * r, r, R


def local_gp(
    q: np.ndarray,
    s: np.ndarray,
    sigma: float,
    amplitude: float,
    length_scale: float,
) -> tuple[float, float, float]:
    """Constant-mean universal kriging at q*=0 and its smoother variance."""
    diff = q[:, None, :] - q[None, :, :]
    distance_sq = np.sum(diff * diff, axis=2)
    kernel = amplitude * np.exp(-distance_sq / (2.0 * length_scale**2))
    covariance = kernel + sigma**2 * np.eye(len(q))
    jitter = 1e-10 * max(1.0, amplitude, sigma**2)
    try:
        factor = cho_factor(covariance + jitter * np.eye(len(q)), check_finite=False)
    except np.linalg.LinAlgError:
        factor = cho_factor(
            covariance + 1e-7 * max(1.0, amplitude, sigma**2) * np.eye(len(q)),
            check_finite=False,
        )
    ones = np.ones(len(q))
    inverse_ones = cho_solve(factor, ones, check_finite=False)
    information = float(ones @ inverse_ones)
    mean_weights = inverse_ones / information
    k0 = amplitude * np.exp(-np.sum(q * q, axis=1) / (2.0 * length_scale**2))
    inverse_k0 = cho_solve(factor, k0, check_finite=False)
    reproduction_gap = 1.0 - float(ones @ inverse_k0)
    weights = inverse_k0 + mean_weights * reproduction_gap
    prediction = float(weights @ s)
    posterior_variance = max(
        amplitude
        - float(k0 @ inverse_k0)
        + reproduction_gap**2 / information,
        0.0,
    )
    frequentist_variance = sigma**2 * float(weights @ weights)
    return prediction, math.sqrt(posterior_variance), math.sqrt(frequentist_variance)


def fit_query(
    sample: np.ndarray,
    tree: cKDTree,
    z: np.ndarray,
    *,
    sigma: float,
    r0: float,
    r: float,
    R: float,
    amplitude: float,
    length_scale: float,
    min_ball: int,
    min_cylinder: int,
) -> LocalFit:
    ball_idx = np.asarray(tree.query_ball_point(z, r0), dtype=int)
    ball_fallback = len(ball_idx) < min_ball
    if ball_fallback:
        ball_idx = np.atleast_1d(tree.query(z, k=min(min_ball, len(sample)))[1]).astype(int)
    ball_mean = np.mean(sample[ball_idx], axis=0)
    direction_vector = ball_mean - z
    direction_signal = float(np.linalg.norm(direction_vector))
    if direction_signal <= 10.0 * np.finfo(float).eps:
        # Data-only fallback: use the first nonzero direction among nearby means.
        near = np.atleast_1d(tree.query(z, k=min(max(min_ball, 12), len(sample)))[1]).astype(int)
        centered_near = sample[near] - z
        candidate = centered_near[np.argmax(np.linalg.norm(centered_near, axis=1))]
        direction_vector = candidate
        direction_signal = float(np.linalg.norm(candidate))
        ball_fallback = True
    direction = direction_vector / max(direction_signal, np.finfo(float).eps)

    search_radius = math.sqrt(R * R + r * r)
    candidate_idx = np.asarray(tree.query_ball_point(z, search_radius), dtype=int)
    centered = sample[candidate_idx] - z
    s_all = centered @ direction
    q_all = centered - s_all[:, None] * direction
    q_norm = np.linalg.norm(q_all, axis=1)
    keep = (np.abs(s_all) <= R) & (q_norm <= r)
    cylinder_idx = candidate_idx[keep]
    cylinder_fallback = len(cylinder_idx) < min_cylinder
    if cylinder_fallback:
        all_centered = sample - z
        all_s = all_centered @ direction
        all_q = all_centered - all_s[:, None] * direction
        scaled_distance = (all_s / R) ** 2 + np.sum(all_q * all_q, axis=1) / r**2
        cylinder_idx = np.argsort(scaled_distance)[: min(min_cylinder, len(sample))]

    centered = sample[cylinder_idx] - z
    s = centered @ direction
    q = centered - s[:, None] * direction
    gp_mean, posterior_sd, frequentist_sd = local_gp(
        q, s, sigma, amplitude, length_scale
    )
    average_mean = float(np.mean(s))
    return LocalFit(
        average_point=z + direction * average_mean,
        gp_point=z + direction * gp_mean,
        direction=direction,
        direction_signal=direction_signal,
        posterior_sd=posterior_sd,
        frequentist_sd=frequentist_sd,
        cylinder_size=len(cylinder_idx),
        ball_fallback=ball_fallback,
        cylinder_fallback=cylinder_fallback,
    )


def point_errors(points: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return cKDTree(truth).query(points, k=1)[0]


def tube_boundary_inside_noise(
    center: np.ndarray,
    direction: np.ndarray,
    width: np.ndarray,
    truth: np.ndarray,
    phi: np.ndarray,
    dense_phi: np.ndarray,
    noise_width: float,
) -> tuple[bool, float]:
    lower = densify_periodic(phi, center - width[:, None] * direction, dense_phi)
    upper = densify_periodic(phi, center + width[:, None] * direction, dense_phi)
    tree = cKDTree(truth)
    maximum = float(max(np.max(tree.query(lower)[0]), np.max(tree.query(upper)[0])))
    return maximum <= noise_width, maximum


def true_curvature(spec: CurveSpec, points: np.ndarray) -> np.ndarray:
    theta = np.arctan2(points[:, 1] / spec.b, points[:, 0] / spec.a)
    denominator = (
        spec.a**2 * np.sin(theta) ** 2 + spec.b**2 * np.cos(theta) ** 2
    ) ** 1.5
    return spec.a * spec.b / denominator


def run_case(
    manifold: str,
    rep: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    spec = CURVES[manifold]
    phi = np.linspace(0.0, 2.0 * np.pi, args.grid_size, endpoint=False)
    dense_phi = np.linspace(0.0, 2.0 * np.pi, args.dense_grid_size, endpoint=False)
    truth_grid = true_curve_polar(spec, phi)
    truth_dense = true_curve_polar(spec, dense_phi)
    rng = np.random.default_rng(seed_for(args.seed, 1701, list(CURVES).index(manifold), rep))
    _, noisy = sample_noisy_curve(rng, spec, args.n, args.sigma)
    permutation = rng.permutation(args.n)
    split = args.n // 2
    scaffold_sample = noisy[permutation[:split]]
    contraction_sample = noisy[permutation[split:]]
    scaffold, scaffold_normal, scaffold_diag = radial_scaffold(
        scaffold_sample, phi, args.scaffold_angle_bandwidth
    )
    query = scaffold + args.c_offset * args.sigma * scaffold_normal
    r0, r, R = yao_bandwidths(len(contraction_sample), args.sigma, args.bandwidth_multiplier)
    amplitude = args.amplitude_factor * args.sigma**2
    length_scale = args.c_ell * r
    tree = cKDTree(contraction_sample)
    fits = [
        fit_query(
            contraction_sample,
            tree,
            z,
            sigma=args.sigma,
            r0=r0,
            r=r,
            R=R,
            amplitude=amplitude,
            length_scale=length_scale,
            min_ball=args.min_ball,
            min_cylinder=args.min_cylinder,
        )
        for z in query
    ]
    average = np.vstack([fit.average_point for fit in fits])
    gp = np.vstack([fit.gp_point for fit in fits])
    direction = np.vstack([fit.direction for fit in fits])
    posterior_sd = np.asarray([fit.posterior_sd for fit in fits])
    frequentist_sd = np.asarray([fit.frequentist_sd for fit in fits])
    signal = np.asarray([fit.direction_signal for fit in fits])
    cylinder_size = np.asarray([fit.cylinder_size for fit in fits])
    ball_fallback = np.asarray([fit.ball_fallback for fit in fits])
    cylinder_fallback = np.asarray([fit.cylinder_fallback for fit in fits])

    scaffold_directed, scaffold_hausdorff = geometric_errors(scaffold, truth_dense, phi, dense_phi)
    avg_directed, avg_hausdorff = geometric_errors(average, truth_dense, phi, dense_phi)
    gp_directed, gp_hausdorff = geometric_errors(gp, truth_dense, phi, dense_phi)
    q_critical = float(norm.ppf(1.0 - args.alpha / (2.0 * args.grid_size)))
    posterior_width = q_critical * posterior_sd
    frequentist_width = q_critical * frequentist_sd
    posterior_cover = variable_tube_contains_truth(gp, posterior_width, truth_dense, phi, dense_phi)
    frequentist_cover = variable_tube_contains_truth(gp, frequentist_width, truth_dense, phi, dense_phi)
    noise_width = args.noise_multiplier * args.sigma
    inside_noise, max_boundary_distance = tube_boundary_inside_noise(
        gp, direction, posterior_width, truth_dense, phi, dense_phi, noise_width
    )
    avg_local_error = point_errors(average, truth_dense)
    gp_local_error = point_errors(gp, truth_dense)
    curvature = true_curvature(spec, truth_grid)
    high = curvature >= np.quantile(curvature, 0.75)
    sd_ratio = posterior_sd / np.maximum(frequentist_sd, np.finfo(float).eps)
    row: dict[str, object] = {
        "manifold": manifold,
        "repeat": rep,
        "n": args.n,
        "n_scaffold": len(scaffold_sample),
        "n_contraction": len(contraction_sample),
        "sigma": args.sigma,
        "scaffold_hausdorff": scaffold_hausdorff,
        "average_hausdorff": avg_hausdorff,
        "gp_hausdorff": gp_hausdorff,
        "gp_better_than_average": int(gp_hausdorff < avg_hausdorff),
        "scaffold_directed_error": scaffold_directed,
        "average_directed_error": avg_directed,
        "gp_directed_error": gp_directed,
        "mean_direction_signal": float(np.mean(signal)),
        "min_direction_signal": float(np.min(signal)),
        "median_cylinder_size": float(np.median(cylinder_size)),
        "min_cylinder_size": int(np.min(cylinder_size)),
        "ball_fallback_count": int(np.sum(ball_fallback)),
        "ball_fallback_fraction": float(np.mean(ball_fallback)),
        "cylinder_fallback_count": int(np.sum(cylinder_fallback)),
        "cylinder_fallback_fraction": float(np.mean(cylinder_fallback)),
        "mean_posterior_sd": float(np.mean(posterior_sd)),
        "mean_frequentist_sd": float(np.mean(frequentist_sd)),
        "mean_posterior_to_frequentist_sd": float(np.mean(sd_ratio)),
        "max_posterior_to_frequentist_sd": float(np.max(sd_ratio)),
        "posterior_band_covers_truth": int(posterior_cover),
        "frequentist_band_covers_truth": int(frequentist_cover),
        "posterior_max_halfwidth": float(np.max(posterior_width)),
        "posterior_max_width_over_noise": float(np.max(posterior_width) / noise_width),
        "posterior_width_below_noise": int(np.max(posterior_width) < noise_width),
        "posterior_band_inside_noise": int(inside_noise),
        "max_posterior_boundary_distance": max_boundary_distance,
        "mean_average_local_error": float(np.mean(avg_local_error)),
        "mean_gp_local_error": float(np.mean(gp_local_error)),
        "high_curvature_average_error": float(np.mean(avg_local_error[high])),
        "high_curvature_gp_error": float(np.mean(gp_local_error[high])),
        "high_curvature_gp_improvement": float(np.mean(avg_local_error[high] - gp_local_error[high])),
        **scaffold_diag,
    }
    detail = {
        "phi": phi,
        "truth_grid": truth_grid,
        "truth_dense": truth_dense,
        "noisy": contraction_sample,
        "scaffold": scaffold,
        "query": query,
        "average": average,
        "gp": gp,
        "direction": direction,
        "posterior_width": posterior_width,
        "curvature": curvature,
        "average_local_error": avg_local_error,
        "gp_local_error": gp_local_error,
    }
    return row, detail


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for manifold in CURVES:
        group = [row for row in rows if row["manifold"] == manifold]
        value = lambda key: np.asarray([float(row[key]) for row in group])
        summary.append({
            "manifold": manifold,
            "mc_reps": len(group),
            "mean_average_hausdorff": float(np.mean(value("average_hausdorff"))),
            "mean_gp_hausdorff": float(np.mean(value("gp_hausdorff"))),
            "median_average_hausdorff": float(np.median(value("average_hausdorff"))),
            "median_gp_hausdorff": float(np.median(value("gp_hausdorff"))),
            "fraction_gp_better": float(np.mean(value("gp_better_than_average"))),
            "posterior_band_coverage": float(np.mean(value("posterior_band_covers_truth"))),
            "frequentist_band_coverage": float(np.mean(value("frequentist_band_covers_truth"))),
            "posterior_band_inside_noise_fraction": float(np.mean(value("posterior_band_inside_noise"))),
            "mean_posterior_to_frequentist_sd": float(np.mean(value("mean_posterior_to_frequentist_sd"))),
            "max_posterior_to_frequentist_sd": float(np.max(value("max_posterior_to_frequentist_sd"))),
            "mean_median_cylinder_size": float(np.mean(value("median_cylinder_size"))),
            "minimum_cylinder_size": int(np.min(value("min_cylinder_size"))),
            "mean_ball_fallback_fraction": float(np.mean(value("ball_fallback_fraction"))),
            "mean_cylinder_fallback_fraction": float(np.mean(value("cylinder_fallback_fraction"))),
            "mean_high_curvature_gp_improvement": float(np.mean(value("high_curvature_gp_improvement"))),
        })
    return summary


def plot_case(path: Path, manifold: str, row: dict[str, object], data: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    closed = lambda values: np.vstack((values, values[:1]))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    ax = axes[0]
    noisy = data["noisy"]
    ax.scatter(noisy[:, 0], noisy[:, 1], s=6, alpha=0.10, color="0.35", label="contraction split")
    for values, style in (
        (data["truth_dense"], {"color": "black", "lw": 2.0, "label": "truth (evaluation)"}),
        (data["scaffold"], {"ls": "--", "lw": 1.5, "label": r"$\Gamma$ scaffold"}),
        (data["query"], {"ls": ":", "lw": 1.6, "label": "offset query"}),
        (data["average"], {"lw": 1.8, "label": "cylinder average"}),
        (data["gp"], {"lw": 1.8, "label": "GP at q=0"}),
    ):
        curve = closed(values)
        ax.plot(curve[:, 0], curve[:, 1], **style)
    stride = max(1, args.grid_size // 12)
    z = data["query"][::stride]
    u = data["direction"][::stride]
    ax.quiver(z[:, 0], z[:, 1], u[:, 0], u[:, 1], angles="xy", scale_units="xy", scale=10.0, width=0.004, color="tab:purple", label=r"ball-step $u_z$")
    ax.set_title(f"{manifold}: H_avg={float(row['average_hausdorff']):.4f}, H_GP={float(row['gp_hausdorff']):.4f}")
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    # Correct two-object return signature.  The truth normal is visualization-only.
    _, truth_normal = frames_from_closed_curve(data["truth_grid"], center=np.zeros(2))
    noise_width = args.noise_multiplier * args.sigma
    lower_noise = data["truth_grid"] - noise_width * truth_normal
    upper_noise = data["truth_grid"] + noise_width * truth_normal
    lower_gp = data["gp"] - data["posterior_width"][:, None] * data["direction"]
    upper_gp = data["gp"] + data["posterior_width"][:, None] * data["direction"]
    noise_polygon = np.vstack((closed(upper_noise), closed(lower_noise)[::-1], upper_noise[:1]))
    gp_polygon = np.vstack((closed(upper_gp), closed(lower_gp)[::-1], upper_gp[:1]))
    ax.fill(noise_polygon[:, 0], noise_polygon[:, 1], color="0.75", alpha=0.45, label=r"truth $\pm1.96\sigma$ reference")
    ax.fill(gp_polygon[:, 0], gp_polygon[:, 1], color="tab:blue", alpha=0.28, label="finite-grid conditional GP band")
    truth_closed = closed(data["truth_dense"])
    gp_closed = closed(data["gp"])
    ax.plot(truth_closed[:, 0], truth_closed[:, 1], color="black", lw=2.0, label="truth")
    ax.plot(gp_closed[:, 0], gp_closed[:, 1], color="tab:blue", lw=1.5, label="GP image")
    ax.set_title(
        f"covered={bool(row['posterior_band_covers_truth'])}, inside_noise={bool(row['posterior_band_inside_noise'])}\n"
        f"max_width/(1.96 sigma)={float(row['posterior_max_width_over_noise']):.2f}"
    )
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_curvature(path: Path, records: list[dict[str, np.ndarray]]) -> None:
    curvature = np.concatenate([record["curvature"] for record in records])
    average_error = np.concatenate([record["average_local_error"] for record in records])
    gp_error = np.concatenate([record["gp_local_error"] for record in records])
    order = np.argsort(curvature)
    bins = np.array_split(order, 18)
    x = np.asarray([np.mean(curvature[idx]) for idx in bins])
    avg = np.asarray([np.mean(average_error[idx]) for idx in bins])
    gp = np.asarray([np.mean(gp_error[idx]) for idx in bins])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(curvature, average_error, s=5, alpha=0.05, color="tab:orange")
    ax.scatter(curvature, gp_error, s=5, alpha=0.05, color="tab:blue")
    ax.plot(x, avg, marker="o", color="tab:orange", label="cylinder average")
    ax.plot(x, gp, marker="o", color="tab:blue", label="GP at q=0")
    ax.set_xlabel("true ellipse curvature (post-hoc only)")
    ax.set_ylabel("point-to-truth geometric error")
    ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def write_report(path: Path, summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    by_name = {str(row["manifold"]): row for row in summary}
    lines = [
        "# Notes-faithful GP contraction diagnostic",
        "",
        "This experiment compares the original cylinder average with a scalar GP",
        "prediction at projected coordinate `q=0`. Both estimators use the same",
        "Yao ball-step direction and exactly the same cylinder observations. The",
        "independent first split supplies only the closed query scaffold.",
        "",
        "## Frozen setup",
        "",
        f"- `n={args.n}`, `sigma={args.sigma}`, `{args.mc_reps}` Monte Carlo replicates per manifold;",
        f"- query offset `c_offset*sigma={args.c_offset}*sigma`;",
        f"- GP amplitude `A={args.amplitude_factor}*sigma^2`, length scale `ell={args.c_ell}*r`;",
        "- constant unknown GP mean handled by universal kriging;",
        "- finite-grid Bonferroni multiplier for UQ visualization.",
        "",
        "## Point-estimation results",
        "",
        "| manifold | mean H_avg | mean H_GP | median H_avg | median H_GP | fraction H_GP < H_avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for manifold in CURVES:
        row = by_name[manifold]
        lines.append(
            f"| {manifold} | {float(row['mean_average_hausdorff']):.5f} | {float(row['mean_gp_hausdorff']):.5f} | "
            f"{float(row['median_average_hausdorff']):.5f} | {float(row['median_gp_hausdorff']):.5f} | {float(row['fraction_gp_better']):.2f} |"
        )
    circle = by_name["circle"]
    ellipse = by_name["ellipse"]
    lines += [
        "",
        "With these frozen hyperparameters, the GP does not improve the primary",
        "Hausdorff criterion on average. It beats the shared-cylinder average in",
        f"{float(circle['fraction_gp_better']):.0%} of circle replicates and",
        f"{float(ellipse['fraction_gp_better']):.0%} of ellipse replicates. The",
        "ellipse top-curvature quartile has only a small positive mean local-error",
        f"difference ({float(ellipse['mean_high_curvature_gp_improvement']):.5f})",
        "in favor of GP. This is weak local evidence and does not overturn the",
        "whole-curve Hausdorff comparison.",
    ]
    lines += [
        "",
        "The comparison estimates the empirical difference between",
        "`E[s | Y in V_z]` and a GP estimate of `E[s | q=0]`. The ellipse curvature",
        "figure is post-hoc: curvature never enters either estimator.",
        "",
        "For a local quadratic graph, the motivating heuristic is that cylinder",
        "averaging contains both an EIV term and a transverse-window term, whereas",
        "the GP target at `q=0` may remove the extra transverse-window contribution.",
        "The remaining population bias can still be of order `kappa*sigma^2/2`;",
        "the experiment does not subtract it and does not claim that GP eliminates bias.",
        "",
        "## Conditional UQ diagnostics",
        "",
        "| manifold | posterior coverage | frequentist-mean coverage | mean s_post/s_F | max s_post/s_F | posterior tube inside noise band |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for manifold in CURVES:
        row = by_name[manifold]
        lines.append(
            f"| {manifold} | {float(row['posterior_band_coverage']):.2f} | {float(row['frequentist_band_coverage']):.2f} | "
            f"{float(row['mean_posterior_to_frequentist_sd']):.3f} | {float(row['max_posterior_to_frequentist_sd']):.3f} | "
            f"{float(row['posterior_band_inside_noise_fraction']):.2f} |"
        )
    lines += [
        "",
        "These are finite-grid conditional GP simultaneous bands. The posterior SD",
        "and `sigma*||a_z||` quantify different uncertainties and are reported",
        "separately. Empirical inclusion here is a simulation diagnostic, not an",
        "honest true-manifold confidence theorem. A maximum half-width below",
        "`1.96*sigma` also does not imply geometric containment; containment is",
        "checked from the dense band boundaries.",
        "",
        "## Neighborhood diagnostics",
        "",
        "| manifold | mean median cylinder n | minimum cylinder n | ball fallback fraction | cylinder fallback fraction | high-curvature GP improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for manifold in CURVES:
        row = by_name[manifold]
        lines.append(
            f"| {manifold} | {float(row['mean_median_cylinder_size']):.1f} | {int(row['minimum_cylinder_size'])} | "
            f"{float(row['mean_ball_fallback_fraction']):.4f} | {float(row['mean_cylinder_fallback_fraction']):.4f} | "
            f"{float(row['mean_high_curvature_gp_improvement']):.5f} |"
        )
    lines += [
        "",
        "The last column uses the top curvature quartile. It is most meaningful for",
        "the ellipse; the circle has constant curvature and acts as a control.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--mc-reps", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--dense-grid-size", type=int, default=2400)
    parser.add_argument("--c-offset", type=float, default=1.0)
    parser.add_argument("--c-ell", type=float, default=1.0)
    parser.add_argument("--amplitude-factor", type=float, default=1.0)
    parser.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    parser.add_argument("--scaffold-angle-bandwidth", type=float, default=0.16)
    parser.add_argument("--min-ball", type=int, default=5)
    parser.add_argument("--min-cylinder", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--noise-multiplier", type=float, default=1.96)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, default=Path("results/notes_gp_contraction_demo"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n = min(args.n, 600)
        args.mc_reps = min(args.mc_reps, 2)
        args.grid_size = min(args.grid_size, 36)
        args.dense_grid_size = min(args.dense_grid_size, 900)
    if args.n < 100 or args.mc_reps < 1 or args.grid_size < 12:
        parser.error("n>=100, mc-reps>=1, and grid-size>=12 are required")
    if args.sigma <= 0 or args.c_ell <= 0 or args.amplitude_factor <= 0:
        parser.error("sigma, c-ell, and amplitude-factor must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    representative: dict[str, tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    ellipse_details: list[dict[str, np.ndarray]] = []
    for manifold in CURVES:
        for rep in range(args.mc_reps):
            row, detail = run_case(manifold, rep, args)
            rows.append(row)
            if rep == 0:
                representative[manifold] = (row, detail)
            if manifold == "ellipse":
                ellipse_details.append(detail)
    summary = summarize(rows)
    write_csv(args.output / "raw_metrics.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    for manifold, (row, detail) in representative.items():
        plot_case(args.output / f"{manifold}_notes_gp_contraction.png", manifold, row, detail, args)
    plot_curvature(args.output / "ellipse_curvature_diagnostic.png", ellipse_details)
    metadata = {
        "algorithm": "Yao ball direction + shared cylinder average/GP(q=0)",
        "truth_used_by_estimator": False,
        "sample_splitting": "half scaffold, half contraction",
        "query_offset_c_sigma": args.c_offset,
        "n": args.n,
        "sigma": args.sigma,
        "mc_reps": args.mc_reps,
        "grid_size": args.grid_size,
        "dense_grid_size": args.dense_grid_size,
        "amplitude": f"{args.amplitude_factor} * sigma^2",
        "length_scale": f"{args.c_ell} * r",
        "gp_mean": "unknown constant estimated by universal kriging",
        "bandwidth_formula": {"r0": "2r", "r": "5 sigma/log10(n_contraction)", "R": "10 sigma sqrt(log(1/sigma))/log10(n_contraction)"},
        "circle_eiv_bias_reference": args.sigma**2 / 2.0,
        "uq_label": "finite-grid conditional GP simultaneous band",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(args.output / "REPORT.md", summary, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
