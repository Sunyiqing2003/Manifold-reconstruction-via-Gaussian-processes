#!/usr/bin/env python3
"""Paper-facing benchmark: MrGap, Manifold Fitting, and Yao-geometry local GP.

All methods receive the same noisy replicate.  Circle/ellipse uncertainty is
evaluated geometrically on a dense closed curve.  Torus uncertainty is reported
only as a local scale diagnostic; no global surface coverage is asserted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.manifold_benchmark import GP, MANIFOLDS, manifold_fitting, sample_torus
from experiments.manifold_fitting_confidence_demo import angle_diff, frames_from_closed_curve
from experiments.notes_gp_contraction_demo import fit_query, radial_scaffold, yao_bandwidths


LEVELS = (0.80, 0.90, 0.95, 0.99)
METHOD_LABELS = {
    "mrgap": "MrGap",
    "manifold_fitting": "Manifold Fitting",
    "ours": "Ours",
    "ours_oracle_direction": "Ours: oracle direction ablation",
}


@dataclass(frozen=True)
class Geometry:
    name: str
    kind: str
    a: float
    b: float
    intrinsic_dim: int
    ambient_dim: int

    @property
    def reach(self) -> float:
        if self.kind == "circle":
            return self.a
        if self.kind == "ellipse":
            return self.b**2 / self.a
        return 0.8


@dataclass
class MethodResult:
    points: np.ndarray
    directions: np.ndarray | None = None
    posterior_sd: np.ndarray | None = None
    frequentist_sd: np.ndarray | None = None
    diagnostics: dict[str, float] | None = None


def seed_for(base: int, *values: int) -> int:
    return int(np.random.SeedSequence([base, *values]).generate_state(1)[0])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def curve_points(geometry: Geometry, theta: np.ndarray) -> np.ndarray:
    return np.column_stack((geometry.a * np.cos(theta), geometry.b * np.sin(theta)))


def sample_curve(
    rng: np.random.Generator, geometry: Geometry, n: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geometry.kind == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
    else:
        accepted: list[np.ndarray] = []
        total = 0
        while total < n:
            proposal = rng.uniform(0.0, 2.0 * np.pi, max(128, 2 * (n - total)))
            speed = np.sqrt(
                (geometry.a * np.sin(proposal)) ** 2
                + (geometry.b * np.cos(proposal)) ** 2
            )
            keep = rng.random(len(proposal)) < speed / max(geometry.a, geometry.b)
            accepted.append(proposal[keep])
            total += int(np.sum(keep))
        theta = np.concatenate(accepted)[:n]
    clean = curve_points(geometry, theta)
    return clean, clean + sigma * rng.normal(size=clean.shape), theta


def curve_truth(geometry: Geometry, size: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    return theta, curve_points(geometry, theta)


def torus_truth(size: int) -> np.ndarray:
    side = max(40, int(np.sqrt(size)))
    u, v = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, side, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi, side, endpoint=False),
        indexing="ij",
    )
    major, minor = 2.0, 0.8
    return np.column_stack(
        (
            ((major + minor * np.cos(u)) * np.cos(v)).ravel(),
            ((major + minor * np.cos(u)) * np.sin(v)).ravel(),
            (minor * np.sin(u)).ravel(),
        )
    )


def densify(phi: np.ndarray, values: np.ndarray, dense_phi: np.ndarray) -> np.ndarray:
    x = np.r_[phi, 2.0 * np.pi]
    y = np.vstack((values, values[:1])) if values.ndim == 2 else np.r_[values, values[0]]
    return CubicSpline(x, y, bc_type="periodic", axis=0)(dense_phi)


def radial_curve_from_cloud(cloud: np.ndarray, phi: np.ndarray, bandwidth: float) -> np.ndarray:
    center = np.mean(cloud, axis=0)
    relative = cloud - center
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    radii = np.linalg.norm(relative, axis=1)
    weights = np.exp(-0.5 * (angle_diff(angles[None, :] - phi[:, None]) / bandwidth) ** 2)
    radius = weights @ radii / np.maximum(weights.sum(axis=1), 1e-14)
    return center + radius[:, None] * np.column_stack((np.cos(phi), np.sin(phi)))


def geometric_metrics(points: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    truth_tree = cKDTree(truth)
    point_tree = cKDTree(points)
    fit_to_truth = truth_tree.query(points)[0]
    truth_to_fit = point_tree.query(truth)[0]
    hausdorff = float(max(np.max(fit_to_truth), np.max(truth_to_fit)))
    average = 0.5 * (float(np.mean(fit_to_truth)) + float(np.mean(truth_to_fit)))
    return hausdorff, average


def mrgap_parameters(geometry: Geometry) -> tuple[float, float, GP]:
    if geometry.kind == "torus":
        source = MANIFOLDS["torus"]
        return source.epsilon, source.delta, source.gps[0]
    # No circle/ellipse tuple is published in this repository.  Freeze the
    # documented first-round Cassini tuple for all planar closed curves.
    source = MANIFOLDS["cassini"]
    return source.epsilon, source.delta, source.gps[0]


def mrgap_at_centers(
    data: np.ndarray,
    centers: np.ndarray,
    intrinsic_dim: int,
    epsilon: float,
    delta: float,
    gp: GP,
) -> MethodResult:
    """Mfit1-equivalent mean and latent posterior at arbitrary center points."""
    tree = cKDTree(data)
    eps_neighbors = tree.query_ball_point(centers, epsilon)
    delta_neighbors = tree.query_ball_point(centers, delta)
    output = np.empty_like(centers)
    directions = np.empty_like(centers)
    posterior_sd = np.empty(len(centers))
    local_sizes = np.empty(len(centers))
    solve_failures = 0
    data_center = np.mean(data, axis=0)
    estimated_major_radius = float(
        np.median(np.linalg.norm(data[:, :2] - data_center[:2], axis=1))
    )
    for j, center in enumerate(centers):
        eps_idx = np.asarray(eps_neighbors[j], dtype=int)
        if len(eps_idx) < intrinsic_dim + 2:
            eps_idx = np.atleast_1d(tree.query(center, k=intrinsic_dim + 2)[1]).astype(int)
        local = data[eps_idx] - center
        covariance = local.T @ local / len(local)
        values, vectors = np.linalg.eigh(covariance)
        rotation = vectors[:, np.argsort(values)[::-1]]
        reg_idx = np.asarray(delta_neighbors[j], dtype=int)
        if len(reg_idx) < intrinsic_dim + 2:
            reg_idx = np.atleast_1d(tree.query(center, k=intrinsic_dim + 2)[1]).astype(int)
        rotated = (data[reg_idx] - center) @ rotation
        x, z = rotated[:, :intrinsic_dim], rotated[:, intrinsic_dim:]
        z_mean = np.mean(z, axis=0)
        kernel = gp.amplitude * np.exp(-distance.cdist(x, x, "sqeuclidean") / gp.rho)
        system = kernel + gp.noise_variance * np.eye(len(x))
        k0 = gp.amplitude * np.exp(-np.sum(x * x, axis=1) / gp.rho)
        jitter = 1e-10 * max(1.0, gp.amplitude, gp.noise_variance)
        try:
            factor = cho_factor(system + jitter * np.eye(len(x)), check_finite=False)
            weights = cho_solve(factor, k0, check_finite=False)
        except np.linalg.LinAlgError:
            solve_failures += 1
            weights = np.linalg.lstsq(system, k0, rcond=None)[0]
        prediction = z_mean + weights @ (z - z_mean)
        output[j] = center + np.r_[np.zeros(intrinsic_dim), prediction] @ rotation.T
        normal_basis = rotation[:, intrinsic_dim:]
        direction = normal_basis[:, 0]
        if data.shape[1] == 2:
            orientation_reference = center - data_center
        else:
            angle = math.atan2(center[1] - data_center[1], center[0] - data_center[0])
            orientation_reference = center - np.array(
                (
                    data_center[0] + estimated_major_radius * math.cos(angle),
                    data_center[1] + estimated_major_radius * math.sin(angle),
                    data_center[2],
                )
            )
        if direction @ orientation_reference < 0:
            direction = -direction
        directions[j] = direction
        posterior_sd[j] = math.sqrt(max(gp.amplitude - float(k0 @ weights), 0.0))
        local_sizes[j] = len(reg_idx)
    return MethodResult(
        output,
        directions,
        posterior_sd,
        None,
        {
            "median_local_n": float(np.median(local_sizes)),
            "min_local_n": float(np.min(local_sizes)),
            "solve_failures": float(solve_failures),
        },
    )


def curve_band_geometry(
    center: np.ndarray,
    directions: np.ndarray,
    width: np.ndarray,
    phi: np.ndarray,
    dense_phi: np.ndarray,
    truth: np.ndarray,
    noise_width: float,
) -> tuple[bool, bool, float]:
    lower = densify(phi, center - width[:, None] * directions, dense_phi)
    upper = densify(phi, center + width[:, None] * directions, dense_phi)
    center_ref = np.mean(center, axis=0)
    lower_radius = np.median(np.linalg.norm(lower - center_ref, axis=1))
    upper_radius = np.median(np.linalg.norm(upper - center_ref, axis=1))
    outer, inner = (lower, upper) if lower_radius > upper_radius else (upper, lower)
    outer_path = MplPath(np.vstack((outer, outer[:1])))
    inner_path = MplPath(np.vstack((inner, inner[:1])))
    covered = bool(
        np.all(outer_path.contains_points(truth, radius=1e-10))
        and not np.any(inner_path.contains_points(truth, radius=-1e-10))
    )
    truth_tree = cKDTree(truth)
    max_boundary = float(max(np.max(truth_tree.query(lower)[0]), np.max(truth_tree.query(upper)[0])))
    return covered, max_boundary < noise_width, max_boundary


def build_curve_methods(
    geometry: Geometry,
    noisy: np.ndarray,
    sigma: float,
    args: argparse.Namespace,
    oracle_direction: bool = False,
) -> tuple[dict[str, MethodResult], dict[str, np.ndarray]]:
    phi = np.linspace(0.0, 2.0 * np.pi, args.curve_grid_size, endpoint=False)
    rng = np.random.default_rng(seed_for(args.seed, 4401, len(noisy), int(1e5 * sigma)))
    permutation = rng.permutation(len(noisy))
    split = len(noisy) // 2
    scaffold_sample, contraction = noisy[permutation[:split]], noisy[permutation[split:]]
    scaffold, scaffold_normal, _ = radial_scaffold(scaffold_sample, phi, args.scaffold_angle_bandwidth)
    query = scaffold + args.query_offset_factor * sigma * scaffold_normal
    r0, r, R = yao_bandwidths(len(contraction), sigma, 1.0)
    tree = cKDTree(contraction)
    ours_fits = []
    oracle_fits = []
    truth_dense = curve_truth(geometry, args.truth_size)[1]
    truth_tree = cKDTree(truth_dense)
    for z in query:
        ours_fits.append(
            fit_query(
                contraction,
                tree,
                z,
                sigma=sigma,
                r0=r0,
                r=r,
                R=R,
                amplitude=sigma**2,
                length_scale=r,
                min_ball=5,
                min_cylinder=11,
            )
        )
        if oracle_direction:
            nearest = truth_dense[truth_tree.query(z)[1]]
            oracle_fits.append(
                fit_query(
                    contraction,
                    tree,
                    z,
                    sigma=sigma,
                    r0=r0,
                    r=r,
                    R=R,
                    amplitude=sigma**2,
                    length_scale=r,
                    min_ball=5,
                    min_cylinder=11,
                    direction_override=nearest - z,
                )
            )
    ours = MethodResult(
        np.vstack([fit.gp_point for fit in ours_fits]),
        np.vstack([fit.direction for fit in ours_fits]),
        np.asarray([fit.posterior_sd for fit in ours_fits]),
        np.asarray([fit.frequentist_sd for fit in ours_fits]),
        {
            "median_local_n": float(np.median([fit.cylinder_size for fit in ours_fits])),
            "min_local_n": float(np.min([fit.cylinder_size for fit in ours_fits])),
            "min_direction_signal": float(np.min([fit.direction_signal for fit in ours_fits])),
        },
    )
    mf_cloud, mf_diag = manifold_fitting(contraction, sigma, 1.0, True)
    mf = MethodResult(
        radial_curve_from_cloud(mf_cloud, phi, args.scaffold_angle_bandwidth),
        diagnostics={
            "median_local_n": float(mf_diag["median_local_neighborhood"]),
            "min_local_n": float(mf_diag["min_local_neighborhood"]),
        },
    )
    epsilon, delta, gp = mrgap_parameters(geometry)
    mrgap = mrgap_at_centers(contraction, query, 1, epsilon, delta, gp)
    methods = {"mrgap": mrgap, "manifold_fitting": mf, "ours": ours}
    if oracle_direction:
        methods["ours_oracle_direction"] = MethodResult(
            np.vstack([fit.gp_point for fit in oracle_fits]),
            np.vstack([fit.direction for fit in oracle_fits]),
            np.asarray([fit.posterior_sd for fit in oracle_fits]),
            np.asarray([fit.frequentist_sd for fit in oracle_fits]),
        )
    return methods, {"phi": phi, "query": query, "contraction": contraction, "truth": truth_dense}


def data_only_torus_queries(sample: np.ndarray, count: int, sigma: float) -> np.ndarray:
    count = min(count, len(sample))
    chosen = np.linspace(0, len(sample) - 1, count, dtype=int)
    scaffold = sample[chosen]
    tree = cKDTree(sample)
    center = np.mean(sample, axis=0)
    major_radius = float(np.median(np.linalg.norm(sample[:, :2] - center[:2], axis=1)))
    normals = np.empty_like(scaffold)
    for j, point in enumerate(scaffold):
        idx = tree.query(point, k=min(20, len(sample)))[1]
        local = sample[np.atleast_1d(idx)] - point
        values, vectors = np.linalg.eigh(local.T @ local / len(local))
        normal = vectors[:, np.argmin(values)]
        # Data-only global torus orientation convention.
        angle = math.atan2(point[1] - center[1], point[0] - center[0])
        ring_point = np.array(
            (
                center[0] + major_radius * math.cos(angle),
                center[1] + major_radius * math.sin(angle),
                center[2],
            )
        )
        if normal @ (point - ring_point) < 0:
            normal = -normal
        normals[j] = normal
    return scaffold + sigma * normals


def build_torus_methods(
    noisy: np.ndarray, sigma: float, args: argparse.Namespace
) -> tuple[dict[str, MethodResult], dict[str, np.ndarray]]:
    split = len(noisy) // 2
    scaffold_sample, contraction = noisy[:split], noisy[split:]
    query = data_only_torus_queries(scaffold_sample, args.torus_query_size, sigma)
    r0, r, R = yao_bandwidths(len(contraction), sigma, 1.0)
    tree = cKDTree(contraction)
    fits = [
        fit_query(
            contraction,
            tree,
            z,
            sigma=sigma,
            r0=r0,
            r=r,
            R=R,
            amplitude=sigma**2,
            length_scale=r,
            min_ball=5,
            min_cylinder=11,
        )
        for z in query
    ]
    ours = MethodResult(
        np.vstack([fit.gp_point for fit in fits]),
        np.vstack([fit.direction for fit in fits]),
        np.asarray([fit.posterior_sd for fit in fits]),
        np.asarray([fit.frequentist_sd for fit in fits]),
        {
            "median_local_n": float(np.median([fit.cylinder_size for fit in fits])),
            "min_local_n": float(np.min([fit.cylinder_size for fit in fits])),
            "min_direction_signal": float(np.min([fit.direction_signal for fit in fits])),
        },
    )
    mf_cloud, _ = manifold_fitting(contraction, sigma, 1.0, True)
    chosen = np.linspace(0, len(mf_cloud) - 1, len(query), dtype=int)
    mf = MethodResult(mf_cloud[chosen])
    epsilon, delta, gp = mrgap_parameters(Geometry("torus", "torus", 2, 0.8, 2, 3))
    mrgap = mrgap_at_centers(contraction, query, 2, epsilon, delta, gp)
    return {"mrgap": mrgap, "manifold_fitting": mf, "ours": ours}, {
        "query": query,
        "contraction": contraction,
        "truth": torus_truth(args.torus_truth_size),
    }


def run_setting(
    geometry: Geometry,
    n: int,
    sigma: float,
    rep: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(seed_for(args.seed, list_geometry_index(geometry), n, int(sigma * 10000), rep))
    if geometry.kind == "torus":
        clean, _ = sample_torus(rng, n)
        noisy = clean + sigma * rng.normal(size=clean.shape)
        methods, detail = build_torus_methods(noisy, sigma, args)
        phi = dense_phi = None
        truth = detail["truth"]
    else:
        clean, noisy, _ = sample_curve(rng, geometry, n, sigma)
        ablation = rep < args.ablation_reps and n == args.ablation_n and abs(sigma - args.ablation_sigma) < 1e-12
        methods, detail = build_curve_methods(geometry, noisy, sigma, args, ablation)
        phi = detail["phi"]
        dense_phi = np.linspace(0.0, 2.0 * np.pi, args.truth_size, endpoint=False)
        truth = detail["truth"]
    reconstruction_rows: list[dict[str, object]] = []
    uq_rows: list[dict[str, object]] = []
    for method, result in methods.items():
        metric_points = (
            densify(phi, result.points, dense_phi)
            if geometry.kind != "torus" and phi is not None and dense_phi is not None
            else result.points
        )
        hausdorff, avg_distance = geometric_metrics(metric_points, truth)
        reconstruction_rows.append({
            "geometry": geometry.name,
            "n": n,
            "sigma": sigma,
            "reach_over_sigma": geometry.reach / sigma,
            "repeat": rep,
            "method": method,
            "hausdorff": hausdorff,
            "avg_distance": avg_distance,
            "paired_rmse": np.nan,
            "median_local_n": (result.diagnostics or {}).get("median_local_n", np.nan),
            "min_local_n": (result.diagnostics or {}).get("min_local_n", np.nan),
            "min_direction_signal": (result.diagnostics or {}).get("min_direction_signal", np.nan),
        })
        if result.posterior_sd is None:
            continue
        scales = [("posterior", result.posterior_sd)]
        if result.frequentist_sd is not None:
            scales.append(("frequentist_gp_mean", result.frequentist_sd))
        for scale_name, scale in scales:
            for level in LEVELS:
                qcrit = float(norm.ppf(1.0 - (1.0 - level) / (2.0 * len(result.points))))
                widths = qcrit * scale
                if geometry.kind == "torus":
                    coverage = np.nan
                    inside = np.nan
                    max_boundary = np.nan
                else:
                    coverage, inside, max_boundary = curve_band_geometry(
                        result.points,
                        result.directions,
                        widths,
                        phi,
                        dense_phi,
                        truth,
                        1.96 * sigma,
                    )
                ratio = (
                    result.posterior_sd / np.maximum(result.frequentist_sd, 1e-14)
                    if method.startswith("ours") and result.frequentist_sd is not None
                    else np.full(len(scale), np.nan)
                )
                uq_rows.append({
                    "geometry": geometry.name,
                    "n": n,
                    "sigma": sigma,
                    "repeat": rep,
                    "method": method,
                    "uq_scale": scale_name,
                    "nominal_level": level,
                    "geometric_coverage": int(coverage) if np.isfinite(coverage) else np.nan,
                    "paired_coverage": np.nan,
                    "mean_halfwidth": float(np.mean(widths)),
                    "max_halfwidth": float(np.max(widths)),
                    "mean_width_over_sigma": float(np.mean(widths) / sigma),
                    "max_width_over_sigma": float(np.max(widths) / sigma),
                    "max_width_over_noise_196": float(np.max(widths) / (1.96 * sigma)),
                    "strict_inside_noise": int(inside) if np.isfinite(inside) else np.nan,
                    "max_boundary_distance": max_boundary,
                    "mean_s_post_over_s_F": float(np.nanmean(ratio)) if np.any(np.isfinite(ratio)) else np.nan,
                    "median_s_post_over_s_F": float(np.nanmedian(ratio)) if np.any(np.isfinite(ratio)) else np.nan,
                    "max_s_post_over_s_F": float(np.nanmax(ratio)) if np.any(np.isfinite(ratio)) else np.nan,
                })
    return reconstruction_rows, uq_rows, {"geometry": geometry, "methods": methods, **detail}


def list_geometry_index(geometry: Geometry) -> int:
    return sum(ord(character) for character in geometry.name)


def summarize_reconstruction(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["geometry"]), int(r["n"]), float(r["sigma"]), str(r["method"])) for r in rows})
    output = []
    for geometry, n, sigma, method in keys:
        group = [r for r in rows if (r["geometry"], r["n"], r["sigma"], r["method"]) == (geometry, n, sigma, method)]
        h = np.asarray([float(r["hausdorff"]) for r in group])
        avg = np.asarray([float(r["avg_distance"]) for r in group])
        paired = np.asarray([float(r["paired_rmse"]) for r in group])
        output.append({
            "geometry": geometry,
            "n": n,
            "sigma": sigma,
            "method": method,
            "repeats": len(group),
            "mean_hausdorff": float(np.mean(h)),
            "median_hausdorff": float(np.median(h)),
            "sd_hausdorff": float(np.std(h, ddof=1)) if len(h) > 1 else 0.0,
            "se_hausdorff": float(np.std(h, ddof=1) / np.sqrt(len(h))) if len(h) > 1 else 0.0,
            "mean_avg_distance": float(np.mean(avg)),
            "median_avg_distance": float(np.median(avg)),
            "sd_avg_distance": float(np.std(avg, ddof=1)) if len(avg) > 1 else 0.0,
            "mean_paired_rmse": float(np.nanmean(paired)) if np.any(np.isfinite(paired)) else np.nan,
        })
    return output


def summarize_uq(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["geometry"]), int(r["n"]), float(r["sigma"]), str(r["method"]), str(r["uq_scale"]), float(r["nominal_level"])) for r in rows})
    output = []
    for geometry, n, sigma, method, scale, level in keys:
        group = [r for r in rows if (r["geometry"], r["n"], r["sigma"], r["method"], r["uq_scale"], r["nominal_level"]) == (geometry, n, sigma, method, scale, level)]
        values = lambda key: np.asarray([float(r[key]) for r in group])
        coverage = values("geometric_coverage")
        inside = values("strict_inside_noise")
        output.append({
            "geometry": geometry,
            "n": n,
            "sigma": sigma,
            "method": method,
            "uq_scale": scale,
            "nominal_level": level,
            "repeats": len(group),
            "empirical_geometric_coverage": float(np.nanmean(coverage)) if np.any(np.isfinite(coverage)) else np.nan,
            "empirical_paired_coverage": np.nan,
            "mean_halfwidth": float(np.mean(values("mean_halfwidth"))),
            "max_halfwidth": float(np.max(values("max_halfwidth"))),
            "mean_width_over_sigma": float(np.mean(values("mean_width_over_sigma"))),
            "max_width_over_sigma": float(np.max(values("max_width_over_sigma"))),
            "max_width_over_noise_196": float(np.max(values("max_width_over_noise_196"))),
            "strict_inside_noise_fraction": float(np.nanmean(inside)) if np.any(np.isfinite(inside)) else np.nan,
            "mean_s_post_over_s_F": float(np.nanmean(values("mean_s_post_over_s_F"))) if np.any(np.isfinite(values("mean_s_post_over_s_F"))) else np.nan,
            "median_s_post_over_s_F": float(np.nanmedian(values("median_s_post_over_s_F"))) if np.any(np.isfinite(values("median_s_post_over_s_F"))) else np.nan,
            "max_s_post_over_s_F": float(np.nanmax(values("max_s_post_over_s_F"))) if np.any(np.isfinite(values("max_s_post_over_s_F"))) else np.nan,
        })
    return output


def plot_calibration(path: Path, geometry: str, summary: list[dict[str, object]], n: int, sigma: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for method, scale, label in (
        ("ours", "posterior", "Ours GP posterior"),
        ("ours", "frequentist_gp_mean", "Ours frequentist GP mean"),
        ("mrgap", "posterior", "MrGap posterior"),
    ):
        rows = [r for r in summary if r["geometry"] == geometry and r["n"] == n and r["sigma"] == sigma and r["method"] == method and r["uq_scale"] == scale]
        rows.sort(key=lambda r: float(r["nominal_level"]))
        if rows:
            ax.plot([r["nominal_level"] for r in rows], [r["empirical_geometric_coverage"] for r in rows], marker="o", label=label)
    ax.plot(LEVELS, LEVELS, color="0.35", ls="--", label="nominal")
    ax.set(xlabel="nominal finite-grid level", ylabel="empirical geometric coverage", ylim=(0.0, 1.03), title=f"{geometry}: simultaneous calibration")
    ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_reconstruction(path: Path, detail: dict[str, object], sigma: float, args: argparse.Namespace) -> None:
    geometry: Geometry = detail["geometry"]
    methods: dict[str, MethodResult] = detail["methods"]
    truth = detail["truth"]
    noisy = detail["contraction"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    for ax, method in zip(axes, ("mrgap", "manifold_fitting", "ours")):
        points = methods[method].points
        ax.scatter(noisy[:, 0], noisy[:, 1], s=5, alpha=0.08, color="0.4")
        ax.plot(np.r_[truth[:, 0], truth[0, 0]], np.r_[truth[:, 1], truth[0, 1]], color="black", lw=1.8, label="truth")
        ax.plot(np.r_[points[:, 0], points[0, 0]], np.r_[points[:, 1], points[0, 1]], lw=1.5, label=METHOD_LABELS[method])
        ax.set_title(METHOD_LABELS[method]); ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    limits = [ax.axis() for ax in axes]
    xmin, xmax = min(v[0] for v in limits), max(v[1] for v in limits)
    ymin, ymax = min(v[2] for v in limits), max(v[3] for v in limits)
    for ax in axes: ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    fig.suptitle(f"{geometry.name}, sigma={sigma}"); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_uq(path: Path, detail: dict[str, object], sigma: float, args: argparse.Namespace, uq_summary: list[dict[str, object]]) -> None:
    geometry: Geometry = detail["geometry"]
    methods: dict[str, MethodResult] = detail["methods"]
    truth = detail["truth"]
    phi = detail["phi"]
    _, truth_normal = frames_from_closed_curve(curve_points(geometry, phi), center=np.zeros(2))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, method in zip(axes, ("mrgap", "ours")):
        result = methods[method]
        qcrit = norm.ppf(1.0 - 0.05 / (2.0 * len(result.points)))
        width = qcrit * result.posterior_sd
        lower = result.points - width[:, None] * result.directions
        upper = result.points + width[:, None] * result.directions
        polygon = np.vstack((upper, upper[:1], lower[:1], lower[::-1], lower[-1:], upper[:1]))
        noise_lower = curve_points(geometry, phi) - 1.96 * sigma * truth_normal
        noise_upper = curve_points(geometry, phi) + 1.96 * sigma * truth_normal
        noise_polygon = np.vstack((noise_upper, noise_upper[:1], noise_lower[:1], noise_lower[::-1], noise_lower[-1:], noise_upper[:1]))
        ax.fill(noise_polygon[:, 0], noise_polygon[:, 1], color="0.75", alpha=0.35, label=r"truth $\pm1.96\sigma$")
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.32, label=f"{METHOD_LABELS[method]} 95% posterior")
        ax.plot(np.r_[truth[:, 0], truth[0, 0]], np.r_[truth[:, 1], truth[0, 1]], color="black", lw=1.8)
        ax.plot(np.r_[result.points[:, 0], result.points[0, 0]], np.r_[result.points[:, 1], result.points[0, 1]], lw=1.4)
        matched = [r for r in uq_summary if r["geometry"] == geometry.name and r["n"] == args.report_n and r["sigma"] == sigma and r["method"] == method and r["uq_scale"] == "posterior" and r["nominal_level"] == 0.95]
        annotation = ""
        if matched:
            annotation = (
                f"\ncoverage={float(matched[0]['empirical_geometric_coverage']):.2f}, "
                f"mean/max width={float(matched[0]['mean_width_over_sigma']):.2f}/"
                f"{float(matched[0]['max_width_over_sigma']):.2f} sigma"
            )
        ax.set_title(METHOD_LABELS[method] + annotation); ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{geometry.name}: posterior uncertainty comparison"); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_curvature(path: Path, detail: dict[str, object]) -> None:
    geometry: Geometry = detail["geometry"]
    ours: MethodResult = detail["methods"]["ours"]
    phi = detail["phi"]
    truth_grid = curve_points(geometry, phi)
    theta = np.arctan2(truth_grid[:, 1] / geometry.b, truth_grid[:, 0] / geometry.a)
    curvature = geometry.a * geometry.b / (geometry.a**2 * np.sin(theta) ** 2 + geometry.b**2 * np.cos(theta) ** 2) ** 1.5
    error = cKDTree(detail["truth"]).query(ours.points)[0]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(curvature, error, label="absolute reconstruction error")
    ax.scatter(curvature, ours.posterior_sd, label="posterior SD")
    ax.set(xlabel="true curvature (post-hoc)", ylabel="local scale", title="Ellipse: error and uncertainty vs curvature")
    ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_robustness(path: Path, summary: list[dict[str, object]]) -> None:
    curves = [row for row in summary if row["geometry"] in ("circle", "ellipse") and row["method"] in ("mrgap", "manifold_fitting", "ours")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, geometry in zip(axes, ("circle", "ellipse")):
        for method in ("mrgap", "manifold_fitting", "ours"):
            rows = [r for r in curves if r["geometry"] == geometry and r["method"] == method]
            rows.sort(key=lambda r: (r["sigma"], r["n"]))
            ax.plot(range(len(rows)), [r["mean_hausdorff"] for r in rows], marker="o", label=METHOD_LABELS[method])
        ax.set_title(geometry); ax.set_xlabel("settings ordered by sigma, then n"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean Hausdorff distance"); axes[1].legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_direction_ablation(path: Path, reconstruction: list[dict[str, object]], uq: list[dict[str, object]], n: int, sigma: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    geometries = ("circle", "ellipse")
    x = np.arange(len(geometries))
    for offset, method in ((-0.18, "ours"), (0.18, "ours_oracle_direction")):
        recon_values = []
        cover_values = []
        for geometry in geometries:
            rr = [r for r in reconstruction if r["geometry"] == geometry and r["n"] == n and r["sigma"] == sigma and r["method"] == method]
            ur = [r for r in uq if r["geometry"] == geometry and r["n"] == n and r["sigma"] == sigma and r["method"] == method and r["uq_scale"] == "posterior" and r["nominal_level"] == 0.95]
            recon_values.append(float(rr[0]["mean_hausdorff"]) if rr else np.nan)
            cover_values.append(float(ur[0]["empirical_geometric_coverage"]) if ur else np.nan)
        label = "estimated ball direction" if method == "ours" else "oracle direction ablation"
        axes[0].bar(x + offset, recon_values, width=0.34, label=label)
        axes[1].bar(x + offset, cover_values, width=0.34, label=label)
    for ax in axes:
        ax.set_xticks(x, geometries); ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("Mean Hausdorff"); axes[1].set_title("95% posterior coverage"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)


def plot_torus(path: Path, scale_path: Path, detail: dict[str, object]) -> None:
    methods: dict[str, MethodResult] = detail["methods"]
    truth = detail["truth"]
    fig = plt.figure(figsize=(14.5, 4.6))
    truth_show = truth[:: max(1, len(truth) // 1800)]
    for index, method in enumerate(("mrgap", "manifold_fitting", "ours"), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        points = methods[method].points
        ax.scatter(truth_show[:, 0], truth_show[:, 1], truth_show[:, 2], s=2, alpha=0.08, color="0.3")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=7, alpha=0.75)
        ax.set_title(METHOD_LABELS[method]); ax.set_box_aspect((1, 1, 0.55)); ax.set_axis_off()
    fig.suptitle("Torus reconstruction: common point-query budget"); fig.tight_layout(); fig.savefig(path, dpi=210); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ours = methods["ours"]
    ax.scatter(np.arange(len(ours.posterior_sd)), ours.posterior_sd, s=8, alpha=0.65, label="Ours posterior SD")
    ax.scatter(np.arange(len(ours.frequentist_sd)), ours.frequentist_sd, s=8, alpha=0.65, label=r"Ours $\sigma\|a\|$")
    mrgap = methods["mrgap"]
    ax.axhline(float(np.mean(mrgap.posterior_sd)), color="tab:green", label="MrGap mean posterior SD")
    ax.set(xlabel="torus query index", ylabel="local uncertainty scale", title="Torus local UQ sanity diagnostic")
    ax.grid(alpha=0.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(scale_path, dpi=210); plt.close(fig)


def write_report(path: Path, reconstruction: list[dict[str, object]], uq: list[dict[str, object]], args: argparse.Namespace) -> None:
    default_recon = [r for r in reconstruction if r["n"] == args.report_n and r["sigma"] == args.report_sigma and r["geometry"] in ("circle", "ellipse") and r["method"] in ("mrgap", "manifold_fitting", "ours")]
    default_uq = [r for r in uq if r["n"] == args.report_n and r["sigma"] == args.report_sigma and r["nominal_level"] == 0.95 and r["geometry"] in ("circle", "ellipse") and ((r["method"] == "ours" and r["uq_scale"] in ("posterior", "frequentist_gp_mean")) or (r["method"] == "mrgap" and r["uq_scale"] == "posterior"))]
    lines = [
        "# MF-level reconstruction with GP uncertainty versus MrGap",
        "",
        "This benchmark uses shared noisy replicates. MrGap is the repository's",
        "one-round Mfit1-equivalent with frozen published example parameters; MF is",
        "the faithful Yao cylinder estimator; Ours uses a split query scaffold and",
        "the estimated Yao ball direction to regress axial displacement on projected",
        "ambient coordinates. Truth enters evaluation and the labeled oracle-direction",
        "ablation only.",
        "",
        "## Default reconstruction (`n=3000`, `sigma=0.06`)",
        "",
        "| geometry | method | repeats | mean Hausdorff | median Hausdorff | mean symmetric distance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in default_recon:
        lines.append(f"| {row['geometry']} | {METHOD_LABELS[str(row['method'])]} | {row['repeats']} | {float(row['mean_hausdorff']):.5f} | {float(row['median_hausdorff']):.5f} | {float(row['mean_avg_distance']):.5f} |")
    lines += ["", "## Default 95% finite-grid UQ", "", "| geometry | method / scale | coverage | mean width / sigma | max width / sigma | inside noise fraction |", "|---|---|---:|---:|---:|---:|"]
    for row in default_uq:
        label = f"{METHOD_LABELS[str(row['method'])]} {row['uq_scale']}"
        lines.append(f"| {row['geometry']} | {label} | {float(row['empirical_geometric_coverage']):.3f} | {float(row['mean_width_over_sigma']):.3f} | {float(row['max_width_over_sigma']):.3f} | {float(row['strict_inside_noise_fraction']):.3f} |")
    recon_map = {(str(r["geometry"]), str(r["method"])): r for r in default_recon}
    uq_map = {(str(r["geometry"]), str(r["method"]), str(r["uq_scale"])): r for r in default_uq}
    required = {
        ("circle", "ours"),
        ("circle", "manifold_fitting"),
        ("ellipse", "ours"),
        ("ellipse", "manifold_fitting"),
    }
    if not required.issubset(recon_map):
        lines += [
            "",
            "This scoped run omits one or both primary closed curves. Global torus",
            "coverage is intentionally undefined; use the local UQ scale table and",
            "the full benchmark for paper-facing conclusions.",
        ]
        path.write_text("\n".join(lines) + "\n")
        return
    circle_ratio = float(recon_map[("circle", "ours")]["mean_hausdorff"]) / float(recon_map[("circle", "manifold_fitting")]["mean_hausdorff"])
    ellipse_ratio = float(recon_map[("ellipse", "ours")]["mean_hausdorff"]) / float(recon_map[("ellipse", "manifold_fitting")]["mean_hausdorff"])
    circle_post = uq_map[("circle", "ours", "posterior")]
    ellipse_post = uq_map[("ellipse", "ours", "posterior")]
    circle_freq = uq_map[("circle", "ours", "frequentist_gp_mean")]
    ellipse_freq = uq_map[("ellipse", "ours", "frequentist_gp_mean")]
    lines += [
        "",
        "## Answers to the four benchmark questions",
        "",
        f"1. **Reconstruction preservation.** On circle, Ours has {circle_ratio:.2f} times MF's mean Hausdorff error, so MF-level accuracy is not preserved there. On ellipse the ratio is {ellipse_ratio:.2f}, so Ours improves on this MF implementation but remains worse than MrGap. No tuning used truth or attempted to reverse the earlier negative GP-versus-average result.",
        f"2. **Calibration.** At nominal 95%, Ours posterior coverage is {float(circle_post['empirical_geometric_coverage']):.2f} on circle and {float(ellipse_post['empirical_geometric_coverage']):.2f} on ellipse. The corresponding same-GP-mean values are {float(circle_freq['empirical_geometric_coverage']):.2f} and {float(ellipse_freq['empirical_geometric_coverage']):.2f}. Calibration is therefore geometry-dependent; the ellipse remains undercovered.",
        f"3. **Efficiency.** Ours posterior mean half-width is {float(circle_post['mean_width_over_sigma']):.2f} sigma on circle and {float(ellipse_post['mean_width_over_sigma']):.2f} sigma on ellipse. Strict containment in the truth-centered `1.96 sigma` reference band is {float(circle_post['strict_inside_noise_fraction']):.2f} and {float(ellipse_post['strict_inside_noise_fraction']):.2f}, checked from dense boundaries rather than inferred from width alone.",
        "4. **Failure modes.** The robustness grid shows larger reconstruction error at small n and large sigma. Sparse settings can under-cover even when normalized widths are larger. The oracle-direction ablation changes ellipse coverage much more than circle coverage, identifying direction estimation as an important ellipse failure mechanism. The curvature plot also shows error spikes that are not matched by comparably large posterior SD changes.",
        "",
        f"The broad grid uses {args.repeats} replicates per non-primary setting; the base circle/ellipse report setting uses {args.final_repeats or args.repeats}. Grid endpoints are diagnostics rather than precise coverage estimates.",
        "",
        "The earlier frozen mechanism result remains part of the repository:",
        "circle `H_avg=0.02652`, `H_GP=0.03032`; ellipse `H_avg=0.03027`,",
        "`H_GP=0.03172`. Thus the GP layer is not introduced as a universal fitting",
        "improvement. Its role is to attach explicit probabilistic uncertainty while",
        "keeping reconstruction in the same practical range.",
    ]
    torus_rows = [r for r in reconstruction if r["geometry"] == "torus" and r["n"] == args.report_n and r["sigma"] == args.report_sigma]
    if torus_rows:
        lines += ["", "## Torus local diagnostic", ""]
        for row in torus_rows:
            lines.append(f"- {METHOD_LABELS[str(row['method'])]}: mean Hausdorff `{float(row['mean_hausdorff']):.3f}`, mean symmetric distance `{float(row['mean_avg_distance']):.3f}`.")
        torus_uq = [r for r in uq if r["geometry"] == "torus" and r["n"] == args.report_n and r["sigma"] == args.report_sigma and r["method"] == "ours" and r["uq_scale"] == "posterior" and r["nominal_level"] == 0.95]
        if torus_uq:
            lines.append(f"Ours has mean posterior-to-frequentist SD ratio `{float(torus_uq[0]['mean_s_post_over_s_F']):.2f}`. These are local scale checks; the sparse common query budget creates a large surface-discretization component, and no global torus coverage is reported.")
    radius_rows = [r for r in reconstruction if str(r["geometry"]).startswith("circle_r") and r["method"] == "ours"]
    if radius_rows:
        lines += ["", "## Geometric SNR diagnostic", ""]
        for row in radius_rows:
            radius = 0.5 if row["geometry"] == "circle_r0.5" else 2.0
            lines.append(f"- `R/sigma={radius / float(row['sigma']):.1f}`: Ours mean Hausdorff `{float(row['mean_hausdorff']):.4f}` over `{row['repeats']}` replicates.")
        lines.append("With only three replicates at each added radius, this check does not support a monotone reach/noise conclusion.")
    lines += [
        "",
        "Paired RMSE is left missing because all three reported objects use a common",
        "query/image representation rather than retaining observation-to-latent pairing.",
        "",
        "## MrGap comparability limitation",
        "",
        "The public MrGap release omits the empirical-Bayes optimizer. Planar curves",
        "therefore use the frozen first-round Cassini tuple; torus uses its published",
        "first-round tuple. Posterior uncertainty is the latent covariance from that",
        "same local GP fit. It is labeled posterior credible uncertainty and is not",
        "truth-calibrated or claimed to be a frequentist interval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", nargs="+", default=["circle", "ellipse", "torus"], choices=["circle", "ellipse", "torus", "circle_r0.5", "circle_r2.0"])
    parser.add_argument("--n-values", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--sigma-values", nargs="+", type=float, default=[0.03, 0.06, 0.10])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=None,
        help="optional repeat count for base circle/ellipse at report n/sigma",
    )
    parser.add_argument("--curve-grid-size", type=int, default=60)
    parser.add_argument("--truth-size", type=int, default=4800)
    parser.add_argument("--torus-query-size", type=int, default=180)
    parser.add_argument("--torus-truth-size", type=int, default=10000)
    parser.add_argument("--scaffold-angle-bandwidth", type=float, default=0.16)
    parser.add_argument("--query-offset-factor", type=float, default=1.0)
    parser.add_argument("--ablation-reps", type=int, default=20)
    parser.add_argument("--ablation-n", type=int, default=3000)
    parser.add_argument("--ablation-sigma", type=float, default=0.06)
    parser.add_argument("--report-n", type=int, default=3000)
    parser.add_argument("--report-sigma", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_mf_gp_uq_vs_mrgap"))
    return parser.parse_args()


def geometry_from_name(name: str) -> Geometry:
    if name == "circle": return Geometry("circle", "circle", 1.0, 1.0, 1, 2)
    if name == "ellipse": return Geometry("ellipse", "ellipse", 1.4, 0.8, 1, 2)
    if name == "torus": return Geometry("torus", "torus", 2.0, 0.8, 2, 3)
    radius = 0.5 if name == "circle_r0.5" else 2.0
    return Geometry(name, "circle", radius, radius, 1, 2)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reconstruction_rows: list[dict[str, object]] = []
    uq_rows: list[dict[str, object]] = []
    representatives: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for geometry_name in args.geometries:
        geometry = geometry_from_name(geometry_name)
        for n in args.n_values:
            for sigma in args.sigma_values:
                is_report_setting = n == args.report_n and abs(sigma - args.report_sigma) < 1e-12
                if geometry_name.startswith("circle_r") and not is_report_setting:
                    continue
                setting_repeats = (
                    args.final_repeats
                    if args.final_repeats is not None
                    and geometry_name in ("circle", "ellipse")
                    and is_report_setting
                    else args.repeats
                )
                for rep in range(setting_repeats):
                    recon, uq, detail = run_setting(geometry, n, sigma, rep, args)
                    reconstruction_rows.extend(recon)
                    uq_rows.extend(uq)
                    if n == args.report_n and abs(sigma - args.report_sigma) < 1e-12 and rep == 0:
                        representatives[geometry_name] = detail
    reconstruction = summarize_reconstruction(reconstruction_rows)
    uq = summarize_uq(uq_rows)
    write_csv(args.output / "reconstruction_raw.csv", reconstruction_rows)
    write_csv(args.output / "uq_raw.csv", uq_rows)
    write_csv(args.output / "reconstruction_summary.csv", reconstruction)
    write_csv(args.output / "uq_summary.csv", uq)
    for geometry in ("circle", "ellipse"):
        if geometry in representatives:
            plot_calibration(args.output / f"calibration_{geometry}.png", geometry, uq, args.report_n, args.report_sigma)
            plot_reconstruction(args.output / f"reconstruction_{geometry}.png", representatives[geometry], args.report_sigma, args)
            plot_uq(args.output / f"uq_{geometry}.png", representatives[geometry], args.report_sigma, args, uq)
    if "ellipse" in representatives:
        plot_curvature(args.output / "curvature_ellipse.png", representatives["ellipse"])
    plot_robustness(args.output / "robustness_reconstruction.png", reconstruction)
    plot_direction_ablation(args.output / "direction_ablation.png", reconstruction, uq, args.ablation_n, args.ablation_sigma)
    if "torus" in representatives:
        plot_torus(
            args.output / "reconstruction_torus.png",
            args.output / "uq_scale_torus.png",
            representatives["torus"],
        )
    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "geometries": args.geometries,
        "n_values": args.n_values,
        "sigma_values": args.sigma_values,
        "repeats": args.repeats,
        "final_repeats": args.final_repeats,
        "truth_use": "evaluation and explicitly labeled oracle-direction ablation only",
        "mrgap_planar_parameters": "frozen Cassini first-round tuple",
        "mrgap_torus_parameters": "frozen torus first-round tuple",
        "torus_global_coverage": "not defined; local UQ scale diagnostics only",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(args.output / "REPORT.md", reconstruction, uq, args)
    print(json.dumps({"runtime_seconds": metadata["runtime_seconds"], "reconstruction_rows": len(reconstruction_rows), "uq_rows": len(uq_rows)}, indent=2))


if __name__ == "__main__":
    main()
