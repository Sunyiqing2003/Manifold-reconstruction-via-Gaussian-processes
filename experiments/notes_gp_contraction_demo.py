#!/usr/bin/env python3
"""Notes-faithful GP replacement of the Manifold Fitting cylinder average.

This script implements the construction in ``Manifold_fitting_notes`` rather than
first running the full Manifold Fitting estimator and then fitting a GP residual.

For each query point z on a coarse data-driven d-dimensional scaffold Gamma:

1. BALL STEP: estimate F(z)-z from nearby noisy observations and normalize it to
   obtain the rank-one contraction direction u_z.
2. CYLINDER STEP: keep observations with a narrow transverse coordinate q_i and a
   longer axial coordinate s_i.
3. GP REPLACEMENT: fit

       s_i = f_z(q_i) + epsilon_i

   using the projected ambient vectors q_i=(I-u_z u_z^T)(Y_i-z), and predict at
   q=0.  No tangent basis is constructed.
4. OUTPUT:

       G_GP(z) = z + u_z * mu_z(0).

For comparison, the original scalar contraction average is

       G_avg(z) = z + u_z * mean_i s_i.

The coarse scaffold is used only to supply a d-dimensional indexing domain and to
place query points in an O(sigma) annulus where the rank-one direction is stable.
Its tangent/normal coordinates are NOT used by the local GP.

The experiment reports circle/ellipse Hausdorff error and a finite-grid simultaneous
GP posterior band.  This is a diagnostic experiment, not a coverage theorem.
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
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.manifold_fitting_confidence_demo import (
    CURVES,
    densify_periodic,
    frames_from_closed_curve,
    geometric_errors,
    sample_noisy_curve,
    seed_for,
    true_curve_polar,
)


@dataclass
class LocalResult:
    direction: np.ndarray
    gp_point: np.ndarray
    avg_point: np.ndarray
    posterior_sd: float
    frequentist_sd: float
    ball_n: int
    cylinder_n: int
    direction_signal: float


def angle_diff(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


def coarse_radial_scaffold(
    sample: np.ndarray,
    phi_grid: np.ndarray,
    angle_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deliberately coarse closed 1-manifold from one sample split.

    This is not the Yao contraction estimator.  It exists only to provide the
    d-dimensional query domain Gamma needed for an image-set experiment.
    """
    center = np.mean(sample, axis=0)
    rel = sample - center
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)
    delta = angle_diff(theta[None, :] - phi_grid[:, None])
    weights = np.exp(-0.5 * (delta / angle_bandwidth) ** 2)
    rhat = (weights @ radius) / np.maximum(weights.sum(axis=1), 1e-14)
    scaffold = center + rhat[:, None] * np.column_stack(
        (np.cos(phi_grid), np.sin(phi_grid))
    )
    tangent, normal_vec = frames_from_closed_curve(scaffold, center=center)
    return scaffold, tangent, normal_vec


def universal_gp_origin(
    q: np.ndarray,
    s: np.ndarray,
    sigma: float,
    amplitude: float,
    length_scale: float,
) -> tuple[float, float, float]:
    """Universal-kriging prediction at q=0 using projected ambient distances."""
    m = len(s)
    if m < 3:
        return float(np.mean(s)), float(sigma), float(sigma / np.sqrt(max(m, 1)))

    sqdist = distance.cdist(q, q, "sqeuclidean")
    kernel = amplitude * np.exp(-0.5 * sqdist / (length_scale**2))
    covariance = kernel + sigma**2 * np.eye(m)
    jitter = 1e-10 * max(1.0, amplitude, sigma**2)
    try:
        factor = cho_factor(covariance + jitter * np.eye(m), lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        factor = cho_factor(
            covariance + 1e-7 * max(1.0, amplitude, sigma**2) * np.eye(m),
            lower=True,
            check_finite=False,
        )

    ones = np.ones(m)
    cinv_ones = cho_solve(factor, ones, check_finite=False)
    denom = float(ones @ cinv_ones)
    base = cinv_ones / max(denom, 1e-14)

    qnorm2 = np.sum(q * q, axis=1)
    k0 = amplitude * np.exp(-0.5 * qnorm2 / (length_scale**2))
    cinv_k0 = cho_solve(factor, k0, check_finite=False)
    smoother = base + cinv_k0 - base * float(ones @ cinv_k0)
    mean = float(smoother @ s)

    posterior_base = amplitude - float(k0 @ cinv_k0)
    mean_correction = (1.0 - float(ones @ cinv_k0)) ** 2 / max(denom, 1e-14)
    posterior_var = max(posterior_base + mean_correction, 0.0)
    freq_var = sigma**2 * float(smoother @ smoother)
    return mean, float(np.sqrt(posterior_var)), float(np.sqrt(max(freq_var, 0.0)))


def contract_one(
    z: np.ndarray,
    sample: np.ndarray,
    tree: cKDTree,
    ball_radius: float,
    transverse_radius: float,
    axial_radius: float,
    sigma: float,
    amplitude: float,
    length_scale: float,
    min_points: int,
    max_points: int,
) -> LocalResult:
    # Step 1: ball mean F(z), hence u_z=(F(z)-z)/||F(z)-z||.
    ball_idx = np.asarray(tree.query_ball_point(z, ball_radius), dtype=int)
    if len(ball_idx) < min_points:
        ball_idx = np.atleast_1d(tree.query(z, k=min(min_points, len(sample)))[1])
    local_mean = np.mean(sample[ball_idx], axis=0)
    direction_raw = local_mean - z
    signal = float(np.linalg.norm(direction_raw))
    if signal <= 1e-12:
        # Fallback: point toward the global sample centre.  This should be rare when
        # the query scaffold is offset by O(sigma).
        direction_raw = np.mean(sample, axis=0) - z
        signal = float(np.linalg.norm(direction_raw))
    direction = direction_raw / max(signal, 1e-14)

    # Step 2: Yao-style cylinder in rank-one coordinates.
    search_radius = math.sqrt(transverse_radius**2 + axial_radius**2)
    cand = np.asarray(tree.query_ball_point(z, search_radius), dtype=int)
    centered = sample[cand] - z
    s_all = centered @ direction
    q_all = centered - s_all[:, None] * direction[None, :]
    qnorm = np.linalg.norm(q_all, axis=1)
    keep = (np.abs(s_all) <= axial_radius) & (qnorm <= transverse_radius)
    idx = cand[keep]
    s = s_all[keep]
    q = q_all[keep]

    if len(idx) < min_points:
        # Preserve the same coordinates but use nearest transverse/cylinder score as
        # a numerical fallback.  Diagnostics record the actual cylinder count.
        score = (qnorm / max(transverse_radius, 1e-14)) ** 2 + (
            s_all / max(axial_radius, 1e-14)
        ) ** 2
        order = np.argsort(score)[: min(min_points, len(cand))]
        s = s_all[order]
        q = q_all[order]

    if len(s) > max_points:
        score = np.sum(q * q, axis=1) / max(transverse_radius**2, 1e-14) + (
            s / max(axial_radius, 1e-14)
        ) ** 2
        order = np.argsort(score)[:max_points]
        s = s[order]
        q = q[order]

    gp_mean, post_sd, freq_sd = universal_gp_origin(
        q=q,
        s=s,
        sigma=sigma,
        amplitude=amplitude,
        length_scale=length_scale,
    )
    avg_mean = float(np.mean(s))
    return LocalResult(
        direction=direction,
        gp_point=z + gp_mean * direction,
        avg_point=z + avg_mean * direction,
        posterior_sd=post_sd,
        frequentist_sd=freq_sd,
        ball_n=len(ball_idx),
        cylinder_n=int(np.sum(keep)),
        direction_signal=signal,
    )


def fit_notes_map(
    queries: np.ndarray,
    sample: np.ndarray,
    sigma: float,
    ball_radius: float,
    transverse_radius: float,
    axial_radius: float,
    amplitude_factor: float,
    length_factor: float,
    min_points: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    tree = cKDTree(sample)
    amplitude = amplitude_factor * sigma**2
    length_scale = length_factor * transverse_radius

    gp = np.empty_like(queries)
    avg = np.empty_like(queries)
    direction = np.empty_like(queries)
    post_sd = np.empty(len(queries))
    freq_sd = np.empty(len(queries))
    ball_n = np.empty(len(queries))
    cyl_n = np.empty(len(queries))
    signal = np.empty(len(queries))

    for j, z in enumerate(queries):
        result = contract_one(
            z=z,
            sample=sample,
            tree=tree,
            ball_radius=ball_radius,
            transverse_radius=transverse_radius,
            axial_radius=axial_radius,
            sigma=sigma,
            amplitude=amplitude,
            length_scale=length_scale,
            min_points=min_points,
            max_points=max_points,
        )
        gp[j] = result.gp_point
        avg[j] = result.avg_point
        direction[j] = result.direction
        post_sd[j] = result.posterior_sd
        freq_sd[j] = result.frequentist_sd
        ball_n[j] = result.ball_n
        cyl_n[j] = result.cylinder_n
        signal[j] = result.direction_signal

    diagnostics = {
        "median_ball_n": float(np.median(ball_n)),
        "median_cylinder_n": float(np.median(cyl_n)),
        "min_cylinder_n": float(np.min(cyl_n)),
        "median_direction_signal": float(np.median(signal)),
        "min_direction_signal": float(np.min(signal)),
    }
    return gp, avg, direction, post_sd, freq_sd, diagnostics


def closed_polygon(lower: np.ndarray, upper: np.ndarray, phi_grid: np.ndarray, dense_phi: np.ndarray) -> np.ndarray:
    lo = densify_periodic(phi_grid, lower, dense_phi)
    hi = densify_periodic(phi_grid, upper, dense_phi)
    return np.vstack((hi, lo[::-1], hi[:1]))


def polygon_covers_truth(polygon: np.ndarray, true_dense: np.ndarray) -> bool:
    path = MplPath(polygon)
    return bool(np.all(path.contains_points(true_dense, radius=1e-10)))


def boundary_inside_noise(
    lower: np.ndarray,
    upper: np.ndarray,
    phi_grid: np.ndarray,
    dense_phi: np.ndarray,
    true_dense: np.ndarray,
    noise_halfwidth: float,
) -> tuple[bool, float]:
    lo = densify_periodic(phi_grid, lower, dense_phi)
    hi = densify_periodic(phi_grid, upper, dense_phi)
    tree = cKDTree(true_dense)
    mx = float(max(np.max(tree.query(lo)[0]), np.max(tree.query(hi)[0])))
    return bool(mx < noise_halfwidth), mx


def run_case(manifold: str, rep: int, args: argparse.Namespace, plot_path: Path | None) -> dict[str, object]:
    spec = CURVES[manifold]
    phi = np.linspace(0.0, 2.0 * np.pi, args.grid_size, endpoint=False)
    dense_phi = np.linspace(0.0, 2.0 * np.pi, args.dense_grid_size, endpoint=False)
    truth = true_curve_polar(spec, dense_phi)
    truth_grid = true_curve_polar(spec, phi)
    _, true_tangent, true_normal = frames_from_closed_curve(truth_grid, center=np.zeros(2))

    rng = np.random.default_rng(seed_for(args.seed, 321, list(CURVES).index(manifold), rep))
    _, noisy = sample_noisy_curve(rng, spec, args.n, args.sigma)
    perm = rng.permutation(args.n)
    n_scaffold = max(50, int(round(args.scaffold_fraction * args.n)))
    scaffold_sample = noisy[perm[:n_scaffold]]
    contraction_sample = noisy[perm[n_scaffold:]]

    scaffold, _, scaffold_normal = coarse_radial_scaffold(
        scaffold_sample, phi, args.scaffold_angle_bandwidth
    )
    # Query annulus: the scaffold is only an indexing manifold.  We move it outward
    # by O(sigma) so the ball step has a non-degenerate contraction signal.
    queries = scaffold + args.query_offset_factor * args.sigma * scaffold_normal

    n_eff = len(contraction_sample)
    base_r = args.bandwidth_multiplier * 5.0 * args.sigma / np.log10(max(n_eff, 10))
    base_R = (
        args.bandwidth_multiplier
        * 10.0
        * args.sigma
        * np.sqrt(np.log(1.0 / args.sigma))
        / np.log10(max(n_eff, 10))
    )
    ball_radius = args.ball_factor * base_r
    transverse_radius = args.transverse_factor * base_r
    axial_radius = args.axial_factor * base_R

    gp, avg, direction, post_sd, freq_sd, diag = fit_notes_map(
        queries=queries,
        sample=contraction_sample,
        sigma=args.sigma,
        ball_radius=ball_radius,
        transverse_radius=transverse_radius,
        axial_radius=axial_radius,
        amplitude_factor=args.amplitude_factor,
        length_factor=args.length_factor,
        min_points=args.min_points,
        max_points=args.max_points,
    )

    scaffold_err = geometric_errors(scaffold, truth, phi, dense_phi)[1]
    query_err = geometric_errors(queries, truth, phi, dense_phi)[1]
    avg_err = geometric_errors(avg, truth, phi, dense_phi)[1]
    gp_err = geometric_errors(gp, truth, phi, dense_phi)[1]

    qcrit = float(norm.ppf(1.0 - args.alpha / (2.0 * args.grid_size)))
    gp_width = qcrit * post_sd
    lower = gp - gp_width[:, None] * direction
    upper = gp + gp_width[:, None] * direction
    polygon = closed_polygon(lower, upper, phi, dense_phi)
    covers = polygon_covers_truth(polygon, truth)

    noise_halfwidth = args.noise_multiplier * args.sigma
    inside_noise, max_boundary = boundary_inside_noise(
        lower, upper, phi, dense_phi, truth, noise_halfwidth
    )

    row = {
        "manifold": manifold,
        "repeat": rep,
        "n": args.n,
        "sigma": args.sigma,
        "n_scaffold": n_scaffold,
        "n_contraction": n_eff,
        "query_offset": args.query_offset_factor * args.sigma,
        "ball_radius": ball_radius,
        "transverse_radius": transverse_radius,
        "axial_radius": axial_radius,
        "scaffold_hausdorff": scaffold_err,
        "query_hausdorff": query_err,
        "average_contraction_hausdorff": avg_err,
        "gp_contraction_hausdorff": gp_err,
        "gp_better_than_average": int(gp_err <= avg_err),
        "gp_better_than_scaffold": int(gp_err <= scaffold_err),
        "gp_tube_covers_truth": int(covers),
        "gp_tube_inside_noise": int(inside_noise),
        "noise_halfwidth": noise_halfwidth,
        "max_gp_halfwidth": float(np.max(gp_width)),
        "mean_gp_halfwidth": float(np.mean(gp_width)),
        "max_gp_width_over_noise": float(np.max(gp_width) / noise_halfwidth),
        "max_boundary_distance_to_truth": max_boundary,
        "strict_containment_margin": float(noise_halfwidth - max_boundary),
        "mean_post_sd": float(np.mean(post_sd)),
        "mean_freq_sd": float(np.mean(freq_sd)),
        **diag,
    }

    if plot_path is not None:
        noise_lo = truth_grid - noise_halfwidth * true_normal
        noise_hi = truth_grid + noise_halfwidth * true_normal
        noise_poly = closed_polygon(noise_lo, noise_hi, phi, dense_phi)

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
        ax = axes[0]
        ax.scatter(contraction_sample[:, 0], contraction_sample[:, 1], s=6, alpha=0.10, label="noisy data")
        ax.plot(truth[:, 0], truth[:, 1], linewidth=2.1, label="truth")
        ax.plot(scaffold[:, 0], scaffold[:, 1], linestyle="--", linewidth=1.2, label="coarse index scaffold")
        ax.plot(queries[:, 0], queries[:, 1], linestyle=":", linewidth=1.1, label="query annulus Gamma")
        ax.plot(avg[:, 0], avg[:, 1], linewidth=1.5, label="cylinder average")
        ax.plot(gp[:, 0], gp[:, 1], linewidth=1.7, label="notes GP contraction")
        stride = max(1, args.grid_size // 20)
        for j in range(0, args.grid_size, stride):
            ax.plot(
                [queries[j, 0], queries[j, 0] + 0.8 * args.sigma * direction[j, 0]],
                [queries[j, 1], queries[j, 1] + 0.8 * args.sigma * direction[j, 1]],
                linewidth=0.7,
                alpha=0.5,
            )
        ax.set_title(
            f"Point estimation: avg H={avg_err:.4f}, GP H={gp_err:.4f}\n"
            "ball direction -> cylinder -> GP q->s at q=0"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)

        ax = axes[1]
        ax.fill(noise_poly[:, 0], noise_poly[:, 1], alpha=0.18, label=r"reference $\pm1.96\sigma$ band")
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.32, label="simultaneous GP contraction band")
        ax.plot(truth[:, 0], truth[:, 1], linewidth=2.1, label="truth")
        ax.plot(gp[:, 0], gp[:, 1], linewidth=1.7, label="GP image curve")
        ax.set_title(
            f"covered={covers}; inside noise={inside_noise}; "
            f"max width/noise={row['max_gp_width_over_noise']:.2f}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)

        fig.suptitle(f"{manifold}: notes-faithful GP replacement", fontsize=13)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=210)
        plt.close(fig)

    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for manifold in sorted({str(r["manifold"]) for r in rows}):
        g = [r for r in rows if r["manifold"] == manifold]

        def mean(name: str) -> float:
            return float(np.mean([float(r[name]) for r in g]))

        out.append(
            {
                "manifold": manifold,
                "repeats": len(g),
                "mean_scaffold_hausdorff": mean("scaffold_hausdorff"),
                "mean_average_hausdorff": mean("average_contraction_hausdorff"),
                "mean_gp_hausdorff": mean("gp_contraction_hausdorff"),
                "gp_better_than_average_fraction": mean("gp_better_than_average"),
                "gp_better_than_scaffold_fraction": mean("gp_better_than_scaffold"),
                "gp_tube_coverage": mean("gp_tube_covers_truth"),
                "gp_tube_inside_noise_fraction": mean("gp_tube_inside_noise"),
                "mean_max_width_over_noise": mean("max_gp_width_over_noise"),
                "mean_strict_containment_margin": mean("strict_containment_margin"),
                "median_direction_signal": mean("median_direction_signal"),
                "median_cylinder_n": mean("median_cylinder_n"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifolds", nargs="+", choices=sorted(CURVES), default=["circle", "ellipse"])
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--sigma", type=float, default=0.06)
    p.add_argument("--scaffold-fraction", type=float, default=0.30)
    p.add_argument("--scaffold-angle-bandwidth", type=float, default=0.22)
    p.add_argument("--query-offset-factor", type=float, default=1.0)
    p.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    p.add_argument("--ball-factor", type=float, default=2.0)
    p.add_argument("--transverse-factor", type=float, default=1.0)
    p.add_argument("--axial-factor", type=float, default=1.0)
    p.add_argument("--amplitude-factor", type=float, default=1.0)
    p.add_argument("--length-factor", type=float, default=1.0)
    p.add_argument("--min-points", type=int, default=25)
    p.add_argument("--max-points", type=int, default=140)
    p.add_argument("--grid-size", type=int, default=72)
    p.add_argument("--dense-grid-size", type=int, default=2400)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--noise-multiplier", type=float, default=1.96)
    p.add_argument("--mc-reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "notes_gp_contraction_demo",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for manifold in args.manifolds:
        for rep in range(args.mc_reps):
            plot_path = args.output / f"{manifold}_notes_gp_contraction.png" if rep == 0 else None
            rows.append(run_case(manifold, rep, args, plot_path))

    summary = summarize(rows)
    write_csv(args.output / "raw_metrics.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "ball direction + rank-one cylinder + GP regression s=f(q) evaluated at q=0",
                "scaffold_role": "coarse d-dimensional query/index domain only; it is not used as the GP coordinate system or final MF estimator",
                "uq": "latent local GP posterior sd with finite-grid Bonferroni simultaneous calibration",
                "caveats": [
                    "The query scaffold is offset by O(sigma) so the rank-one ball direction is non-degenerate.",
                    "The GP posterior band is conditional on the estimated ball directions and cylinder selections.",
                    "The local GP target under noisy q remains an errors-in-variables target; this script does not deconvolve tangential noise.",
                    "The closed-image and band checks are circle/ellipse diagnostics, not a general manifold theorem.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for r in summary:
        print(
            r["manifold"],
            f"scaffold_H={float(r['mean_scaffold_hausdorff']):.4f}",
            f"avg_H={float(r['mean_average_hausdorff']):.4f}",
            f"gp_H={float(r['mean_gp_hausdorff']):.4f}",
            f"GP<avg={float(r['gp_better_than_average_fraction']):.3f}",
            f"coverage={float(r['gp_tube_coverage']):.3f}",
            f"inside_noise={float(r['gp_tube_inside_noise_fraction']):.3f}",
        )


if __name__ == "__main__":
    main()
