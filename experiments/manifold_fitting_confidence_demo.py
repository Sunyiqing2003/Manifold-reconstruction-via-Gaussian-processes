#!/usr/bin/env python3
"""Circle/ellipse confidence-band demo for a Manifold-Fitting-style refinement.

This experiment is deliberately separate from the published MrGap benchmark.

The final estimator is a closed one-dimensional image curve

    M_hat = G_hat_h(M_tilde),

not merely a collection of independently denoised anchor points.  Two pilot modes
are provided:

* ``oracle``: use the true curve only as the preliminary parameter domain, isolating
  the refinement/UQ layer;
* ``data``: use an independent pilot split, run the repository's Manifold Fitting
  port, and convert the fitted point cloud to a closed radial curve.

The refinement bootstrap is conditional on the pilot curve.  In oracle-pilot mode
we also approximate the population normal displacement with a very large independent
Monte Carlo sample.  That population bias is simulation-only and is used to compare

* a stochastic-only simultaneous tube;
* stochastic radius + oracle bias envelope;
* an oracle bias-corrected center + stochastic radius.

The goal is diagnostic: determine whether undercoverage is driven mainly by sampling
fluctuation or by the population smoothing/EIV bias.  The script does not itself
claim an honest confidence theorem.

Run from the repository root:

    python experiments/manifold_fitting_confidence_demo.py --quick

or, for a more stable run:

    python experiments/manifold_fitting_confidence_demo.py \
        --mc-reps 50 --bootstrap 300 --population-n 150000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.manifold_benchmark import manifold_fitting


@dataclass(frozen=True)
class CurveSpec:
    name: str
    a: float
    b: float

    @property
    def reach_proxy(self) -> float:
        if self.name == "circle":
            return self.a
        # Minimum radius of curvature for x=a cos(t), y=b sin(t), a >= b.
        return self.b * self.b / self.a


CURVES = {
    "circle": CurveSpec("circle", 1.0, 1.0),
    "ellipse": CurveSpec("ellipse", 1.4, 0.8),
}


@dataclass
class BootstrapBand:
    pointwise_sd: np.ndarray
    simultaneous_halfwidth: np.ndarray
    simultaneous_constant_radius: float
    q_studentized: float


def angle_diff(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


def true_curve_polar(spec: CurveSpec, phi: np.ndarray) -> np.ndarray:
    """Curve indexed by polar angle, useful for both circle and centered ellipse."""
    phi = np.asarray(phi)
    c, s = np.cos(phi), np.sin(phi)
    if spec.name == "circle":
        r = np.full_like(phi, spec.a, dtype=float)
    else:
        r = 1.0 / np.sqrt((c / spec.a) ** 2 + (s / spec.b) ** 2)
    return np.column_stack((r * c, r * s))


def parametric_curve(spec: CurveSpec, theta: np.ndarray) -> np.ndarray:
    return np.column_stack((spec.a * np.cos(theta), spec.b * np.sin(theta)))


def sample_arclength_theta(
    rng: np.random.Generator, spec: CurveSpec, n: int
) -> np.ndarray:
    """Rejection sample approximately uniform arclength parameters."""
    if spec.name == "circle":
        return rng.uniform(0.0, 2.0 * np.pi, n)

    accepted: list[np.ndarray] = []
    total = 0
    max_speed = max(spec.a, spec.b)
    while total < n:
        m = max(128, 2 * (n - total))
        theta = rng.uniform(0.0, 2.0 * np.pi, m)
        speed = np.sqrt(
            (spec.a * np.sin(theta)) ** 2 + (spec.b * np.cos(theta)) ** 2
        )
        keep = rng.random(m) < speed / max_speed
        vals = theta[keep]
        accepted.append(vals)
        total += len(vals)
    return np.concatenate(accepted)[:n]


def sample_noisy_curve(
    rng: np.random.Generator, spec: CurveSpec, n: int, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    theta = sample_arclength_theta(rng, spec, n)
    clean = parametric_curve(spec, theta)
    noisy = clean + sigma * rng.normal(size=clean.shape)
    return clean, noisy


def frames_from_closed_curve(
    points: np.ndarray, center: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic finite-difference tangent and consistently outward normal."""
    deriv = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent = deriv / np.maximum(np.linalg.norm(deriv, axis=1, keepdims=True), 1e-14)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    if center is None:
        center = np.mean(points, axis=0)
    radial = points - center
    flip = np.sum(normal * radial, axis=1) < 0
    normal[flip] *= -1.0
    return tangent, normal


def build_data_pilot(
    noisy_pilot: np.ndarray,
    sigma: float,
    phi_grid: np.ndarray,
    mf_multiplier: float,
    angle_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Convert a Manifold Fitting point cloud into a closed radial 1-manifold.

    Circle and ellipse are star-shaped around their center, so this construction is a
    convenient demo device that guarantees the preliminary estimator is one-dimensional
    and closed.
    """
    denoised, diag = manifold_fitting(
        noisy_pilot,
        sigma=sigma,
        bandwidth_multiplier=mf_multiplier,
        average=True,
    )

    center = np.mean(denoised, axis=0)
    rel = denoised - center
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)

    delta = angle_diff(theta[None, :] - phi_grid[:, None])
    weights = np.exp(-0.5 * (delta / angle_bandwidth) ** 2)
    rhat = (weights @ radius) / np.maximum(weights.sum(axis=1), 1e-14)

    pilot = center + rhat[:, None] * np.column_stack(
        (np.cos(phi_grid), np.sin(phi_grid))
    )
    tangent, normal = frames_from_closed_curve(pilot, center=center)
    diagnostics = {
        "pilot_center_x": float(center[0]),
        "pilot_center_y": float(center[1]),
        "pilot_min_radius": float(np.min(rhat)),
        "pilot_max_radius": float(np.max(rhat)),
        "pilot_mf_r": float(diag.get("mf_r", np.nan)),
        "pilot_mf_R": float(diag.get("mf_R", np.nan)),
        "pilot_median_neighborhood": float(
            diag.get("median_local_neighborhood", np.nan)
        ),
    }
    return pilot, tangent, normal, diagnostics


def oracle_pilot(
    spec: CurveSpec, phi_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pilot = true_curve_polar(spec, phi_grid)
    tangent, normal = frames_from_closed_curve(pilot, center=np.zeros(2))
    return pilot, tangent, normal


def refinement_arrays(
    sample: np.ndarray,
    pilot: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    h: float,
    normal_bandwidth: float,
    kernel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normal-only image refinement and reusable weight arrays."""
    centered = sample[None, :, :] - pilot[:, None, :]
    u = np.einsum("gni,gi->gn", centered, tangent)
    v = np.einsum("gni,gi->gn", centered, normal)

    if kernel == "gaussian":
        weights = np.exp(
            -0.5 * (u / h) ** 2 - 0.5 * (v / normal_bandwidth) ** 2
        )
    elif kernel == "biweight":
        wt = np.maximum(1.0 - (u / h) ** 2, 0.0) ** 2
        wn = np.maximum(1.0 - (v / normal_bandwidth) ** 2, 0.0) ** 2
        weights = wt * wn
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    denom = weights.sum(axis=1)
    correction = np.sum(weights * v, axis=1) / np.maximum(denom, 1e-14)
    return correction, weights, v, denom


def bootstrap_refinement_band(
    weights: np.ndarray,
    normal_coordinate: np.ndarray,
    correction: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> BootstrapBand:
    """Conditional nonparametric bootstrap of the refinement pool."""
    n = weights.shape[1]
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_bootstrap)

    denom_star = counts @ weights.T
    numer_star = counts @ (weights * normal_coordinate).T
    corr_star = numer_star / np.maximum(denom_star, 1e-14)
    diffs = corr_star - correction[None, :]

    pointwise_sd = np.std(diffs, axis=0, ddof=1)
    scale = np.maximum(pointwise_sd, 1e-12)
    q_studentized = float(
        np.quantile(np.max(np.abs(diffs) / scale[None, :], axis=1), 1.0 - alpha)
    )
    halfwidth = q_studentized * pointwise_sd
    constant_radius = float(
        np.quantile(np.max(np.abs(diffs), axis=1), 1.0 - alpha)
    )
    return BootstrapBand(
        pointwise_sd=pointwise_sd,
        simultaneous_halfwidth=halfwidth,
        simultaneous_constant_radius=constant_radius,
        q_studentized=q_studentized,
    )


def densify_periodic(
    phi_grid: np.ndarray, values: np.ndarray, dense_phi: np.ndarray
) -> np.ndarray:
    x = np.r_[phi_grid, 2.0 * np.pi]
    if values.ndim == 1:
        y = np.r_[values, values[0]]
    else:
        y = np.vstack((values, values[0]))
    return CubicSpline(x, y, bc_type="periodic", axis=0)(dense_phi)


def geometric_errors(
    fitted_grid: np.ndarray,
    true_dense: np.ndarray,
    phi_grid: np.ndarray,
    dense_phi: np.ndarray,
) -> tuple[float, float]:
    fitted_dense = densify_periodic(phi_grid, fitted_grid, dense_phi)
    true_tree = cKDTree(true_dense)
    fit_tree = cKDTree(fitted_dense)
    true_to_fit = float(np.max(fit_tree.query(true_dense, k=1)[0]))
    fit_to_true = float(np.max(true_tree.query(fitted_dense, k=1)[0]))
    return true_to_fit, max(true_to_fit, fit_to_true)


def variable_tube_contains_truth(
    fitted_grid: np.ndarray,
    width_grid: np.ndarray,
    true_dense: np.ndarray,
    phi_grid: np.ndarray,
    dense_phi: np.ndarray,
) -> bool:
    fitted_dense = densify_periodic(phi_grid, fitted_grid, dense_phi)
    width_dense = np.maximum(densify_periodic(phi_grid, width_grid, dense_phi), 0.0)
    fit_tree = cKDTree(fitted_dense)
    distance_to_fit, idx = fit_tree.query(true_dense, k=1)
    return bool(np.all(distance_to_fit <= width_dense[idx] + 1e-12))


def population_oracle_bias(
    spec: CurveSpec,
    phi_grid: np.ndarray,
    sigma: float,
    h: float,
    normal_bandwidth: float,
    kernel: str,
    n_population: int,
    seed: int,
    chunk_size: int = 4000,
) -> np.ndarray:
    """Monte Carlo population normal correction with the true curve as pilot."""
    pilot, tangent, normal = oracle_pilot(spec, phi_grid)
    rng = np.random.default_rng(seed)
    sum_w = np.zeros(len(phi_grid))
    sum_wv = np.zeros(len(phi_grid))
    remaining = n_population

    while remaining > 0:
        m = min(chunk_size, remaining)
        _, noisy = sample_noisy_curve(rng, spec, m, sigma)
        _, weights, v, _ = refinement_arrays(
            noisy, pilot, tangent, normal, h, normal_bandwidth, kernel
        )
        sum_w += weights.sum(axis=1)
        sum_wv += np.sum(weights * v, axis=1)
        remaining -= m

    return sum_wv / np.maximum(sum_w, 1e-14)


def seed_for(base: int, *values: int) -> int:
    return int(np.random.SeedSequence([base, *values]).generate_state(1)[0])


def plot_representative(
    path: Path,
    noisy_refine: np.ndarray,
    true_dense: np.ndarray,
    pilot: np.ndarray,
    fitted: np.ndarray,
    normal: np.ndarray,
    stochastic_width: np.ndarray,
    bias_width: np.ndarray | None,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(noisy_refine[:, 0], noisy_refine[:, 1], s=8, alpha=0.18, label="refinement sample")
    ax.plot(true_dense[:, 0], true_dense[:, 1], linewidth=2.0, label="true curve")
    ax.plot(pilot[:, 0], pilot[:, 1], linestyle="--", linewidth=1.4, label="pilot")
    ax.plot(fitted[:, 0], fitted[:, 1], linewidth=2.0, label="refined curve")

    lower = fitted - stochastic_width[:, None] * normal
    upper = fitted + stochastic_width[:, None] * normal
    ax.plot(lower[:, 0], lower[:, 1], linewidth=1.0, label="stochastic band")
    ax.plot(upper[:, 0], upper[:, 1], linewidth=1.0)

    if bias_width is not None:
        lower_b = fitted - bias_width[:, None] * normal
        upper_b = fitted + bias_width[:, None] * normal
        ax.plot(lower_b[:, 0], lower_b[:, 1], linestyle=":", linewidth=1.2, label="bias-aware diagnostic")
        ax.plot(upper_b[:, 0], upper_b[:, 1], linestyle=":", linewidth=1.2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (str(r["manifold"]), str(r["pilot_mode"]), float(r["sigma"]), float(r["h_factor"]))
            for r in rows
        }
    )
    out: list[dict[str, object]] = []
    for manifold, pilot_mode, sigma, h_factor in keys:
        g = [
            r for r in rows
            if r["manifold"] == manifold
            and r["pilot_mode"] == pilot_mode
            and float(r["sigma"]) == sigma
            and float(r["h_factor"]) == h_factor
        ]

        def mean_field(name: str) -> float:
            vals = [float(r[name]) for r in g if r[name] != ""]
            return float(np.mean(vals)) if vals else float("nan")

        out.append(
            {
                "manifold": manifold,
                "pilot_mode": pilot_mode,
                "sigma": sigma,
                "h_factor": h_factor,
                "repeats": len(g),
                "naive_constant_coverage": mean_field("naive_constant_covered"),
                "naive_variable_coverage": mean_field("naive_variable_covered"),
                "bias_aware_coverage": mean_field("bias_aware_covered"),
                "bias_corrected_coverage": mean_field("bias_corrected_covered"),
                "mean_directed_error": mean_field("directed_true_to_fit"),
                "mean_hausdorff_error": mean_field("hausdorff_error"),
                "mean_stochastic_radius": mean_field("stochastic_constant_radius"),
                "mean_population_bias_envelope": mean_field("population_bias_envelope"),
                "mean_bias_to_stochastic_ratio": mean_field("bias_to_stochastic_ratio"),
            }
        )
    return out


def plot_coverage_summary(path: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    labels = [
        f"{r['manifold']}\n{r['pilot_mode']}\ns={float(r['sigma']):g}, h/s={float(r['h_factor']):g}"
        for r in summary
    ]
    x = np.arange(len(summary))
    width = 0.23
    naive = np.array([float(r["naive_constant_coverage"]) for r in summary])
    bias_aware = np.array([float(r["bias_aware_coverage"]) for r in summary])
    bias_corrected = np.array([float(r["bias_corrected_coverage"]) for r in summary])

    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(summary)), 5.2))
    ax.bar(x - width, naive, width=width, label="stochastic only")
    mask_a = np.isfinite(bias_aware)
    ax.bar(x[mask_a], bias_aware[mask_a], width=width, label="+ oracle bias envelope")
    mask_c = np.isfinite(bias_corrected)
    ax.bar(x[mask_c] + width, bias_corrected[mask_c], width=width, label="oracle bias-corrected center")
    ax.axhline(0.95, linestyle="--", linewidth=1.0, label="nominal 0.95")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("empirical simultaneous coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bias_vs_stochastic(path: Path, summary: list[dict[str, object]]) -> None:
    oracle = [
        r for r in summary
        if r["pilot_mode"] == "oracle"
        and np.isfinite(float(r["mean_bias_to_stochastic_ratio"]))
    ]
    if not oracle:
        return
    labels = [
        f"{r['manifold']}\ns={float(r['sigma']):g}\nh/s={float(r['h_factor']):g}"
        for r in oracle
    ]
    vals = [float(r["mean_bias_to_stochastic_ratio"]) for r in oracle]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(oracle)), 4.8))
    ax.bar(np.arange(len(vals)), vals)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("population-bias envelope / bootstrap radius")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifolds", nargs="+", choices=sorted(CURVES), default=["circle", "ellipse"])
    parser.add_argument("--pilot-modes", nargs="+", choices=["oracle", "data"], default=["oracle", "data"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--pilot-fraction", type=float, default=0.35)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.05])
    parser.add_argument("--h-factors", nargs="+", type=float, default=[1.5])
    parser.add_argument("--normal-factor", type=float, default=2.5)
    parser.add_argument("--kernel", choices=["gaussian", "biweight"], default="gaussian")
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--dense-grid-size", type=int, default=2400)
    parser.add_argument("--pilot-angle-bandwidth", type=float, default=0.16)
    parser.add_argument("--pilot-mf-multiplier", type=float, default=1.0)
    parser.add_argument("--mc-reps", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--population-n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "manifold_fitting_confidence_demo",
    )
    parser.add_argument("--quick", action="store_true", help="Use a small preflight configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = 600
        args.mc_reps = 8
        args.bootstrap = 80
        args.population_n = 30000
        args.grid_size = 120
        args.dense_grid_size = 1200

    if not (0.0 < args.pilot_fraction < 1.0):
        raise ValueError("--pilot-fraction must lie in (0,1)")
    if any(s <= 0 for s in args.sigmas):
        raise ValueError("this demo assumes positive Gaussian noise")
    if args.bootstrap < 20:
        raise ValueError("use at least 20 bootstrap replicates")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, args.grid_size, endpoint=False)
    dense_phi = np.linspace(0.0, 2.0 * np.pi, args.dense_grid_size, endpoint=False)

    population_bias: dict[tuple[str, float, float], np.ndarray] = {}
    for m_idx, manifold in enumerate(args.manifolds):
        spec = CURVES[manifold]
        for s_idx, sigma in enumerate(args.sigmas):
            for h_idx, h_factor in enumerate(args.h_factors):
                population_bias[(manifold, sigma, h_factor)] = population_oracle_bias(
                    spec=spec,
                    phi_grid=phi_grid,
                    sigma=sigma,
                    h=h_factor * sigma,
                    normal_bandwidth=args.normal_factor * sigma,
                    kernel=args.kernel,
                    n_population=args.population_n,
                    seed=seed_for(args.seed, 900, m_idx, s_idx, h_idx),
                )

    rows: list[dict[str, object]] = []
    representatives_written: set[tuple[str, str, float, float]] = set()

    for m_idx, manifold in enumerate(args.manifolds):
        spec = CURVES[manifold]
        true_grid = true_curve_polar(spec, phi_grid)
        true_dense = true_curve_polar(spec, dense_phi)

        for s_idx, sigma in enumerate(args.sigmas):
            for h_idx, h_factor in enumerate(args.h_factors):
                h = h_factor * sigma
                normal_bw = args.normal_factor * sigma
                pop_bias = population_bias[(manifold, sigma, h_factor)]
                pop_envelope = float(np.max(np.abs(pop_bias)))

                for p_idx, pilot_mode in enumerate(args.pilot_modes):
                    for rep in range(args.mc_reps):
                        rng = np.random.default_rng(
                            seed_for(args.seed, 100, m_idx, s_idx, h_idx, p_idx, rep)
                        )
                        _, noisy = sample_noisy_curve(rng, spec, args.n, sigma)
                        perm = rng.permutation(args.n)
                        n_pilot = max(20, int(round(args.pilot_fraction * args.n)))
                        noisy_pilot = noisy[perm[:n_pilot]]
                        noisy_refine = noisy[perm[n_pilot:]]

                        pilot_diag: dict[str, float] = {}
                        if pilot_mode == "oracle":
                            pilot, tangent, normal = oracle_pilot(spec, phi_grid)
                        else:
                            pilot, tangent, normal, pilot_diag = build_data_pilot(
                                noisy_pilot,
                                sigma,
                                phi_grid,
                                args.pilot_mf_multiplier,
                                args.pilot_angle_bandwidth,
                            )

                        correction, weights, v, denom = refinement_arrays(
                            noisy_refine,
                            pilot,
                            tangent,
                            normal,
                            h,
                            normal_bw,
                            args.kernel,
                        )
                        fitted = pilot + correction[:, None] * normal
                        boot = bootstrap_refinement_band(
                            weights,
                            v,
                            correction,
                            rng,
                            args.bootstrap,
                            args.alpha,
                        )

                        directed, hausdorff = geometric_errors(
                            fitted, true_dense, phi_grid, dense_phi
                        )
                        naive_constant_covered = int(
                            directed <= boot.simultaneous_constant_radius
                        )
                        naive_variable_covered = int(
                            variable_tube_contains_truth(
                                fitted,
                                boot.simultaneous_halfwidth,
                                true_dense,
                                phi_grid,
                                dense_phi,
                            )
                        )

                        normal_error = np.sum((fitted - true_grid) * normal, axis=1)
                        tangent_error = np.sum((fitted - true_grid) * tangent, axis=1)

                        bias_aware_covered: int | str = ""
                        bias_corrected_covered: int | str = ""
                        bias_to_stochastic: float | str = ""
                        bias_width_for_plot: np.ndarray | None = None

                        if pilot_mode == "oracle":
                            bias_aware_radius = boot.simultaneous_constant_radius + pop_envelope
                            bias_aware_covered = int(directed <= bias_aware_radius)
                            bias_to_stochastic = pop_envelope / max(
                                boot.simultaneous_constant_radius, 1e-14
                            )

                            corrected = fitted - pop_bias[:, None] * normal
                            corrected_directed, _ = geometric_errors(
                                corrected, true_dense, phi_grid, dense_phi
                            )
                            bias_corrected_covered = int(
                                corrected_directed <= boot.simultaneous_constant_radius
                            )
                            bias_width_for_plot = boot.simultaneous_halfwidth + np.abs(pop_bias)

                        rows.append(
                            {
                                "manifold": manifold,
                                "pilot_mode": pilot_mode,
                                "repeat": rep,
                                "n": args.n,
                                "n_pilot": len(noisy_pilot),
                                "n_refine": len(noisy_refine),
                                "sigma": sigma,
                                "reach_proxy": spec.reach_proxy,
                                "geometric_snr": spec.reach_proxy / sigma,
                                "h": h,
                                "h_factor": h_factor,
                                "normal_bandwidth": normal_bw,
                                "kernel": args.kernel,
                                "min_effective_weight": float(np.min(denom)),
                                "median_effective_weight": float(np.median(denom)),
                                "mean_abs_normal_error": float(np.mean(np.abs(normal_error))),
                                "max_abs_normal_error": float(np.max(np.abs(normal_error))),
                                "mean_abs_tangent_error": float(np.mean(np.abs(tangent_error))),
                                "directed_true_to_fit": directed,
                                "hausdorff_error": hausdorff,
                                "stochastic_constant_radius": boot.simultaneous_constant_radius,
                                "mean_pointwise_stochastic_halfwidth": float(
                                    np.mean(boot.simultaneous_halfwidth)
                                ),
                                "q_studentized": boot.q_studentized,
                                "naive_constant_covered": naive_constant_covered,
                                "naive_variable_covered": naive_variable_covered,
                                "population_bias_envelope": pop_envelope if pilot_mode == "oracle" else "",
                                "bias_to_stochastic_ratio": bias_to_stochastic,
                                "bias_aware_covered": bias_aware_covered,
                                "bias_corrected_covered": bias_corrected_covered,
                                "pilot_center_x": pilot_diag.get("pilot_center_x", ""),
                                "pilot_center_y": pilot_diag.get("pilot_center_y", ""),
                                "pilot_median_neighborhood": pilot_diag.get(
                                    "pilot_median_neighborhood", ""
                                ),
                            }
                        )

                        rep_key = (manifold, pilot_mode, sigma, h_factor)
                        if rep_key not in representatives_written:
                            representatives_written.add(rep_key)
                            plot_representative(
                                output
                                / (
                                    f"representative_{manifold}_{pilot_mode}"
                                    f"_sigma{sigma:g}_h{h_factor:g}.png"
                                ),
                                noisy_refine,
                                true_dense,
                                pilot,
                                fitted,
                                normal,
                                boot.simultaneous_halfwidth,
                                bias_width_for_plot,
                                title=(
                                    f"{manifold}, pilot={pilot_mode}, "
                                    f"sigma={sigma:g}, h/sigma={h_factor:g}"
                                ),
                            )

    summary = summarize_rows(rows)
    write_csv(output / "raw_metrics.csv", rows)
    write_csv(output / "summary.csv", summary)
    plot_coverage_summary(output / "coverage_summary.png", summary)
    plot_bias_vs_stochastic(output / "bias_vs_stochastic.png", summary)

    metadata = {
        "purpose": "Diagnostic simultaneous confidence-band experiment for a Manifold-Fitting-style image refinement.",
        "manifolds": args.manifolds,
        "pilot_modes": args.pilot_modes,
        "n": args.n,
        "pilot_fraction": args.pilot_fraction,
        "sigmas": args.sigmas,
        "h_factors": args.h_factors,
        "normal_factor": args.normal_factor,
        "kernel": args.kernel,
        "grid_size": args.grid_size,
        "dense_grid_size": args.dense_grid_size,
        "pilot_angle_bandwidth": args.pilot_angle_bandwidth,
        "pilot_mf_multiplier": args.pilot_mf_multiplier,
        "mc_reps": args.mc_reps,
        "bootstrap": args.bootstrap,
        "alpha": args.alpha,
        "population_n": args.population_n,
        "seed": args.seed,
        "interpretation": [
            "The bootstrap is conditional on the preliminary manifold. Data-pilot coverage therefore exposes unaccounted pilot-stage uncertainty.",
            "Oracle population-bias quantities use simulation truth and a large independent Monte Carlo sample. They are diagnostic only.",
            "Coverage is approximated on a dense polar-angle grid. No asymptotic coverage claim is made by this script.",
            "The final estimator is a closed one-dimensional image curve, not merely a collection of independently denoised anchor points.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote results to {output}")
    for r in summary:
        bias_aware = float(r["bias_aware_coverage"])
        print(
            r["manifold"],
            r["pilot_mode"],
            f"sigma={float(r['sigma']):g}",
            f"h/sigma={float(r['h_factor']):g}",
            f"naive={float(r['naive_constant_coverage']):.3f}",
            f"bias-aware={bias_aware:.3f}" if np.isfinite(bias_aware) else "bias-aware=NA",
        )


if __name__ == "__main__":
    main()
