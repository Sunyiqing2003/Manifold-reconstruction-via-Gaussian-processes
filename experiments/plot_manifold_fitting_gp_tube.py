#!/usr/bin/env python3
"""Plot the local-GP manifold confidence tube inside the observation-noise band.

This is a visualization-only companion to ``manifold_fitting_gp_confidence_demo.py``.
For circle and ellipse, it runs one representative simulation, constructs a closed
preliminary curve, applies local GP normal refinement, and plots:

* noisy observations;
* the true manifold;
* the GP-refined manifold;
* a simultaneous GP-posterior tube around the refined manifold;
* a reference Gaussian observation-noise band of half-width ``noise_multiplier*sigma``
  around the true manifold.

The code also checks numerically on a dense grid whether (i) the GP tube covers the
true manifold and (ii) the two GP tube boundaries stay strictly inside the reference
noise band.  The latter is a visualization diagnostic, not a theorem.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

from experiments.manifold_fitting_confidence_demo import (
    CURVES,
    build_data_pilot,
    densify_periodic,
    oracle_pilot,
    sample_noisy_curve,
    seed_for,
    true_curve_polar,
    variable_tube_contains_truth,
)
from experiments.manifold_fitting_gp_confidence_demo import fit_gp_curve


def closed_band_polygon(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.vstack((upper, lower[::-1], upper[:1]))


def boundary_inside_noise_band(
    lower: np.ndarray,
    upper: np.ndarray,
    true_dense: np.ndarray,
    phi_grid: np.ndarray,
    dense_phi: np.ndarray,
    noise_halfwidth: float,
) -> tuple[bool, float]:
    lower_dense = densify_periodic(phi_grid, lower, dense_phi)
    upper_dense = densify_periodic(phi_grid, upper, dense_phi)
    tree = cKDTree(true_dense)
    d_lower = tree.query(lower_dense, k=1)[0]
    d_upper = tree.query(upper_dense, k=1)[0]
    max_distance = float(max(np.max(d_lower), np.max(d_upper)))
    return bool(max_distance < noise_halfwidth), max_distance


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(
    outpath: Path,
    manifold: str,
    pilot_mode: str,
    n: int,
    sigma: float,
    h_factor: float,
    normal_factor: float,
    amplitude_factor: float,
    length_factor: float,
    max_points: int,
    min_points: int,
    weight_floor: float,
    grid_size: int,
    dense_grid_size: int,
    pilot_fraction: float,
    pilot_angle_bandwidth: float,
    pilot_mf_multiplier: float,
    alpha: float,
    noise_multiplier: float,
    seed: int,
) -> dict[str, object]:
    spec = CURVES[manifold]
    phi_grid = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    dense_phi = np.linspace(0.0, 2.0 * np.pi, dense_grid_size, endpoint=False)
    true_grid = true_curve_polar(spec, phi_grid)
    true_dense = true_curve_polar(spec, dense_phi)
    _, true_tangent, true_normal = oracle_pilot(spec, phi_grid)

    rng = np.random.default_rng(seed)
    _, noisy = sample_noisy_curve(rng, spec, n, sigma)
    perm = rng.permutation(n)
    n_pilot = max(30, int(round(pilot_fraction * n)))
    noisy_pilot = noisy[perm[:n_pilot]]
    noisy_refine = noisy[perm[n_pilot:]]

    if pilot_mode == "oracle":
        pilot, tangent, normal_vec = oracle_pilot(spec, phi_grid)
    else:
        pilot, tangent, normal_vec, _ = build_data_pilot(
            noisy_pilot,
            sigma,
            phi_grid,
            pilot_mf_multiplier,
            pilot_angle_bandwidth,
        )

    h = h_factor * sigma
    fitted, post_sd, freq_sd, diag = fit_gp_curve(
        noisy_refine,
        pilot,
        tangent,
        normal_vec,
        sigma,
        h,
        normal_factor * sigma,
        amplitude_factor,
        length_factor,
        max_points,
        min_points,
        weight_floor,
    )

    # Finite-grid simultaneous GP credible band.  We deliberately use the same
    # conservative Bonferroni calibration as the main GP demo.
    q_gp = float(norm.ppf(1.0 - alpha / (2.0 * grid_size)))
    gp_width = q_gp * post_sd
    gp_lower = fitted - gp_width[:, None] * normal_vec
    gp_upper = fitted + gp_width[:, None] * normal_vec

    noise_halfwidth = noise_multiplier * sigma
    noise_lower = true_grid - noise_halfwidth * true_normal
    noise_upper = true_grid + noise_halfwidth * true_normal

    gp_covers_truth = variable_tube_contains_truth(
        fitted,
        gp_width,
        true_dense,
        phi_grid,
        dense_phi,
    )
    gp_inside_noise, max_gp_boundary_distance = boundary_inside_noise_band(
        gp_lower,
        gp_upper,
        true_dense,
        phi_grid,
        dense_phi,
        noise_halfwidth,
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    noise_poly = closed_band_polygon(noise_lower, noise_upper)
    gp_poly = closed_band_polygon(gp_lower, gp_upper)

    ax.fill(
        noise_poly[:, 0],
        noise_poly[:, 1],
        alpha=0.18,
        label=rf"reference noise band: $\pm {noise_multiplier:g}\sigma$",
    )
    ax.fill(
        gp_poly[:, 0],
        gp_poly[:, 1],
        alpha=0.28,
        label="simultaneous GP posterior tube",
    )
    ax.scatter(
        noisy_refine[:, 0],
        noisy_refine[:, 1],
        s=7,
        alpha=0.12,
        label="noisy observations",
    )
    ax.plot(true_dense[:, 0], true_dense[:, 1], linewidth=2.2, label="true manifold")
    ax.plot(fitted[:, 0], fitted[:, 1], linewidth=1.8, label="GP-refined manifold")
    if pilot_mode == "data":
        ax.plot(pilot[:, 0], pilot[:, 1], linestyle="--", linewidth=1.0, label="MF pilot")

    status = (
        f"truth covered={gp_covers_truth}; "
        f"GP tube inside noise band={gp_inside_noise}"
    )
    ax.set_title(
        f"{manifold}: Manifold Fitting + GP tube\n"
        f"pilot={pilot_mode}, n={n}, sigma={sigma:g}, h/sigma={h_factor:g}\n{status}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return {
        "manifold": manifold,
        "pilot_mode": pilot_mode,
        "n": n,
        "n_pilot": n_pilot,
        "n_refine": len(noisy_refine),
        "sigma": sigma,
        "h_factor": h_factor,
        "noise_multiplier": noise_multiplier,
        "noise_halfwidth": noise_halfwidth,
        "gp_q": q_gp,
        "max_gp_halfwidth": float(np.max(gp_width)),
        "mean_gp_halfwidth": float(np.mean(gp_width)),
        "max_freq_halfwidth": float(np.max(q_gp * freq_sd)),
        "gp_covers_truth": int(gp_covers_truth),
        "gp_tube_inside_noise_band": int(gp_inside_noise),
        "max_gp_boundary_distance_to_truth": max_gp_boundary_distance,
        "strict_containment_margin": float(noise_halfwidth - max_gp_boundary_distance),
        "median_local_n": diag["median_local_n"],
        "median_effective_n": diag["median_effective_n"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifolds", nargs="+", choices=sorted(CURVES), default=["circle", "ellipse"])
    p.add_argument("--pilot-mode", choices=["oracle", "data"], default="data")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--sigma", type=float, default=0.06)
    p.add_argument("--h-factor", type=float, default=1.5)
    p.add_argument("--normal-factor", type=float, default=2.5)
    p.add_argument("--amplitude-factor", type=float, default=1.0)
    p.add_argument("--length-factor", type=float, default=1.0)
    p.add_argument("--max-points", type=int, default=140)
    p.add_argument("--min-points", type=int, default=30)
    p.add_argument("--weight-floor", type=float, default=1e-3)
    p.add_argument("--grid-size", type=int, default=60)
    p.add_argument("--dense-grid-size", type=int, default=2400)
    p.add_argument("--pilot-fraction", type=float, default=0.35)
    p.add_argument("--pilot-angle-bandwidth", type=float, default=0.16)
    p.add_argument("--pilot-mf-multiplier", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--noise-multiplier", type=float, default=1.96)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "manifold_fitting_gp_tube_visualization",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for j, manifold in enumerate(args.manifolds):
        row = plot_case(
            args.output / f"{manifold}_{args.pilot_mode}_gp_tube_vs_noise_band.png",
            manifold=manifold,
            pilot_mode=args.pilot_mode,
            n=args.n,
            sigma=args.sigma,
            h_factor=args.h_factor,
            normal_factor=args.normal_factor,
            amplitude_factor=args.amplitude_factor,
            length_factor=args.length_factor,
            max_points=args.max_points,
            min_points=args.min_points,
            weight_floor=args.weight_floor,
            grid_size=args.grid_size,
            dense_grid_size=args.dense_grid_size,
            pilot_fraction=args.pilot_fraction,
            pilot_angle_bandwidth=args.pilot_angle_bandwidth,
            pilot_mf_multiplier=args.pilot_mf_multiplier,
            alpha=args.alpha,
            noise_multiplier=args.noise_multiplier,
            seed=seed_for(args.seed, 777, j),
        )
        rows.append(row)
        print(
            manifold,
            f"coverage={row['gp_covers_truth']}",
            f"inside_noise={row['gp_tube_inside_noise_band']}",
            f"gp_max={row['max_gp_halfwidth']:.4f}",
            f"noise_halfwidth={row['noise_halfwidth']:.4f}",
            f"margin={row['strict_containment_margin']:.4f}",
        )

    write_metrics(args.output / "visualization_metrics.csv", rows)


if __name__ == "__main__":
    main()
