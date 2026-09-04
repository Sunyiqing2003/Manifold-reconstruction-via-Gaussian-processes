#!/usr/bin/env python3
"""Use Manifold Fitting as the geometric center and GP only for normal UQ.

This experiment tests the design suggested by the previous circle/ellipse runs:

    1. estimate a closed preliminary manifold M_MF with Manifold Fitting;
    2. fit local GPs only to learn a possible residual normal displacement field;
    3. KEEP M_MF as the point-estimator center;
    4. form a residual-aware simultaneous tube around M_MF with half-width

           |m_hat_GP(z)| + q * s_GP(z),

       where m_hat_GP(z) is the local GP posterior-mean normal correction and
       s_GP(z) is the latent GP posterior standard deviation at tangent coordinate 0.

The experiment compares the point-estimation error of the MF preliminary curve with
that of the GP-refined curve and checks whether the MF-centered GP tube covers the
truth while remaining strictly inside the reference observation-noise band.

The simultaneous critical value is the same finite-grid Bonferroni calibration used
in the existing GP demo.  Hence this is a diagnostic visualization, not an asymptotic
confidence theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    geometric_errors,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(r["manifold"]), float(r["sigma"]), float(r["h_factor"])) for r in rows})
    out: list[dict[str, object]] = []
    for manifold, sigma, h_factor in keys:
        group = [
            r
            for r in rows
            if r["manifold"] == manifold
            and float(r["sigma"]) == sigma
            and float(r["h_factor"]) == h_factor
        ]

        def avg(name: str) -> float:
            return float(np.mean([float(r[name]) for r in group]))

        out.append(
            {
                "manifold": manifold,
                "sigma": sigma,
                "h_factor": h_factor,
                "repeats": len(group),
                "mf_mean_hausdorff": avg("mf_hausdorff"),
                "gp_refined_mean_hausdorff": avg("gp_refined_hausdorff"),
                "mf_better_fraction": avg("mf_better_than_gp"),
                "mf_centered_gp_tube_coverage": avg("mf_centered_gp_tube_covers_truth"),
                "mf_centered_freq_tube_coverage": avg("mf_centered_freq_tube_covers_truth"),
                "gp_refined_tube_coverage": avg("gp_refined_gp_tube_covers_truth"),
                "mf_centered_gp_inside_noise_fraction": avg("mf_centered_gp_tube_inside_noise"),
                "mean_mf_centered_gp_max_halfwidth": avg("mf_centered_gp_max_halfwidth"),
                "mean_gp_refined_gp_max_halfwidth": avg("gp_refined_gp_max_halfwidth"),
                "mean_noise_halfwidth": avg("noise_halfwidth"),
                "mean_mf_centered_width_ratio": avg("mf_centered_gp_width_over_noise"),
                "mean_strict_containment_margin": avg("strict_containment_margin"),
                "mean_abs_gp_residual": avg("mean_abs_gp_residual"),
                "mean_gp_posterior_sd": avg("mean_gp_posterior_sd"),
            }
        )
    return out


def plot_representative(
    path: Path,
    manifold: str,
    noisy_refine: np.ndarray,
    true_dense: np.ndarray,
    true_grid: np.ndarray,
    true_normal: np.ndarray,
    pilot: np.ndarray,
    fitted: np.ndarray,
    normal_vec: np.ndarray,
    mf_width: np.ndarray,
    gp_refined_width: np.ndarray,
    noise_halfwidth: float,
    metrics: dict[str, object],
) -> None:
    noise_lower = true_grid - noise_halfwidth * true_normal
    noise_upper = true_grid + noise_halfwidth * true_normal
    mf_lower = pilot - mf_width[:, None] * normal_vec
    mf_upper = pilot + mf_width[:, None] * normal_vec
    gp_lower = fitted - gp_refined_width[:, None] * normal_vec
    gp_upper = fitted + gp_refined_width[:, None] * normal_vec

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))

    ax = axes[0]
    ax.scatter(noisy_refine[:, 0], noisy_refine[:, 1], s=7, alpha=0.12, label="noisy observations")
    ax.plot(true_dense[:, 0], true_dense[:, 1], linewidth=2.2, label="true manifold")
    ax.plot(pilot[:, 0], pilot[:, 1], linestyle="--", linewidth=1.8, label="MF center")
    ax.plot(fitted[:, 0], fitted[:, 1], linewidth=1.5, label="GP-refined center")
    ax.set_title(
        "Point estimator comparison\n"
        f"MF Hausdorff={float(metrics['mf_hausdorff']):.4f}; "
        f"GP-refined={float(metrics['gp_refined_hausdorff']):.4f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    noise_poly = closed_band_polygon(noise_lower, noise_upper)
    mf_poly = closed_band_polygon(mf_lower, mf_upper)
    ax.fill(
        noise_poly[:, 0],
        noise_poly[:, 1],
        alpha=0.18,
        label=rf"reference observation band: $\pm 1.96\sigma$",
    )
    ax.fill(
        mf_poly[:, 0],
        mf_poly[:, 1],
        alpha=0.32,
        label=r"MF-centered GP tube: $|\hat m_{GP}|+q s_{GP}$",
    )
    ax.plot(true_dense[:, 0], true_dense[:, 1], linewidth=2.2, label="true manifold")
    ax.plot(pilot[:, 0], pilot[:, 1], linestyle="--", linewidth=1.8, label="MF center")
    # Keep the GP-refined center visible only as a thin diagnostic line.
    ax.plot(fitted[:, 0], fitted[:, 1], linewidth=0.9, alpha=0.7, label="GP-refined diagnostic")
    ax.set_title(
        "GP used for UQ, not as the center\n"
        f"truth covered={bool(metrics['mf_centered_gp_tube_covers_truth'])}; "
        f"inside noise band={bool(metrics['mf_centered_gp_tube_inside_noise'])}; "
        f"max width/noise={float(metrics['mf_centered_gp_width_over_noise']):.2f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(manifold, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def run_case(
    manifold: str,
    rep: int,
    args: argparse.Namespace,
    representative_path: Path | None,
) -> dict[str, object]:
    spec = CURVES[manifold]
    phi_grid = np.linspace(0.0, 2.0 * np.pi, args.grid_size, endpoint=False)
    dense_phi = np.linspace(0.0, 2.0 * np.pi, args.dense_grid_size, endpoint=False)
    true_grid = true_curve_polar(spec, phi_grid)
    true_dense = true_curve_polar(spec, dense_phi)
    _, _, true_normal = oracle_pilot(spec, phi_grid)

    rng = np.random.default_rng(seed_for(args.seed, 991, list(CURVES).index(manifold), rep))
    _, noisy = sample_noisy_curve(rng, spec, args.n, args.sigma)
    perm = rng.permutation(args.n)
    n_pilot = max(30, int(round(args.pilot_fraction * args.n)))
    noisy_pilot = noisy[perm[:n_pilot]]
    noisy_refine = noisy[perm[n_pilot:]]

    pilot, tangent, normal_vec, _ = build_data_pilot(
        noisy_pilot,
        args.sigma,
        phi_grid,
        args.pilot_mf_multiplier,
        args.pilot_angle_bandwidth,
    )

    fitted, post_sd, freq_sd, diag = fit_gp_curve(
        noisy_refine,
        pilot,
        tangent,
        normal_vec,
        args.sigma,
        args.h_factor * args.sigma,
        args.normal_factor * args.sigma,
        args.amplitude_factor,
        args.length_factor,
        args.max_points,
        args.min_points,
        args.weight_floor,
    )

    # GP posterior-mean residual relative to the MF center.
    gp_residual = np.sum((fitted - pilot) * normal_vec, axis=1)
    q = float(norm.ppf(1.0 - args.alpha / (2.0 * args.grid_size)))

    # Main proposal: MF remains the point-estimator center.  The estimated residual
    # magnitude is treated as a correction/bias allowance rather than forcing a move
    # of the center, while q*s_GP supplies the simultaneous stochastic allowance.
    mf_gp_width = np.abs(gp_residual) + q * post_sd
    mf_freq_width = np.abs(gp_residual) + q * freq_sd
    gp_refined_width = q * post_sd

    mf_directed, mf_hausdorff = geometric_errors(pilot, true_dense, phi_grid, dense_phi)
    gp_directed, gp_hausdorff = geometric_errors(fitted, true_dense, phi_grid, dense_phi)

    mf_gp_cover = variable_tube_contains_truth(
        pilot, mf_gp_width, true_dense, phi_grid, dense_phi
    )
    mf_freq_cover = variable_tube_contains_truth(
        pilot, mf_freq_width, true_dense, phi_grid, dense_phi
    )
    gp_refined_cover = variable_tube_contains_truth(
        fitted, gp_refined_width, true_dense, phi_grid, dense_phi
    )

    mf_lower = pilot - mf_gp_width[:, None] * normal_vec
    mf_upper = pilot + mf_gp_width[:, None] * normal_vec
    noise_halfwidth = args.noise_multiplier * args.sigma
    inside_noise, max_boundary_distance = boundary_inside_noise_band(
        mf_lower,
        mf_upper,
        true_dense,
        phi_grid,
        dense_phi,
        noise_halfwidth,
    )

    row = {
        "manifold": manifold,
        "repeat": rep,
        "n": args.n,
        "n_pilot": n_pilot,
        "n_refine": len(noisy_refine),
        "sigma": args.sigma,
        "h_factor": args.h_factor,
        "q_bonferroni": q,
        "noise_halfwidth": noise_halfwidth,
        "mf_directed_error": mf_directed,
        "mf_hausdorff": mf_hausdorff,
        "gp_refined_directed_error": gp_directed,
        "gp_refined_hausdorff": gp_hausdorff,
        "mf_better_than_gp": int(mf_hausdorff <= gp_hausdorff),
        "mean_abs_gp_residual": float(np.mean(np.abs(gp_residual))),
        "max_abs_gp_residual": float(np.max(np.abs(gp_residual))),
        "mean_gp_posterior_sd": float(np.mean(post_sd)),
        "mean_freq_gp_mean_sd": float(np.mean(freq_sd)),
        "mf_centered_gp_max_halfwidth": float(np.max(mf_gp_width)),
        "mf_centered_gp_mean_halfwidth": float(np.mean(mf_gp_width)),
        "mf_centered_freq_max_halfwidth": float(np.max(mf_freq_width)),
        "gp_refined_gp_max_halfwidth": float(np.max(gp_refined_width)),
        "mf_centered_gp_width_over_noise": float(np.max(mf_gp_width) / noise_halfwidth),
        "mf_centered_gp_tube_covers_truth": int(mf_gp_cover),
        "mf_centered_freq_tube_covers_truth": int(mf_freq_cover),
        "gp_refined_gp_tube_covers_truth": int(gp_refined_cover),
        "mf_centered_gp_tube_inside_noise": int(inside_noise),
        "max_mf_centered_boundary_distance_to_truth": max_boundary_distance,
        "strict_containment_margin": float(noise_halfwidth - max_boundary_distance),
        "median_local_n": diag["median_local_n"],
        "median_effective_n": diag["median_effective_n"],
    }

    if representative_path is not None:
        plot_representative(
            representative_path,
            manifold,
            noisy_refine,
            true_dense,
            true_grid,
            true_normal,
            pilot,
            fitted,
            normal_vec,
            mf_gp_width,
            gp_refined_width,
            noise_halfwidth,
            row,
        )

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifolds", nargs="+", choices=sorted(CURVES), default=["circle", "ellipse"])
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
    p.add_argument("--mc-reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "manifold_fitting_gp_mfcenter_demo",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for manifold in args.manifolds:
        for rep in range(args.mc_reps):
            rep_path = (
                args.output / f"{manifold}_mf_center_gp_uq.png" if rep == 0 else None
            )
            row = run_case(manifold, rep, args, rep_path)
            rows.append(row)

    summary = summarize(rows)
    write_csv(args.output / "raw_metrics.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "purpose": "Keep Manifold Fitting as point-estimator center; use local GP residual mean and posterior variance only to form a normal confidence tube.",
                "tube_halfwidth": "abs(GP posterior-mean normal residual) + Bonferroni_q * GP posterior sd",
                "reference_noise_band": "noise_multiplier * sigma around the true manifold; visualization benchmark only",
                "caveats": [
                    "The GP residual mean is used as a conservative correction allowance, not as a proven bias estimator.",
                    "The simultaneous critical value is Bonferroni over the finite pilot grid.",
                    "Pilot-stage uncertainty is not analytically propagated; its realized error is included empirically because the tube is centered on the data-driven MF pilot.",
                    "This is a circle/ellipse diagnostic, not a general-manifold theorem.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for r in summary:
        print(
            r["manifold"],
            f"MF_H={float(r['mf_mean_hausdorff']):.4f}",
            f"GP_H={float(r['gp_refined_mean_hausdorff']):.4f}",
            f"MF_better={float(r['mf_better_fraction']):.3f}",
            f"coverage={float(r['mf_centered_gp_tube_coverage']):.3f}",
            f"inside_noise={float(r['mf_centered_gp_inside_noise_fraction']):.3f}",
            f"width/noise={float(r['mean_mf_centered_width_ratio']):.3f}",
        )


if __name__ == "__main__":
    main()
