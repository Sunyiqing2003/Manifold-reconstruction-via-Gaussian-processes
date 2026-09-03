#!/usr/bin/env python3
"""Manifold Fitting + local GP confidence-band diagnostic on circle and ellipse.

This experiment keeps the geometric idea from the image-set construction:

    M_hat = G_hat(M_tilde),

where ``M_tilde`` is a closed preliminary 1-manifold.  The second-stage normal
correction is now a local GP prediction at tangent coordinate zero.  The main UQ
quantities are therefore GP quantities rather than bootstrap quantities:

1. latent GP posterior standard deviation at the chart origin;
2. frequentist standard deviation of the GP posterior mean,
       sigma * ||a||,
   where ``a`` is the exact linear-smoother weight vector;
3. simulation-only oracle population-bias diagnostics.

The local GP uses a smooth localization weight through a heteroskedastic nugget,
so every retained local observation has noise variance sigma^2 / omega_i.  This is
an empirical device for the demo; it is not claimed to solve the full chart/EIV
selection problem.

The simultaneous bands use a Bonferroni critical value over the finite curve grid.
This deliberately avoids pretending that independent local GPs define a coherent
joint posterior process across charts.

Run from repository root:

    python experiments/manifold_fitting_gp_confidence_demo.py --quick

A more stable oracle-pilot diagnostic is for example

    python experiments/manifold_fitting_gp_confidence_demo.py \
        --pilot-modes oracle --n 2500 --mc-reps 20 --grid-size 60
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
from scipy.linalg import cho_factor, cho_solve
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


@dataclass
class GPFit:
    prediction: float
    posterior_sd: float
    frequentist_sd: float
    local_n: int
    effective_n: float
    smoother_norm: float
    weight_max: float


def local_gp_origin(
    sample: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    sigma: float,
    h: float,
    normal_bandwidth: float,
    amplitude: float,
    length_scale: float,
    max_points: int,
    min_points: int,
    weight_floor: float,
) -> GPFit:
    """Fit a localized universal-kriging GP and predict the normal graph at w=0."""
    centered = sample - center
    w_all = centered @ tangent
    z_all = centered @ normal
    loc_weight = np.exp(
        -0.5 * (w_all / h) ** 2 - 0.5 * (z_all / normal_bandwidth) ** 2
    )

    # Keep the highest-weight observations.  This cap is purely computational;
    # all reported local/effective sample sizes are recorded for diagnostics.
    candidate = np.flatnonzero(loc_weight >= weight_floor)
    if len(candidate) < min_points:
        candidate = np.argsort(loc_weight)[-min(min_points, len(sample)) :]
    if len(candidate) > max_points:
        order = np.argpartition(loc_weight[candidate], -max_points)[-max_points:]
        candidate = candidate[order]

    w = w_all[candidate]
    z = z_all[candidate]
    omega = np.maximum(loc_weight[candidate], weight_floor)
    m = len(w)
    if m < 3:
        return GPFit(np.nan, np.nan, np.nan, m, float(omega.sum()), np.nan, np.nan)

    pairwise = w[:, None] - w[None, :]
    correlation = np.exp(-0.5 * (pairwise / length_scale) ** 2)
    # Smooth localization: low-weight observations are assigned a large nugget.
    noise_diag = sigma**2 / omega
    covariance = amplitude * correlation + np.diag(noise_diag)
    jitter = 1e-10 * max(1.0, amplitude, sigma**2)

    try:
        factor = cho_factor(
            covariance + jitter * np.eye(m), lower=True, check_finite=False
        )
    except np.linalg.LinAlgError:
        factor = cho_factor(
            covariance + 1e-7 * max(1.0, amplitude, sigma**2) * np.eye(m),
            lower=True,
            check_finite=False,
        )

    ones = np.ones(m)
    cinv_ones = cho_solve(factor, ones, check_finite=False)
    denom_mean = float(ones @ cinv_ones)
    base = cinv_ones / denom_mean

    k0 = amplitude * np.exp(-0.5 * (w / length_scale) ** 2)
    cinv_k0 = cho_solve(factor, k0, check_finite=False)
    smoother = base + cinv_k0 - base * float(ones @ cinv_k0)
    prediction = float(smoother @ z)

    posterior_base = amplitude - float(k0 @ cinv_k0)
    posterior_mean_correction = (1.0 - float(ones @ cinv_k0)) ** 2 / denom_mean
    posterior_variance = max(posterior_base + posterior_mean_correction, 0.0)

    # Conditional-on-chart leading sampling variance of the GP mean.  The actual
    # response noise in the data generator is homoskedastic sigma^2; localization
    # affects the estimator weights but not this repeated-sampling variance formula.
    frequentist_variance = sigma**2 * float(smoother @ smoother)

    return GPFit(
        prediction=prediction,
        posterior_sd=float(np.sqrt(posterior_variance)),
        frequentist_sd=float(np.sqrt(max(frequentist_variance, 0.0))),
        local_n=m,
        effective_n=float(omega.sum()),
        smoother_norm=float(np.linalg.norm(smoother)),
        weight_max=float(np.max(np.abs(smoother))),
    )


def fit_gp_curve(
    sample: np.ndarray,
    pilot: np.ndarray,
    tangent: np.ndarray,
    normal_vec: np.ndarray,
    sigma: float,
    h: float,
    normal_bandwidth: float,
    amplitude_factor: float,
    length_factor: float,
    max_points: int,
    min_points: int,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    amplitude = amplitude_factor * sigma**2
    length_scale = length_factor * h
    prediction = np.empty(len(pilot))
    posterior_sd = np.empty(len(pilot))
    frequentist_sd = np.empty(len(pilot))
    local_n = np.empty(len(pilot))
    effective_n = np.empty(len(pilot))
    smoother_norm = np.empty(len(pilot))
    weight_max = np.empty(len(pilot))

    for j in range(len(pilot)):
        fit = local_gp_origin(
            sample,
            pilot[j],
            tangent[j],
            normal_vec[j],
            sigma,
            h,
            normal_bandwidth,
            amplitude,
            length_scale,
            max_points,
            min_points,
            weight_floor,
        )
        prediction[j] = fit.prediction
        posterior_sd[j] = fit.posterior_sd
        frequentist_sd[j] = fit.frequentist_sd
        local_n[j] = fit.local_n
        effective_n[j] = fit.effective_n
        smoother_norm[j] = fit.smoother_norm
        weight_max[j] = fit.weight_max

    fitted = pilot + prediction[:, None] * normal_vec
    diagnostics = {
        "median_local_n": float(np.nanmedian(local_n)),
        "median_effective_n": float(np.nanmedian(effective_n)),
        "median_smoother_norm": float(np.nanmedian(smoother_norm)),
        "max_smoother_weight": float(np.nanmax(weight_max)),
    }
    return fitted, posterior_sd, frequentist_sd, diagnostics


def population_gp_bias(
    spec,
    phi_grid: np.ndarray,
    sigma: float,
    h: float,
    normal_bandwidth: float,
    amplitude_factor: float,
    length_factor: float,
    max_points: int,
    min_points: int,
    weight_floor: float,
    n_population: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Simulation-only estimate of E[m_hat(z)] with the true curve as pilot."""
    pilot, tangent, normal_vec = oracle_pilot(spec, phi_grid)
    acc = np.zeros(len(phi_grid))
    for b in range(repeats):
        rng = np.random.default_rng(seed_for(seed, 1000, b))
        _, noisy = sample_noisy_curve(rng, spec, n_population, sigma)
        fitted, _, _, _ = fit_gp_curve(
            noisy,
            pilot,
            tangent,
            normal_vec,
            sigma,
            h,
            normal_bandwidth,
            amplitude_factor,
            length_factor,
            max_points,
            min_points,
            weight_floor,
        )
        acc += np.sum((fitted - pilot) * normal_vec, axis=1)
    return acc / repeats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(r["manifold"]),
                str(r["pilot_mode"]),
                float(r["sigma"]),
                float(r["h_factor"]),
            )
            for r in rows
        }
    )
    out = []
    for manifold, pilot_mode, sigma, h_factor in keys:
        g = [
            r
            for r in rows
            if r["manifold"] == manifold
            and r["pilot_mode"] == pilot_mode
            and float(r["sigma"]) == sigma
            and float(r["h_factor"]) == h_factor
        ]

        def mean(name: str) -> float:
            vals = [float(r[name]) for r in g if r[name] != ""]
            return float(np.mean(vals)) if vals else float("nan")

        out.append(
            {
                "manifold": manifold,
                "pilot_mode": pilot_mode,
                "sigma": sigma,
                "h_factor": h_factor,
                "repeats": len(g),
                "gp_posterior_coverage": mean("gp_posterior_covered"),
                "freq_gp_mean_coverage": mean("freq_gp_mean_covered"),
                "gp_posterior_biasaware_coverage": mean("gp_posterior_biasaware_covered"),
                "freq_biasaware_coverage": mean("freq_biasaware_covered"),
                "mean_gp_radius": mean("gp_posterior_constant_radius"),
                "mean_freq_radius": mean("freq_constant_radius"),
                "mean_population_bias_envelope": mean("population_bias_envelope"),
                "mean_bias_over_gp_radius": mean("bias_over_gp_radius"),
                "mean_bias_over_freq_radius": mean("bias_over_freq_radius"),
                "mean_hausdorff_error": mean("hausdorff_error"),
                "median_local_n": mean("median_local_n"),
            }
        )
    return out


def plot_representative(
    path: Path,
    noisy: np.ndarray,
    true_dense: np.ndarray,
    pilot: np.ndarray,
    fitted: np.ndarray,
    normal_vec: np.ndarray,
    gp_width: np.ndarray,
    freq_width: np.ndarray,
    bias: np.ndarray | None,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(noisy[:, 0], noisy[:, 1], s=8, alpha=0.15, label="refinement sample")
    ax.plot(true_dense[:, 0], true_dense[:, 1], linewidth=2, label="truth")
    ax.plot(pilot[:, 0], pilot[:, 1], linestyle="--", linewidth=1.2, label="pilot")
    ax.plot(fitted[:, 0], fitted[:, 1], linewidth=2, label="GP-refined")

    for width, label, linestyle in (
        (gp_width, "GP posterior simultaneous", ":"),
        (freq_width, "frequentist GP-mean simultaneous", "-."),
    ):
        lo = fitted - width[:, None] * normal_vec
        hi = fitted + width[:, None] * normal_vec
        ax.plot(lo[:, 0], lo[:, 1], linestyle=linestyle, linewidth=1.1, label=label)
        ax.plot(hi[:, 0], hi[:, 1], linestyle=linestyle, linewidth=1.1)

    if bias is not None:
        corrected = fitted - bias[:, None] * normal_vec
        ax.plot(corrected[:, 0], corrected[:, 1], linewidth=1.2, label="oracle pop-bias corrected")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(path: Path, summary: list[dict[str, object]]) -> None:
    labels = [
        f"{r['manifold']}\n{r['pilot_mode']}\ns={float(r['sigma']):g}"
        for r in summary
    ]
    x = np.arange(len(summary))
    width = 0.20
    gp = np.array([float(r["gp_posterior_coverage"]) for r in summary])
    fr = np.array([float(r["freq_gp_mean_coverage"]) for r in summary])
    gp_b = np.array([float(r["gp_posterior_biasaware_coverage"]) for r in summary])
    fr_b = np.array([float(r["freq_biasaware_coverage"]) for r in summary])

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(summary)), 5.2))
    ax.bar(x - 1.5 * width, gp, width, label="GP posterior")
    ax.bar(x - 0.5 * width, fr, width, label="freq GP mean")
    mask = np.isfinite(gp_b)
    ax.bar(x[mask] + 0.5 * width, gp_b[mask], width, label="GP + oracle bias")
    mask2 = np.isfinite(fr_b)
    ax.bar(x[mask2] + 1.5 * width, fr_b[mask2], width, label="freq + oracle bias")
    ax.axhline(0.95, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("simultaneous truth coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifolds", nargs="+", choices=sorted(CURVES), default=["circle", "ellipse"])
    p.add_argument("--pilot-modes", nargs="+", choices=["oracle", "data"], default=["oracle", "data"])
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--pilot-fraction", type=float, default=0.35)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.05])
    p.add_argument("--h-factors", nargs="+", type=float, default=[1.5])
    p.add_argument("--normal-factor", type=float, default=2.5)
    p.add_argument("--amplitude-factor", type=float, default=1.0)
    p.add_argument("--length-factor", type=float, default=1.0)
    p.add_argument("--max-points", type=int, default=140)
    p.add_argument("--min-points", type=int, default=30)
    p.add_argument("--weight-floor", type=float, default=1e-3)
    p.add_argument("--grid-size", type=int, default=60)
    p.add_argument("--dense-grid-size", type=int, default=1800)
    p.add_argument("--pilot-angle-bandwidth", type=float, default=0.16)
    p.add_argument("--pilot-mf-multiplier", type=float, default=1.0)
    p.add_argument("--mc-reps", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--population-n", type=int, default=3000)
    p.add_argument("--population-repeats", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "manifold_fitting_gp_confidence_demo",
    )
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = 600
        args.mc_reps = 5
        args.grid_size = 36
        args.dense_grid_size = 900
        args.max_points = 80
        args.population_n = 1200
        args.population_repeats = 3

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    phi_grid = np.linspace(0, 2 * np.pi, args.grid_size, endpoint=False)
    dense_phi = np.linspace(0, 2 * np.pi, args.dense_grid_size, endpoint=False)
    bonf = float(norm.ppf(1.0 - args.alpha / (2.0 * args.grid_size)))

    pop_bias: dict[tuple[str, float, float], np.ndarray] = {}
    for mi, manifold in enumerate(args.manifolds):
        spec = CURVES[manifold]
        for si, sigma in enumerate(args.sigmas):
            for hi, h_factor in enumerate(args.h_factors):
                pop_bias[(manifold, sigma, h_factor)] = population_gp_bias(
                    spec,
                    phi_grid,
                    sigma,
                    h_factor * sigma,
                    args.normal_factor * sigma,
                    args.amplitude_factor,
                    args.length_factor,
                    args.max_points,
                    args.min_points,
                    args.weight_floor,
                    args.population_n,
                    args.population_repeats,
                    seed_for(args.seed, 800, mi, si, hi),
                )

    rows: list[dict[str, object]] = []
    plotted: set[tuple[str, str, float, float]] = set()

    for mi, manifold in enumerate(args.manifolds):
        spec = CURVES[manifold]
        true_grid = true_curve_polar(spec, phi_grid)
        true_dense = true_curve_polar(spec, dense_phi)
        for si, sigma in enumerate(args.sigmas):
            for hi, h_factor in enumerate(args.h_factors):
                h = h_factor * sigma
                normal_bw = args.normal_factor * sigma
                bias = pop_bias[(manifold, sigma, h_factor)]
                bias_env = float(np.max(np.abs(bias)))

                for pi, pilot_mode in enumerate(args.pilot_modes):
                    for rep in range(args.mc_reps):
                        rng = np.random.default_rng(seed_for(args.seed, 20, mi, si, hi, pi, rep))
                        _, noisy = sample_noisy_curve(rng, spec, args.n, sigma)
                        perm = rng.permutation(args.n)
                        n_pilot = max(30, int(round(args.pilot_fraction * args.n)))
                        noisy_pilot = noisy[perm[:n_pilot]]
                        noisy_refine = noisy[perm[n_pilot:]]

                        if pilot_mode == "oracle":
                            pilot, tangent, normal_vec = oracle_pilot(spec, phi_grid)
                        else:
                            pilot, tangent, normal_vec, _ = build_data_pilot(
                                noisy_pilot,
                                sigma,
                                phi_grid,
                                args.pilot_mf_multiplier,
                                args.pilot_angle_bandwidth,
                            )

                        fitted, post_sd, freq_sd, diag = fit_gp_curve(
                            noisy_refine,
                            pilot,
                            tangent,
                            normal_vec,
                            sigma,
                            h,
                            normal_bw,
                            args.amplitude_factor,
                            args.length_factor,
                            args.max_points,
                            args.min_points,
                            args.weight_floor,
                        )
                        gp_width = bonf * post_sd
                        freq_width = bonf * freq_sd
                        directed, hausdorff = geometric_errors(
                            fitted, true_dense, phi_grid, dense_phi
                        )

                        gp_cov = int(
                            variable_tube_contains_truth(
                                fitted, gp_width, true_dense, phi_grid, dense_phi
                            )
                        )
                        freq_cov = int(
                            variable_tube_contains_truth(
                                fitted, freq_width, true_dense, phi_grid, dense_phi
                            )
                        )

                        gp_bias_cov: int | str = ""
                        freq_bias_cov: int | str = ""
                        bias_over_gp: float | str = ""
                        bias_over_freq: float | str = ""
                        if pilot_mode == "oracle":
                            gp_bias_cov = int(
                                variable_tube_contains_truth(
                                    fitted,
                                    gp_width + np.abs(bias),
                                    true_dense,
                                    phi_grid,
                                    dense_phi,
                                )
                            )
                            freq_bias_cov = int(
                                variable_tube_contains_truth(
                                    fitted,
                                    freq_width + np.abs(bias),
                                    true_dense,
                                    phi_grid,
                                    dense_phi,
                                )
                            )
                            bias_over_gp = bias_env / max(float(np.max(gp_width)), 1e-14)
                            bias_over_freq = bias_env / max(float(np.max(freq_width)), 1e-14)

                        rows.append(
                            {
                                "manifold": manifold,
                                "pilot_mode": pilot_mode,
                                "repeat": rep,
                                "n": args.n,
                                "n_refine": len(noisy_refine),
                                "sigma": sigma,
                                "reach_proxy": spec.reach_proxy,
                                "h": h,
                                "h_factor": h_factor,
                                "bonferroni_q": bonf,
                                "amplitude": args.amplitude_factor * sigma**2,
                                "length_scale": args.length_factor * h,
                                "median_local_n": diag["median_local_n"],
                                "median_effective_n": diag["median_effective_n"],
                                "median_smoother_norm": diag["median_smoother_norm"],
                                "max_smoother_weight": diag["max_smoother_weight"],
                                "mean_gp_posterior_sd": float(np.mean(post_sd)),
                                "mean_freq_gp_mean_sd": float(np.mean(freq_sd)),
                                "gp_to_freq_sd_ratio": float(np.mean(post_sd / np.maximum(freq_sd, 1e-14))),
                                "gp_posterior_constant_radius": float(np.max(gp_width)),
                                "freq_constant_radius": float(np.max(freq_width)),
                                "directed_true_to_fit": directed,
                                "hausdorff_error": hausdorff,
                                "gp_posterior_covered": gp_cov,
                                "freq_gp_mean_covered": freq_cov,
                                "population_bias_envelope": bias_env if pilot_mode == "oracle" else "",
                                "bias_over_gp_radius": bias_over_gp,
                                "bias_over_freq_radius": bias_over_freq,
                                "gp_posterior_biasaware_covered": gp_bias_cov,
                                "freq_biasaware_covered": freq_bias_cov,
                            }
                        )

                        key = (manifold, pilot_mode, sigma, h_factor)
                        if key not in plotted:
                            plotted.add(key)
                            plot_representative(
                                outdir / f"representative_{manifold}_{pilot_mode}_sigma{sigma:g}_h{h_factor:g}.png",
                                noisy_refine,
                                true_dense,
                                pilot,
                                fitted,
                                normal_vec,
                                gp_width,
                                freq_width,
                                bias if pilot_mode == "oracle" else None,
                                f"{manifold}; pilot={pilot_mode}; sigma={sigma:g}; h/sigma={h_factor:g}",
                            )

    summary = summarize(rows)
    write_csv(outdir / "raw_metrics.csv", rows)
    write_csv(outdir / "summary.csv", summary)
    plot_summary(outdir / "coverage_summary.png", summary)

    metadata = {
        "purpose": "Manifold Fitting preliminary curve + local GP normal refinement; compare GP posterior and frequentist GP-mean variance.",
        "critical_value": "Bonferroni over finite pilot grid; no cross-chart posterior independence assumption.",
        "gp_posterior_variance": "Universal-kriging latent posterior variance at tangent coordinate zero.",
        "frequentist_variance": "sigma^2 ||a||^2 for exact local GP linear-smoother weights.",
        "population_bias": "Oracle-pilot simulation-only repeated large-sample estimate; not a practical bias estimator.",
        "caveats": [
            "Local GP is an empirical high-SNR diagnostic, not a completed EIV correction.",
            "Smooth localization is implemented through a heteroskedastic nugget and a computational top-weight cap.",
            "Data-pilot UQ conditions on the estimated preliminary manifold and does not add pilot-stage uncertainty.",
            "Coverage is evaluated on a dense curve grid; this is not itself an asymptotic theorem.",
        ],
        "args": vars(args) | {"output": str(args.output)},
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote results to {outdir}")
    for r in summary:
        print(
            r["manifold"], r["pilot_mode"],
            f"sigma={float(r['sigma']):g}",
            f"GP={float(r['gp_posterior_coverage']):.3f}",
            f"freq={float(r['freq_gp_mean_coverage']):.3f}",
            f"GP+bias={float(r['gp_posterior_biasaware_coverage']):.3f}" if np.isfinite(float(r["gp_posterior_biasaware_coverage"])) else "GP+bias=NA",
            f"sd_ratio={float(r['mean_gp_radius']) / max(float(r['mean_freq_radius']), 1e-14):.3f}",
        )


if __name__ == "__main__":
    main()
