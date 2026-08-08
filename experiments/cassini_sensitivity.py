#!/usr/bin/env python3
"""Reproduce and stress-test MrGap on the Cassini oval.

This is a small, transparent Python translation of Mfit1.m and Mfit2.m.  The
main experiment keeps the paper's hyperparameters fixed while changing either
the observation noise or the number of observations.  This intentionally
measures sensitivity of the published setting; it is not an oracle-retuned
benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree, distance


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GPParams:
    amplitude: float
    length_scale_squared: float
    noise_variance: float


ROUND_1 = GPParams(0.014, 0.2, 0.002)
ROUND_2 = GPParams(0.048, 0.3, 2e-5)
EPSILON = 0.3
DELTA = 0.6
DIMENSION = 1


def cassini_oval(theta: np.ndarray) -> np.ndarray:
    """Equation (19) of the paper, embedded in R^3."""
    inner = np.cos(2 * theta)
    radius = np.sqrt(inner + np.sqrt(inner**2 + 0.2))
    return np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta), -0.3 * np.sin(theta))
    )


def grmse(points: np.ndarray, reference: np.ndarray) -> float:
    """One-sided geometric RMSE used for the paper's dense-reference estimate."""
    nearest, _ = cKDTree(reference).query(points, k=1, workers=-1)
    return float(np.sqrt(np.mean(nearest**2)))


def squared_exponential(x: np.ndarray, y: np.ndarray, params: GPParams) -> np.ndarray:
    return params.amplitude * np.exp(
        -(distance.cdist(x, y, metric="sqeuclidean") / params.length_scale_squared)
    )


def local_coordinates(
    data: np.ndarray, center: np.ndarray, epsilon: float, delta: float, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    distances = np.linalg.norm(data - center, axis=1)
    epsilon_ball = data[distances < epsilon]
    delta_ball = data[distances < delta] - center

    # Mfit*.m divides by the local count. Scaling does not change eigenvectors.
    covariance = (epsilon_ball - center).T @ (epsilon_ball - center)
    covariance /= len(epsilon_ball)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    rotated = delta_ball @ rotation
    return rotation, rotated[:, :d], rotated[:, d:], len(epsilon_ball), len(delta_ball)


def mfit1(
    data: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    delta: float,
    d: int,
    params: GPParams,
) -> tuple[np.ndarray, dict[str, float]]:
    """Faithful vectorized equivalent of Mfit1.m (one denoising round)."""
    fitted = np.empty_like(centers, dtype=float)
    epsilon_counts: list[int] = []
    delta_counts: list[int] = []

    for i, center in enumerate(centers):
        rotation, x_train, z_train, n_eps, n_delta = local_coordinates(
            data, center, epsilon, delta, d
        )
        epsilon_counts.append(n_eps)
        delta_counts.append(n_delta)

        x_query = np.zeros((1, d))
        k_train = squared_exponential(x_train, x_train, params)
        k_query = squared_exponential(x_query, x_train, params)
        z_mean = z_train.mean(axis=0, keepdims=True)
        alpha = np.linalg.solve(
            k_train + params.noise_variance * np.eye(len(x_train)),
            z_train - z_mean,
        )
        z_query = z_mean + k_query @ alpha
        fitted[i] = np.hstack((x_query, z_query)) @ rotation.T + center

    diagnostics = {
        "min_epsilon_neighbors": int(np.min(epsilon_counts)),
        "median_epsilon_neighbors": float(np.median(epsilon_counts)),
        "min_delta_neighbors": int(np.min(delta_counts)),
        "median_delta_neighbors": float(np.median(delta_counts)),
    }
    return fitted, diagnostics


def sample_unit_ball(rng: np.random.Generator, count: int, d: int) -> np.ndarray:
    directions = rng.normal(size=(count, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(count) ** (1 / d)
    return directions * radii[:, None]


def mfit2(
    data: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    delta: float,
    d: int,
    points_per_patch: int,
    params: GPParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Faithful equivalent of Mfit2.m (sequential interpolation)."""
    generated: list[np.ndarray] = []

    for center in centers:
        rotation, x_data, z_data, _, _ = local_coordinates(
            data, center, epsilon, delta, d
        )

        if generated:
            previous = np.vstack(generated)
            previous = previous[np.linalg.norm(previous - center, axis=1) < delta] - center
            previous_rotated = previous @ rotation
            x_train = np.vstack((x_data, previous_rotated[:, :d]))
            z_train = np.vstack((z_data, previous_rotated[:, d:]))
        else:
            x_train, z_train = x_data, z_data

        radial_distance = np.linalg.norm(x_data - x_data.mean(axis=0), axis=1)
        # MATLAB sqrt(var(a)) uses the sample standard deviation (ddof=1).
        spread = np.std(radial_distance, ddof=1) if len(radial_distance) > 1 else 0.0
        radius = float(np.mean(radial_distance) - spread)
        x_query = x_data.mean(axis=0) + radius * sample_unit_ball(
            rng, points_per_patch, d
        )

        k_train = squared_exponential(x_train, x_train, params)
        k_query = squared_exponential(x_query, x_train, params)
        alpha = np.linalg.solve(
            k_train + params.noise_variance * np.eye(len(x_train)), z_train
        )
        z_query = k_query @ alpha
        generated.append(np.hstack((x_query, z_query)) @ rotation.T + center)

    return np.vstack(generated)


def denoise_twice(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    first, _ = mfit1(data, data, EPSILON, DELTA, DIMENSION, ROUND_1)
    second, diagnostics = mfit1(first, first, EPSILON, DELTA, DIMENSION, ROUND_2)
    return first, second, diagnostics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], variable: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    values = sorted({float(row[variable]) for row in rows})
    for value in values:
        group = [row for row in rows if float(row[variable]) == value]
        record: dict[str, object] = {variable: int(value) if variable == "sample_size" else value}
        for metric in ("raw_grmse", "round1_grmse", "round2_grmse"):
            numbers = np.array([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = float(numbers.mean())
            record[f"{metric}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        record["median_min_epsilon_neighbors"] = float(
            np.median([float(row["min_epsilon_neighbors"]) for row in group])
        )
        record["median_epsilon_neighbors"] = float(
            np.median([float(row["median_epsilon_neighbors"]) for row in group])
        )
        output.append(record)
    return output


def sensitivity_experiments(repeats: int, seed: int) -> tuple[list[dict], list[dict]]:
    reference = cassini_oval(np.linspace(0, 2 * np.pi, 100_000, endpoint=False))
    noise_levels = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12)
    sample_sizes = (30, 50, 75, 102, 150, 250)
    noise_rows: list[dict] = []
    size_rows: list[dict] = []

    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)

        clean_noise = cassini_oval(rng.uniform(0, 2 * np.pi, 102))
        standard_noise = rng.normal(size=clean_noise.shape)
        for sigma in noise_levels:
            observed = clean_noise + sigma * standard_noise
            first, second, diagnostics = denoise_twice(observed)
            noise_rows.append(
                {
                    "repeat": repeat,
                    "noise_sigma": sigma,
                    "sample_size": 102,
                    "raw_grmse": grmse(observed, reference),
                    "round1_grmse": grmse(first, reference),
                    "round2_grmse": grmse(second, reference),
                    **diagnostics,
                }
            )

        max_size = max(sample_sizes)
        clean_size = cassini_oval(rng.uniform(0, 2 * np.pi, max_size))
        observed_size = clean_size + 0.04 * rng.normal(size=clean_size.shape)
        for size in sample_sizes:
            observed = observed_size[:size]
            first, second, diagnostics = denoise_twice(observed)
            size_rows.append(
                {
                    "repeat": repeat,
                    "noise_sigma": 0.04,
                    "sample_size": size,
                    "raw_grmse": grmse(observed, reference),
                    "round1_grmse": grmse(first, reference),
                    "round2_grmse": grmse(second, reference),
                    **diagnostics,
                }
            )
    return noise_rows, size_rows


def plot_sensitivity(summary: list[dict], variable: str, label: str, path: Path) -> None:
    x = np.array([row[variable] for row in summary], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = (("raw", "Observed", "#777777"), ("round1", "After round 1", "#2a6fbb"),
              ("round2", "After round 2", "#c43c39"))
    for prefix, legend, color in styles:
        mean = np.array([row[f"{prefix}_grmse_mean"] for row in summary])
        std = np.array([row[f"{prefix}_grmse_std"] for row in summary])
        ax.plot(x, mean, marker="o", label=legend, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel(label)
    ax.set_ylabel("GRMSE to dense Cassini oval")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def existing_dataset_baseline(output_dir: Path, seed: int) -> dict[str, float]:
    dataset = loadmat(ROOT / "Cassini oval.mat")
    observed = np.asarray(dataset["X"], dtype=float)
    reference = np.asarray(dataset["M11"], dtype=float)
    first, second, diagnostics = denoise_twice(observed)
    # Algorithm 2 uses the previous-round samples with the last-round parameters.
    interpolated = mfit2(
        first, first, EPSILON, DELTA, DIMENSION, 20, ROUND_2, np.random.default_rng(seed)
    )
    np.savez_compressed(
        output_dir / "existing_baseline_points.npz",
        observed=observed,
        round1=first,
        round2=second,
        interpolated=interpolated,
    )

    metrics = {
        "observed_grmse": grmse(observed, reference),
        "round1_grmse": grmse(first, reference),
        "round2_grmse": grmse(second, reference),
        "interpolation_grmse": grmse(interpolated, reference),
        **diagnostics,
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, points, title in (
        (axes[0], observed, f"Observed ({metrics['observed_grmse']:.4f})"),
        (axes[1], second, f"Denoised ({metrics['round2_grmse']:.4f})"),
        (axes[2], interpolated, f"Interpolated ({metrics['interpolation_grmse']:.4f})"),
    ):
        ax.plot(reference[::100, 0], reference[::100, 1], color="#bbbbbb", lw=1)
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.7)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "existing_baseline_xy.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cassini")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = existing_dataset_baseline(args.output, args.seed)
    noise_rows, size_rows = sensitivity_experiments(args.repeats, args.seed)
    noise_summary = summarize(noise_rows, "noise_sigma")
    size_summary = summarize(size_rows, "sample_size")

    write_rows(args.output / "noise_raw.csv", noise_rows)
    write_rows(args.output / "sample_size_raw.csv", size_rows)
    write_rows(args.output / "noise_summary.csv", noise_summary)
    write_rows(args.output / "sample_size_summary.csv", size_summary)
    plot_sensitivity(noise_summary, "noise_sigma", "Noise standard deviation", args.output / "noise_sensitivity.png")
    plot_sensitivity(size_summary, "sample_size", "Number of observed points", args.output / "sample_size_sensitivity.png")

    metadata = {
        "seed": args.seed,
        "repeats": args.repeats,
        "epsilon": EPSILON,
        "delta": DELTA,
        "dimension": DIMENSION,
        "round_1": asdict(ROUND_1),
        "round_2": asdict(ROUND_2),
        "existing_dataset_baseline": baseline,
        "note": "Sensitivity scans use uniform theta samples and fixed paper hyperparameters.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
