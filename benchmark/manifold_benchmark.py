#!/usr/bin/env python3
"""Unified diagnostic benchmark for MrGap and Yao et al. Manifold Fitting.

The Manifold Fitting port follows the official MATLAB ``manfit_ours.m``.  MrGap
follows this repository's ``Mfit1.m``.  See README.md for fairness conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree, distance

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GP:
    amplitude: float
    rho: float
    noise_variance: float


@dataclass(frozen=True)
class Manifold:
    name: str
    intrinsic_dim: int
    ambient_dim: int
    default_n: int
    default_sigma: float
    epsilon: float
    delta: float
    gps: tuple[GP, GP]
    epsilon_grid: tuple[float, ...]
    sample: Callable[[np.random.Generator, int], tuple[np.ndarray, object]]
    tangent: Callable[[object], np.ndarray]


def _cassini_map(theta: np.ndarray) -> np.ndarray:
    q = np.cos(2 * theta)
    radius = np.sqrt(q + np.sqrt(q * q + 0.2))
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), -0.3 * np.sin(theta)))


def sample_cassini(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(0, 2 * np.pi, n)
    return _cassini_map(theta), theta


def tangent_cassini(theta: object) -> np.ndarray:
    t = np.asarray(theta)
    h = 1e-6
    derivative = (_cassini_map(t + h) - _cassini_map(t - h)) / (2 * h)
    return derivative[:, :, None] / np.linalg.norm(derivative, axis=1)[:, None, None]


def _sample_torus_parameters(
    rng: np.random.Generator, n: int, major: float, minor: float, half: bool
) -> tuple[np.ndarray, np.ndarray]:
    accepted: list[np.ndarray] = []
    while sum(len(x) for x in accepted) < n:
        u = rng.uniform(0, 2 * np.pi, max(64, 2 * n))
        keep = rng.random(len(u)) < (major + minor * np.cos(u)) / (major + minor)
        accepted.append(u[keep])
    u = np.concatenate(accepted)[:n]
    v = rng.uniform(-np.pi / 2, np.pi / 2, n) if half else rng.uniform(0, 2 * np.pi, n)
    return u, v


def _torus_sample(
    rng: np.random.Generator, n: int, major: float, minor: float, half: bool
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float, float]]:
    u, v = _sample_torus_parameters(rng, n, major, minor, half)
    points = np.column_stack(
        ((major + minor * np.cos(u)) * np.cos(v),
         (major + minor * np.cos(u)) * np.sin(v), minor * np.sin(u))
    )
    return points, (u, v, major, minor)


def tangent_torus(latent: object) -> np.ndarray:
    u, v, major, minor = latent  # type: ignore[misc]
    du = np.column_stack((-minor * np.sin(u) * np.cos(v), -minor * np.sin(u) * np.sin(v), minor * np.cos(u)))
    dv = np.column_stack((-(major + minor * np.cos(u)) * np.sin(v), (major + minor * np.cos(u)) * np.cos(v), np.zeros_like(u)))
    du /= np.linalg.norm(du, axis=1, keepdims=True)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    return np.stack((du, dv), axis=2)


def sample_torus(rng: np.random.Generator, n: int) -> tuple[np.ndarray, object]:
    return _torus_sample(rng, n, 2.0, 0.8, False)


def sample_half_torus(rng: np.random.Generator, n: int) -> tuple[np.ndarray, object]:
    return _torus_sample(rng, n, 3.0, 0.8, True)


def _rp3_map(z: np.ndarray) -> np.ndarray:
    a, b, c, d = z.T
    return np.column_stack((a*a, b*b, c*c, d*d, a*b, a*c, a*d, b*c, b*d, c*d))


def sample_rp3(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    z = rng.normal(size=(n, 4))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return _rp3_map(z), z


def tangent_rp3(latent: object) -> np.ndarray:
    zs = np.asarray(latent)
    output = np.empty((len(zs), 10, 3))
    for i, z in enumerate(zs):
        _, _, vh = np.linalg.svd(z.reshape(1, 4), full_matrices=True)
        sphere_tangent = vh[1:].T
        a, b, c, d = z
        jacobian = np.array([
            [2*a, 0, 0, 0], [0, 2*b, 0, 0], [0, 0, 2*c, 0], [0, 0, 0, 2*d],
            [b, a, 0, 0], [c, 0, a, 0], [d, 0, 0, a], [0, c, b, 0],
            [0, d, 0, b], [0, 0, d, c],
        ])
        output[i], _ = np.linalg.qr(jacobian @ sphere_tangent)
    return output


MANIFOLDS = {
    "cassini": Manifold("cassini", 1, 3, 102, 0.04, 0.3, 0.6,
        (GP(0.014, 0.2, 0.002), GP(0.048, 0.3, 2e-5)),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), sample_cassini, tangent_cassini),
    "rp3": Manifold("rp3", 3, 10, 1200, 0.04, 0.5, 0.5,
        (GP(0.05, 1.0, 0.0015), GP(0.08, 1.2, 0.0004)),
        (0.2, 0.3, 0.4, 0.5), sample_rp3, tangent_rp3),
    "torus": Manifold("torus", 2, 3, 558, 0.12, 0.8, 1.0,
        (GP(0.06, 0.2, 0.03), GP(0.2, 1.1, 0.007)),
        (0.3, 0.5, 0.7, 0.8, 1.0), sample_torus, tangent_torus),
    # The paper reports only the final-round GP values for this example; they
    # are used in both rounds and explicitly recorded in metadata.
    "half_torus": Manifold("half_torus", 2, 3, 400, 0.12, 1.0, 1.0,
        (GP(1.3, 5.0, 0.002), GP(1.3, 5.0, 0.002)),
        (0.4, 0.6, 0.8, 1.0), sample_half_torus, tangent_torus),
}

NOISE_LEVELS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12)
SAMPLE_SIZES = (50, 100, 250, 500, 1000, 5000)
MF_BANDWIDTH_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MRGAP_ITERATIONS = (1, 2, 3, 4, 5)


def kernel(x: np.ndarray, y: np.ndarray, gp: GP) -> np.ndarray:
    return gp.amplitude * np.exp(-distance.cdist(x, y, "sqeuclidean") / gp.rho)


def complete_rotation(tangent: np.ndarray) -> np.ndarray:
    rotation, _ = np.linalg.qr(tangent, mode="complete")
    return rotation


def mrgap_round(
    data: np.ndarray,
    epsilon: float,
    delta: float,
    d: int,
    gp: GP,
    oracle_tangents: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    tree = cKDTree(data)
    eps_neighbors = tree.query_ball_point(data, epsilon)
    delta_neighbors = tree.query_ball_point(data, delta)
    output = np.empty_like(data)
    eps_counts = np.fromiter(map(len, eps_neighbors), dtype=int)
    delta_counts = np.fromiter(map(len, delta_neighbors), dtype=int)
    failures = 0
    for i, center in enumerate(data):
        if oracle_tangents is None:
            local = data[eps_neighbors[i]] - center
            covariance = local.T @ local / len(local)
            values, vectors = np.linalg.eigh(covariance)
            rotation = vectors[:, np.argsort(values)[::-1]]
        else:
            rotation = complete_rotation(oracle_tangents[i])
        rotated = (data[delta_neighbors[i]] - center) @ rotation
        x, z = rotated[:, :d], rotated[:, d:]
        z_mean = z.mean(axis=0, keepdims=True)
        kxx = kernel(x, x, gp) + gp.noise_variance * np.eye(len(x))
        k0x = kernel(np.zeros((1, d)), x, gp)
        try:
            prediction = z_mean + k0x @ np.linalg.solve(kxx, z - z_mean)
        except np.linalg.LinAlgError:
            failures += 1
            prediction = z_mean + k0x @ np.linalg.lstsq(kxx, z - z_mean, rcond=None)[0]
        output[i] = np.hstack((np.zeros((1, d)), prediction)) @ rotation.T + center
    return output, {
        "median_local_neighborhood": float(np.median(eps_counts)),
        "min_local_neighborhood": int(np.min(eps_counts)),
        "median_regression_neighborhood": float(np.median(delta_counts)),
        "linear_solve_failures": failures,
    }


def manifold_fitting(
    sample: np.ndarray,
    sigma: float,
    bandwidth_multiplier: float = 1.0,
    average: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Port of official MATLAB manfit_ours(sample,sig,sample_init,op_average).

    For a paired denoising comparison sample_init=sample. At sigma=0 the
    official bandwidth formula is undefined; identity is used as its continuous
    no-noise convention and status is recorded.
    """
    n = len(sample)
    if sigma == 0:
        return sample.copy(), {
            "median_local_neighborhood": 1.0, "min_local_neighborhood": 1,
            "mf_r": 0.0, "mf_R": 0.0, "sigma_zero_identity": 1,
        }
    r = bandwidth_multiplier * 5 * sigma / np.log10(n)
    R = bandwidth_multiplier * 10 * sigma * np.sqrt(np.log(1 / sigma)) / np.log10(n)
    tree = cKDTree(sample)
    nearest_five = tree.query(sample, k=min(5, n))[1]
    ball_neighbors = tree.query_ball_point(sample, 2 * r)
    output = np.empty_like(sample)
    cylinder_counts = np.empty(n, dtype=int)
    search_radius = math.sqrt(R * R + r * r)
    candidates = tree.query_ball_point(sample, search_radius)
    for i, x in enumerate(sample):
        base_idx = np.union1d(ball_neighbors[i], np.atleast_1d(nearest_five[i]))
        xbar = sample[base_idx].mean(axis=0) + np.finfo(float).eps
        direction = x - xbar
        norm = np.linalg.norm(direction)
        if norm <= np.finfo(float).eps:
            output[i] = xbar
            cylinder_counts[i] = 0
            continue
        direction /= norm
        centered = sample[candidates[i]] - x
        axial = centered @ direction
        radial_sq = np.sum(centered * centered, axis=1) - axial * axial
        mask = (np.abs(axial) < R) & (radial_sq < r * r)
        cylinder_idx = np.asarray(candidates[i], dtype=int)[mask]
        cylinder_counts[i] = len(cylinder_idx)
        output[i] = sample[cylinder_idx].mean(axis=0) if len(cylinder_idx) > 10 else xbar
    if average:
        out_tree = cKDTree(output)
        knn = out_tree.query(output, k=min(5, n))[1]
        close = out_tree.query_ball_point(output, r / 4)
        smoothed = np.empty_like(output)
        for i in range(n):
            idx = np.union1d(close[i], np.atleast_1d(knn[i]))
            smoothed[i] = output[idx].mean(axis=0)
        output = smoothed
    return output, {
        "median_local_neighborhood": float(np.median(cylinder_counts)),
        "min_local_neighborhood": int(np.min(cylinder_counts)),
        "mf_r": r, "mf_R": R, "sigma_zero_identity": 0,
    }


def error_to_reference(points: np.ndarray, reference_tree: cKDTree) -> float:
    nearest = reference_tree.query(points, k=1, workers=-1)[0]
    return float(np.sqrt(np.mean(nearest * nearest)))


def augmented_reference(reference_tree: cKDTree, clean: np.ndarray) -> cKDTree:
    """Add the trial's exact clean samples to the independent dense reference.

    This removes avoidable Monte Carlo floor (especially for RP3) without ever
    using the reference cloud to generate observations.
    """
    return cKDTree(np.vstack((np.asarray(reference_tree.data), clean)))


def paired_rmse(points: np.ndarray, clean: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - clean) ** 2, axis=1))))


def measured(call: Callable[[], tuple[np.ndarray, dict]]) -> tuple[np.ndarray, dict, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    output, diagnostics = call()
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return output, diagnostics, runtime, peak / 1024**2


def seed_for(base: int, *values: int) -> int:
    return int(np.random.SeedSequence([base, *values]).generate_state(1)[0])


def make_data(manifold: Manifold, n: int, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray, object]:
    rng = np.random.default_rng(seed)
    clean, latent = manifold.sample(rng, n)
    noisy = clean + sigma * rng.normal(size=clean.shape)
    return clean, noisy, latent


def base_record(experiment: str, manifold: Manifold, repeat: int, n: int, sigma: float,
                raw_error: float) -> dict[str, object]:
    return {"experiment": experiment, "method": "", "manifold": manifold.name,
            "repeat": repeat, "n": n, "sigma": sigma, "iterations": 0,
            "oracle_tangent": 0, "epsilon": "", "delta": "", "bandwidth_multiplier": "",
            "median_local_neighborhood": "", "min_local_neighborhood": "",
            "raw_error": raw_error, "reconstruction_error": "", "paired_rmse": "",
            "runtime_sec": "", "peak_memory_mb": "", "status": "ok"}


def evaluate_methods(experiment: str, manifold: Manifold, repeat: int, clean: np.ndarray,
                     noisy: np.ndarray, latent: object, reference: cKDTree, sigma: float,
                     iterations: tuple[int, ...] = MRGAP_ITERATIONS, oracle: bool = False,
                     epsilon: float | None = None, mf_multiplier: float = 1.0) -> list[dict[str, object]]:
    raw = error_to_reference(noisy, reference)
    rows: list[dict[str, object]] = []
    eps = manifold.epsilon if epsilon is None else epsilon
    tangents = manifold.tangent(latent) if oracle else None
    current = noisy.copy()
    cumulative_time = 0.0
    peak_memory = 0.0
    wanted = set(iterations)
    for iteration in range(1, max(iterations) + 1):
        gp = manifold.gps[0] if iteration == 1 else manifold.gps[1]
        current, diag, runtime, memory = measured(
            lambda cur=current, par=gp: mrgap_round(cur, eps, manifold.delta,
                                                    manifold.intrinsic_dim, par, tangents)
        )
        cumulative_time += runtime
        peak_memory = max(peak_memory, memory)
        if iteration in wanted:
            row = base_record(experiment, manifold, repeat, len(noisy), sigma, raw)
            row.update({"method": "mrgap_oracle" if oracle else "mrgap",
                        "iterations": iteration, "oracle_tangent": int(oracle),
                        "epsilon": eps, "delta": manifold.delta,
                        "median_local_neighborhood": diag["median_local_neighborhood"],
                        "min_local_neighborhood": diag["min_local_neighborhood"],
                        "reconstruction_error": error_to_reference(current, reference),
                        "paired_rmse": paired_rmse(current, clean),
                        "runtime_sec": cumulative_time, "peak_memory_mb": peak_memory})
            rows.append(row)
    if not oracle:
        fitted, diag, runtime, memory = measured(
            lambda: manifold_fitting(noisy, sigma, mf_multiplier, True)
        )
        row = base_record(experiment, manifold, repeat, len(noisy), sigma, raw)
        row.update({"method": "manifold_fitting", "iterations": 1,
                    "bandwidth_multiplier": mf_multiplier,
                    "median_local_neighborhood": diag["median_local_neighborhood"],
                    "min_local_neighborhood": diag["min_local_neighborhood"],
                    "reconstruction_error": error_to_reference(fitted, reference),
                    "paired_rmse": paired_rmse(fitted, clean), "runtime_sec": runtime,
                    "peak_memory_mb": memory,
                    "status": "sigma_zero_identity" if diag["sigma_zero_identity"] else "ok"})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def mean_curve(rows: list[dict[str, object]], xkey: str, method: str, manifold: str,
               iterations: int = 1) -> tuple[np.ndarray, np.ndarray]:
    chosen = [r for r in rows if r["method"] == method and r["manifold"] == manifold
              and int(r["iterations"]) == iterations]
    xs = sorted({float(r[xkey]) for r in chosen})
    ys = [np.mean([float(r["reconstruction_error"]) for r in chosen if float(r[xkey]) == x]) for x in xs]
    return np.asarray(xs), np.asarray(ys)


def plot_grid(rows: list[dict[str, object]], experiment: str, xkey: str, xlabel: str,
              path: Path, include_oracle: bool = False,
              ylabel: str = "reconstruction GRMSE") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    styles = [("mrgap", 2, "MrGap (2)", "#c43c39"),
              ("manifold_fitting", 1, "Manifold Fitting", "#2a6fbb")]
    if include_oracle:
        styles.append(("mrgap_oracle", 2, "MrGap oracle (2)", "#2f8f46"))
    subset = [r for r in rows if r["experiment"] == experiment]
    for ax, name in zip(axes.flat, MANIFOLDS):
        for method, iteration, label, color in styles:
            x, y = mean_curve(subset, xkey, method, name, iteration)
            if len(x): ax.plot(x, y, marker="o", label=label, color=color)
        ax.set_title(name); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_sample_neighbors(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    subset = [r for r in rows if r["experiment"] == "sample_size"]
    for ax, name in zip(axes.flat, MANIFOLDS):
        for method, iteration, label, color in (("mrgap", 2, "MrGap (2)", "#c43c39"),
                                                  ("manifold_fitting", 1, "Manifold Fitting", "#2a6fbb")):
            chosen = [r for r in subset if r["manifold"] == name and r["method"] == method
                      and int(r["iterations"]) == iteration]
            grouped: dict[float, list[tuple[float, float]]] = {}
            for r in chosen:
                grouped.setdefault(float(r["n"]), []).append((float(r["median_local_neighborhood"]),
                                                               float(r["reconstruction_error"])))
            x = [np.mean(grouped[n], axis=0)[0] for n in sorted(grouped)]
            y = [np.mean(grouped[n], axis=0)[1] for n in sorted(grouped)]
            ax.plot(x, y, marker="o", label=label, color=color)
        ax.set_title(name); ax.set_xlabel("median local neighborhood size")
        ax.set_ylabel("reconstruction GRMSE"); ax.grid(alpha=.25)
    axes.flat[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_runtime(rows: list[dict[str, object]], path: Path) -> None:
    plot_grid(rows, "sample_size", "n", "number of observations", path)
    # Replace plotted error with runtime by temporarily mapping the field.


def runtime_plot(rows: list[dict[str, object]], path: Path) -> None:
    copied = [dict(r, reconstruction_error=r["runtime_sec"]) for r in rows]
    plot_grid(copied, "sample_size", "n", "number of observations", path,
              ylabel="cumulative runtime (s)")


def iteration_plot(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (name, manifold) in zip(axes.flat, MANIFOLDS.items()):
        chosen = [r for r in rows if r["experiment"] == "noise" and r["method"] == "mrgap"
                  and r["manifold"] == name and
                  abs(float(r["sigma"]) - manifold.default_sigma) < 1e-12]
        xs = sorted({int(r["iterations"]) for r in chosen})
        ys = [np.mean([float(r["reconstruction_error"]) for r in chosen
                       if int(r["iterations"]) == x]) for x in xs]
        ax.plot(xs, ys, "o-", color="#c43c39")
        ax.set_xticks(xs); ax.set_title(name); ax.set_xlabel("MrGap rounds")
        ax.set_ylabel("reconstruction GRMSE"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def report(output: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    baseline = [r for r in rows if r["experiment"] == "noise" and
                abs(float(r["sigma"]) - MANIFOLDS[str(r["manifold"])].default_sigma) < 1e-12 and
                ((r["method"] == "mrgap" and int(r["iterations"]) == 2) or r["method"] == "manifold_fitting")]
    lines = ["# MrGap vs Manifold Fitting: benchmark summary", "",
             "This is an empirical diagnostic benchmark; no theoretical claim is made.", "",
             "## Default-condition means", "",
             "| method | manifold | n | sigma | error | runtime (s) | median neighborhood |", "|---|---|---:|---:|---:|---:|---:|"]
    keys = sorted({(str(r["method"]), str(r["manifold"]), int(r["n"]), float(r["sigma"])) for r in baseline})
    for key in keys:
        group = [r for r in baseline if (str(r["method"]), str(r["manifold"]), int(r["n"]), float(r["sigma"])) == key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:.3g} | "
                     f"{np.mean([float(r['reconstruction_error']) for r in group]):.5f} | "
                     f"{np.mean([float(r['runtime_sec']) for r in group]):.3f} | "
                     f"{np.mean([float(r['median_local_neighborhood']) for r in group]):.1f} |")
    lines += ["", "## Empirical diagnostic highlights", ""]
    noise_rows = [r for r in rows if r["experiment"] == "noise" and
                  ((r["method"] == "mrgap" and int(r["iterations"]) == 2) or
                   r["method"] == "manifold_fitting")]
    for name in sorted({str(r["manifold"]) for r in noise_rows}):
        lines.append(f"### {name}")
        for method in ("mrgap", "manifold_fitting"):
            group = sorted([r for r in noise_rows if r["manifold"] == name and r["method"] == method],
                           key=lambda r: float(r["sigma"]))
            failed = [float(r["sigma"]) for r in group
                      if float(r["sigma"]) > 0 and r["status"] == "ok" and
                      float(r["reconstruction_error"]) >= float(r["raw_error"]) - 1e-12]
            failure_text = "none in positive-noise scan" if not failed else ", ".join(f"{x:g}" for x in failed)
            lines.append(f"- `{method}` non-improving positive-noise values: {failure_text}.")
        oracle = [r for r in rows if r["experiment"] == "oracle" and r["manifold"] == name]
        by_sigma: dict[float, dict[str, float]] = {}
        for r in oracle:
            by_sigma.setdefault(float(r["sigma"]), {})[str(r["method"])] = float(r["reconstruction_error"])
        gains = [(s, v["mrgap"] - v["mrgap_oracle"]) for s, v in by_sigma.items()
                 if "mrgap" in v and "mrgap_oracle" in v]
        if gains:
            s, gain = max(gains, key=lambda item: item[1])
            lines.append(f"- Largest absolute oracle-tangent gain: {gain:.5f} at sigma={s:g}.")
        lines.append("")
    lines += ["A non-improving endpoint means reconstruction GRMSE is no smaller than the noisy input; it is a diagnostic convention, not a universal failure definition.",
              "", "## Interpretation constraints", "",
              "- Observation samples are freshly generated for every `(manifold,n,sigma,repeat)`; they are not prefixes or subsets of a 100,000-point cloud.",
              "- Dense reference samples are independent and used only for evaluation; each trial's exact clean observations are added to that reference to remove Monte Carlo distance floor.",
              "- Manifold Fitting is given the true simulation noise `sigma`, as required by its official implementation. MrGap uses fixed paper hyperparameters.",
              "- The official Manifold Fitting bandwidth is undefined at `sigma=0`; the benchmark records an explicit identity convention there.",
              "- Manifold Fitting is one official pass, not an iterative optimizer. MrGap rows report cumulative runtime for every round from 1 through 5.",
              "- For half-torus MrGap, the paper publishes only last-round GP parameters; the benchmark reuses them for every round.",
              "", "## Plots", "",
              "- `noise_sensitivity.png`: failure as ambient noise increases.",
              "- `sample_size_sensitivity.png`: error against independently generated sample size.",
              "- `error_vs_local_neighbors.png`: distinguishes global `n` from effective local sample size.",
              "- `bandwidth_sensitivity.png`: method-specific neighborhood scans.",
              "- `oracle_tangent.png`: local-PCA MrGap against true-tangent MrGap.",
              "- `runtime_scaling.png`: cumulative method runtime against `n`.", "",
              "- `iteration_comparison.png`: MrGap 1--5-round scan at each default condition.", "",
              "See `benchmark_rows.csv` for the requested `method | manifold | n | sigma | error | runtime` data and all diagnostics."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark")
    parser.add_argument("--manifolds", nargs="+", choices=tuple(MANIFOLDS), default=list(MANIFOLDS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repeats = args.repeats if args.repeats is not None else (1 if args.profile == "pilot" else 3)
    reference_n = 20_000 if args.profile == "pilot" else 100_000
    selected = [MANIFOLDS[name] for name in args.manifolds]
    references: dict[str, cKDTree] = {}
    for j, manifold in enumerate(selected):
        ref, _ = manifold.sample(np.random.default_rng(seed_for(args.seed, 900, j)), reference_n)
        references[manifold.name] = cKDTree(ref)

    rows: list[dict[str, object]] = []
    for j, manifold in enumerate(selected):
        reference = references[manifold.name]
        noise_n = manifold.default_n if args.profile == "full" else min(manifold.default_n, 500)
        for repeat in range(repeats):
            for k, sigma in enumerate(NOISE_LEVELS):
                clean, noisy, latent = make_data(manifold, noise_n, sigma, seed_for(args.seed, j, 1, repeat, k))
                trial_reference = augmented_reference(reference, clean)
                normal_rows = evaluate_methods("noise", manifold, repeat, clean, noisy, latent, trial_reference, sigma)
                rows += normal_rows
                # Put the matched ordinary two-round result beside the oracle
                # result under one experiment label for direct plotting.
                rows += [dict(r, experiment="oracle") for r in normal_rows
                         if r["method"] == "mrgap" and int(r["iterations"]) == 2]
                rows += evaluate_methods("oracle", manifold, repeat, clean, noisy, latent, trial_reference, sigma,
                                         iterations=(2,), oracle=True)
            for k, n in enumerate(SAMPLE_SIZES):
                # Every n gets a fresh draw. No prefix/subsampling coupling.
                clean, noisy, latent = make_data(manifold, n, manifold.default_sigma,
                                                  seed_for(args.seed, j, 2, repeat, k))
                trial_reference = augmented_reference(reference, clean)
                if args.profile == "pilot" and n > 1000:
                    # Manifold Fitting remains cheap; exact MrGap's dense local GP is
                    # deliberately marked rather than silently approximated.
                    raw = error_to_reference(noisy, trial_reference)
                    local_counts = np.fromiter(
                        map(len, cKDTree(noisy).query_ball_point(noisy, manifold.epsilon)), dtype=int
                    )
                    for iteration in MRGAP_ITERATIONS:
                        skipped = base_record("sample_size", manifold, repeat, n,
                                              manifold.default_sigma, raw)
                        skipped.update({"method": "mrgap", "iterations": iteration,
                                        "epsilon": manifold.epsilon, "delta": manifold.delta,
                                        "median_local_neighborhood": float(np.median(local_counts)),
                                        "min_local_neighborhood": int(np.min(local_counts)),
                                        "reconstruction_error": float("nan"),
                                        "paired_rmse": float("nan"), "runtime_sec": float("nan"),
                                        "peak_memory_mb": float("nan"),
                                        "status": "skipped_resource_guard"})
                        rows.append(skipped)
                    fitted, diag, runtime, memory = measured(lambda: manifold_fitting(noisy, manifold.default_sigma))
                    row = base_record("sample_size", manifold, repeat, n, manifold.default_sigma, raw)
                    row.update({"method": "manifold_fitting", "iterations": 1,
                                "median_local_neighborhood": diag["median_local_neighborhood"],
                                "min_local_neighborhood": diag["min_local_neighborhood"],
                                "reconstruction_error": error_to_reference(fitted, trial_reference),
                                "paired_rmse": paired_rmse(fitted, clean), "runtime_sec": runtime,
                                "peak_memory_mb": memory})
                    rows.append(row)
                else:
                    rows += evaluate_methods("sample_size", manifold, repeat, clean, noisy, latent,
                                             trial_reference, manifold.default_sigma)
            # Bandwidth scans use independently generated default-condition data.
            clean, noisy, latent = make_data(manifold, noise_n, manifold.default_sigma,
                                              seed_for(args.seed, j, 3, repeat))
            trial_reference = augmented_reference(reference, clean)
            for epsilon in manifold.epsilon_grid:
                rows += evaluate_methods("bandwidth_mrgap", manifold, repeat, clean, noisy, latent,
                                         trial_reference, manifold.default_sigma, iterations=(2,), epsilon=epsilon)
                rows.pop()  # remove MF duplicate from the MrGap epsilon scan
            for multiplier in MF_BANDWIDTH_MULTIPLIERS:
                result = evaluate_methods("bandwidth_mf", manifold, repeat, clean, noisy, latent,
                                          trial_reference, manifold.default_sigma, iterations=(1,), mf_multiplier=multiplier)
                rows.append(result[-1])

    write_csv(args.output / "benchmark_rows.csv", rows)
    metadata = {"profile": args.profile, "repeats": repeats, "seed": args.seed,
                "reference_n": reference_n, "noise_levels": NOISE_LEVELS,
                "sample_sizes": SAMPLE_SIZES, "mrgap_iterations": MRGAP_ITERATIONS,
                "fresh_sample_for_every_n": True,
                "official_manifold_fitting_source": "https://github.com/zhigang-yao/manifold-fitting/tree/master/Manifold%20Fitting/Matlab",
                "manifolds": args.manifolds}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_grid(rows, "noise", "sigma", "ambient noise sigma", args.output / "noise_sensitivity.png")
    plot_grid(rows, "sample_size", "n", "number of observations", args.output / "sample_size_sensitivity.png")
    plot_sample_neighbors(rows, args.output / "error_vs_local_neighbors.png")
    plot_grid(rows, "oracle", "sigma", "ambient noise sigma", args.output / "oracle_tangent.png", True)
    runtime_plot(rows, args.output / "runtime_scaling.png")
    iteration_plot(rows, args.output / "iteration_comparison.png")
    # Bandwidth methods have different x axes; normalized multiplier is used for MF.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, name in zip(axes.flat, MANIFOLDS):
        mr = [r for r in rows if r["experiment"] == "bandwidth_mrgap" and r["manifold"] == name]
        mf = [r for r in rows if r["experiment"] == "bandwidth_mf" and r["manifold"] == name]
        if mr:
            x = sorted({float(r["epsilon"]) for r in mr}); y = [np.mean([float(r["reconstruction_error"]) for r in mr if float(r["epsilon"]) == z]) for z in x]
            ax.plot(np.asarray(x) / MANIFOLDS[name].epsilon, y, "o-", label="MrGap epsilon/default")
        if mf:
            x = sorted({float(r["bandwidth_multiplier"]) for r in mf}); y = [np.mean([float(r["reconstruction_error"]) for r in mf if float(r["bandwidth_multiplier"]) == z]) for z in x]
            ax.plot(x, y, "o-", label="MF r,R/default")
        ax.set_title(name); ax.set_xlabel("normalized neighborhood scale"); ax.set_ylabel("GRMSE"); ax.grid(alpha=.25)
    axes.flat[0].legend(frameon=False); fig.tight_layout(); fig.savefig(args.output / "bandwidth_sensitivity.png", dpi=170); plt.close(fig)
    report(args.output, rows, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
