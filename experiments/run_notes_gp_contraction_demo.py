#!/usr/bin/env python3
"""Compatibility runner for notes_gp_contraction_demo.

The underlying experiment uses ``frames_from_closed_curve`` in two contexts with
slightly different unpacking.  Keep the main experiment unchanged and normalize the
helper API here so CI can run the intended notes-faithful experiment.
"""
from __future__ import annotations

import numpy as np

import experiments.notes_gp_contraction_demo as demo

_orig_frames = demo.frames_from_closed_curve


def _frames3(points: np.ndarray, center=None):
    tangent, normal = _orig_frames(points, center=center)
    return None, tangent, normal


def _coarse_radial_scaffold(sample, phi_grid, angle_bandwidth):
    center = np.mean(sample, axis=0)
    rel = sample - center
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)
    delta = demo.angle_diff(theta[None, :] - phi_grid[:, None])
    weights = np.exp(-0.5 * (delta / angle_bandwidth) ** 2)
    rhat = (weights @ radius) / np.maximum(weights.sum(axis=1), 1e-14)
    scaffold = center + rhat[:, None] * np.column_stack(
        (np.cos(phi_grid), np.sin(phi_grid))
    )
    tangent, normal = _orig_frames(scaffold, center=center)
    return scaffold, tangent, normal


demo.frames_from_closed_curve = _frames3
demo.coarse_radial_scaffold = _coarse_radial_scaffold

if __name__ == "__main__":
    demo.main()
