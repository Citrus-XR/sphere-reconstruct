"""Robust similarity alignment with explicit degeneracy checks."""

from __future__ import annotations

import numpy as np


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("similarity requires matching finite Nx3 coordinates with at least three points")
    centered = source - source.mean(axis=0)
    other = target - target.mean(axis=0)
    for values in (centered, other):
        singular = np.linalg.svd(values, compute_uv=False)
        if singular[0] < 1e-12 or singular[1] < singular[0] * 1e-3:
            raise ValueError("collinear camera centers cannot constrain a similarity rotation")
    left, singular, right = np.linalg.svd(centered.T @ other)
    signs = np.array([1.0, 1.0, np.linalg.det(left @ right)])
    rotation = left @ np.diag(signs) @ right
    scale = float(np.sum(singular * signs) / np.sum(centered ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("similarity scale must be finite and positive")
    return scale, rotation, target.mean(axis=0) - scale * source.mean(axis=0) @ rotation


def robust_similarity(source: np.ndarray, target: np.ndarray, threshold: float) -> tuple:
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (len(source) < 8 or source.shape != target.shape or source.shape[1:] != (3,)
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("robust similarity requires eight finite pairs and a positive threshold")
    rng = np.random.default_rng(0)
    minimum_inliers = max(8, len(source) // 2 + 1)
    best = None
    for _ in range(1500):
        indices = rng.choice(len(source), 4, replace=False)
        try:
            candidate = fit_similarity(source[indices], target[indices])
        except ValueError:
            continue
        scale, rotation, translation = candidate
        error = np.linalg.norm(scale * source @ rotation + translation - target, axis=1)
        inliers = error < threshold
        score = (int(inliers.sum()), -float(np.median(error)))
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[0][0] < minimum_inliers:
        raise ValueError("no majority-supported similarity found")
    inliers = best[1]
    for _ in range(5):
        transform = fit_similarity(source[inliers], target[inliers])
        scale, rotation, translation = transform
        errors = np.linalg.norm(scale * source @ rotation + translation - target, axis=1)
        updated = errors < threshold
        if np.array_equal(updated, inliers):
            break
        if updated.sum() < minimum_inliers:
            raise ValueError("similarity refinement lost majority support")
        inliers = updated
    return transform, errors, inliers
