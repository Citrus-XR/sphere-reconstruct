from __future__ import annotations

import numpy as np
import pytest
from sphere_reconstruct.colmap.sequence_alignment import fit_similarity, robust_similarity


def test_robust_alignment_ignores_repeated_object_offsets_without_fitting_heldout_points():
    rng = np.random.default_rng(40)
    source = rng.normal(size=(200, 3))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    truth = 2.4 * source @ rotation + [2, -3, 4]
    noisy = truth[:150].copy()
    noisy[:35] += [1.6, 0, 0]
    (scale, actual_rotation, translation), errors, inliers = robust_similarity(source[:150], noisy, 0.05)
    assert int(inliers.sum()) == 115
    assert np.all(errors[:35] > 1.5)
    np.testing.assert_allclose(scale * source[150:] @ actual_rotation + translation, truth[150:], atol=1e-10)


def test_planar_path_constrains_rotation_but_line_does_not():
    source = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    fit_similarity(source, source + [1, 2, 3])
    with pytest.raises(ValueError, match="collinear"):
        fit_similarity(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))


def test_alignment_rejects_nonfinite_data():
    source = np.zeros((10, 3))
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        robust_similarity(source, source, 0.1)
