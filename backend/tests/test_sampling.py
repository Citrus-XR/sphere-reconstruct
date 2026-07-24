"""imaging.sampling の単体テスト (cv2/numpy のみ, torch 不要)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.imaging import sampling  # noqa: E402


def test_laplacian_sharpness_blurred_is_lower():
    # ランダムノイズ画像 (鮮鋭) vs ぼかした画像.
    rng = np.random.default_rng(0)
    sharp = (rng.random((128, 128)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (11, 11), 5)
    s_sharp = sampling.laplacian_sharpness(sharp)
    s_blur = sampling.laplacian_sharpness(blurred)
    assert s_sharp > s_blur


def test_candidate_indices_basic():
    idx = sampling.candidate_indices(center_index=100, span=20, count=5)
    assert len(idx) == 5
    assert idx[0] == 90
    assert idx[-1] == 110
    assert idx == sorted(idx)


def test_candidate_indices_clamps_negative_and_bound():
    idx = sampling.candidate_indices(center_index=5, span=20, count=5, fps_bound=8)
    assert min(idx) >= 0
    assert max(idx) <= 8


def test_candidate_indices_count_one():
    assert sampling.candidate_indices(10, 20, 1) == [10]


def test_pick_sharpest():
    assert sampling.pick_sharpest([1.0, 5.0, 3.0]) == 1
    assert sampling.pick_sharpest([]) == 0


def test_sharpness_of_file(tmp_path):
    rng = np.random.default_rng(1)
    sharp = (rng.random((256, 256)) * 255).astype(np.uint8)
    p = tmp_path / "sharp.jpg"
    cv2.imwrite(str(p), sharp)
    s = sampling.sharpness_of_file(p, downscale=2)
    assert s > 0
