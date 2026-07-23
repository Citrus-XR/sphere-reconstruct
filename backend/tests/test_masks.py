"""imaging.masks の単体テスト. cv2/numpy のみで torch 不要."""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.imaging import masks


def test_plan_downsample_no_shrink_when_within_limit():
    plan = masks.plan_downsample(1000, 800, max_size=1024)
    assert plan.scale == 1.0
    assert (plan.dst_w, plan.dst_h) == (1000, 800)


def test_plan_downsample_shrinks_long_edge():
    plan = masks.plan_downsample(3840, 3840, max_size=1024)
    assert plan.scale == 1024 / 3840
    assert plan.dst_w == 1024
    assert plan.dst_h == 1024


def test_plan_downsample_preserves_aspect():
    plan = masks.plan_downsample(4000, 2000, max_size=1000)
    assert plan.dst_w == 1000
    assert plan.dst_h == 500


def test_plan_downsample_disabled_with_zero():
    plan = masks.plan_downsample(8000, 4000, max_size=0)
    assert plan.scale == 1.0


def test_upscale_mask_roundtrip_size():
    small = np.zeros((256, 256), dtype=np.uint8)
    small[64:192, 64:192] = 1  # center square
    big = masks.upscale_mask(small, 1024, 1024)
    assert big.shape == (1024, 1024)
    assert big.dtype == np.uint8
    # 中央は 1, 隅は 0.
    assert big[512, 512] == 1
    assert big[0, 0] == 0


def test_union_masks():
    a = np.zeros((10, 10), dtype=np.uint8)
    a[:5, :] = 1
    b = np.zeros((10, 10), dtype=np.uint8)
    b[5:, :] = 1
    u = masks.union_masks([a, b])
    assert u is not None
    assert u.sum() == 100  # 全面 1


def test_union_masks_empty_returns_none():
    assert masks.union_masks([]) is None


def test_coverage_ratio():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[:2, :] = 1  # 20 / 100
    assert abs(masks.coverage_ratio(m) - 0.2) < 1e-9


def test_write_mask_png_invert_for_colmap(tmp_path):
    import cv2

    m = np.zeros((8, 8), dtype=np.uint8)
    m[0:4, :] = 1  # 検出物体
    out = tmp_path / "mask.png"
    masks.write_mask_png(m, out, invert=True)
    loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # invert: 検出物体 (m=1) は 0, それ以外 255.
    assert loaded[0, 0] == 0
    assert loaded[7, 7] == 255
