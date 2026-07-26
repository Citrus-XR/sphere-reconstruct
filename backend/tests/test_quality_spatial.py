"""imaging.quality と sampling.select_spatial の単体テスト (cv2/numpy, torch 不要)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.imaging import quality, sampling  # noqa: E402


def test_exposure_stats_normal():
    gray = np.full((64, 64), 128, dtype=np.uint8)
    st = quality.exposure_stats(gray)
    assert st.under_ratio == 0.0
    assert st.over_ratio == 0.0
    assert st.is_ok()


def test_exposure_stats_overexposed():
    gray = np.full((64, 64), 255, dtype=np.uint8)
    st = quality.exposure_stats(gray)
    assert st.over_ratio == 1.0
    assert not st.is_ok()


def test_exposure_stats_underexposed():
    gray = np.zeros((64, 64), dtype=np.uint8)
    st = quality.exposure_stats(gray)
    assert st.under_ratio == 1.0
    assert not st.is_ok()


def test_sift_feature_count_more_on_textured():
    rng = np.random.default_rng(0)
    textured = (rng.random((256, 256)) * 255).astype(np.uint8)
    flat = np.full((256, 256), 128, dtype=np.uint8)
    assert quality.sift_feature_count(textured, downscale=1) > quality.sift_feature_count(flat, downscale=1)


def test_optical_flow_median_zero_on_identical():
    rng = np.random.default_rng(1)
    img = (rng.random((128, 128)) * 255).astype(np.uint8)
    m = quality.optical_flow_median(img, img, downscale=1)
    assert m < 0.5  # ほぼ 0


def test_optical_flow_median_positive_on_shift():
    rng = np.random.default_rng(2)
    img = (rng.random((128, 128)) * 255).astype(np.uint8)
    shifted = np.roll(img, 10, axis=1)
    m = quality.optical_flow_median(img, shifted, downscale=1)
    assert m > 0.0


# -- spatial selector ---------------------------------------------------------


def _cand(idx, ts, sharp, exp_ok=True, feats=100):
    return sampling.Candidate(
        index=idx, timestamp_us=ts, sharpness=sharp, exposure_ok=exp_ok, feature_count=feats
    )


def test_select_spatial_fast_layer_rejects_blur_and_exposure():
    cands = [
        _cand(0, 0, 100.0),
        _cand(1, 1000, 5.0),  # blur (below min_sharpness)
        _cand(2, 2000, 100.0, exp_ok=False),  # exposure
        _cand(3, 3000, 100.0),
    ]
    # motion: 常に target を超える -> 各 valid を選ぶ
    cfg = sampling.SpatialConfig(min_sharpness=10.0, target_motion=0.0)
    res = sampling.select_spatial(cands, lambda a, b: 999.0, cfg)
    assert res.reasons["blur"] == 1
    assert res.reasons["exposure"] == 1
    assert 1 not in res.selected_indices
    assert 2 not in res.selected_indices


def test_select_spatial_greedy_spacing_by_motion():
    # 10 候補, motion = index 差. target=3 -> 約 3 間隔で選ぶ.
    cands = [_cand(i, i * 1000, 100.0) for i in range(10)]
    cfg = sampling.SpatialConfig(target_motion=3.0)
    res = sampling.select_spatial(cands, lambda a, b: float(b - a), cfg)
    # 0 を seed, motion>=3 で 3, 6, 9 ... 実際は窓の quality 最大 (全部同じなので最初).
    assert res.selected_indices[0] == 0
    # 選ばれた間隔が概ね target*min_spacing_frac 以上.
    diffs = [
        b - a for a, b in zip(res.selected_indices[:-1], res.selected_indices[1:], strict=True)
    ]
    assert all(d >= 2 for d in diffs)


def test_select_spatial_picks_best_quality_in_window():
    # 間隔帯 [target*0.7, ...] に入る候補の中で feature_count が大きいものを選ぶ.
    # target=4, min_spacing=2.8. motion=index 差. 窓で m>=2.8 は index 3,4.
    # その中で feats 最大を選ぶ.
    cands = [
        _cand(0, 0, 100.0, feats=100),
        _cand(1, 1000, 100.0, feats=100),
        _cand(2, 2000, 100.0, feats=100),
        _cand(3, 3000, 100.0, feats=500),  # 間隔帯内で高品質
        _cand(4, 4000, 100.0, feats=100),
    ]
    cfg = sampling.SpatialConfig(target_motion=4.0)
    res = sampling.select_spatial(cands, lambda a, b: float(b - a), cfg)
    # 0 seed, motion>=4 は index 4 で発火, 帯 [2.8,4] = index 3,4 の中で feats 最大の 3.
    assert 3 in res.selected_indices


def test_select_spatial_max_frames_cap():
    cands = [_cand(i, i * 1000, 100.0) for i in range(20)]
    cfg = sampling.SpatialConfig(target_motion=1.0, max_frames=3)
    res = sampling.select_spatial(cands, lambda a, b: float(b - a), cfg)
    assert len(res.selected_indices) <= 3
    assert res.selected_indices[0] == 0
    assert res.selected_indices[-1] == 19
