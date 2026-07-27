"""フレーム品質・運動の指標 (空間抽出の 2 層フィルタ用).

快速層 (安価):
  - laplacian_sharpness (sampling.py)  … ブレ
  - exposure_stats                     … 過曝 / 欠曝
精確層 (高価):
  - sift_feature_count                 … 特徴量の豊富さ
  - optical_flow_median                … 前選択フレームからの運動量

cv2 / numpy のみ. torch 不要なので単体テスト可能. ALIKED (学習特徴) は将来
別モジュールで追加する余地を残し, ここでは SIFT を使う.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


@dataclass
class ExposureStats:
    under_ratio: float  # 暗部 (<= low) に張り付いた画素の割合
    over_ratio: float  # 明部 (>= high) に張り付いた画素の割合
    mean: float

    def is_ok(self, max_clip: float = 0.25) -> bool:
        """過曝/欠曝が閾値以下なら True."""
        return self.under_ratio <= max_clip and self.over_ratio <= max_clip


def exposure_stats(gray: np.ndarray, low: int = 8, high: int = 247) -> ExposureStats:
    """グレースケール画像の露出統計. 過曝/欠曝の張り付き割合を測る."""
    cv2 = _cv2()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    total = gray.size
    under = float((gray <= low).sum()) / total
    over = float((gray >= high).sum()) / total
    return ExposureStats(under_ratio=under, over_ratio=over, mean=float(gray.mean()))


# SIFT instance は thread 間で共有せず、candidate worker ごとに再利用する。
_sift_by_thread = threading.local()


def _get_sift(max_features: int = 4000):
    sift = getattr(_sift_by_thread, "value", None)
    if sift is None:
        cv2 = _cv2()
        sift = cv2.SIFT_create(nfeatures=max_features)
        _sift_by_thread.value = sift
    return sift


def sift_feature_count(gray: np.ndarray, downscale: int = 2) -> int:
    """SIFT キーポイント数. downscale で縮小して高速化 (相対比較には十分)."""
    cv2 = _cv2()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if downscale > 1:
        h, w = gray.shape[:2]
        gray = cv2.resize(
            gray, (max(1, w // downscale), max(1, h // downscale)), interpolation=cv2.INTER_AREA
        )
    kp = _get_sift().detect(gray, None)
    return len(kp)


def optical_flow_median(prev_gray: np.ndarray, cur_gray: np.ndarray, downscale: int = 4) -> float:
    """2 フレーム間の dense optical flow (Farneback) の中央値ノルム.

    運動量の代理指標. downscale で縮小してから計算する. 単位は「縮小後の画素」.
    値が小さい = ほぼ静止 (フレーム間が近すぎ), 大きい = 大きく動いた.
    """
    cv2 = _cv2()
    if prev_gray.ndim == 3:
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    if cur_gray.ndim == 3:
        cur_gray = cv2.cvtColor(cur_gray, cv2.COLOR_BGR2GRAY)
    if downscale > 1:
        h, w = prev_gray.shape[:2]
        size = (max(1, w // downscale), max(1, h // downscale))
        prev_gray = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        cur_gray = cv2.resize(cur_gray, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        cur_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.median(mag))
