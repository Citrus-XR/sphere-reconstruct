"""フレーム選択の指標.

「空間抽出」= 固定時間間隔ではなく, 各区間で最も鮮鋭なフレームを選ぶ. 鮮鋭度は
Laplacian の分散 (Tenengrad より軽量で実用的) で測る. ブレたフレームは分散が低い.

cv2 / numpy のみ. torch 不要なので単体テスト可能.
"""

from __future__ import annotations

import numpy as np


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


def laplacian_sharpness(gray: np.ndarray) -> float:
    """グレースケール画像の Laplacian 分散. 大きいほど鮮鋭.

    入力が BGR/RGB の場合は呼び出し側でグレー化しておくこと.
    """
    cv2 = _cv2()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def sharpness_of_file(path, downscale: int = 4) -> float:
    """JPEG を読み, (高速化のため) 縮小してから鮮鋭度を測る.

    downscale=4 で長辺 1/4 に縮小. ブレ判定には十分で, 8K でも高速.
    """
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    if downscale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w // downscale), max(1, h // downscale)), interpolation=cv2.INTER_AREA)
    return laplacian_sharpness(img)


def candidate_indices(center_index: int, span: int, count: int, fps_bound: int | None = None) -> list[int]:
    """center を中心に ±span/2 の範囲へ count 個の候補 frame index を均等配置する.

    負の index は 0 に丸める. fps_bound (総フレーム数-1) が与えられれば上限で clamp.
    重複は除いて昇順で返す.
    """
    if count <= 1:
        return [max(0, center_index)]
    half = span // 2
    start = center_index - half
    step = span / (count - 1) if count > 1 else 0
    idx = sorted({max(0, int(round(start + i * step))) for i in range(count)})
    if fps_bound is not None:
        idx = sorted({min(fps_bound, i) for i in idx})
    return idx


def pick_sharpest(scores: list[float]) -> int:
    """スコア列から最大の index を返す. 空なら 0."""
    if not scores:
        return 0
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


# -----------------------------------------------------------------------------
# 空間抽出 (2 層多基準選択)
# -----------------------------------------------------------------------------
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Candidate:
    """候補フレーム 1 枚の計測結果.

    快速層: sharpness (Laplacian 分散), exposure_ok (過曝/欠曝でない)
    精確層: feature_count (SIFT 特徴数)
    """

    index: int  # source frame index
    timestamp_us: int
    sharpness: float
    exposure_ok: bool
    feature_count: int = 0


@dataclass
class SpatialConfig:
    min_sharpness: float = 0.0  # これ未満は快速層で棄却 (0 = 無効)
    min_features: int = 0  # これ未満は精確層で棄却
    target_motion: float = 1.5  # 前選択フレームからの運動量がこれを超えたら次を選ぶ
    min_spacing_frac: float = 0.7  # 最小間隔 = target_motion * これ (これ未満の候補は近すぎ)
    max_frames: int = 0  # 0 = 無制限


@dataclass
class SpatialResult:
    selected_indices: list[int]
    rejected_fast: int  # 快速層で落ちた数
    reasons: dict[str, int] = field(default_factory=dict)


def _quality(c: Candidate) -> float:
    """選択優先度. 特徴数と鮮鋭度を組み合わせる (正規化は呼び出し前提の相対比較)."""
    return c.sharpness * (1.0 + c.feature_count)


def select_spatial(
    candidates: list[Candidate],
    motion_fn: Callable[[int, int], float],
    config: SpatialConfig,
) -> SpatialResult:
    """2 層 + 貪欲な空間間隔でフレームを選ぶ.

    1. 快速層: sharpness >= min_sharpness かつ exposure_ok.
    2. 精確層: feature_count >= min_features.
    3. 貪欲間隔: 最初の valid を選び, 前選択からの motion が target_motion を超えたら,
       「間隔帯 [target*min_spacing_frac, ...]」に入る候補の中で _quality 最大のものを
       選ぶ. これにより最小間隔を保ちつつ, その付近で最も高品質なフレームを採る.
       「時間」ではなく「視覚/運動の変化量」で等間隔サンプルする.

    motion_fn(a_index, b_index): a と b の運動量 (光流中央値 or IMU 回転角). 各候補につき
    1 回だけ呼ぶよう内部でキャッシュする. candidates は時刻昇順であること.
    """
    reasons = {"blur": 0, "exposure": 0, "few_features": 0}
    valid: list[Candidate] = []
    for c in candidates:
        if config.min_sharpness > 0 and c.sharpness < config.min_sharpness:
            reasons["blur"] += 1
            continue
        if not c.exposure_ok:
            reasons["exposure"] += 1
            continue
        if config.min_features > 0 and c.feature_count < config.min_features:
            reasons["few_features"] += 1
            continue
        valid.append(c)

    rejected = len(candidates) - len(valid)
    if not valid:
        return SpatialResult(selected_indices=[], rejected_fast=rejected, reasons=reasons)

    min_spacing = config.target_motion * config.min_spacing_frac
    selected = [valid[0]]
    last = valid[0]
    window: list[tuple[Candidate, float]] = []  # (候補, last からの motion)
    for c in valid[1:]:
        m = motion_fn(last.index, c.index)
        window.append((c, m))
        if m >= config.target_motion:
            # 最小間隔を満たす候補の中で quality 最大を選ぶ.
            eligible = [cand for cand, mm in window if mm >= min_spacing]
            if not eligible:
                eligible = [c]
            best = max(eligible, key=_quality)
            selected.append(best)
            last = best
            window = []
    if selected[-1].index != valid[-1].index:
        selected.append(valid[-1])
    if config.max_frames and len(selected) > config.max_frames:
        # 先頭から N 枚で打ち切ると動画後半を失う. 全選択列を均等に間引き, 始終端を残す.
        positions = np.linspace(0, len(selected) - 1, config.max_frames)
        selected = [selected[round(position)] for position in positions]

    return SpatialResult(
        selected_indices=[c.index for c in selected],
        rejected_fast=rejected,
        reasons=reasons,
    )
