"""フレーム選択の指標.

「空間抽出」= 固定時間間隔ではなく, 各区間で最も鮮鋭なフレームを選ぶ. 鮮鋭度は
Laplacian の分散 (Tenengrad より軽量で実用的) で測る. ブレたフレームは分散が低い.

cv2 / numpy のみ. torch 不要なので単体テスト可能.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
    rolling_shutter_motion_deg: float = 0.0


@dataclass
class SpatialConfig:
    min_sharpness: float = 0.0  # これ未満は快速層で棄却 (0 = 無効)
    min_features: int = 0  # これ未満は精確層で棄却
    target_motion: float = 1.5  # 前選択フレームからの運動量がこれを超えたら次を選ぶ
    min_spacing_frac: float = 0.7  # 最小間隔 = target_motion * これ (これ未満の候補は近すぎ)
    max_frames: int = 0  # 0 = 無制限
    max_rolling_shutter_motion_deg: float = float("inf")
    max_temporal_gap_sec: float = 4.0  # 0 = continuity bridge を無効化
    continuity_strategy: str = "balanced"


@dataclass
class SpatialResult:
    selected_indices: list[int]
    rejected_fast: int  # 快速層で落ちた数
    reasons: dict[str, int] = field(default_factory=dict)
    bridge_frames: int = 0
    bridge_relaxed_sharpness: int = 0
    bridge_relaxed_rolling_shutter: int = 0
    unresolved_gaps: int = 0
    maximum_gap_sec: float = 0.0


def _quality(c: Candidate) -> float:
    """選択優先度. 特徴数と鮮鋭度を組み合わせる (正規化は呼び出し前提の相対比較)."""
    return c.sharpness * (1.0 + c.feature_count)


def _strict_rejection(c: Candidate, config: SpatialConfig) -> str | None:
    if config.min_sharpness > 0 and c.sharpness < config.min_sharpness:
        return "blur"
    if not c.exposure_ok:
        return "exposure"
    if config.min_features > 0 and c.feature_count < config.min_features:
        return "few_features"
    if c.rolling_shutter_motion_deg > config.max_rolling_shutter_motion_deg:
        return "rolling_shutter"
    return None


def _bridgeable(c: Candidate, config: SpatialConfig) -> bool:
    """Return whether a frame can rescue a temporal hole.

    Exposure failure and too few features make a frame unsuitable even as a bridge.
    Sharpness and rolling-shutter thresholds are deliberately soft here: a locally
    suboptimal image is preferable to leaving a reconstruction graph disconnected.
    """
    return c.exposure_ok and (config.min_features <= 0 or c.feature_count >= config.min_features)


def _maximum_gap_sec(selected: list[Candidate]) -> float:
    if len(selected) < 2:
        return 0.0
    return max(
        (right.timestamp_us - left.timestamp_us) / 1_000_000
        for left, right in zip(selected, selected[1:], strict=False)
    )


def _bridge_candidate(
    candidates: list[Candidate],
    previous: Candidate,
    config: SpatialConfig,
    motion: Callable[[int, int], float],
) -> Candidate:
    """Pick one bridge frame according to the requested continuity policy."""
    if config.continuity_strategy == "quality":
        return max(candidates, key=lambda candidate: (_quality(candidate), candidate.timestamp_us))

    scored = [(candidate, motion(previous.index, candidate.index)) for candidate in candidates]
    minimum_motion = config.target_motion * config.min_spacing_frac
    sufficient = [(candidate, value) for candidate, value in scored if value >= minimum_motion]
    if not sufficient:
        return max(scored, key=lambda item: (item[1], _quality(item[0]), item[0].timestamp_us))[0]

    if config.continuity_strategy == "parallax":
        return min(
            sufficient,
            key=lambda item: (abs(item[1] - config.target_motion), -_quality(item[0]), -item[0].timestamp_us),
        )[0]

    # Balanced policy: retain adequate parallax, then prefer the sharpest and
    # best-featured image within the useful motion band. If all viable choices
    # overshoot the band, use the one nearest the target instead.
    upper_motion = max(minimum_motion, config.target_motion * 1.5)
    balanced = [(candidate, value) for candidate, value in sufficient if value <= upper_motion]
    if balanced:
        return max(balanced, key=lambda item: (_quality(item[0]), item[0].timestamp_us))[0]
    return min(
        sufficient,
        key=lambda item: (abs(item[1] - config.target_motion), -_quality(item[0]), -item[0].timestamp_us),
    )[0]


def _insert_continuity_bridges(
    selected: list[Candidate],
    candidates: list[Candidate],
    config: SpatialConfig,
    motion: Callable[[int, int], float],
) -> tuple[list[Candidate], int, int]:
    """Fill long selected-frame gaps without changing normal spatial sampling.

    A bridge is selected from the latter half of each maximum-gap window where
    possible. This advances the trajectory while leaving room to choose a sharp
    candidate with useful parallax, rather than repeatedly selecting near the
    previous frame.
    """
    if config.max_temporal_gap_sec <= 0 or len(selected) < 2:
        return selected, 0, 0

    max_gap_us = int(config.max_temporal_gap_sec * 1_000_000)
    candidates_by_time = sorted(candidates, key=lambda candidate: candidate.timestamp_us)
    result = [selected[0]]
    bridges = 0
    unresolved = 0
    remaining_budget = max(0, config.max_frames - len(selected)) if config.max_frames else None

    for target in selected[1:]:
        previous = result[-1]
        while target.timestamp_us - previous.timestamp_us > max_gap_us:
            if remaining_budget is not None and remaining_budget <= 0:
                unresolved += 1
                break
            deadline = previous.timestamp_us + max_gap_us
            pool = [
                candidate
                for candidate in candidates_by_time
                if previous.timestamp_us < candidate.timestamp_us <= deadline
            ]
            if not pool:
                unresolved += 1
                break
            tail_start = previous.timestamp_us + max_gap_us // 2
            tail = [candidate for candidate in pool if candidate.timestamp_us >= tail_start]
            bridge = _bridge_candidate(tail or pool, previous, config, motion)
            result.append(bridge)
            previous = bridge
            bridges += 1
            if remaining_budget is not None:
                remaining_budget -= 1
        result.append(target)
    return result, bridges, unresolved


def select_spatial(
    candidates: list[Candidate],
    motion_fn: Callable[[int, int], float],
    config: SpatialConfig,
    progress: Callable[[int, int], None] | None = None,
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
    if config.max_temporal_gap_sec < 0:
        raise ValueError("max_temporal_gap_sec must be >= 0")
    if config.continuity_strategy not in {"quality", "parallax", "balanced"}:
        raise ValueError(f"unsupported continuity_strategy: {config.continuity_strategy}")

    reasons = {"blur": 0, "exposure": 0, "few_features": 0, "rolling_shutter": 0}
    valid: list[Candidate] = []
    bridgeable: list[Candidate] = []
    rejection_by_index: dict[int, str | None] = {}
    for c in candidates:
        rejection = _strict_rejection(c, config)
        rejection_by_index[c.index] = rejection
        if rejection is None:
            valid.append(c)
        else:
            reasons[rejection] += 1
        if _bridgeable(c, config):
            bridgeable.append(c)

    rejected = len(candidates) - len(valid)
    if not valid:
        return SpatialResult(selected_indices=[], rejected_fast=rejected, reasons=reasons)

    min_spacing = config.target_motion * config.min_spacing_frac
    selected = [valid[0]]
    last = valid[0]
    window: list[tuple[Candidate, float]] = []  # (候補, last からの motion)
    motion_total = max(0, len(valid) - 1)
    for motion_number, c in enumerate(valid[1:], 1):
        m = motion_fn(last.index, c.index)
        if progress is not None:
            progress(motion_number, motion_total)
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

    motion_cache: dict[tuple[int, int], float] = {}

    def cached_motion(first: int, second: int) -> float:
        key = (first, second)
        if key not in motion_cache:
            motion_cache[key] = motion_fn(first, second)
        return motion_cache[key]

    selected, bridge_frames, unresolved_gaps = _insert_continuity_bridges(
        selected,
        bridgeable,
        config,
        cached_motion,
    )
    strict_indices = {candidate.index for candidate in valid}
    relaxed_sharpness = sum(
        rejection_by_index.get(candidate.index) == "blur"
        for candidate in selected
        if candidate.index not in strict_indices
    )
    relaxed_rolling_shutter = sum(
        rejection_by_index.get(candidate.index) == "rolling_shutter"
        for candidate in selected
        if candidate.index not in strict_indices
    )

    return SpatialResult(
        selected_indices=[c.index for c in selected],
        rejected_fast=rejected,
        reasons=reasons,
        bridge_frames=bridge_frames,
        bridge_relaxed_sharpness=relaxed_sharpness,
        bridge_relaxed_rolling_shutter=relaxed_rolling_shutter,
        unresolved_gaps=unresolved_gaps,
        maximum_gap_sec=_maximum_gap_sec(selected),
    )


def prefer_low_rolling_shutter_motion(
    centers: list[int],
    span: int,
    motion_by_index: dict[int, float],
    maximum_motion_deg: float,
    *,
    frame_bound: int | None = None,
) -> list[int]:
    """Move only unsafe interval samples to the nearest low-readout-motion frame."""
    if not motion_by_index or not np.isfinite(maximum_motion_deg):
        return centers
    selected = []
    for center in centers:
        center_motion = motion_by_index.get(center)
        if center_motion is not None and center_motion <= maximum_motion_deg:
            selected.append(center)
            continue
        candidates = candidate_indices(
            center,
            max(1, span),
            max(2, span + 1),
            fps_bound=frame_bound,
        )
        measured = [candidate for candidate in candidates if candidate in motion_by_index]
        if not measured:
            selected.append(center)
            continue
        safe = [
            candidate
            for candidate in measured
            if motion_by_index[candidate] <= maximum_motion_deg
        ]
        if safe:
            chosen = min(safe, key=lambda candidate: (abs(candidate - center), motion_by_index[candidate]))
        else:
            chosen = min(
                measured,
                key=lambda candidate: (motion_by_index[candidate], abs(candidate - center)),
            )
        selected.append(chosen)
    return selected
