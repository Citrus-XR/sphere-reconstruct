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
