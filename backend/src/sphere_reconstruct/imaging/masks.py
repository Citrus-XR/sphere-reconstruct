"""マスクのダウンサンプル / アップスケール / 合成ユーティリティ.

8K フレームをそのまま SAM3 に渡すと極端に遅い (特徴抽出の計算量が解像度に対して
急増する) ため, 「縮小画像で検出 -> mask を元解像度へ bilinear 拡大」という流れを取る.

このモジュールは cv2 / numpy のみ. torch も sam3 も import しない (純粋なので
CUDA 無し環境でも単体テスト可能).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cv2():
    import cv2  # noqa: PLC0415

    return cv2


@dataclass
class DownsamplePlan:
    """縮小計画. 元サイズと縮小後サイズ, スケール比を持つ."""

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float  # dst / src (<= 1.0)


def plan_downsample(width: int, height: int, max_size: int) -> DownsamplePlan:
    """長辺が max_size を超えるなら, アスペクト比維持で縮小する計画を返す.

    max_size <= 0 なら縮小しない (scale=1.0).
    """
    long_edge = max(width, height)
    if max_size <= 0 or long_edge <= max_size:
        return DownsamplePlan(width, height, width, height, 1.0)
    scale = max_size / long_edge
    dst_w = max(1, round(width * scale))
    dst_h = max(1, round(height * scale))
    return DownsamplePlan(width, height, dst_w, dst_h, scale)


def downsample_image(image: np.ndarray, plan: DownsamplePlan) -> np.ndarray:
    """plan に従い画像を縮小する. scale==1.0 ならコピーせずそのまま返す."""
    if plan.scale >= 1.0:
        return image
    cv2 = _cv2()
    # 縮小には INTER_AREA が最も綺麗.
    return cv2.resize(image, (plan.dst_w, plan.dst_h), interpolation=cv2.INTER_AREA)


def upscale_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """低解像度 bool/uint8 mask を target サイズへ bilinear 拡大して 2 値化する.

    bilinear で拡大した後 0.5 でしきい値化することで, 縁を滑らかにしつつ 2 値を保つ.
    入力が既に target サイズなら bool 変換だけ行う.
    """
    cv2 = _cv2()
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if (m.shape[1], m.shape[0]) != (target_w, target_h):
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return (m > 0.5).astype(np.uint8)


def union_masks(masks: list[np.ndarray]) -> np.ndarray | None:
    """複数の 2 値 mask を論理和で合成する. 空リストなら None."""
    if not masks:
        return None
    acc = masks[0].astype(bool)
    for m in masks[1:]:
        acc |= m.astype(bool)
    return acc.astype(np.uint8)


def coverage_ratio(mask: np.ndarray) -> float:
    """mask が画像に占める面積比 (0.0 - 1.0)."""
    if mask.size == 0:
        return 0.0
    return float((mask > 0).sum()) / float(mask.size)


def write_mask_png(mask: np.ndarray, out_path, invert: bool = False) -> None:
    """2 値 mask を PNG で書き出す. COLMAP 用は「マスクしたい所を 0, 使う所を 255」.

    invert=False: mask=1 の画素を 255 (検出物体の可視化用).
    invert=True:  mask=1 の画素を 0, それ以外を 255 (COLMAP が無視する = 検出物体を除外).
    """
    cv2 = _cv2()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask > 0).astype(np.uint8)
    img = (1 - m) * 255 if invert else m * 255
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")
