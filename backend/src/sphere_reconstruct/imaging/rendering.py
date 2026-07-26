"""fisheye -> pinhole の実 remap.

`projection.py` は数値核だけ. こちらは cv2.remap で実画像に対する backward
remap を行う. cv2 は opencv-python-headless (imaging extra) からロードする.

依存を明示するため, cv2 の import はこのモジュール内でのみ行う.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .projection import (
    LensIntrinsics,
    PinholeView,
    pinhole_backproject,
    project_mei,
    yaw_pitch_rotation,
)


def _cv2():
    """cv2 を遅延 import. imaging extra が入っていない環境で projection.py だけ使えるように."""
    import cv2  # noqa: PLC0415

    return cv2


@dataclass
class RenderStats:
    valid_ratio: float  # remap で source から拾えた画素の比率
    src_size: tuple[int, int]
    dst_size: tuple[int, int]


def build_remap(
    view: PinholeView,
    src_intr: LensIntrinsics,
    *,
    extra_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole `view` の全画素について, fisheye source の (u, v) と有効フラグを返す.

    - `view.yaw_deg`, `view.pitch_deg` は rig 座標系での回転.
    - `extra_rotation` はさらに lens 座標系への追加回転 (offset_v3 の angles を適用したい場合).
    """
    # (H, W, 3) の pinhole 射線 (view カメラ座標).
    rays = pinhole_backproject(view)
    H, W, _ = rays.shape
    # rig 座標系 = view 座標系を yaw/pitch で戻したもの.
    R_view = yaw_pitch_rotation(view.yaw_deg, view.pitch_deg)
    # view 座標 -> rig 座標 = R_view^T @ ray (ここでは backprojection なので view で作った射線を
    # 「view -> rig -> lens」に持っていく).
    rays_rig = rays.reshape(-1, 3) @ R_view

    if extra_rotation is not None:
        rays_lens = rays_rig @ extra_rotation.T
    else:
        rays_lens = rays_rig

    uv, valid = project_mei(rays_lens, src_intr)
    map_x = uv[:, 0].reshape(H, W).astype(np.float32)
    map_y = uv[:, 1].reshape(H, W).astype(np.float32)
    valid_mask = valid.reshape(H, W)
    # invalid 画素は remap で外に飛ばして BORDER_CONSTANT で 0 になるようにする.
    map_x = np.where(valid_mask, map_x, -1.0)
    map_y = np.where(valid_mask, map_y, -1.0)
    return map_x, map_y, valid_mask


def render_pinhole(
    src_image_path: Path,
    view: PinholeView,
    src_intr: LensIntrinsics,
    *,
    extra_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, RenderStats]:
    """fisheye JPEG を読んで, view の pinhole 画像 (uint8 HxWx3 BGR) を返す."""
    cv2 = _cv2()
    src = cv2.imread(str(src_image_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"cannot read {src_image_path}")
    if src.shape[1] != src_intr.width or src.shape[0] != src_intr.height:
        raise ValueError(
            f"image size {src.shape[1]}x{src.shape[0]} != intrinsics {src_intr.width}x{src_intr.height}"
        )

    map_x, map_y, valid = build_remap(view, src_intr, extra_rotation=extra_rotation)
    dst = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    stats = RenderStats(
        valid_ratio=float(valid.mean()),
        src_size=(src.shape[1], src.shape[0]),
        dst_size=(view.width, view.height),
    )
    return dst, stats


def render_perspective_from_equirect(
    src_image_path: Path,
    view: PinholeView,
) -> tuple[np.ndarray, RenderStats]:
    """Equirectangular (全天球) 画像から view の pinhole 画像 (uint8 HxWx3 BGR) を作る.

    ERP は既に球面を平面展開したものなので, 各 pinhole 画素の視線を rig(=ワールド)座標へ
    回し, その方向を経度/緯度に変換して equirect を bilinear サンプルする. 光学中心は
    パノラマ中心を全 view で共有する (並進ゼロ). 経度は横方向に周回するので BORDER_WRAP.
    """
    cv2 = _cv2()
    src = cv2.imread(str(src_image_path), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"cannot read {src_image_path}")
    src_h, src_w = src.shape[:2]

    rays = pinhole_backproject(view)  # (H, W, 3), +X 右 +Y 下 +Z 前
    h, w, _ = rays.shape
    R_view = yaw_pitch_rotation(view.yaw_deg, view.pitch_deg)
    d = rays.reshape(-1, 3) @ R_view  # view -> rig(world) 方向.
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    lon = np.arctan2(dx, dz)  # +Z 前で 0, +X 右で +pi/2.
    lat = np.arcsin(np.clip(dy / r, -1.0, 1.0))  # +Y 下 -> lat>0 が下.
    u = (lon / (2.0 * math.pi) + 0.5) * src_w
    v = (lat / math.pi + 0.5) * src_h
    # 経度 u は横方向に周回する (mod), 緯度 v は極でクランプする (縦は周回させない).
    # BORDER_WRAP は u/v 両方に効くため, v を先に範囲内へ収めて縦の巻き込みを防ぐ.
    map_x = np.mod(u, src_w).reshape(h, w).astype(np.float32)
    map_y = np.clip(v, 0.0, src_h - 1.0).reshape(h, w).astype(np.float32)
    dst = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    stats = RenderStats(valid_ratio=1.0, src_size=(src_w, src_h), dst_size=(view.width, view.height))
    return dst, stats


def write_jpeg(dst: np.ndarray, out_path: Path, quality: int = 92) -> None:
    cv2 = _cv2()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), dst, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")


def lens_local_rotation(lens) -> np.ndarray:
    """offset_v3 の (yaw, pitch, roll) 角度から 3x3 rotation.

    観測された roll ≈ 90 deg (実際は lens 自体の物理向き) を含む. rig 座標系
    -> lens 座標系 の回転として使う.
    """
    y = math.radians(lens.yaw)
    p = math.radians(lens.pitch)
    r = math.radians(lens.roll)
    # ZYX 順で組む (yaw -> pitch -> roll).
    Rz = np.array(
        [
            [math.cos(y), -math.sin(y), 0.0],
            [math.sin(y), math.cos(y), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    Ry = np.array(
        [
            [math.cos(p), 0.0, math.sin(p)],
            [0.0, 1.0, 0.0],
            [-math.sin(p), 0.0, math.cos(p)],
        ]
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(r), -math.sin(r)],
            [0.0, math.sin(r), math.cos(r)],
        ]
    )
    return Rx @ Ry @ Rz
