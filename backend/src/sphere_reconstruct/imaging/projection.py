"""MEI (Mei-Rives) 拡張畜れみモデルの投影 / 逆投影.

参考: PIPELINE.md (insv-stitch) の MEI 説明を独立に実装 + サンプル観測.

方針:
- 「目標 pinhole 画素 -> 目標射線 -> 物理镜头座標系 -> MEI 投影 -> 一回 backward
  remap」で最終画素を得る. fisheye を先に等距柱状 (ERP) に展開してから再投影する
  複数段パイプは避ける.
- 単体テストはリファレンス解像度と抽出解像度が違っても成立するように, 抜きだし
  時のスケールを常に明示引数で渡す.

このモジュールは numpy 依存. cv2 は使わず, 画像 remap は
`imaging/rendering.py` に分離する. 数値核だけをここに置く.

座標系:
- 世界座標: rig の原点 (レンズ A の光心) 起点, 右手系, +Z 前方 / +Y 下向き / +X 右向き
  (offset_v3 の座標系を採用. lens A tx=ty=tz=0 を基準に取ると
  lens B tz=-0.032273 が観測されるので, lens B が背面側なら「+Z 前方」に一致).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..insta360.calibration import MeiLensCalibration


@dataclass
class LensIntrinsics:
    """レンズ内部パラメータをネイティブ解像度に変換したもの.

    offset_v3 の cx, cy は「合成 10752x5376 座標」で表現されている. lens B は
    cx が 5376 だけシフトされているため, 実際の 5376x5376 単眼画像に使うときは
    lens_native_cx = raw_cx - lens_offset_x, lens_native_cy = raw_cy を使う.

    さらに ffmpeg で抜いた実画像 (例: 3840x3840) に使うときは (native ref -> 実画像)
    のスケールを fx, fy, cx, cy に掛ける.
    """

    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    width: int  # 実画像の幅
    height: int  # 実画像の高さ


@dataclass(frozen=True)
class OpenCVFisheyeApproximation:
    params: tuple[float, float, float, float, float, float, float, float]
    rms_error_px: float
    maximum_error_px: float
    forward_radius_px: float


def lens_to_intrinsics(
    lens: MeiLensCalibration,
    *,
    lens_index: int,
    single_lens_native_width: int,
    target_width: int,
    target_height: int,
) -> LensIntrinsics:
    """offset_v3 の生キャリブから, 単眼 target_width x target_height 用の内部パラメータへ.

    - lens_index=0 のレンズは cx をそのまま使う.
    - lens_index=1 以降は raw cx から lens_offset_x = lens_index * single_lens_native_width を引く.
      (X5 では 「lens0:5376, lens1:8064 - 5376 = 2688」と合成画像上で並んでいると観測)
    - 最後に (target_width / single_lens_native_width) スケールで fx,fy,cx,cy を縮小する.
    """
    if lens.xi <= 0:
        raise ValueError(f"lens.xi must be > 0 (got {lens.xi})")
    if single_lens_native_width <= 0 or target_width <= 0:
        raise ValueError("native/target width must be > 0")

    offset_x = lens_index * single_lens_native_width
    native_cx = lens.cx - offset_x
    native_cy = lens.cy

    scale = target_width / single_lens_native_width
    return LensIntrinsics(
        xi=lens.xi,
        fx=lens.fx * scale,
        fy=lens.fy * scale,
        cx=native_cx * scale,
        cy=native_cy * scale,
        k1=lens.k1,
        k2=lens.k2,
        k3=lens.k3,
        p1=lens.p1,
        p2=lens.p2,
        width=target_width,
        height=target_height,
    )


def approximate_opencv_fisheye(
    intr: LensIntrinsics,
    *,
    maximum_theta_rad: float = math.pi / 2,
    theta_samples: int = 120,
    azimuth_samples: int = 96,
) -> OpenCVFisheyeApproximation:
    """MEI calibration を COLMAP の forward-hemisphere OPENCV_FISHEYE へ近似する。"""
    # COLMAP の perspective fisheye ray は FOV <= 180° の時だけ正しく、常に rz > 0 を返す。
    # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/sensor/models.h#L290-L347
    if not 0 < maximum_theta_rad <= math.pi / 2:
        raise ValueError("OPENCV_FISHEYE approximation must stay within the forward hemisphere")
    theta_values = np.linspace(1e-4, maximum_theta_rad, theta_samples)
    azimuth_values = np.linspace(0.0, 2.0 * math.pi, azimuth_samples, endpoint=False)
    theta, azimuth = np.meshgrid(theta_values, azimuth_values, indexing="ij")
    rays = np.stack(
        (
            np.sin(theta) * np.cos(azimuth),
            np.sin(theta) * np.sin(azimuth),
            np.cos(theta),
        ),
        axis=-1,
    )
    projected, _valid = project_mei(rays.reshape(-1, 3), intr)
    if not np.all(np.isfinite(projected)):
        raise ValueError("MEI calibration produced non-finite forward-hemisphere coordinates")
    u = projected[:, 0].reshape(theta.shape)
    v = projected[:, 1].reshape(theta.shape)
    powers = np.stack([theta**power for power in (1, 3, 5, 7, 9)], axis=-1)
    horizontal_design = (np.cos(azimuth)[..., np.newaxis] * powers).reshape(-1, 5)
    vertical_design = (np.sin(azimuth)[..., np.newaxis] * powers).reshape(-1, 5)
    horizontal = np.linalg.lstsq(horizontal_design, (u - intr.cx).reshape(-1), rcond=None)[0]
    vertical = np.linalg.lstsq(vertical_design, (v - intr.cy).reshape(-1), rcond=None)[0]
    fx, fy = float(horizontal[0]), float(vertical[0])
    distortion = tuple(float((horizontal[index] / fx + vertical[index] / fy) * 0.5) for index in range(1, 5))
    radial = 1.0 + sum(distortion[index] * theta ** (2 * (index + 1)) for index in range(4))
    predicted_u = intr.cx + fx * np.cos(azimuth) * theta * radial
    predicted_v = intr.cy + fy * np.sin(azimuth) * theta * radial
    error = np.hypot(predicted_u - u, predicted_v - v)
    edge_radial = maximum_theta_rad * (
        1.0 + sum(distortion[index] * maximum_theta_rad ** (2 * (index + 1)) for index in range(4))
    )
    return OpenCVFisheyeApproximation(
        params=(fx, fy, intr.cx, intr.cy, *distortion),
        rms_error_px=float(np.sqrt(np.mean(error**2))),
        maximum_error_px=float(np.max(error)),
        forward_radius_px=min(fx, fy) * edge_radial,
    )


# -----------------------------------------------------------------------------
# MEI 投影 / 歪み補正
# -----------------------------------------------------------------------------


def project_mei(rays: np.ndarray, intr: LensIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """3D 射線 (N,3) -> 画素座標 (N,2). 有効フラグ (N,) も返す.

    有効フラグ = 「MEI 前方射影 (Z + xi > 0)」and 「画素座標が画像内」.
    """
    if rays.ndim != 2 or rays.shape[1] != 3:
        raise ValueError(f"rays must be (N,3), got {rays.shape}")

    # 単位球面上へ正規化.
    norm = np.linalg.norm(rays, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    normalized = rays / norm

    denom = normalized[:, 2] + intr.xi
    # 分母 <= 0 (球の裏側) は不可視.
    valid_front = denom > 1e-9
    denom_safe = np.where(valid_front, denom, 1.0)

    x = normalized[:, 0] / denom_safe
    y = normalized[:, 1] / denom_safe

    # 拡張畜れみ (radial + tangential).
    xd, yd = _apply_distortion(x, y, intr)

    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy

    in_frame = (u >= 0) & (u <= intr.width - 1) & (v >= 0) & (v <= intr.height - 1)
    valid = valid_front & in_frame

    uv = np.stack([u, v], axis=1)
    return uv, valid


def _apply_distortion(x: np.ndarray, y: np.ndarray, intr: LensIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + intr.k1 * r2 + intr.k2 * r4 + intr.k3 * r6
    # OpenCV 系 tangential: p1, p2 の順で:
    #   dx = 2*p1*x*y + p2*(r2 + 2*x^2)
    #   dy = p1*(r2 + 2*y^2) + 2*p2*x*y
    dx = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
    dy = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
    return x * radial + dx, y * radial + dy


# -----------------------------------------------------------------------------
# Pinhole rig 生成
# -----------------------------------------------------------------------------


@dataclass
class PinholeView:
    """1 個の virtual pinhole 视图. rig 内で lens A の光心を共有する."""

    name: str  # "front" / "back" / "left" / ...
    fov_deg: float  # 水平 FoV.
    width: int
    height: int
    yaw_deg: float  # rig 座標系上の水平回転 (right-handed, +Y 下向き)
    pitch_deg: float  # 垂直回転


def cubemap_views(size: int = 1024, fov_deg: float = 90.0) -> list[PinholeView]:
    """立方体展開の 6 面. 前 / 後 / 左 / 右 / 上 / 下. 主に COLMAP 用."""
    return [
        PinholeView("front", fov_deg, size, size, yaw_deg=0.0, pitch_deg=0.0),
        PinholeView("right", fov_deg, size, size, yaw_deg=90.0, pitch_deg=0.0),
        PinholeView("back", fov_deg, size, size, yaw_deg=180.0, pitch_deg=0.0),
        PinholeView("left", fov_deg, size, size, yaw_deg=-90.0, pitch_deg=0.0),
        PinholeView("up", fov_deg, size, size, yaw_deg=0.0, pitch_deg=-90.0),
        PinholeView("down", fov_deg, size, size, yaw_deg=0.0, pitch_deg=90.0),
    ]


def pinhole_focal_from_fov(width: int, fov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def pinhole_backproject(view: PinholeView) -> np.ndarray:
    """View の全画素について, カメラ座標での 3D 射線 (H, W, 3) を返す (未正規化).

    +X 右, +Y 下, +Z 前方.
    """
    f = pinhole_focal_from_fov(view.width, view.fov_deg)
    # 主点は中心.
    cx = view.width / 2.0
    cy = view.height / 2.0
    us = np.arange(view.width, dtype=np.float64)
    vs = np.arange(view.height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - cx) / f
    y = (vv - cy) / f
    z = np.ones_like(x)
    return np.stack([x, y, z], axis=-1)


def yaw_pitch_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """rig 座標 -> lens 座標 の回転. yaw が先, pitch が後.

    Y 軸右手系, 回転は右手ネジ方向.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    yaw_rotation = np.array(
        [
            [math.cos(y), 0.0, math.sin(y)],
            [0.0, 1.0, 0.0],
            [-math.sin(y), 0.0, math.cos(y)],
        ]
    )
    pitch_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(p), -math.sin(p)],
            [0.0, math.sin(p), math.cos(p)],
        ]
    )
    return pitch_rotation @ yaw_rotation
