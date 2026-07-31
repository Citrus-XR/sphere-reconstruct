"""MEI (Mei-Rives) 拡張歪みモデルの投影 / 逆投影。

参考: PIPELINE.md (insv-stitch) の MEI 説明を独立に実装 + サンプル観測.

方針:
- 「目標 pinhole 画素 -> 目標射線 -> 物理镜头座標系 -> MEI 投影 -> 一回 backward
  remap」で最終画素を得る. fisheye を先に等距柱状 (ERP) に展開してから再投影する
  複数段パイプは避ける.
- 単体テストはリファレンス解像度と抽出解像度が違っても成立するように, 抜きだし
  時のスケールを常に明示引数で渡す.

このモジュールは numpy 依存. cv2 は使わず, 画像 remap は
`imaging/rendering.py` に分離する. 数値核だけをここに置く.

入力校正は adapter が sensor-local 画像座標へ正規化済みでなければならない。container
内の合成画布、crop、sensor 順序などの規約をこの数値核へ持ち込まない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..domain.camera_system import MeiIntrinsics


@dataclass(frozen=True)
class ColmapFisheyeApproximation:
    camera_model: str
    params: tuple[float, ...]
    rms_error_px: float
    maximum_error_px: float
    colmap_rms_error_px: float
    colmap_maximum_error_px: float
    lichtfeld_rms_error_px: float
    lichtfeld_maximum_error_px: float
    forward_radius_px: float


def approximate_opencv_fisheye(
    intr: MeiIntrinsics,
    *,
    maximum_theta_rad: float = math.pi / 2,
    theta_samples: int = 120,
    azimuth_samples: int = 96,
) -> ColmapFisheyeApproximation:
    """MEI を non-radial 項のない互換性重視の OPENCV_FISHEYE へ近似する。"""
    if not 0 < maximum_theta_rad <= math.pi / 2:
        raise ValueError("fisheye approximation は forward hemisphere 内でなければなりません")
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
        raise ValueError("MEI calibration が non-finite な forward ray を生成しました")
    u = projected[:, 0].reshape(theta.shape)
    v = projected[:, 1].reshape(theta.shape)
    powers = np.stack([theta**power for power in (1, 3, 5, 7, 9)], axis=-1)
    horizontal_design = (np.cos(azimuth)[..., np.newaxis] * powers).reshape(-1, 5)
    vertical_design = (np.sin(azimuth)[..., np.newaxis] * powers).reshape(-1, 5)
    horizontal = np.linalg.lstsq(horizontal_design, (u - intr.cx).reshape(-1), rcond=None)[0]
    vertical = np.linalg.lstsq(vertical_design, (v - intr.cy).reshape(-1), rcond=None)[0]
    fx, fy = float(horizontal[0]), float(vertical[0])
    distortion = tuple(
        float((horizontal[index] / fx + vertical[index] / fy) * 0.5)
        for index in range(1, 5)
    )
    radial = 1.0 + sum(
        distortion[index] * theta ** (2 * (index + 1)) for index in range(4)
    )
    predicted_u = intr.cx + fx * np.cos(azimuth) * theta * radial
    predicted_v = intr.cy + fy * np.sin(azimuth) * theta * radial
    error = np.hypot(predicted_u - u, predicted_v - v)
    edge_radial = maximum_theta_rad * (
        1.0
        + sum(
            distortion[index] * maximum_theta_rad ** (2 * (index + 1))
            for index in range(4)
        )
    )
    rms = float(np.sqrt(np.mean(error**2)))
    maximum = float(np.max(error))
    return ColmapFisheyeApproximation(
        camera_model="OPENCV_FISHEYE",
        params=(fx, fy, intr.cx, intr.cy, *distortion),
        rms_error_px=rms,
        maximum_error_px=maximum,
        colmap_rms_error_px=rms,
        colmap_maximum_error_px=maximum,
        lichtfeld_rms_error_px=rms,
        lichtfeld_maximum_error_px=maximum,
        forward_radius_px=min(fx, fy) * edge_radial,
    )


def approximate_thin_prism_fisheye(
    intr: MeiIntrinsics,
    *,
    maximum_theta_rad: float = math.pi / 2,
    theta_samples: int = 120,
    azimuth_samples: int = 96,
) -> ColmapFisheyeApproximation:
    """MEI calibration を COLMAP THIN_PRISM_FISHEYE へ近似する。"""
    # COLMAP の perspective fisheye ray は FOV <= 180° の時だけ正しく、常に rz > 0 を返す。
    # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/sensor/models.h#L290-L347
    if not 0 < maximum_theta_rad <= math.pi / 2:
        raise ValueError("fisheye approximation must stay within the forward hemisphere")
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
    distorted_x = np.cos(azimuth) * theta * radial
    distorted_y = np.sin(azimuth) * theta * radial
    radius_squared = distorted_x * distorted_x + distorted_y * distorted_y
    count = distorted_x.size
    design = np.zeros((count * 2, 6), dtype=np.float64)
    target = np.empty(count * 2, dtype=np.float64)
    x = distorted_x.reshape(-1)
    y = distorted_y.reshape(-1)
    radius2 = radius_squared.reshape(-1)
    design[:count, 0] = 1.0
    design[:count, 2] = fx * 2.0 * x * y
    design[:count, 3] = fx * (radius2 + 2.0 * x * x)
    design[:count, 4] = fx * radius2
    target[:count] = u.reshape(-1) - fx * x
    design[count:, 1] = 1.0
    design[count:, 2] = fy * (radius2 + 2.0 * y * y)
    design[count:, 3] = fy * 2.0 * x * y
    design[count:, 5] = fy * radius2
    target[count:] = v.reshape(-1) - fy * y
    cx, cy, p1, p2, sx1, sy1 = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = _refine_thin_prism_parameters(
        theta,
        azimuth,
        np.asarray(
            (
                fx,
                fy,
                cx,
                cy,
                distortion[0],
                distortion[1],
                p1,
                p2,
                distortion[2],
                distortion[3],
                sx1,
                sy1,
            )
        ),
        u,
        v,
    )
    colmap_u, colmap_v = _thin_prism_prediction(theta, azimuth, fitted)
    lichtfeld_u, lichtfeld_v = _lichtfeld_thin_prism_prediction(theta, azimuth, fitted)
    colmap_error = np.hypot(colmap_u - u, colmap_v - v)
    lichtfeld_error = np.hypot(lichtfeld_u - u, lichtfeld_v - v)
    boundary_radius = np.hypot(
        colmap_u[-1] - fitted[2],
        colmap_v[-1] - fitted[3],
    )
    return ColmapFisheyeApproximation(
        camera_model="THIN_PRISM_FISHEYE",
        params=tuple(float(value) for value in fitted),
        rms_error_px=float(max(np.sqrt(np.mean(colmap_error**2)), np.sqrt(np.mean(lichtfeld_error**2)))),
        maximum_error_px=float(max(np.max(colmap_error), np.max(lichtfeld_error))),
        colmap_rms_error_px=float(np.sqrt(np.mean(colmap_error**2))),
        colmap_maximum_error_px=float(np.max(colmap_error)),
        lichtfeld_rms_error_px=float(np.sqrt(np.mean(lichtfeld_error**2))),
        lichtfeld_maximum_error_px=float(np.max(lichtfeld_error)),
        forward_radius_px=float(np.median(boundary_radius)),
    )


def _refine_thin_prism_parameters(
    theta: np.ndarray,
    azimuth: np.ndarray,
    initial: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> np.ndarray:
    parameters = initial.astype(np.float64, copy=True)
    target = np.concatenate((target_u.ravel(), target_v.ravel()))
    for _ in range(8):
        predicted = _combined_prediction(theta, azimuth, parameters)
        combined_target = np.concatenate((target, target))
        residual = predicted - combined_target
        jacobian = np.empty((len(residual), len(parameters)), dtype=np.float64)
        for index in range(len(parameters)):
            step = 1e-3 if index < 4 else 1e-7
            shifted = parameters.copy()
            shifted[index] += step
            shifted_prediction = _combined_prediction(theta, azimuth, shifted)
            jacobian[:, index] = (shifted_prediction - predicted) / step
        delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        original_cost = float(np.dot(residual, residual))
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = parameters + fraction * delta
            candidate_prediction = _combined_prediction(theta, azimuth, candidate)
            candidate_residual = candidate_prediction - combined_target
            if float(np.dot(candidate_residual, candidate_residual)) < original_cost:
                parameters = candidate
                break
        else:
            break
        if np.linalg.norm(delta) < 1e-10:
            break
    return parameters


def _flatten_prediction(theta: np.ndarray, azimuth: np.ndarray, parameters: np.ndarray):
    u, v = _thin_prism_prediction(theta, azimuth, parameters)
    return u.ravel(), v.ravel()


def _combined_prediction(theta: np.ndarray, azimuth: np.ndarray, parameters: np.ndarray):
    # COLMAP は全 distortion を等距離座標へ同時適用するが、LFStudio v0.5.3 は radial 後の
    # 座標へ tangential / prism を適用する。共有 dataset が両方で同じ pixel ray を近似するよう
    # 二つの residual を同時に最小化する。
    # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/sensor/models.h#L2116-L2169
    # https://github.com/MrNeRF/LichtFeld-Studio/blob/d8c50c6a3e2273cb74130a6e9023de8d068af52d/src/training/rasterization/gsplat/Cameras.cuh#L1042-L1136
    return np.concatenate(
        (
            np.concatenate(_flatten_prediction(theta, azimuth, parameters)),
            np.concatenate(
                tuple(
                    array.ravel()
                    for array in _lichtfeld_thin_prism_prediction(theta, azimuth, parameters)
                )
            ),
        )
    )


def _thin_prism_prediction(
    theta: np.ndarray,
    azimuth: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = parameters
    x = np.cos(azimuth) * theta
    y = np.sin(azimuth) * theta
    radius_squared = x * x + y * y
    radial = radius_squared * (
        k1 + radius_squared * (k2 + radius_squared * (k3 + radius_squared * k4))
    )
    u = cx + fx * (
        x
        + x * radial
        + 2.0 * p1 * x * y
        + p2 * (radius_squared + 2.0 * x * x)
        + sx1 * radius_squared
    )
    v = cy + fy * (
        y
        + y * radial
        + p1 * (radius_squared + 2.0 * y * y)
        + 2.0 * p2 * x * y
        + sy1 * radius_squared
    )
    return u, v


def _lichtfeld_thin_prism_prediction(
    theta: np.ndarray,
    azimuth: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = parameters
    theta2 = theta * theta
    radial = 1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
    x = np.cos(azimuth) * theta * radial
    y = np.sin(azimuth) * theta * radial
    radius_squared = x * x + y * y
    u = cx + fx * (
        x + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x) + sx1 * radius_squared
    )
    v = cy + fy * (
        y + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y + sy1 * radius_squared
    )
    return u, v


# -----------------------------------------------------------------------------
# MEI 投影 / 歪み補正
# -----------------------------------------------------------------------------


def project_mei(rays: np.ndarray, intr: MeiIntrinsics) -> tuple[np.ndarray, np.ndarray]:
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

    # 拡張歪み (radial + tangential).
    xd, yd = _apply_distortion(x, y, intr)

    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy

    in_frame = (u >= 0) & (u <= intr.width - 1) & (v >= 0) & (v <= intr.height - 1)
    valid = valid_front & in_frame

    uv = np.stack([u, v], axis=1)
    return uv, valid


def unproject_mei(pixels: np.ndarray, intr: MeiIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-local pixel を MEI camera ray へ戻す。"""
    pixels = np.asarray(pixels, dtype=np.float64).reshape((-1, 2))
    distorted = np.column_stack(
        ((pixels[:, 0] - intr.cx) / intr.fx, (pixels[:, 1] - intr.cy) / intr.fy)
    )
    normalized = distorted.copy()
    maximum_radius = 1.0 / intr.xi
    _clip_planar_radius(normalized, maximum_radius)
    for _ in range(20):
        applied_x, applied_y = _apply_distortion(
            normalized[:, 0],
            normalized[:, 1],
            intr,
        )
        normalized += distorted - np.column_stack((applied_x, applied_y))
        _clip_planar_radius(normalized, maximum_radius)
    applied_x, applied_y = _apply_distortion(normalized[:, 0], normalized[:, 1], intr)
    inversion_error = np.linalg.norm(
        np.column_stack((applied_x, applied_y)) - distorted,
        axis=1,
    )
    radius2 = np.sum(normalized * normalized, axis=1)
    discriminant = 1.0 + (1.0 - intr.xi * intr.xi) * radius2
    safe_discriminant = np.maximum(discriminant, 0.0)
    scale = (intr.xi + np.sqrt(safe_discriminant)) / (1.0 + radius2)
    rays = np.column_stack(
        (
            scale * normalized[:, 0],
            scale * normalized[:, 1],
            scale - intr.xi,
        )
    )
    norms = np.linalg.norm(rays, axis=1)
    rays = np.divide(
        rays,
        norms[:, None],
        out=np.zeros_like(rays),
        where=norms[:, None] > 1e-12,
    )
    in_frame = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= intr.width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= intr.height - 1)
    )
    valid = (
        (discriminant >= 0.0)
        & (rays[:, 2] >= -1e-9)
        & (inversion_error < 1e-6)
        & in_frame
    )
    return rays, valid


def _apply_distortion(x: np.ndarray, y: np.ndarray, intr: MeiIntrinsics) -> tuple[np.ndarray, np.ndarray]:
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


def _clip_planar_radius(points: np.ndarray, maximum_radius: float) -> None:
    radius = np.linalg.norm(points, axis=1)
    scale = np.divide(
        np.minimum(radius, maximum_radius),
        radius,
        out=np.ones_like(radius),
        where=radius > 1e-12,
    )
    points *= scale[:, None]


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
