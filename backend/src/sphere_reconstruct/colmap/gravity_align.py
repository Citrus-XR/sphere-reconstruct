"""露光時刻と同期した INSV 加速度から sparse model の物理的な上方向を求める.

カメラが動く映像では device 座標の加速度を全期間平均してはいけない. 各 front image の
timestamp 周辺だけを robust 平均し, COLMAP の world->camera pose で world へ戻す. encoded
fisheye image と IMU の軸対応は右手系の signed axis permutation 24 通りから, 全 frame の
角度残差が最小のものを選ぶ. 連続回転を自由推定すると world up と mounting が共に未知で
非一意になるため行わない.

aligned dataset は LichtFeld/COLMAP 規約の -Y up. Web preview だけは表示時に
diag(1,-1,-1) を掛けて Three.js の +Y up へ変換する.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Callable

import numpy as np

from .model import Reconstruction

# COLMAP/LichtFeld dataset 座標では -Y が物理的な上. LF viewer は dataset world を
# diag(1,-1,-1) で表示座標へ移し, -Y を画面上の +Y にする.
TARGET_UP = np.array([0.0, -1.0, 0.0])
DATASET_TO_VIEWER = np.diag([1.0, -1.0, -1.0])

# 旧 API 用. 時刻同期版は撮影ごとに mounting を robust 推定する.
MOUNTING_R = np.eye(3)

_FRONT_FRAME = re.compile(r"(?:^|/)(?:lens0|front_lens0)/frame_(\d+)\.[^.]+$")


def _quat_to_R(qvec) -> np.ndarray:
    """world->cam クォータニオン (qw,qx,qy,qz) -> 回転行列."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """回転行列 -> (qw,qx,qy,qz). 数値安定な分岐版."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
    return (float(qw / n), float(qx / n), float(qy / n), float(qz / n))


def _rotation_aligning(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """単位ベクトル u を t に写す最小回転 (Rodrigues)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    t = t / (np.linalg.norm(t) + 1e-12)
    v = np.cross(u, t)
    s = np.linalg.norm(v)
    c = float(np.dot(u, t))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 反対向き: u に直交する任意軸で 180 度.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, axis)
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2 * K @ K  # 180 度回転.
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _is_front(name: str) -> bool:
    return "lens0" in name


def compute_align_rotation(
    recon: Reconstruction,
    gravity_imu,
    *,
    mounting: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict]:
    """重力整列の大域回転 R_align (3x3) と診断情報を返す.

    front 画像が無い、または一致度が悪い場合は (None, info) を返して整列を見送る。
    """
    if gravity_imu is None:
        return None, {"reason": "no_gravity"}
    mount = MOUNTING_R if mounting is None else mounting
    g_cam = mount @ np.asarray(gravity_imu, dtype=float)

    ups = []
    for img in recon.images.values():
        if not _is_front(img.name):
            continue
        R_wc = _quat_to_R(img.qvec)
        up = R_wc.T @ g_cam
        n = np.linalg.norm(up)
        if n > 1e-9:
            ups.append(up / n)
    if len(ups) < 2:
        return None, {"reason": "too_few_front_images", "count": len(ups)}

    mean_up = np.mean(ups, axis=0)
    mn = np.linalg.norm(mean_up)
    if mn < 1e-6:
        return None, {"reason": "inconsistent_up"}
    mean_up /= mn
    # 一致度: 各 up と平均のなす角の中央値 (deg). 大きいとモデル/重力が信用できない.
    angles = [np.degrees(np.arccos(np.clip(float(np.dot(u, mean_up)), -1, 1))) for u in ups]
    spread = float(np.median(angles))

    R_align = _rotation_aligning(mean_up, TARGET_UP)
    info = {
        "up_world": [round(float(x), 4) for x in mean_up],
        "spread_deg": round(spread, 2),
        "front_images": len(ups),
    }
    if spread > 20.0:
        # ばらつきが大きい場合は整列によって歪む恐れがあるため見送る。
        return None, {**info, "reason": "up_spread_too_large"}
    return R_align, info


def compute_timed_align_rotation(
    recon: Reconstruction,
    frame_times: dict[int, float],
    imu_samples,
    *,
    imu_timestamps_sec: list[float] | None = None,
    offset_min: float = -0.5,
    offset_max: float = 0.5,
    offset_step: float = 0.02,
    window_seconds: float = 0.025,
    image_prefix: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[np.ndarray | None, dict]:
    """時刻同期した加速度列から world up と IMU mounting を同時推定する.

    カメラが動く素材では device 座標の重力を全期間平均できない. 各露光時刻の加速度を使い,
    video/IMU の小さな開始時刻差も grid search し, signed axis mounting と world up の
    robust consensus を評価する.
    """
    if len(imu_samples) < 2:
        return None, {"reason": "no_timed_imu"}
    pairs = []
    for image in recon.images.values():
        normalized_name = image.name.replace("\\", "/")
        if image_prefix is not None and not normalized_name.startswith(image_prefix):
            continue
        match = _FRONT_FRAME.search(normalized_name)
        if match is None:
            continue
        frame_index = int(match.group(1))
        if frame_index in frame_times:
            pairs.append((image, frame_times[frame_index]))
    if len(pairs) < 4:
        return None, {"reason": "too_few_timed_front_images", "count": len(pairs)}

    timestamps = np.asarray(
        imu_timestamps_sec
        if imu_timestamps_sec is not None
        else [(sample.timestamp_us - imu_samples[0].timestamp_us) / 1_000_000.0 for sample in imu_samples]
    )
    raw_acceleration = np.asarray([sample.accel_xyz for sample in imu_samples], dtype=float)
    if timestamps.ndim != 1 or len(timestamps) != len(raw_acceleration):
        raise ValueError("IMU timestamps and accelerations must have matching one-dimensional lengths")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("IMU timestamps must be sorted")
    rotations = np.asarray([_quat_to_R(image.qvec) for image, _time in pairs])
    exposure_times = np.asarray([time for _image, time in pairs])

    offsets = np.arange(offset_min, offset_max + offset_step * 0.5, offset_step)
    estimates = []
    for offset_number, offset in enumerate(offsets, 1):
        estimate = _estimate_at_offset(
            timestamps,
            raw_acceleration,
            rotations,
            exposure_times,
            float(offset),
            window_seconds,
        )
        if estimate is not None:
            estimates.append(estimate)
        if progress is not None:
            progress("coarse", offset_number, len(offsets))
    if not estimates:
        return None, {"reason": "imu_time_range_mismatch"}
    best = min(estimates, key=lambda estimate: estimate["score"])

    fine_step = offset_step / 4.0
    fine_offsets = np.arange(
        best["time_offset_sec"] - offset_step,
        best["time_offset_sec"] + offset_step + fine_step * 0.5,
        fine_step,
    )
    fine = []
    for offset_number, offset in enumerate(fine_offsets, 1):
        estimate = _estimate_at_offset(
            timestamps,
            raw_acceleration,
            rotations,
            exposure_times,
            float(offset),
            window_seconds,
        )
        if estimate is not None:
            fine.append(estimate)
        if progress is not None:
            progress("fine", offset_number, len(fine_offsets))
    if fine:
        best = min(fine, key=lambda estimate: estimate["score"])

    required_inliers = max(4, round(len(pairs) * 0.6))
    if best["inlier_count"] < required_inliers:
        return None, {**_public_estimate(best, len(pairs)), "reason": "too_few_gravity_inliers"}
    if best["spread_deg"] > 10.0 or best["p90_deg"] > 20.0:
        return None, {**_public_estimate(best, len(pairs)), "reason": "gravity_residual_too_large"}
    return _rotation_aligning(best["up_world"], TARGET_UP), _public_estimate(best, len(pairs))


def reference_trajectory_diameter(recon: Reconstruction, image_prefix: str | None = None) -> float:
    centers = reference_camera_centers(recon, image_prefix)
    if len(centers) < 2:
        return 0.0
    points = np.asarray(centers, dtype=float)
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def reference_camera_centers(
    recon: Reconstruction, image_prefix: str | None = None
) -> list[tuple[float, float, float]]:
    """Rig の reference sensor だけの camera center を返す。

    Native / pinhole rig は ``lens0`` / ``front_lens0`` を reference として生成する。未知の naming
    では登録数が最大の単一 camera ID を選び、複数 sensor の baseline を軌跡移動と誤認しない。
    """
    centers = [
        image.camera_center
        for image in recon.images.values()
        if (image_prefix is None or image.name.replace("\\", "/").startswith(image_prefix))
        and _FRONT_FRAME.search(image.name.replace("\\", "/"))
    ]
    if centers:
        return centers
    eligible_images = [
        image
        for image in recon.images.values()
        if image_prefix is None or image.name.replace("\\", "/").startswith(image_prefix)
    ]
    if not eligible_images:
        return []
    counts = Counter(image.camera_id for image in eligible_images)
    reference_camera_id = min(counts, key=lambda camera_id: (-counts[camera_id], camera_id))
    return [image.camera_center for image in eligible_images if image.camera_id == reference_camera_id]


def _estimate_at_offset(
    timestamps: np.ndarray,
    accelerations: np.ndarray,
    rotations: np.ndarray,
    exposure_times: np.ndarray,
    offset: float,
    window: float,
):
    gravities = []
    selected_rotations = []
    for rotation, exposure in zip(rotations, exposure_times, strict=True):
        target = exposure + offset
        left = int(np.searchsorted(timestamps, target - window, side="left"))
        right = int(np.searchsorted(timestamps, target + window, side="right"))
        if left == right:
            continue
        gravity = np.median(accelerations[left:right], axis=0)
        norm = np.linalg.norm(gravity)
        if norm <= 1e-9:
            continue
        gravities.append(gravity / norm)
        selected_rotations.append(rotation)
    if len(gravities) < 4:
        return None
    gravities = np.asarray(gravities)
    selected_rotations = np.asarray(selected_rotations)

    candidates = [
        _mounting_consensus(gravities, selected_rotations, mounting)
        for mounting in _right_handed_axis_rotations()
    ]
    best = min(candidates, key=lambda candidate: candidate["score"])
    missing_fraction = 1.0 - len(gravities) / len(rotations)
    return {
        **best,
        "time_offset_sec": offset,
        "sampled_frames": len(gravities),
        "score": best["score"] + 10.0 * missing_fraction,
    }


def _mounting_consensus(
    gravities: np.ndarray,
    rotations: np.ndarray,
    mounting: np.ndarray,
) -> dict:
    world_ups = np.asarray(
        [rotation.T @ mounting @ gravity for gravity, rotation in zip(gravities, rotations, strict=True)]
    )
    world_ups /= np.linalg.norm(world_ups, axis=1)[:, None]
    inliers = np.ones(len(world_ups), dtype=bool)
    for _ in range(10):
        up_world = np.mean(world_ups[inliers], axis=0)
        up_world /= np.linalg.norm(up_world)
        angles = _angular_errors(world_ups, up_world)
        median = float(np.median(angles))
        mad = float(np.median(np.abs(angles - median)))
        next_inliers = angles <= max(8.0, median + 2.5 * mad)
        if np.array_equal(next_inliers, inliers):
            break
        if np.count_nonzero(next_inliers) < 4:
            break
        inliers = next_inliers

    up_world = np.mean(world_ups[inliers], axis=0)
    up_world /= np.linalg.norm(up_world)
    angles = _angular_errors(world_ups, up_world)
    spread = float(np.median(angles[inliers]))
    p90 = float(np.percentile(angles[inliers], 90))
    outlier_fraction = 1.0 - np.count_nonzero(inliers) / len(inliers)
    return {
        "up_world": up_world,
        "mounting": mounting,
        "spread_deg": spread,
        "p90_deg": p90,
        "inlier_count": int(np.count_nonzero(inliers)),
        "score": spread + 0.1 * p90 + 10.0 * outlier_fraction,
    }


def _right_handed_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                rotation[row, column] = signs[row]
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    return rotations


def _angular_errors(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(vectors @ reference, -1.0, 1.0)))


def _public_estimate(estimate: dict, total_frames: int) -> dict:
    return {
        "up_world": [round(float(value), 6) for value in estimate["up_world"]],
        "mounting": [[round(float(value), 6) for value in row] for row in estimate["mounting"]],
        "time_offset_sec": round(float(estimate["time_offset_sec"]), 4),
        "spread_deg": round(float(estimate["spread_deg"]), 3),
        "p90_deg": round(float(estimate["p90_deg"]), 3),
        "inlier_count": estimate["inlier_count"],
        "sampled_frames": estimate["sampled_frames"],
        "front_images": total_frames,
    }


def apply_alignment(recon: Reconstruction, R_align: np.ndarray) -> None:
    """R_align を再構成に in-place 適用する. 点 p'=R p, カメラ R_wc'=R_wc R^T, t 不変."""
    Rt = R_align.T
    for p in recon.points3D.values():
        p.xyz = tuple(float(x) for x in (R_align @ np.asarray(p.xyz)))
    for img in recon.images.values():
        R_wc = _quat_to_R(img.qvec)
        img.qvec = _R_to_quat(R_wc @ Rt)
        # tvec は不変 (x_cam = R_wc p + t = (R_wc R^T)(R p) + t).
