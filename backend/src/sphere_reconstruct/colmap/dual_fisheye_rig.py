"""Calibrated multi-sensor rig configuration for raw dual-fisheye cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..insta360.calibration import MeiLensCalibration


@dataclass(frozen=True)
class SensorExtrinsic:
    """COLMAP ``cam_from_rig`` rigid transform."""

    rotation_wxyz: tuple[float, float, float, float]
    translation_xyz: tuple[float, float, float]

    @classmethod
    def identity(cls) -> SensorExtrinsic:
        return cls((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def offset_v3_sensor_extrinsics(lenses: list[MeiLensCalibration]) -> dict[str, SensorExtrinsic]:
    """Convert a two-lens offset_v3 calibration into lens0-relative transforms."""
    if len(lenses) != 2:
        raise ValueError(f"dual-fisheye calibration requires exactly 2 lenses, got {len(lenses)}")
    rotations = [_offset_v3_rotation(lens, index) for index, lens in enumerate(lenses)]
    translations = [np.asarray((lens.tx, lens.ty, lens.tz), dtype=np.float64) for lens in lenses]
    secondary_rotation = rotations[1] @ rotations[0].T
    secondary_translation = translations[1] - secondary_rotation @ translations[0]
    return {
        "lens0": SensorExtrinsic.identity(),
        "lens1": SensorExtrinsic(
            rotation_wxyz=_rotation_to_quaternion(secondary_rotation),
            translation_xyz=tuple(float(value) for value in secondary_translation),
        ),
    }


def build_rig_config(
    camera_model_name: str,
    camera_params_by_sensor: dict[str, list[float]],
    sensor_extrinsics: dict[str, SensorExtrinsic],
    *,
    prefix: str = "",
) -> list[dict]:
    """Build COLMAP rig_configurator input from adapter-provided sensor calibration."""
    if not sensor_extrinsics:
        raise ValueError("rig must contain at least one sensor")
    if camera_params_by_sensor.keys() != sensor_extrinsics.keys():
        raise ValueError("camera parameters and sensor extrinsics must have identical sensor IDs")
    cameras = []
    for index, (sensor_id, extrinsic) in enumerate(sensor_extrinsics.items()):
        camera = {
            "image_prefix": f"{prefix}{sensor_id}/",
            "camera_model_name": camera_model_name,
            "camera_params": list(camera_params_by_sensor[sensor_id]),
        }
        if index == 0:
            camera["ref_sensor"] = True
        else:
            camera["cam_from_rig_rotation"] = list(extrinsic.rotation_wxyz)
            camera["cam_from_rig_translation"] = list(extrinsic.translation_xyz)
        cameras.append(camera)
    return [{"cameras": cameras}]


def _offset_v3_rotation(lens: MeiLensCalibration, lens_index: int) -> np.ndarray:
    # offset_v3 の軸順と dual-lens 固有の lens0 反転は公開実装で相互検証済み。
    # https://github.com/kya8/slate/blob/3c6658644265304b975d3dad9afd7b69592260af/src/slate/extra/insta360_tf.cpp#L48-L79
    rotation = (
        _axis_rotation("y", math.pi / 2.0 + math.radians(lens.pitch))
        @ _axis_rotation("z", math.radians(lens.yaw))
        @ _axis_rotation("x", math.radians(lens.roll))
    )
    return rotation @ _axis_rotation("z", math.pi) if lens_index == 0 else rotation


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))
    if axis == "y":
        return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))
    if axis == "z":
        return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    raise ValueError(f"unsupported rotation axis: {axis}")


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
            )
        elif diagonal == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
            )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)
