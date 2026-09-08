"""Insta360 メタデータを vendor-neutral camera system へ変換する。"""

from __future__ import annotations

import math

import numpy as np

from ..domain.camera_system import (
    CalibratedCameraSystem,
    CalibratedSensor,
    CalibrationImageTransform,
    OmniIntrinsics,
    ScanDirection,
    SensorExtrinsic,
    ShutterCalibration,
    ShutterType,
    TimestampReference,
)
from .calibration import DualLensCalibration, OmniLensCalibration
from .metadata import WindowCropInfo


def from_calibration(
    calibration: DualLensCalibration,
    *,
    window_crop: WindowCropInfo | None,
    rolling_shutter_readout_ms: float | None,
) -> CalibratedCameraSystem:
    """Versioned offset 固有の合成画布と軸規約を canonical contract へ正規化する。"""
    if not calibration.is_valid() or len(calibration.lenses) != 2:
        raise ValueError("Insta360 offset には有効な 2 sensor calibration が必要です")
    reference_widths = {lens.ref_image_width for lens in calibration.lenses}
    reference_heights = {lens.ref_image_height for lens in calibration.lenses}
    if len(reference_widths) != 1 or len(reference_heights) != 1:
        raise ValueError("Insta360 offset の sensor 間で参照サイズが一致しません")
    combined_width = reference_widths.pop()
    sensor_height = reference_heights.pop()
    if combined_width <= 0 or sensor_height <= 0 or combined_width % 2:
        raise ValueError(f"Insta360 offset の参照画布サイズが不正です: {combined_width}x{sensor_height}")
    sensor_width = combined_width // 2
    if window_crop is None:
        image_transform = CalibrationImageTransform.identity(sensor_width, sensor_height)
    else:
        if (
            window_crop.source_width != sensor_width
            or window_crop.source_height != sensor_height
        ):
            raise ValueError(
                "window crop の入力 size が calibration の sensor-local reference と一致しません: "
                f"{window_crop.source_width}x{window_crop.source_height} != "
                f"{sensor_width}x{sensor_height}"
            )
        image_transform = CalibrationImageTransform(
            reference_width=sensor_width,
            reference_height=sensor_height,
            crop_x=(window_crop.source_width - window_crop.cropped_width) / 2.0,
            crop_y=(window_crop.source_height - window_crop.cropped_height) / 2.0,
            crop_width=window_crop.cropped_width,
            crop_height=window_crop.cropped_height,
        )
    extrinsics = _sensor_extrinsics(calibration.lenses)
    sensors = tuple(
        CalibratedSensor(
            id=f"lens{index}",
            image_key=f"lens{index}",
            intrinsics=OmniIntrinsics(
                width=image_transform.crop_width,
                height=image_transform.crop_height,
                xi=lens.xi,
                fx=lens.fx,
                fy=lens.fy,
                cx=lens.cx - index * sensor_width - image_transform.crop_x,
                cy=lens.cy - image_transform.crop_y,
                distortion_model=lens.distortion_model,
                distortion_parameters=lens.distortion_parameters,
            ),
            calibration_image_transform=image_transform,
            cam_from_rig=extrinsics[f"lens{index}"],
            shutter=(
                ShutterCalibration(
                    type=ShutterType.ROLLING,
                    readout_time_ms=rolling_shutter_readout_ms,
                    scan_direction=ScanDirection.UNKNOWN,
                    timestamp_reference=TimestampReference.UNKNOWN,
                )
                if rolling_shutter_readout_ms is not None
                else ShutterCalibration.unknown()
            ),
        )
        for index, lens in enumerate(calibration.lenses)
    )
    return CalibratedCameraSystem(
        calibration_source=f"{calibration.source.value}_v{calibration.version}",
        reference_sensor_id="lens0",
        sensors=sensors,
    )


def _sensor_extrinsics(lenses: tuple[OmniLensCalibration, OmniLensCalibration]) -> dict[str, SensorExtrinsic]:
    rotations = [_offset_rotation(lens, index) for index, lens in enumerate(lenses)]
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


def _offset_rotation(lens: OmniLensCalibration, lens_index: int) -> np.ndarray:
    # Dual-lens offset の軸順では lens0 だけ optical frame を Z 軸まわりに反転する。
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
    raise ValueError(f"未対応の回転軸です: {axis}")


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
