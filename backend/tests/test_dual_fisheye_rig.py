"""Versioned calibration から generic dual-fisheye rig 外参を生成する。"""

from __future__ import annotations

import numpy as np
import pytest

from sphere_reconstruct.colmap import calibrated_rig
from sphere_reconstruct.domain.camera_system import (
    CalibratedCameraSystem,
    OmniDistortionModel,
    ScanDirection,
    SensorExtrinsic,
    ShutterType,
    TimestampReference,
)
from sphere_reconstruct.insta360 import camera_system
from sphere_reconstruct.insta360.calibration import (
    CalibSource,
    DualLensCalibration,
    OmniLensCalibration,
)
from sphere_reconstruct.insta360.metadata import WindowCropInfo


def _lens(yaw, pitch, roll, tx, ty, tz):
    return OmniLensCalibration(
        xi=2.0,
        fx=4200.0,
        fy=4200.0,
        cx=2700.0,
        cy=2688.0,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        tx=tx,
        ty=ty,
        tz=tz,
        distortion_model=OmniDistortionModel.RADTAN,
        distortion_parameters=(0.1, 0.01, -0.01, 0.0, 0.0),
        ref_image_width=10752,
        ref_image_height=5376,
        lens_flags=113,
    )


def test_calibration_uses_full_relative_rotation_and_translation():
    system = camera_system.from_calibration(
        DualLensCalibration(
            source=CalibSource.OFFSET,
            version=3,
            lenses=(
                _lens(0.615, 0.016, 89.937, 0.0, 0.0, 0.0),
                _lens(-0.718, 0.211, 89.840, -0.000048, 0.000131, -0.032273),
            ),
        ),
        window_crop=WindowCropInfo(5376, 5376, 5312, 5312),
        rolling_shutter_readout_ms=21.244001,
    )
    extrinsics = {sensor.id: sensor.cam_from_rig for sensor in system.sensors}

    assert extrinsics["lens0"].rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert extrinsics["lens0"].translation_xyz == (0.0, 0.0, 0.0)
    secondary = extrinsics["lens1"]
    expected = np.asarray([-0.0017243283, -0.0019476860, 0.9999962161, -0.0008949862])
    actual = np.asarray(secondary.rotation_wxyz)
    if np.dot(actual, expected) < 0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_allclose(secondary.translation_xyz, [-0.000048, 0.000131, -0.032273])
    assert all(sensor.shutter.type == ShutterType.ROLLING for sensor in system.sensors)
    assert all(sensor.shutter.scan_direction == ScanDirection.UNKNOWN for sensor in system.sensors)
    assert all(
        sensor.shutter.timestamp_reference == TimestampReference.UNKNOWN
        for sensor in system.sensors
    )
    assert CalibratedCameraSystem.from_dict(system.to_dict()) == system


def test_rig_config_accepts_calibration_from_any_dual_fisheye_adapter():
    extrinsics = {
        "left": SensorExtrinsic.identity(),
        "right": SensorExtrinsic(
            rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
            translation_xyz=(0.001, -0.002, -0.03),
        ),
    }
    config = calibrated_rig.build_rig_config(
        camera_model_name="THIN_PRISM_FISHEYE",
        camera_params_by_sensor={"left": [1.0] * 12, "right": [2.0] * 12},
        sensor_extrinsics=extrinsics,
        prefix="sources/camera/",
    )

    cameras = config[0]["cameras"]
    assert cameras[0]["image_prefix"] == "sources/camera/left/"
    assert cameras[0]["ref_sensor"] is True
    assert cameras[1]["image_prefix"] == "sources/camera/right/"
    assert cameras[1]["cam_from_rig_translation"] == [0.001, -0.002, -0.03]


def test_camera_system_rejects_non_unit_rig_quaternion():
    with pytest.raises(ValueError, match="unit"):
        SensorExtrinsic(
            rotation_wxyz=(2.0, 0.0, 0.0, 0.0),
            translation_xyz=(0.0, 0.0, 0.0),
        )


def test_camera_system_rejects_implicit_coordinate_conventions():
    system = camera_system.from_calibration(
        DualLensCalibration(
            source=CalibSource.OFFSET,
            version=3,
            lenses=(
                _lens(0.615, 0.016, 89.937, 0.0, 0.0, 0.0),
                _lens(-0.718, 0.211, 89.840, -0.000048, 0.000131, -0.032273),
            ),
        ),
        window_crop=WindowCropInfo(5376, 5376, 5312, 5312),
        rolling_shutter_readout_ms=21.244001,
    ).to_dict()
    del system["coordinate_system"]["length_unit"]

    with pytest.raises(ValueError, match="座標系"):
        CalibratedCameraSystem.from_dict(system)
