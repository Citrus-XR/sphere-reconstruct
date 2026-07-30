"""offset_v3 から generic dual-fisheye rig 外参を生成する。"""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.colmap import dual_fisheye_rig
from sphere_reconstruct.insta360.calibration import MeiLensCalibration


def _lens(yaw, pitch, roll, tx, ty, tz):
    return MeiLensCalibration(
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
    )


def test_offset_v3_uses_full_relative_rotation_and_translation():
    extrinsics = dual_fisheye_rig.offset_v3_sensor_extrinsics(
        [
            _lens(0.615, 0.016, 89.937, 0.0, 0.0, 0.0),
            _lens(-0.718, 0.211, 89.840, -0.000048, 0.000131, -0.032273),
        ]
    )

    assert extrinsics["lens0"].rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert extrinsics["lens0"].translation_xyz == (0.0, 0.0, 0.0)
    secondary = extrinsics["lens1"]
    expected = np.asarray([-0.0017243283, -0.0019476860, 0.9999962161, -0.0008949862])
    actual = np.asarray(secondary.rotation_wxyz)
    if np.dot(actual, expected) < 0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    np.testing.assert_allclose(secondary.translation_xyz, [-0.000048, 0.000131, -0.032273])


def test_rig_config_accepts_calibration_from_any_dual_fisheye_adapter():
    extrinsics = {
        "left": dual_fisheye_rig.SensorExtrinsic.identity(),
        "right": dual_fisheye_rig.SensorExtrinsic(
            rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
            translation_xyz=(0.001, -0.002, -0.03),
        ),
    }
    config = dual_fisheye_rig.build_rig_config(
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
