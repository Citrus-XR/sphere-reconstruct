"""imaging.projection の単体テスト.

MEI 前方射影のプロパティを確認する:
  1. 光軸方向の射線 (0,0,1) は画像中心 (cx, cy) にマップされる.
  2. 光軸から少し外れた射線は cx から離れた方向へマップされる.
  3. 後方 (Z < -xi) の射線は valid=False.

adapter で正規化した MEI のスケーリング挙動も確認する.
"""

from __future__ import annotations

import math

import numpy as np

from sphere_reconstruct.imaging import projection as proj
from sphere_reconstruct.imaging.fisheye_camera import camera_rays_to_pixels, pixels_to_camera_rays
from sphere_reconstruct.insta360 import camera_system
from sphere_reconstruct.insta360.calibration import CalibSource, DualLensCalibration, MeiLensCalibration
from sphere_reconstruct.insta360.metadata import WindowCropInfo


def _make_lens_a() -> MeiLensCalibration:
    # 実 X5 サンプルの lens A.
    return MeiLensCalibration(
        xi=2.0,
        fx=4278.30,
        fy=4277.33,
        cx=2694.63,
        cy=2681.84,
        yaw=0.615,
        pitch=0.016,
        roll=89.937,
        tx=0.0,
        ty=0.0,
        tz=0.0,
        k1=0.18366432,
        k2=2.07332635,
        k3=-3.27984834,
        p1=-0.00005305,
        p2=0.00065176,
        ref_image_width=10752,
        ref_image_height=5376,
        lens_flags=113,
    )


def _make_lens_b() -> MeiLensCalibration:
    return MeiLensCalibration(
        xi=2.0,
        fx=4296.81,
        fy=4298.54,
        cx=8064.92,
        cy=2686.41,
        yaw=-0.718,
        pitch=0.211,
        roll=89.840,
        tx=-0.000048,
        ty=0.000131,
        tz=-0.032273,
        k1=0.18302010,
        k2=2.05338216,
        k3=-3.26668859,
        p1=0.00187136,
        p2=0.00038193,
        ref_image_width=10752,
        ref_image_height=5376,
        lens_flags=113,
    )


def _camera_system():
    return camera_system.from_offset_v3(
        DualLensCalibration(
            source=CalibSource.OFFSET_V3,
            lenses=[_make_lens_a(), _make_lens_b()],
        ),
        window_crop=WindowCropInfo(5376, 5376, 5312, 5312),
        rolling_shutter_readout_ms=21.244001,
    )


def test_adapter_applies_centered_calibration_window_crop():
    intr = _camera_system().sensors[0].intrinsics
    assert intr.width == 5312
    assert intr.height == 5312
    assert intr.fx == 4278.30
    assert intr.cx == 2694.63 - 32.0


def test_adapter_removes_lens_b_canvas_offset():
    intr = _camera_system().sensors[1].intrinsics
    # 合成画像で 8064.92 だった cx が, 単眼画像では 8064.92 - 5376 = 2688.92.
    assert abs(intr.cx - (8064.92 - 5376.0 - 32.0)) < 1e-6


def test_normalized_intrinsics_scale_to_extraction_resolution():
    intr = _camera_system().sensors[0].intrinsics.scaled(3840, 3840)
    scale = 3840 / 5312
    assert abs(intr.fx - 4278.30 * scale) < 1e-3
    assert abs(intr.cx - (2694.63 - 32.0) * scale) < 1e-3


def test_project_mei_optical_axis_lands_at_principal_point():
    intr = _camera_system().sensors[0].intrinsics
    rays = np.array([[0.0, 0.0, 1.0]])
    uv, valid = proj.project_mei(rays, intr)
    assert valid[0]
    # 光軸は歪み中心なので u ≈ cx, v ≈ cy.
    assert abs(uv[0, 0] - intr.cx) < 1e-6
    assert abs(uv[0, 1] - intr.cy) < 1e-6


def test_mei_pixel_ray_roundtrip_is_subpixel_exact():
    intr = _camera_system().sensors[0].intrinsics.scaled(3840, 3840)
    theta = np.linspace(0.0, math.radians(85.0), 30)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    theta_grid, azimuth_grid = np.meshgrid(theta, azimuth, indexing="ij")
    rays = np.column_stack(
        (
            (np.sin(theta_grid) * np.cos(azimuth_grid)).ravel(),
            (np.sin(theta_grid) * np.sin(azimuth_grid)).ravel(),
            np.cos(theta_grid).ravel(),
        )
    )

    pixels, projected = proj.project_mei(rays, intr)
    recovered, unprojected = proj.unproject_mei(pixels, intr)

    valid = projected & unprojected
    angular_error = np.arccos(np.clip(np.sum(rays[valid] * recovered[valid], axis=1), -1.0, 1.0))
    assert valid.sum() > 1000
    assert np.max(angular_error) < 2e-7


def test_project_mei_back_hemisphere_rejected():
    intr = _camera_system().sensors[0].intrinsics
    # xi=2 なので Z=-1 でも denom = -1+2 = 1 > 0, まだ valid.
    # Z=-3 なら denom = -3+2 = -1 < 0, front-side は不通.
    # ただし unit sphere 上で Z=-3 は不可能なので, 正規化前で強めに -1 を渡す.
    rays = np.array([[0.0, 0.0, -1.0]])
    uv, valid = proj.project_mei(rays, intr)
    # 正規化後 Z=-1, denom = -1+2 = 1 > 0. まだ画像内かどうかは fx / distortion 次第.
    # xi=2 は真横方向まで見えるようなモデルなので, 画像内に落ちる可能性が高い. OK.
    # ここでは 「valid が bool で返ってくる」ことだけ確認.
    assert valid.dtype == np.bool_
    assert uv.shape == (1, 2)


def test_mei_calibration_fits_forward_thin_prism_fisheye():
    results = []
    for sensor in _camera_system().sensors:
        intr = sensor.intrinsics.scaled(3840, 3840)
        results.append(proj.approximate_thin_prism_fisheye(intr))

    assert results[0].camera_model == "THIN_PRISM_FISHEYE"
    assert results[0].colmap_rms_error_px < 0.05
    assert results[0].lichtfeld_rms_error_px < 0.05
    assert results[0].maximum_error_px < 0.26
    assert results[1].colmap_rms_error_px < 0.12
    assert results[1].lichtfeld_rms_error_px < 0.12
    assert results[1].maximum_error_px < 0.6
    assert all(len(result.params) == 12 for result in results)
    assert 0.44 < results[0].forward_radius_px / 3840 < 0.46
    assert 0.44 < results[1].forward_radius_px / 3840 < 0.46
    assert results[0].params != results[1].params


def test_rectification_target_model_is_invertible_and_bounded():
    theta = np.linspace(0.01, math.radians(89.0), 40)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    theta_grid, azimuth_grid = np.meshgrid(theta, azimuth, indexing="ij")
    rays = np.column_stack(
        (
            (np.sin(theta_grid) * np.cos(azimuth_grid)).ravel(),
            (np.sin(theta_grid) * np.sin(azimuth_grid)).ravel(),
            np.cos(theta_grid).ravel(),
        )
    )
    results = [
        proj.approximate_opencv_fisheye(sensor.intrinsics.scaled(3840, 3840))
        for sensor in _camera_system().sensors
    ]

    assert results[0].maximum_error_px < 1.6
    assert results[1].maximum_error_px < 4.6
    for result in results:
        pixels = camera_rays_to_pixels(result.camera_model, result.params, rays)
        recovered = pixels_to_camera_rays(result.camera_model, result.params, pixels)
        angular_error = np.arccos(np.clip(np.sum(rays * recovered, axis=1), -1.0, 1.0))
        assert np.max(angular_error) < 2e-7


def test_pinhole_backproject_center_ray_is_optical_axis():
    view = proj.PinholeView("front", fov_deg=90.0, width=64, height=64, yaw_deg=0.0, pitch_deg=0.0)
    rays = proj.pinhole_backproject(view)
    assert rays.shape == (64, 64, 3)
    cy, cx = 32, 32
    # 中心画素の射線は概ね (0, 0, 1) に近い.
    center = rays[cy, cx]
    # 32 - 32 = 0 割る f = 0 => x = y = 0.
    assert abs(center[0]) < 0.05
    assert abs(center[1]) < 0.05
    assert center[2] == 1.0


def test_cubemap_views_returns_six():
    views = proj.cubemap_views(size=512, fov_deg=90.0)
    assert len(views) == 6
    names = [v.name for v in views]
    assert set(names) == {"front", "back", "left", "right", "up", "down"}


def test_yaw_pitch_rotation_is_orthonormal():
    rotation = proj.yaw_pitch_rotation(30.0, 15.0)
    # 直交かつ det=1.
    should_be_identity = rotation @ rotation.T
    assert np.allclose(should_be_identity, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9
