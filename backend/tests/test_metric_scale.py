"""物理 rig baseline が global similarity scale を一意に復元することを検証する。"""

from __future__ import annotations

from sphere_reconstruct.colmap.metric_scale import estimate_metric_scale
from sphere_reconstruct.colmap.model import Camera, Image, Point3D, Reconstruction
from sphere_reconstruct.stages.restore_metric_scale import _load_rig_config


def _reconstruction(model_units_per_meter: float) -> tuple[Reconstruction, dict, list[dict]]:
    cameras = {
        1: Camera(1, "OPENCV_FISHEYE", 64, 64, [20, 20, 32, 32, 0, 0, 0, 0], 5),
        2: Camera(2, "OPENCV_FISHEYE", 64, 64, [20, 20, 32, 32, 0, 0, 0, 0], 5),
    }
    physical_baseline = 0.032
    images = {}
    points = {}
    records = []
    for capture in range(12):
        front_center = (float(capture), 0.0, 0.0)
        back_center = (
            float(capture),
            0.0,
            physical_baseline * model_units_per_meter,
        )
        for sensor, camera_id, center in (
            ("front", 1, front_center),
            ("back", 2, back_center),
        ):
            image_id = capture * 2 + camera_id
            name = f"sources/primary/{sensor}/frame_{capture:06d}.jpg"
            images[image_id] = Image(
                image_id,
                (1.0, 0.0, 0.0, 0.0),
                tuple(-value for value in center),
                camera_id,
                name,
            )
            records.append(
                {
                    "name": name,
                    "source_id": "primary",
                    "capture_index": capture,
                }
            )
        point_id = capture + 1
        points[point_id] = Point3D(
            point_id,
            (float(capture), 0.0, 4.0),
            (128, 128, 128),
            0.5,
            [(capture * 2 + 1, 0), (capture * 2 + 2, 0)],
        )
    rig = [
        {
            "cameras": [
                {
                    "image_prefix": "sources/primary/front/",
                    "ref_sensor": True,
                },
                {
                    "image_prefix": "sources/primary/back/",
                    "cam_from_rig_rotation": [1.0, 0.0, 0.0, 0.0],
                    "cam_from_rig_translation": [0.0, 0.0, -physical_baseline],
                },
            ]
        }
    ]
    return Reconstruction(cameras, images, points), {"images": records}, rig


def test_normalized_rig_recovers_meter_scale():
    reconstruction, catalog, rig = _reconstruction(0.0025)

    result = estimate_metric_scale(reconstruction, catalog, rig)

    assert result["metric"] is True
    assert result["applied"] is True
    assert result["model_units_per_meter"] == 0.0025
    assert result["scale_factor"] == 400.0
    assert result["baseline_pairs"] == 12
    assert result["stereo_points"] == 12
    assert result["stereo_captures"] == 12
    assert result["relative_mad"] < 1e-12


def test_already_metric_rig_is_validation_only():
    reconstruction, catalog, rig = _reconstruction(1.0)

    result = estimate_metric_scale(reconstruction, catalog, rig)

    assert result["metric"] is True
    assert result["applied"] is False
    assert result["scale_factor"] == 1.0


def test_monocular_rig_does_not_guess_scale_from_imu():
    reconstruction, catalog, _rig = _reconstruction(1.0)

    result = estimate_metric_scale(reconstruction, catalog, [{"cameras": []}])

    assert result == {
        "applied": False,
        "metric": False,
        "reason": "rig_has_no_physical_baseline",
    }


def test_back_to_back_rig_without_shared_capture_points_is_not_metric():
    reconstruction, catalog, rig = _reconstruction(1.0)
    reconstruction.points3D.clear()

    result = estimate_metric_scale(reconstruction, catalog, rig)

    assert result["metric"] is False
    assert result["reason"] == "rig_baseline_not_observable"
    assert result["baseline_pairs"] == 12
    assert result["stereo_points"] == 0
    assert result["stereo_captures"] == 0


def test_dataset_without_rig_config_has_no_physical_baseline(tmp_path):
    assert _load_rig_config(tmp_path / "rig_config.json") == []
