"""魚眼学習 crop が有効画素と COLMAP 幾何を同じ座標系へ移すことを検証する。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image as PilImage

from sphere_reconstruct.colmap import model, training_crop


def _reconstruction() -> model.Reconstruction:
    camera = model.Camera(
        camera_id=1,
        model="OPENCV_FISHEYE",
        width=64,
        height=64,
        params=[20.0, 21.0, 32.0, 33.0, 0.0, 0.0, 0.0, 0.0],
        model_id=5,
    )
    image = model.Image(
        image_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=1,
        name="front/frame.jpg",
        points2D=[model.ImagePoint2D(20.0, 21.0, 7)],
    )
    return model.Reconstruction(cameras={1: camera}, images={1: image}, points3D={})


def _catalog() -> dict:
    return {
        "images": [
            {
                "name": "front/frame.jpg",
                "width": 64,
                "height": 64,
                "valid_region": {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.25},
            }
        ]
    }


def test_crop_plan_updates_principal_point_and_observations(tmp_path: Path):
    plan = training_crop.build_plan(_reconstruction(), _catalog())
    cropped = training_crop.crop_reconstruction(_reconstruction(), plan)

    assert plan.images["front/frame.jpg"] == training_crop.CropRect(16, 16, 48, 48)
    assert plan.source_pixels == 64 * 64
    assert plan.cropped_pixels == 32 * 32
    assert cropped.cameras[1].width == 32
    assert cropped.cameras[1].height == 32
    assert cropped.cameras[1].params[2:4] == [16.0, 17.0]
    assert (cropped.images[1].points2D[0].x, cropped.images[1].points2D[0].y) == (4.0, 5.0)

    model.write_cameras_bin(tmp_path / "cameras.bin", cropped.cameras)
    model.write_images_bin(tmp_path / "images.bin", cropped.images)
    assert model.read_cameras_bin(tmp_path / "cameras.bin") == cropped.cameras
    assert model.read_images_bin(tmp_path / "images.bin") == cropped.images


def test_crop_plan_uses_calibrated_fisheye_forward_bounds():
    catalog = {
        "images": [
            {
                "name": "front/frame.jpg",
                "width": 64,
                "height": 64,
                "valid_region": {
                    "kind": "opencv_fisheye",
                    "params": [20.0, 21.0, 32.0, 33.0, 0.0, 0.0, 0.0, 0.0],
                    "max_theta_rad": 0.5,
                    "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.49},
                },
            }
        ]
    }

    plan = training_crop.build_plan(_reconstruction(), catalog)

    assert plan.images["front/frame.jpg"] == training_crop.CropRect(16, 16, 48, 48)


def test_inconsistent_crop_for_shared_camera_is_rejected():
    reconstruction = _reconstruction()
    reconstruction.images[2] = model.Image(
        image_id=2,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=1,
        name="front/other.jpg",
    )
    catalog = _catalog()
    catalog["images"].append(
        {
            "name": "front/other.jpg",
            "width": 64,
            "height": 64,
            "valid_region": {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.4},
        }
    )

    with pytest.raises(ValueError, match="inconsistent"):
        training_crop.build_plan(reconstruction, catalog)


def test_jpeg_and_mask_crop_keep_planned_dimensions(tmp_path: Path):
    jpegtran = shutil.which("jpegtran")
    if jpegtran is None:
        pytest.skip("jpegtran is not installed on this test host")
    source_images = tmp_path / "source-images"
    source_masks = tmp_path / "source-masks"
    image_path = source_images / "front" / "frame.jpg"
    mask_path = source_masks / "front" / "frame.jpg.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    PilImage.new("RGB", (64, 64), (30, 60, 90)).save(image_path, format="JPEG", quality=90)
    PilImage.new("L", (64, 64), 255).save(mask_path, format="PNG")
    plan = training_crop.build_plan(_reconstruction(), _catalog())

    training_crop.crop_images(source_images, tmp_path / "images", plan, jpegtran)
    training_crop.crop_masks(source_masks, tmp_path / "masks", plan)

    with PilImage.open(tmp_path / "images" / "front" / "frame.jpg") as image:
        assert image.size == (32, 32)
    with PilImage.open(tmp_path / "masks" / "front" / "frame.jpg.png") as mask:
        assert mask.size == (32, 32)
