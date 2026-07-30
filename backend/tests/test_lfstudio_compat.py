"""Stock LFStudio 用 camera-model workaround を検証する。"""

from sphere_reconstruct.colmap import lfstudio_compat, model, training_crop
from sphere_reconstruct.stages.export_dataset import ExportDataset


def test_camera_only_stock_workaround_is_not_default():
    assert ExportDataset().normalize_params({})["lfstudio_stock_thin_prism_workaround"] is False


def test_thin_prism_export_uses_consistent_opencv_model_after_crop():
    camera = model.Camera(
        camera_id=1,
        model="THIN_PRISM_FISHEYE",
        width=32,
        height=32,
        params=[20.0, 20.0, 16.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        model_id=10,
    )
    image = model.Image(
        image_id=1,
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=1,
        name="sources/source/lens0/frame_000000.jpg",
    )
    reconstruction = model.Reconstruction(cameras={1: camera}, images={1: image}, points3D={})
    catalog = {
        "camera_groups": [
            {
                "id": "source:native-fisheye:lens0",
                "consumer_compatibility": {
                    "lichtfeld_stock": {
                        "camera_model": "OPENCV_FISHEYE",
                        "camera_params": [21.0, 22.0, 32.0, 33.0, 0.1, 0.2, 0.3, 0.4],
                        "rms_error_px": 0.5,
                        "maximum_error_px": 2.0,
                        "reason": "thin_prism_inverse_inconsistent",
                    }
                },
            }
        ],
        "images": [
            {
                "name": image.name,
                "camera_group_id": "source:native-fisheye:lens0",
            }
        ],
    }
    crop = training_crop.CropRect(16, 16, 48, 48)
    plan = training_crop.CropPlan(
        images={image.name: crop},
        cameras={1: crop},
        source_sizes={image.name: (64, 64)},
        source_pixels=4096,
        cropped_pixels=1024,
        alignment_px=16,
    )

    converted, report = lfstudio_compat.apply_stock_camera_workarounds(
        reconstruction,
        catalog,
        plan,
    )

    assert converted.cameras[1].model == "OPENCV_FISHEYE"
    assert converted.cameras[1].model_id == 5
    assert converted.cameras[1].params == [21.0, 22.0, 16.0, 17.0, 0.1, 0.2, 0.3, 0.4]
    assert converted.images is reconstruction.images
    assert report["applied"] is True
    assert report["camera_reports"][0]["maximum_error_px"] == 2.0
