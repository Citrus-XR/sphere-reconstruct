"""Mandatory fisheye rectification が image / mask / camera / rig を同時に変換することを検証する。"""

import json

import numpy as np
from PIL import Image

from sphere_reconstruct.domain.camera_system import OmniDistortionModel, OmniIntrinsics
from sphere_reconstruct.imaging import projection
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.rectify_fisheye import RectifyFisheye


def test_rectify_fisheye_materializes_consistent_opencv_png(tmp_path):
    project = tmp_path / "project"
    prepared = project / "prepare_images"
    prepared.mkdir(parents=True)
    source_path = prepared / "source.jpg"
    grid_y, grid_x = np.mgrid[:64, :64]
    source_rgb = np.stack((grid_x * 4, grid_y * 4, (grid_x + grid_y) * 2), axis=2).astype(np.uint8)
    Image.fromarray(source_rgb, mode="RGB").save(source_path, format="JPEG", quality=100, subsampling=0)
    source_projection = OmniIntrinsics(
        width=64,
        height=64,
        xi=2.0,
        fx=72.0,
        fy=71.0,
        cx=32.0,
        cy=31.5,
        distortion_model=OmniDistortionModel.RADTAN,
        distortion_parameters=(0.02, 0.01, -0.002, 0.001, -0.001),
    )
    thin = projection.approximate_thin_prism_fisheye(source_projection)
    target = projection.approximate_opencv_fisheye(source_projection)
    name = "sources/source/lens0/frame_000000.jpg"
    catalog = {
        "version": 3,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "source",
        "sources": [
            {
                "id": "source",
                "label": "Source",
                "role": "primary",
                "adapter": "insta360",
                "media_kind": "video",
                "projection": "dual_fisheye",
                "count": 1,
            }
        ],
        "camera_groups": [
            {
                "id": "source:native-fisheye:lens0",
                "source_id": "source",
                "camera_model": thin.camera_model,
                "camera_params": list(thin.params),
                "single_camera": True,
                "single_camera_per_folder": False,
                "refine_intrinsics": False,
                "image_names": [name],
                "rectification": {
                    "required": True,
                    "source_projection": source_projection.to_dict(),
                    "target_camera_model": target.camera_model,
                    "target_camera_params": list(target.params),
                    "rms_error_px_before_resampling": target.rms_error_px,
                    "maximum_error_px_before_resampling": target.maximum_error_px,
                },
            }
        ],
        "rig_config_path": "rig_config.json",
        "images": [
            {
                "name": name,
                "path": "prepare_images/source.jpg",
                "source_id": "source",
                "source_label": "Source",
                "source_role": "primary",
                "capture_index": 0,
                "source_index": 0,
                "timestamp_sec": 0.0,
                "sensor_id": "lens0",
                "projection": "dual_fisheye",
                "width": 64,
                "height": 64,
                "camera_group_id": "source:native-fisheye:lens0",
                "valid_region": {
                    "kind": "fisheye",
                    "camera_model": thin.camera_model,
                    "params": list(thin.params),
                    "max_theta_rad": 1.4,
                    "physical_circle": {
                        "cx": 0.5,
                        "cy": 0.5,
                        "r": 0.45,
                        "operations": [
                            {"mode": "subtract", "x": 0.5, "y": 0.5, "r": 0.05}
                        ],
                    },
                },
            }
        ],
        "source_calibrations": [{"source_id": "source"}],
    }
    (prepared / "image_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (prepared / "rig_config.json").write_text(
        json.dumps(
            [
                {
                    "cameras": [
                        {
                            "image_prefix": "sources/source/lens0/",
                            "camera_model_name": thin.camera_model,
                            "camera_params": list(thin.params),
                            "ref_sensor": True,
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    output = project / ".rectify_fisheye.tmp"
    output.mkdir()
    stage = RectifyFisheye()
    context = StageContext(
        project_id="project",
        project_dir=project,
        stage_out_dir=output,
        params={},
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )

    manifest = stage.execute(context)

    result = json.loads((output / "image_catalog.json").read_text(encoding="utf-8"))
    image = result["images"][0]
    group = result["camera_groups"][0]
    assert result["version"] == 3
    assert image["name"].endswith(".png")
    assert image["valid_mask_path"].endswith("lens0.png")
    assert group["camera_model"] == "OPENCV_FISHEYE"
    assert len(group["camera_params"]) == 8
    with Image.open(output / "images" / image["name"]) as rectified:
        assert rectified.size == (64, 64)
        assert rectified.format == "PNG"
    with Image.open(output / "validity" / "source" / "lens0.png") as validity:
        mask = np.asarray(validity)
        assert mask[32, 32] == 0
        assert mask.sum() > 0
    rig = json.loads((output / "rig_config.json").read_text(encoding="utf-8"))
    assert rig[0]["cameras"][0]["camera_model_name"] == "OPENCV_FISHEYE"
    assert manifest.extra["rectified_images"] == 1
    assert manifest.extra["image_format"] == "png"
    quality = manifest.extra["roundtrip_quality"][0]
    assert quality["samples"] > 100
    assert quality["psnr_db"] > 25.0
