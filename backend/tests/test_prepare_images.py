"""Mixed source を canonical image catalog へ変換する。"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sphere_reconstruct.domain.camera_system import OmniDistortionModel
from sphere_reconstruct.imaging import source_region, valid_region
from sphere_reconstruct.insta360 import camera_system
from sphere_reconstruct.insta360.calibration import (
    CalibSource,
    DualLensCalibration,
    OmniLensCalibration,
)
from sphere_reconstruct.insta360.metadata import WindowCropInfo
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.prepare_images import PrepareImages


def _valid_dual_fisheye_calibration() -> dict:
    common = {
        "xi": 2.0,
        "tx": 0.0,
        "ty": 0.0,
        "distortion_model": OmniDistortionModel.RADTAN,
        "distortion_parameters": (0.183, 2.06, -3.27, 0.0, 0.0),
        "ref_image_width": 10752,
        "ref_image_height": 5376,
        "lens_flags": 113,
    }
    return {
        "valid": True,
        "lenses": [
            {
                **common,
                "yaw": 0.615,
                "pitch": 0.016,
                "roll": 89.937,
                "fx": 4278.3,
                "fy": 4277.33,
                "cx": 2694.63,
                "cy": 2681.84,
                "tz": 0.0,
            },
            {
                **common,
                "yaw": -0.718,
                "pitch": 0.211,
                "roll": 89.840,
                "fx": 4296.81,
                "fy": 4298.54,
                "cx": 8064.92,
                "cy": 2686.41,
                "tz": -0.032273,
            },
        ],
    }


def test_perspective_exif_orientation_and_focal_are_normalized(tmp_path):
    project = tmp_path / "project"
    source = project / "phone.jpg"
    source.parent.mkdir(parents=True)
    image = Image.new("RGB", (120, 80), (20, 40, 60))
    exif = Image.Exif()
    exif[274] = 6
    exif[271] = "Apple"
    exif[272] = "Phone"
    exif[41989] = 25
    image.save(source, exif=exif)
    _write_documents(
        project,
        sources=[
            {
                "id": "phone",
                "label": "Phone",
                "role": "primary",
                "adapter": "generic_images",
                "media_kind": "images",
                "projection": "perspective",
                "kind": "perspective_images",
                "width": None,
                "height": None,
                "count": 1,
                "frames": [
                    {
                        "index": 0,
                        "source_id": "phone",
                        "source_index": 0,
                        "timestamp_sec": None,
                        "image_source": str(source),
                    }
                ],
            }
        ],
    )
    output = project / ".prepare_images.tmp"
    output.mkdir()
    stage = PrepareImages()
    context = StageContext(
        project_id="project",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params({"reconstruction_mode": "native_fisheye"}),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )

    stage.execute(context)

    catalog = json.loads((output / "image_catalog.json").read_text())
    prepared = catalog["images"][0]
    assert (prepared["width"], prepared["height"]) == (80, 120)
    group = catalog["camera_groups"][0]
    assert group["camera_model"] == "SIMPLE_RADIAL"
    assert 75 < group["camera_params"][0] < 85
    with Image.open(output / Path(prepared["path"]).relative_to("prepare_images")) as normalized:
        assert normalized.getexif().get(274) is None


def test_phone_exclusion_reaches_feature_and_export_masks_without_sam(tmp_path):
    from sphere_reconstruct.colmap import input_workspace
    from sphere_reconstruct.stages.export_dataset import _copy_masks

    project = tmp_path / "project"
    project.mkdir()
    photo = project / "phone.jpg"
    image = Image.new("RGB", (120, 80), (20, 40, 60))
    exif = Image.Exif()
    exif[274] = 6
    image.save(photo, exif=exif)
    source = {
        "id": "phone", "label": "Phone", "role": "primary", "adapter": "generic_images",
        "media_kind": "images", "projection": "perspective", "kind": "perspective_images",
        "width": None, "height": None, "count": 1,
        "frames": [{"index": 0, "source_id": "phone", "source_index": 0, "timestamp_sec": None,
                    "image_source": str(photo)}],
    }
    _write_documents(project, sources=[source])
    region = source_region.default_region("perspective")
    region["views"]["main"]["operations"] = [
        {"mode": "subtract", "x": 0.75, "y": 0.25, "r": 0.1, "stroke_id": 1},
    ]
    source_region.save_region(project, "phone", "perspective", region)
    output = project / ".prepare_images.tmp"
    output.mkdir()
    stage = PrepareImages()
    context = StageContext(project_id="project", project_dir=project, stage_out_dir=output,
                           params=stage.normalize_params({}), sources=(),
                           progress=ProgressReporter(lambda *_args: None))
    stage.execute(context)
    record = json.loads((output / "image_catalog.json").read_text())["images"][0]
    assert (record["width"], record["height"]) == (80, 120)
    mask = valid_region.render_mask(record["valid_region"], 80, 120)
    assert mask[30, 60] == 0
    assert mask[90, 20] == mask[0, 0] == 1
    output.rename(project / "rectify_fisheye")
    catalog_path = project / "rectify_fisheye" / "image_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for item in catalog["images"]:
        item["path"] = item["path"].replace("prepare_images", "rectify_fisheye")
    catalog_path.write_text(json.dumps(catalog))
    spec = input_workspace.build(project, project / "extract_features", use_feature_masks=False)
    assert spec.mask_path is not None
    with Image.open(project / "extract_features" / "masks" / (record["name"] + ".png")) as generated:
        assert generated.getpixel((60, 30)) == 0
    count, sizes = _copy_masks(project, None, project / "export" / "masks", {record["name"]},
                               {record["name"]: (80, 120)})
    assert count == 1 and sizes[record["name"]] == (80, 120)
    with Image.open(project / "export" / "masks" / (record["name"] + ".png")) as exported:
        assert exported.getpixel((60, 30)) == 0
        assert exported.getpixel((0, 0)) == 255


@pytest.mark.parametrize("projection_name", ["dual_fisheye", "equirectangular"])
def test_source_region_is_remapped_with_pinhole_views(tmp_path, projection_name):
    project = tmp_path / "project"
    project.mkdir()
    image = project / "source.jpg"
    width, height = (100, 100) if projection_name == "dual_fisheye" else (200, 100)
    Image.new("RGB", (width, height), (80, 80, 80)).save(image)
    frame = {"index": 0, "source_id": "primary", "source_index": 0, "timestamp_sec": 0.0}
    frame.update({"lens0": "source.jpg", "lens1": "source.jpg"} if projection_name == "dual_fisheye"
                 else {"image": "source.jpg"})
    source = {"id": "primary", "label": "360", "role": "primary", "adapter": "insta360",
              "media_kind": "video", "projection": projection_name, "kind": "insv_dual",
              "width": width, "height": height, "count": 1, "frames": [frame]}
    if projection_name == "dual_fisheye":
        source["calibration"] = _valid_dual_fisheye_calibration()
    _write_documents(project, sources=[source])
    region = source_region.default_region(projection_name)
    for view in region["views"].values():
        view["operations"] = [{"mode": "subtract", "x": 0.5, "y": 0.5, "r": 0.12, "stroke_id": 1}]
    stage = PrepareImages()
    masks_by_run = []
    for run, current in enumerate((source_region.default_region(projection_name), region)):
        source_region.save_region(project, "primary", projection_name, current)
        output = project / f"output{run}"
        output.mkdir()
        context = StageContext(project_id="project", project_dir=project, stage_out_dir=output,
                               params=stage.normalize_params({"reconstruction_mode": "pinhole_rig", "size": 32}),
                               sources=(), progress=ProgressReporter(lambda *_args: None))
        stage.execute(context)
        catalog = json.loads((output / "image_catalog.json").read_text())
        masks = []
        for record in catalog["images"]:
            with Image.open(project / record["valid_mask_path"]) as mask:
                masks.append(np.asarray(mask).copy())
        masks_by_run.append(masks)
    assert len(masks_by_run[0]) == len(masks_by_run[1])
    assert all(np.all(mask <= original) for original, mask in zip(*masks_by_run, strict=True))
    assert sum(np.count_nonzero(mask) for mask in masks_by_run[1]) < sum(
        np.count_nonzero(mask) for mask in masks_by_run[0]
    )


def test_native_fisheye_and_phone_create_separate_camera_groups(tmp_path):
    project = tmp_path / "project"
    extract = project / "extract_frames"
    extract.mkdir(parents=True)
    front = extract / "front.jpg"
    back = extract / "back.jpg"
    phone = project / "phone.jpg"
    for path in (front, back, phone):
        Image.new("RGB", (100, 100), (80, 80, 80)).save(path)
    _write_documents(
        project,
        sources=[
            {
                "id": "primary",
                "label": "360",
                "role": "primary",
                "adapter": "insta360",
                "media_kind": "video",
                "projection": "dual_fisheye",
                "kind": "insv_dual",
                "width": 100,
                "height": 100,
                "count": 1,
                "calibration": _valid_dual_fisheye_calibration(),
                "frames": [
                    {
                        "index": 0,
                        "source_id": "primary",
                        "source_index": 0,
                        "timestamp_sec": 0.0,
                        "lens0": str(front.relative_to(project)),
                        "lens1": str(back.relative_to(project)),
                    }
                ],
            },
            {
                "id": "phone",
                "label": "Phone",
                "role": "supplemental",
                "adapter": "generic_images",
                "media_kind": "images",
                "projection": "perspective",
                "kind": "perspective_images",
                "width": None,
                "height": None,
                "count": 1,
                "frames": [
                    {
                        "index": 1,
                        "source_id": "phone",
                        "source_index": 0,
                        "timestamp_sec": None,
                        "image_source": str(phone),
                    }
                ],
            },
        ],
    )
    output = project / ".prepare_images.tmp"
    output.mkdir()
    stage = PrepareImages()
    progress_calls = []
    context = StageContext(
        project_id="project",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params({"reconstruction_mode": "native_fisheye"}),
        sources=(),
        progress=ProgressReporter(lambda *args: progress_calls.append(args)),
    )

    stage.execute(context)

    catalog = json.loads((output / "image_catalog.json").read_text())
    assert len(catalog["images"]) == 3
    assert {group["camera_model"] for group in catalog["camera_groups"]} == {
        "THIN_PRISM_FISHEYE",
        "SIMPLE_RADIAL",
    }
    assert len(json.loads((output / "rig_config.json").read_text())) == 1
    native_groups = [
        group for group in catalog["camera_groups"] if group["camera_model"] == "THIN_PRISM_FISHEYE"
    ]
    assert len(native_groups) == 2
    assert native_groups[0]["camera_params"] != native_groups[1]["camera_params"]
    assert all(group["refine_intrinsics"] is False for group in native_groups)
    native_regions = [
        image["valid_region"]
        for image in catalog["images"]
        if image["projection"] == "dual_fisheye"
    ]
    assert all(region["kind"] == "fisheye" for region in native_regions)
    assert all(region["camera_model"] == "THIN_PRISM_FISHEYE" for region in native_regions)
    assert all(region["max_theta_rad"] < math.pi / 2 for region in native_regions)
    assert all(len(region["params"]) == 12 for region in native_regions)
    assert all(region["physical_circle"]["r"] > 0 for region in native_regions)
    numeric = [call[1] for call in progress_calls if call[1] is not None]
    assert numeric == sorted(numeric)
    assert numeric[-1] == 0.99
    assert any(call[3] == "log.prepare_source_progress" for call in progress_calls)
    rig = json.loads((output / "rig_config.json").read_text())[0]["cameras"]
    assert rig[0]["image_prefix"].endswith("/lens0/")
    assert rig[1]["image_prefix"].endswith("/lens1/")
    assert rig[1]["cam_from_rig_rotation"] != [0.0, 0.0, 1.0, 0.0]


def _write_documents(project, *, sources):
    inspect = project / "inspect_source"
    extract = project / "extract_frames"
    inspect.mkdir(parents=True, exist_ok=True)
    extract.mkdir(parents=True, exist_ok=True)
    inspections = []
    for source in sources:
        inspection = {key: value for key, value in source.items() if key != "frames"}
        if source["projection"] == "dual_fisheye":
            calibration = source.get("calibration", {"valid": False})
            inspection["calibration"] = calibration
            if calibration.get("valid"):
                system = camera_system.from_calibration(
                    DualLensCalibration(
                        source=CalibSource.OFFSET,
                        version=3,
                        lenses=tuple(
                            OmniLensCalibration(**lens) for lens in calibration["lenses"]
                        ),
                    ),
                    window_crop=WindowCropInfo(5376, 5376, 5312, 5312),
                    rolling_shutter_readout_ms=21.244001,
                )
                relative = f"sources/{source['id']}/camera_system.json"
                path = inspect / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(system.to_dict()))
                inspection["camera_system_path"] = relative
        inspections.append(inspection)
    (inspect / "sources.json").write_text(json.dumps({"version": 4, "sources": inspections}))
    (extract / "manifest_frames.json").write_text(
        json.dumps(
            {
                "version": 2,
                "count": sum(len(source["frames"]) for source in sources),
                "sources": sources,
                "frames": [frame for source in sources for frame in source["frames"]],
            }
        )
    )
