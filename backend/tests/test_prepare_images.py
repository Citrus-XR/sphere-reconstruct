"""Mixed source を canonical image catalog へ変換する。"""

import json
from pathlib import Path

from PIL import Image

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.prepare_images import PrepareImages


def _valid_x5_calibration() -> dict:
    common = {
        "xi": 2.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 90.0,
        "tx": 0.0,
        "ty": 0.0,
        "k1": 0.183,
        "k2": 2.06,
        "k3": -3.27,
        "p1": 0.0,
        "p2": 0.0,
        "ref_image_width": 10752,
        "ref_image_height": 5376,
        "lens_flags": 113,
    }
    return {
        "valid": True,
        "lenses": [
            {
                **common,
                "fx": 4278.3,
                "fy": 4277.33,
                "cx": 2694.63,
                "cy": 2681.84,
                "tz": 0.0,
            },
            {
                **common,
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
                "adapter": "insta360_insv",
                "media_kind": "video",
                "projection": "dual_fisheye",
                "kind": "insv_dual",
                "width": 100,
                "height": 100,
                "count": 1,
                "offset_v3": _valid_x5_calibration(),
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
        "OPENCV_FISHEYE",
        "SIMPLE_RADIAL",
    }
    assert len(json.loads((output / "rig_config.json").read_text())) == 1
    native_groups = [group for group in catalog["camera_groups"] if group["camera_model"] == "OPENCV_FISHEYE"]
    assert len(native_groups) == 2
    assert native_groups[0]["camera_params"] != native_groups[1]["camera_params"]
    assert all(group["refine_intrinsics"] is False for group in native_groups)
    assert all(
        image["valid_region"]["r"] < 0.46
        for image in catalog["images"]
        if image["projection"] == "dual_fisheye"
    )
    numeric = [call[1] for call in progress_calls if call[1] is not None]
    assert numeric == sorted(numeric)
    assert numeric[-1] == 0.99
    assert any(call[3] == "log.prepare_source_progress" for call in progress_calls)


def _write_documents(project, *, sources):
    inspect = project / "inspect_source"
    extract = project / "extract_frames"
    inspect.mkdir(parents=True, exist_ok=True)
    extract.mkdir(parents=True, exist_ok=True)
    inspections = []
    for source in sources:
        inspection = {key: value for key, value in source.items() if key != "frames"}
        if source["projection"] == "dual_fisheye":
            inspection["offset_v3"] = source.get("offset_v3", {"valid": False})
        inspections.append(inspection)
    (inspect / "sources.json").write_text(json.dumps({"version": 2, "sources": inspections}))
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
