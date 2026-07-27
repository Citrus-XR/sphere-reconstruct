"""Mixed source を canonical image catalog へ変換する。"""

import json
from pathlib import Path

from PIL import Image

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.prepare_images import PrepareImages


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
    assert len(catalog["images"]) == 3
    assert {group["camera_model"] for group in catalog["camera_groups"]} == {
        "OPENCV_FISHEYE",
        "SIMPLE_RADIAL",
    }
    assert len(json.loads((output / "rig_config.json").read_text())) == 1


def _write_documents(project, *, sources):
    inspect = project / "inspect_source"
    extract = project / "extract_frames"
    inspect.mkdir(parents=True, exist_ok=True)
    extract.mkdir(parents=True, exist_ok=True)
    inspections = []
    for source in sources:
        inspection = {key: value for key, value in source.items() if key != "frames"}
        if source["projection"] == "dual_fisheye":
            inspection["offset_v3"] = {"valid": False}
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
