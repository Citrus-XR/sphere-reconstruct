import json
import sqlite3
from types import SimpleNamespace

from PIL import Image

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages import extract_features


def test_extracted_focal_priors_are_corrected_before_rig_configuration(tmp_path, monkeypatch):
    project = tmp_path / "project"
    prepared = project / "rectify_fisheye"
    prepared.mkdir(parents=True)
    groups = []
    images = []
    sources = []
    for source_id, known in (("native", True), ("phone", False)):
        name = f"{source_id}/frame.jpg"
        image_path = prepared / name
        image_path.parent.mkdir()
        Image.new("RGB", (16, 16)).save(image_path)
        groups.append({
            "id": source_id,
            "source_id": source_id,
            "camera_model": "OPENCV_FISHEYE" if known else "SIMPLE_RADIAL",
            "camera_params": [8, 8, 8, 8, 0, 0, 0, 0] if known else [19.2, 8, 8, 0],
            "single_camera": True,
            "single_camera_per_folder": False,
            "image_names": [name],
            "refine_intrinsics": not known,
            "has_prior_focal_length": known,
        })
        source = {
            "id": source_id,
            "label": source_id,
            "role": "primary" if known else "supplemental",
            "projection": "dual_fisheye" if known else "perspective",
        }
        sources.append(source)
        images.append({
            "name": name,
            "path": str(image_path.relative_to(project)),
            "source_id": source_id,
            "source_role": source["role"],
            "capture_index": 0,
            "timestamp_sec": 0.0,
            "sensor_id": "main",
            "width": 16,
            "height": 16,
            "valid_region": {"kind": "full"},
        })
    (prepared / "image_catalog.json").write_text(json.dumps({
        "version": 3,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "native",
        "sources": sources,
        "camera_groups": groups,
        "images": images,
        "rig_config_path": "rig_config.json",
    }), encoding="utf-8")
    (prepared / "rig_config.json").write_text("[]", encoding="utf-8")

    def fake_extractor(_binary, *, database_path, image_list_path, **_kwargs):
        with sqlite3.connect(database_path) as connection:
            connection.executescript(
                "CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY, prior_focal_length INTEGER);"
                "CREATE TABLE IF NOT EXISTS images (name TEXT UNIQUE, camera_id INTEGER);"
                "CREATE TABLE IF NOT EXISTS keypoints (rows INTEGER);"
                "CREATE TABLE IF NOT EXISTS descriptors (rows INTEGER);"
            )
            camera_id = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] + 1
            connection.execute("INSERT INTO cameras VALUES (?, 1)", (camera_id,))
            for name in image_list_path.read_text().splitlines():
                connection.execute("INSERT INTO images VALUES (?, ?)", (name, camera_id))
                connection.execute("INSERT INTO keypoints VALUES (20)")
                connection.execute("INSERT INTO descriptors VALUES (20)")

    calls = []

    def fake_rig_configurator(_binary, *, database_path, **_kwargs):
        with sqlite3.connect(database_path) as connection:
            flags = connection.execute(
                "SELECT name, prior_focal_length FROM images JOIN cameras USING (camera_id) ORDER BY name"
            ).fetchall()
        assert flags == [("native/frame.jpg", 1), ("phone/frame.jpg", 0)]
        calls.append("rig")

    monkeypatch.setattr(extract_features.colmap_runner, "feature_extractor", fake_extractor)
    monkeypatch.setattr(extract_features.colmap_runner, "rig_configurator", fake_rig_configurator)
    monkeypatch.setattr(extract_features.colmap_runner, "resolve_colmap_bin", lambda _configured: "colmap")
    monkeypatch.setattr(
        extract_features, "get_settings", lambda: SimpleNamespace(binaries=SimpleNamespace(colmap="colmap"))
    )
    output = project / ".extract_features.tmp"
    output.mkdir()
    stage = extract_features.ExtractFeatures()
    context = StageContext(
        project_id="test",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params({"use_feature_masks": False}),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )

    manifest = stage.execute(context)

    assert calls == ["rig"]
    assert manifest.extra["images"] == 2
