from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from sphere_reconstruct.colmap import model as cm
from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.domain.source import MediaKind, Projection, SourceRole
from sphere_reconstruct.pipeline.stage import ProgressReporter, SourceContext, StageContext
from sphere_reconstruct.stages import reconstruct as module


def _spec() -> InputSpec:
    images = [
        {"name": "fish/a.jpg", "source_id": "fish", "capture_index": 0, "sensor_id": "front"},
        {"name": "fish/b.jpg", "source_id": "fish", "capture_index": 0, "sensor_id": "back"},
        {"name": "fish/c.jpg", "source_id": "fish", "capture_index": 1, "sensor_id": "front"},
        {"name": "phone/a.jpg", "source_id": "phone", "capture_index": 0, "sensor_id": "main"},
    ]
    return InputSpec(
        version=3, reconstruction_mode="native_fisheye", image_count=4, source_count=2,
        primary_source_id="fish", primary_image_names=[image["name"] for image in images[:3]],
        sources=[
            {"id": "fish", "label": "360", "role": "primary", "projection": "dual_fisheye", "media_kind": "video"},
            {"id": "phone", "label": "Phone", "role": "supplemental", "projection": "perspective", "media_kind": "video"},
        ],
        images=images, feature_batches=[], image_path="images", mask_path=None,
        feature_masks_enabled=False, rig_config_path=None, refine_intrinsics=True,
        refine_rig=False, multiple_models=False,
    )


def _model(*, phone: bool, phone_supported: bool = True) -> cm.Reconstruction:
    spec = _spec()
    camera = cm.Camera(1, "PINHOLE", 100, 100, [50, 50, 50, 50], 1)
    images = {
        index: cm.Image(
            index, (1, 0, 0, 0), (-float(index), 0, 0), 1 if index < 4 else 2,
            record["name"],
            [] if index == 2 or (index == 4 and not phone_supported) else [cm.ImagePoint2D(50, 50, 1)],
        )
        for index, record in enumerate(spec.images, 1) if phone or index < 4
    }
    cameras = {1: camera}
    if phone:
        cameras[2] = cm.Camera(2, "SIMPLE_RADIAL", 100, 100, [80, 50, 50, 0], 2)
    point = cm.Point3D(1, (0, 0, 5), (120, 120, 120), 0.2, [
        (index, 0) for index, image in images.items() if image.points2D
    ])
    return cm.Reconstruction(cameras, images, {1: point})


def _write_model(path: Path, model: cm.Reconstruction) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cm.write_cameras_bin(path / "cameras.bin", model.cameras)
    cm.write_images_bin(path / "images.bin", model.images)
    cm.write_points3D_bin(path / "points3D.bin", model.points3D)


def test_empty_rig_sensor_is_supported_by_other_sensor_but_empty_phone_is_rejected(tmp_path):
    _write_model(tmp_path / "0", _model(phone=True, phone_supported=False))
    _, summary = module._select_largest_model(tmp_path, _spec(), max_step_ratio=10)
    assert summary["unsupported_registered_frames"] == [
        {"source_id": "phone", "source_label": "Phone", "capture_index": 0}
    ]
    params = module.Reconstruct().normalize_params({"min_points3D": 1})
    assert not module._summary_passes(summary, params, "fish", require_trajectory_continuity=True)
    summary["registered_ratio"] = 1.0
    with pytest.raises(RuntimeError, match="Phone:0"):
        module._validate_summary(summary, params)


def test_supplemental_jump_fails_even_when_primary_passes():
    summary = {
        "registered_ratio": 1.0, "num_points3D": 1000,
        "source_registration": {"fish": {"registered": 100, "total": 100}},
        "primary_trajectory": {"passed": True},
        "supplemental_trajectories": {"phone": {
            "label": "Phone", "trajectory": {"available": True, "passed": False, "outlier_steps": 2, "maximum_to_p95_ratio": 200},
        }},
    }
    params = module.Reconstruct().normalize_params({})
    assert not module._summary_passes(summary, params, "fish", require_trajectory_continuity=True)
    with pytest.raises(RuntimeError, match="supplemental trajectory Phone"):
        module._validate_summary(summary, params, require_trajectory_continuity=True)


def test_mixed_stage_anchors_primary_and_preserves_per_camera_policy(tmp_path, monkeypatch):
    spec = _spec()
    project = tmp_path / "project"
    output = tmp_path / "experiment"
    output.mkdir()
    for directory in ("match_features", "extract_features", "rectify_fisheye"):
        (project / directory).mkdir(parents=True)
    spec.write(project / "extract_features/input_spec.json")
    catalog = {"images": spec.images, "camera_groups": [
        {"id": "fish", "source_id": "fish", "refine_intrinsics": False, "has_prior_focal_length": True, "image_names": spec.primary_image_names},
        {"id": "phone", "source_id": "phone", "refine_intrinsics": True, "has_prior_focal_length": False, "image_names": ["phone/a.jpg"]},
    ]}
    (project / "rectify_fisheye/image_catalog.json").write_text(json.dumps(catalog))
    with sqlite3.connect(project / "match_features/database.db") as db:
        db.executescript("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, prior_focal_length INTEGER); CREATE TABLE images(image_id INTEGER, name TEXT, camera_id INTEGER);")
        db.executemany("INSERT INTO cameras VALUES (?, 1)", [(1,), (2,)])
        db.executemany("INSERT INTO images VALUES (?, ?, ?)", [
            (index, image["name"], 1 if index < 4 else 2) for index, image in enumerate(spec.images, 1)
        ])
    events = []
    sources = tuple(SourceContext(
        id=source["id"], label=source["label"], role=SourceRole(source["role"]),
        adapter="test", media_kind=MediaKind.VIDEO, projection=Projection(source["projection"]),
        path=project, ordinal=index, enabled=True,
    ) for index, source in enumerate(spec.sources))
    context = StageContext(
        "test", project, output, module.Reconstruct().normalize_params({"mapper": "global", "min_points3D": 1}),
        sources, ProgressReporter(lambda *args: events.append(args)),
    )

    def global_mapper(_binary, **kwargs):
        assert kwargs["refine_intrinsics"] is False
        args = kwargs["extra_args"]
        names = Path(args[args.index("--GlobalMapper.image_list_path") + 1]).read_text().splitlines()
        assert names == spec.primary_image_names
        _write_model(kwargs["output_path"] / "0", _model(phone=False))
        kwargs["on_line"]("Global bundle adjustment iteration 3 / 3 finished")

    def mapper(_binary, **kwargs):
        args = kwargs["extra_args"]
        assert args[args.index("--Mapper.fix_existing_frames") + 1] == "1"
        assert args[args.index("--Mapper.structure_less_registration_fallback") + 1] == "0"
        fixed = Path(args[args.index("--Mapper.constant_camera_list_path") + 1]).read_text().splitlines()
        assert fixed == ["1"]
        assert kwargs["refine_intrinsics"] is True
        with sqlite3.connect(kwargs["database_path"]) as db:
            assert db.execute("SELECT camera_id, prior_focal_length FROM cameras ORDER BY camera_id").fetchall() == [(1, 1), (2, 0)]
        seed = cm.read_model(Path(args[args.index("--input_path") + 1]))
        assert len(seed.images) == 3
        _write_model(kwargs["output_path"], _model(phone=True))

    def preview(ctx, _model, **_kwargs):
        folder = ctx.stage_out_dir / "preview"
        folder.mkdir()
        (folder / "reconstruction.json").write_text("{}")
        (folder / "points.bin").write_bytes(b"points")
        return SimpleNamespace(num_points_written=1)

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(binaries=SimpleNamespace(colmap="colmap")))
    monkeypatch.setattr(module.colmap_runner, "resolve_colmap_bin", lambda binary: binary)
    monkeypatch.setattr(module.colmap_runner, "global_mapper", global_mapper)
    monkeypatch.setattr(module.colmap_runner, "mapper", mapper)
    monkeypatch.setattr(module.colmap_runner, "view_graph_calibrator", lambda *_args, **_kwargs: pytest.fail("frozen primary must not calibrate phone view graph"))
    monkeypatch.setattr(module, "_complete_incremental_point_colors", lambda *_args, **_kwargs: ({}, {"applied": True}))
    monkeypatch.setattr(module.similarity_transform, "write_preview", preview)

    result = module.Reconstruct().execute(context)

    assert result.extra["num_images"] == 4
    assert result.extra["registered_total_ratio"] == 1
    assert result.extra["source_registration"]["phone"]["registered"] == 1
    assert result.extra["mixed_registration"]["primary_poses_fixed"] is True
    assert result.extra["unsupported_registered_frames"] == []
    assert result.extra["point_color_completion"]["applied"] is True
    with sqlite3.connect(project / "match_features/database.db") as db:
        assert db.execute("SELECT prior_focal_length FROM cameras WHERE camera_id=2").fetchone() == (1,)
