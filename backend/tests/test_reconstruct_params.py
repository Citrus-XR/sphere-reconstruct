"""分割した SfM stages の parameter contract を検証する."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.pipeline.stage import ProgressReporter
from sphere_reconstruct.stages.extract_features import ExtractFeatures
from sphere_reconstruct.stages.match_features import (
    MatchFeatures,
    _dominant_temporal_offset,
    _filter_cross_source_temporal_pairs,
    _filter_same_source_temporal_pairs,
    _rig_verification_args,
    _matching_summary,
)
from sphere_reconstruct.stages.reconstruct import (
    Reconstruct,
    _complete_incremental_point_colors,
    _incremental_args,
    _requested_primary_initialization,
    _run_global_mapper_with_retries,
    _summary_passes,
    _validate_summary,
)


def test_feature_defaults_use_sift_for_normal_footage():
    params = ExtractFeatures().normalize_params({})
    assert params["reconstruction_mode"] == "native_fisheye"
    assert params["feature_type"] == "SIFT"
    assert params["max_image_size"] == 0
    assert params["max_num_features"] == 8192


def test_zero_feature_limits_delegate_to_colmap_defaults():
    params = ExtractFeatures().normalize_params({"max_image_size": 0, "max_num_features": 0})
    args = ExtractFeatures._feature_args(params, None)
    assert "--FeatureExtraction.max_image_size" not in args
    assert "--SiftExtraction.max_num_features" not in args


def test_matching_defaults_to_mixed_source_auto_pairing():
    params = MatchFeatures().normalize_params({})
    assert params["matcher_type"] == "bruteforce"
    assert params["pairing"] == "auto"
    assert params["overlap"] == 4
    assert params["loop_closure"] is False
    assert params["transitive_matching"] is False
    assert params["transitive_iterations"] == 1
    assert params["cross_source_temporal_filter"] is False
    assert params["cross_source_temporal_window_sec"] == 3.0
    assert params["same_source_max_time_gap_sec"] == 0.0


def test_rig_verification_args_are_only_emitted_for_a_rig():
    assert _rig_verification_args(True, enabled=False) == []
    assert _rig_verification_args(True, enabled=True) == [
        "--FeatureMatching.rig_verification",
        "1",
    ]
    assert _rig_verification_args(False, enabled=True) == []


def test_matching_summary_separates_rig_inliers_below_per_pair_threshold(tmp_path):
    import sqlite3

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO matches VALUES (2147483649, 20), (2147483650, 30), (4294967297, 0);"
            "INSERT INTO two_view_geometries VALUES (2147483649, 1), (2147483650, 15), (4294967297, 0);"
        )
    summary = _matching_summary(database, SimpleNamespace(images=[]), 15)
    assert summary["verified_pairs"] == 2
    assert summary["pairs_meeting_inlier_threshold"] == 1
    assert summary["pairs_below_inlier_threshold"] == 1
    assert summary["pair_inlier_threshold"] == 15


def test_temporal_filter_keeps_dominant_video_offset_and_removes_outliers(tmp_path: Path):
    import sqlite3

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        images = []
        for index in range(30):
            images.extend(
                [
                    (index * 2 + 1, f"a/{index}.jpg"),
                    (index * 2 + 2, f"b/{index}.jpg"),
                ]
        )
        connection.executemany("INSERT INTO images VALUES (?, ?)", images)
        maximum = 2_147_483_647

        def pair(first, second):
            return min(first, second) * maximum + max(first, second)

        rows = []
        for index in range(25):
            # The dominant offset is A-B = +2 seconds.
            rows.append((pair(index * 2 + 1, index * 2 + 2), 100))
        for index in range(25, 30):
            rows.append((pair(index * 2 + 1, index * 2 + 2), 100))
        connection.executemany("INSERT INTO matches VALUES (?, ?)", rows)
        connection.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", rows)

    images_spec = []
    for index in range(30):
        images_spec.extend(
            [
                {
                    "name": f"a/{index}.jpg",
                    "source_id": "a",
                    "timestamp_sec": float(index + (2 if index < 25 else 30)),
                },
                {"name": f"b/{index}.jpg", "source_id": "b", "timestamp_sec": float(index)},
            ]
        )
    from sphere_reconstruct.colmap.input_workspace import InputSpec

    spec = InputSpec(
        version=3,
        reconstruction_mode="perspective",
        image_count=len(images_spec),
        source_count=2,
        primary_source_id="a",
        primary_image_names=[],
        sources=[],
        images=images_spec,
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path=None,
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )

    result = _filter_cross_source_temporal_pairs(database, spec, window_sec=3.0)

    assert result["removed_pairs"] == 5
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 25


def test_temporal_filter_ignores_photo_sources(tmp_path: Path):
    import sqlite3

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        connection.execute("INSERT INTO images VALUES (1, 'a.jpg')")
        connection.execute("INSERT INTO images VALUES (2, 'b.jpg')")
        pair_id = 2_147_483_647 + 2
        connection.execute("INSERT INTO matches VALUES (?, 30)", (pair_id,))
        connection.execute("INSERT INTO two_view_geometries VALUES (?, 30)", (pair_id,))

    from sphere_reconstruct.colmap.input_workspace import InputSpec

    spec = InputSpec(
        version=3,
        reconstruction_mode="perspective",
        image_count=2,
        source_count=2,
        primary_source_id="a",
        primary_image_names=[],
        sources=[],
        images=[
            {"name": "a.jpg", "source_id": "a", "timestamp_sec": None},
            {"name": "b.jpg", "source_id": "b", "timestamp_sec": None},
        ],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path=None,
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )

    result = _filter_cross_source_temporal_pairs(database, spec, window_sec=3.0)

    assert result["enabled"] is False
    assert result["removed_pairs"] == 0


def test_same_source_temporal_filter_keeps_rig_and_cross_source_edges(tmp_path: Path):
    import sqlite3

    database = tmp_path / "database.db"
    maximum = 2_147_483_647

    def pair(first, second):
        return min(first, second) * maximum + max(first, second)

    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?)",
            [
                (1, "video/front_000.jpg"),
                (2, "video/back_000.jpg"),
                (3, "video/front_001.jpg"),
                (4, "video/front_002.jpg"),
                (5, "other/frame.jpg"),
            ],
        )
        rows = [(pair(1, 2), 50), (pair(1, 3), 50), (pair(1, 4), 50), (pair(1, 5), 50)]
        connection.executemany("INSERT INTO matches VALUES (?, ?)", rows)
        connection.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", rows)

    spec = InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=5,
        source_count=2,
        primary_source_id="video",
        primary_image_names=[],
        sources=[],
        images=[
            {"name": "video/front_000.jpg", "source_id": "video", "timestamp_sec": 0.0},
            {"name": "video/back_000.jpg", "source_id": "video", "timestamp_sec": 0.0},
            {"name": "video/front_001.jpg", "source_id": "video", "timestamp_sec": 7.0},
            {"name": "video/front_002.jpg", "source_id": "video", "timestamp_sec": 11.0},
            {"name": "other/frame.jpg", "source_id": "other", "timestamp_sec": 100.0},
        ],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path=None,
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )

    result = _filter_same_source_temporal_pairs(database, spec, max_gap_sec=10.0)

    assert result == {
        "enabled": True,
        "max_gap_sec": 10.0,
        "removed_pairs": 1,
        "sources": {"video": 1},
    }
    with sqlite3.connect(database) as connection:
        remaining = {
            row[0] for row in connection.execute("SELECT pair_id FROM two_view_geometries")
        }
    assert remaining == {pair(1, 2), pair(1, 3), pair(1, 5)}


def test_temporal_offset_detection_rejects_ambiguous_video_pair():
    values = [(index, float(index), 30) for index in range(25)]

    assert _dominant_temporal_offset(values) is None


def test_rig_verification_can_be_disabled_for_diagnostics():
    assert MatchFeatures().normalize_params({"rig_verification": False})["rig_verification"] is False


def test_rig_verification_runs_on_pairing_graph_before_transitive(tmp_path: Path, monkeypatch):
    from sphere_reconstruct.stages import match_features as module

    extract_dir = tmp_path / "extract_features"
    extract_dir.mkdir()
    (extract_dir / "database.db").write_bytes(b"features")
    output = tmp_path / ".match_features.tmp"
    output.mkdir()
    spec = InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=10,
        source_count=1,
        primary_source_id="primary",
        primary_image_names=[],
        sources=[{"id": "primary", "label": "Primary", "role": "primary"}],
        images=[],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path="rig_config.json",
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )
    calls = {}

    monkeypatch.setattr(module.InputSpec, "read", lambda _path: spec)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(
            binaries=SimpleNamespace(colmap="colmap", vocab_tree="tree"),
            aliked=SimpleNamespace(matcher_path=""),
        ),
    )
    monkeypatch.setattr(module.colmap_runner, "resolve_colmap_bin", lambda _value: "colmap")
    monkeypatch.setattr(
        module.colmap_runner,
        "resolve_vocab_tree_path",
        lambda _value: tmp_path / "tree.bin",
    )
    monkeypatch.setattr(
        module.colmap_runner,
        "sequential_matcher",
        lambda _binary, **kwargs: calls.setdefault("pairing", kwargs),
    )
    monkeypatch.setattr(
        module.colmap_runner,
        "transitive_matcher",
        lambda _binary, **kwargs: calls.setdefault("transitive", kwargs),
    )
    monkeypatch.setattr(
        module,
        "_matching_summary",
        lambda *_args: {
            "raw_pairs": 10,
            "verified_pairs": 8,
            "minimum_inliers": 20,
            "average_inliers": 30.0,
            "maximum_inliers": 40,
            "total_inliers": 240,
            "cross_source_verified_pairs": 0,
            "source_pair_counts": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_filter_same_source_temporal_pairs",
        lambda *_args, **_kwargs: {"enabled": True, "removed_pairs": 0, "sources": {}},
    )
    context = SimpleNamespace(
        project_dir=tmp_path,
        stage_out_dir=output,
        params=MatchFeatures().normalize_params(
            {"pairing": "sequential", "transitive_matching": True}
        ),
        progress=ProgressReporter(lambda *_args: None),
        inputs_for=lambda _stage: [],
    )

    manifest = MatchFeatures().execute(context)

    assert calls["pairing"]["extra_args"][-2:] == [
        "--FeatureMatching.rig_verification",
        "1",
    ]
    assert "--FeatureMatching.rig_verification" not in calls["transitive"]["extra_args"]
    assert manifest.extra["rig_verification_scope"] == "pairing_graph"


def test_reconstruction_defaults_to_incremental_without_optional_gpu_solvers():
    params = Reconstruct().normalize_params({})
    assert params["mapper"] == "incremental"
    assert params["view_graph_calibration"] is False
    assert params["ba_use_gpu"] is False
    assert params["global_positioning_use_gpu"] is False
    assert params["random_seed"] == 0
    assert params["min_registered_ratio"] == 0.8
    assert params["min_points3D"] == 100
    assert params["incremental_fallback"] is True
    assert params["max_adjacent_step_ratio"] == 10.0
    assert {
        key: params[key]
        for key in (
            "filter_max_reproj_error",
            "filter_min_tri_angle",
            "tri_create_max_angle_error",
            "tri_continue_max_angle_error",
            "tri_merge_max_reproj_error",
            "tri_complete_max_reproj_error",
            "tri_min_angle",
        )
    } == {
        "filter_max_reproj_error": 4.0,
        "filter_min_tri_angle": 1.5,
        "tri_create_max_angle_error": 2.0,
        "tri_continue_max_angle_error": 2.0,
        "tri_merge_max_reproj_error": 4.0,
        "tri_complete_max_reproj_error": 4.0,
        "tri_min_angle": 1.5,
    }

    gpu_params = Reconstruct().normalize_params({"ba_use_gpu": True})
    assert gpu_params["ba_use_gpu"] is True
    assert gpu_params["global_positioning_use_gpu"] is True


def test_reconstruction_rejects_invalid_triangulation_values():
    with pytest.raises(ValueError, match="triangulation parameters"):
        Reconstruct().normalize_params({"tri_min_angle": -1})
    with pytest.raises(ValueError, match="triangulation parameters"):
        Reconstruct().normalize_params({"filter_max_reproj_error": float("nan")})


def test_incremental_does_not_run_view_graph_calibration_by_default():
    params = Reconstruct().normalize_params({"mapper": "incremental"})
    assert params["view_graph_calibration"] is False


def test_incremental_defers_all_point_colors_to_final_parallel_extraction():
    args = _incremental_args(Reconstruct().normalize_params({"mapper": "incremental"}))

    assert args[args.index("--Mapper.extract_colors") + 1] == "0"


def test_incremental_args_use_colmap_video_global_ba_schedule():
    args = _incremental_args(Reconstruct().normalize_params({}), video_data=True)

    assert args[-4:] == [
        "--Mapper.ba_global_frames_ratio",
        "1.4",
        "--Mapper.ba_global_points_ratio",
        "1.4",
    ]


def test_incremental_lets_colmap_select_default_initialization_pair(tmp_path: Path):
    import sqlite3

    from sphere_reconstruct.colmap.database import pair_id_from_image_ids

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?)",
            [
                (1, "primary/lens0_000.jpg"),
                (2, "primary/lens1_000.jpg"),
                (3, "primary/lens0_001.jpg"),
                (4, "supplemental/frame_000.jpg"),
            ],
        )
        connection.executemany(
            "INSERT INTO two_view_geometries VALUES (?, ?)",
            [
                (pair_id_from_image_ids(1, 2), 1000),
                (pair_id_from_image_ids(1, 3), 120),
                (pair_id_from_image_ids(1, 4), 2000),
            ],
        )

    spec = InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=4,
        source_count=2,
        primary_source_id="primary",
        primary_image_names=[],
        sources=[],
        images=[
            {
                "name": "primary/lens0_000.jpg",
                "source_id": "primary",
                "sensor_id": "lens0",
                "capture_index": 0,
            },
            {
                "name": "primary/lens1_000.jpg",
                "source_id": "primary",
                "sensor_id": "lens1",
                "capture_index": 0,
            },
            {
                "name": "primary/lens0_001.jpg",
                "source_id": "primary",
                "sensor_id": "lens0",
                "capture_index": 1,
            },
            {
                "name": "supplemental/frame_000.jpg",
                "source_id": "supplemental",
                "sensor_id": "camera",
                "capture_index": 0,
            },
        ],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path="rig_config.json",
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )

    initialization = _requested_primary_initialization(
        database, spec, Reconstruct().normalize_params({})
    )

    assert initialization is None
    args = _incremental_args(Reconstruct().normalize_params({}), initialization)
    assert "--Mapper.init_image_id1" not in args

    initialization = _requested_primary_initialization(
        database,
        spec,
        Reconstruct().normalize_params({"init_image_id1": 1, "init_image_id2": 3}),
    )

    assert initialization == {
        "mode": "user_selected_primary_pair",
        "image_ids": [1, 3],
        "inliers": None,
    }
    args = _incremental_args(Reconstruct().normalize_params({}), initialization)
    assert args[args.index("--Mapper.init_image_id1") : args.index("--Mapper.init_image_id1") + 4] == [
        "--Mapper.init_image_id1",
        "1",
        "--Mapper.init_image_id2",
        "3",
    ]
    assert args[-2:] == ["--Mapper.init_num_trials", "1"]


def test_incremental_initialization_rejects_supplemental_pair(tmp_path: Path):
    import sqlite3

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER);"
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?)", [(1, "primary.jpg"), (2, "supplemental.jpg")],
        )

    spec = InputSpec(
        version=3,
        reconstruction_mode="perspective",
        image_count=2,
        source_count=2,
        primary_source_id="primary",
        primary_image_names=[],
        sources=[],
        images=[
            {"name": "primary.jpg", "source_id": "primary"},
            {"name": "supplemental.jpg", "source_id": "supplemental"},
        ],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path=None,
        refine_intrinsics=False,
        refine_rig=False,
        multiple_models=False,
    )

    with pytest.raises(ValueError, match="supplemental sources"):
        _requested_primary_initialization(
            database,
            spec,
            Reconstruct().normalize_params({"init_image_id1": 1, "init_image_id2": 2}),
        )


def test_final_point_color_extraction_replaces_only_the_temporary_model(tmp_path: Path, monkeypatch):
    from sphere_reconstruct.stages import reconstruct as module

    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    (model_dir / "original.txt").write_text("original", encoding="utf-8")
    (model_dir / "points3D.bin").write_text("uncolored", encoding="utf-8")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    before = {
        "num_cameras": 2,
        "num_images": 4,
        "num_points3D": 3,
        "num_observations": 8,
        "mean_reprojection_error": 1.0,
        "median_reprojection_error": 0.9,
        "p95_reprojection_error": 1.8,
        "mean_track_length": 8 / 3,
        "median_track_length": 2.0,
        "camera_trajectory_diameter": 4.0,
        "exact_black_points": 2,
    }
    after = {**before, "exact_black_points": 0, "exact_black_point_ratio": 0.0}
    calls = {}

    def fake_color_extractor(_binary, *, output_path, **kwargs):
        calls.update(kwargs)
        output_path.mkdir()
        (output_path / "points3D.bin").write_text("colored", encoding="utf-8")

    monkeypatch.setattr(module.colmap_runner, "color_extractor", fake_color_extractor)
    monkeypatch.setattr(
        module.colmap_model,
        "read_model",
        lambda path: SimpleNamespace(summary=lambda: after),
    )
    context = SimpleNamespace(progress=ProgressReporter(lambda *_args: None, tick_min_interval=0))

    summary, result = _complete_incremental_point_colors(
        context,
        colmap_bin="colmap",
        image_path=tmp_path / "images",
        model_dir=model_dir,
        logs_dir=logs_dir,
        before_summary=before,
    )

    assert summary == after
    assert result == {
        "applied": True,
        "exact_black_points_before": 2,
        "exact_black_points_after": 0,
        "resolved_points": 2,
    }
    assert calls["input_path"] == model_dir
    assert calls["num_threads"] == -1
    assert (model_dir / "points3D.bin").read_text(encoding="utf-8") == "colored"
    assert (model_dir / "original.txt").read_text(encoding="utf-8") == "original"
    assert not (model_dir / ".points3D-before-color-extractor.bin").exists()
    assert not (model_dir.parent / ".color-extractor").exists()


def test_continuity_ratio_must_be_disabled_explicitly_or_greater_than_one():
    assert Reconstruct().normalize_params({"max_adjacent_step_ratio": 0})["max_adjacent_step_ratio"] == 0
    with pytest.raises(ValueError, match="greater than one"):
        Reconstruct().normalize_params({"max_adjacent_step_ratio": 1})


def test_quality_gate_rejects_camera_only_reconstruction():
    with pytest.raises(RuntimeError, match="points3D=0"):
        _validate_summary(
            {"registered_ratio": 1.0, "num_points3D": 0},
            {"min_registered_ratio": 0.8, "min_points3D": 100},
        )


def test_continuity_gate_rejects_fully_registered_teleporting_trajectory():
    trajectory = {
        "available": True,
        "passed": False,
        "maximum_to_p95_ratio": 56.6,
        "outlier_threshold": 10.0,
        "largest_steps": [{"from_capture": 261, "to_capture": 262, "distance": 62.9}],
    }
    summary = {
        "registered_ratio": 1.0,
        "num_points3D": 1000,
        "primary_trajectory": trajectory,
        "source_registration": {"primary": {"total": 100, "registered": 100, "role": "primary"}},
    }
    params = {
        "min_registered_ratio": 0.8,
        "min_points3D": 100,
        "max_adjacent_step_ratio": 10.0,
    }

    with pytest.raises(RuntimeError, match="261->262"):
        _validate_summary(summary, params, require_trajectory_continuity=True)
    assert _summary_passes(summary, params, "primary", require_trajectory_continuity=True) is False


def test_global_mapper_retries_camera_only_result(tmp_path: Path, monkeypatch):
    from sphere_reconstruct.stages import reconstruct as module

    calls = []

    def fake_mapper(_binary, *, output_path, extra_args, **_kwargs):
        seed = int(extra_args[extra_args.index("--GlobalMapper.random_seed") + 1])
        calls.append(seed)
        model = output_path / "0"
        model.mkdir(parents=True)
        (model / "cameras.bin").write_bytes(b"camera")

    def fake_select(output_path, _spec, **_kwargs):
        seed = calls[-1]
        points = 0 if seed == 0 else 1000
        return output_path / "0", {
            "num_images": 10,
            "num_points3D": points,
            "mean_reprojection_error": 1.0,
            "source_registration": {
                "primary": {"total": 10, "registered": 10, "ratio": 1.0, "connected": True}
            },
            "primary_trajectory": {"available": False, "passed": True},
        }

    monkeypatch.setattr(module.colmap_runner, "global_mapper", fake_mapper)
    monkeypatch.setattr(module, "_select_largest_model", fake_select)
    spec = InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=10,
        source_count=1,
        primary_source_id="primary",
        primary_image_names=[],
        sources=[{"id": "primary", "label": "Primary", "role": "primary", "projection": "dual_fisheye"}],
        images=[],
        feature_batches=[],
        image_path="images",
        mask_path=None,
        feature_masks_enabled=False,
        rig_config_path=None,
        refine_intrinsics=True,
        refine_rig=False,
        multiple_models=False,
    )
    context = type(
        "Context",
        (),
        {
            "params": {
                **Reconstruct().normalize_params({}),
                "min_registered_ratio": 0.8,
                "min_points3D": 100,
            },
            "progress": ProgressReporter(lambda *_args: None),
        },
    )()
    sparse = tmp_path / "sparse"
    logs = tmp_path / "logs"
    sparse.mkdir()
    logs.mkdir()

    model, summary, attempts = _run_global_mapper_with_retries(
        context,
        spec,
        colmap_bin="colmap",
        database_path=tmp_path / "database.db",
        image_path=tmp_path / "images",
        sparse_dir=sparse,
        logs_dir=logs,
    )

    assert calls == [0, 1]
    assert summary["num_points3D"] == 1000
    assert [attempt["seed"] for attempt in attempts] == [0, 1]
    assert model == sparse / "0"
