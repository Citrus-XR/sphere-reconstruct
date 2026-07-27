"""分割した SfM stages の parameter contract を検証する."""

from pathlib import Path

import pytest

from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.pipeline.stage import ProgressReporter
from sphere_reconstruct.stages.extract_features import ExtractFeatures
from sphere_reconstruct.stages.match_features import MatchFeatures
from sphere_reconstruct.stages.reconstruct import (
    Reconstruct,
    _run_global_mapper_with_retries,
    _validate_summary,
)


def test_feature_defaults_use_sift_for_normal_footage():
    params = ExtractFeatures().normalize_params({})
    assert params["reconstruction_mode"] == "native_fisheye"
    assert params["feature_type"] == "SIFT"
    assert params["max_image_size"] == 2048
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


def test_reconstruction_defaults_to_global_cpu_when_cudss_is_unknown():
    params = Reconstruct().normalize_params({})
    assert params["mapper"] == "global"
    assert params["view_graph_calibration"] is True
    assert params["ba_use_gpu"] is False
    assert params["random_seed"] == 0
    assert params["min_registered_ratio"] == 0.8
    assert params["min_points3D"] == 100


def test_incremental_does_not_run_view_graph_calibration_by_default():
    params = Reconstruct().normalize_params({"mapper": "incremental"})
    assert params["view_graph_calibration"] is False


def test_quality_gate_rejects_camera_only_reconstruction():
    with pytest.raises(RuntimeError, match="points3D=0"):
        _validate_summary(
            {"registered_ratio": 1.0, "num_points3D": 0},
            {"min_registered_ratio": 0.8, "min_points3D": 100},
        )


def test_global_mapper_retries_camera_only_result(tmp_path: Path, monkeypatch):
    from sphere_reconstruct.stages import reconstruct as module

    calls = []

    def fake_mapper(_binary, *, output_path, extra_args, **_kwargs):
        seed = int(extra_args[extra_args.index("--GlobalMapper.random_seed") + 1])
        calls.append(seed)
        model = output_path / "0"
        model.mkdir(parents=True)
        (model / "cameras.bin").write_bytes(b"camera")

    def fake_select(output_path, _spec):
        seed = calls[-1]
        points = 0 if seed == 0 else 1000
        return output_path / "0", {
            "num_images": 10,
            "num_points3D": points,
            "mean_reprojection_error": 1.0,
            "source_registration": {
                "primary": {"total": 10, "registered": 10, "ratio": 1.0, "connected": True}
            },
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
