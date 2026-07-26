"""分割した SfM stages の parameter contract を検証する."""

import pytest

from sphere_reconstruct.stages.extract_features import ExtractFeatures
from sphere_reconstruct.stages.match_features import MatchFeatures
from sphere_reconstruct.stages.reconstruct import Reconstruct, _validate_summary


def test_feature_defaults_use_sift_for_normal_footage():
    params = ExtractFeatures().normalize_params({})
    assert params["reconstruction_mode"] == "native_fisheye"
    assert params["feature_type"] == "SIFT"
    assert params["max_image_size"] == 2048
    assert params["max_num_features"] == 8192


def test_matching_defaults_to_fast_bruteforce():
    params = MatchFeatures().normalize_params({})
    assert params["matcher_type"] == "bruteforce"
    assert params["pairing"] == "sequential"
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
