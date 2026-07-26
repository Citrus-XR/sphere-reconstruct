"""COLMAP 4.1 の CLI option namespace へ正しく変換されることを検証する."""

from sphere_reconstruct.colmap import quality


def test_feature_image_size_uses_feature_extraction_namespace():
    args = quality.feature_extra_args(
        {"sift_max_num_features": 8192, "sift_max_image_size": 2048},
        "sift",
    )
    assert "--FeatureExtraction.max_image_size" in args
    assert "--SiftExtraction.max_image_size" not in args
    assert "--SiftExtraction.max_num_features" in args


def test_common_matching_options_use_feature_matching_namespace():
    args = quality.matcher_extra_args(
        {"max_num_matches": 16384, "guided_matching": True},
        "sift",
    )
    assert args == [
        "--FeatureMatching.max_num_matches",
        "16384",
        "--FeatureMatching.guided_matching",
        "1",
    ]
