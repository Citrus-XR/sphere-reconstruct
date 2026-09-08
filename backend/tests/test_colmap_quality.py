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


def test_incremental_triangulation_options_use_mapper_namespace():
    args = quality.mapper_extra_args(
        {
            "filter_max_reproj_error": 1.5,
            "filter_min_tri_angle": 3.0,
            "tri_create_max_angle_error": 1.0,
            "tri_continue_max_angle_error": 1.0,
            "tri_merge_max_reproj_error": 1.5,
            "tri_complete_max_reproj_error": 1.5,
            "tri_min_angle": 3.0,
        }
    )
    assert args == [
        "--Mapper.filter_max_reproj_error", "1.5",
        "--Mapper.filter_min_tri_angle", "3",
        "--Mapper.tri_create_max_angle_error", "1",
        "--Mapper.tri_continue_max_angle_error", "1",
        "--Mapper.tri_merge_max_reproj_error", "1.5",
        "--Mapper.tri_complete_max_reproj_error", "1.5",
        "--Mapper.tri_min_angle", "3",
    ]


def test_global_mapper_does_not_receive_incremental_triangulation_options():
    args = quality.global_mapper_extra_args(
        {
            "random_seed": 0,
            "filter_max_reproj_error": 1.0,
            "filter_min_tri_angle": 5.0,
            "tri_create_max_angle_error": 0.75,
            "tri_continue_max_angle_error": 0.75,
            "tri_merge_max_reproj_error": 1.0,
            "tri_complete_max_reproj_error": 1.0,
            "tri_min_angle": 5.0,
        }
    )
    assert args == ["--GlobalMapper.random_seed", "0"]
