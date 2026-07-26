"""lichtfeld_config: camera model × strategy × GUT 互換と cap 導出を検証する.

最重要: igs+ に gut=true を絶対に付けない (LichtFeld が起動時にハードエラーにするため).
"""

from __future__ import annotations

from sphere_reconstruct.colmap import lichtfeld_config as lc


def _profile(models, npts=200_000, reproj=0.7):
    return {
        "camera_models": models,
        "num_points3D": npts,
        "num_images": 120,
        "mean_reprojection_error": reproj,
        "scene_scale_point": 3.2,
        "scene_scale_camera": 5.0,
    }


def test_igsplus_never_enables_gut_on_fisheye():
    configs, info = lc.build_configs(_profile(["OPENCV_FISHEYE"]))
    assert info["camera_class"] == "fisheye"
    # igs+ は GUT 不可 → undistort に倒す, gut は false.
    assert configs["igsplus"]["strategy"] == "igs+"
    assert configs["igsplus"]["gut"] is False
    assert configs["igsplus"]["undistort"] is True
    # mcmc / mrnf は GUT を使う.
    assert configs["mcmc"]["gut"] is True and configs["mcmc"]["undistort"] is False
    assert configs["mrnf"]["gut"] is True


def test_pinhole_needs_neither():
    configs, info = lc.build_configs(_profile(["PINHOLE"]))
    assert info["camera_class"] == "pinhole"
    for c in configs.values():
        assert c["gut"] is False and c["undistort"] is False


def test_equirect_trains_via_gut_except_igsplus():
    configs, info = lc.build_configs(_profile(["EQUIRECTANGULAR"]))
    assert info["camera_class"] == "equirect"
    # equirect は gut=true (gsplat backend) で mrnf/mcmc は訓練可能.
    assert configs["mrnf"]["gut"] is True and configs["mrnf"]["undistort"] is False
    assert configs["mcmc"]["gut"] is True
    # igs+ は gut 不可 → equirect 学習不能なので config 自体を出さない.
    assert "igsplus" not in configs
    assert "equirect_igsplus_unsupported" in info["warnings"]


def test_cap_derivation_clamped():
    # 少点でも有効な SfM point は保持し、cap だけ floor にする。
    _, info = lc.build_configs(_profile(["PINHOLE"], npts=1000))
    assert info["max_cap"] == lc._CAP_FLOOR
    assert info["random_init"] is False
    assert "sparse_point_cloud" in info["warnings"]
    configs, _ = lc.build_configs(_profile(["PINHOLE"], npts=1000))
    assert all(config["random"] is False for config in configs.values())
    # 多点 → ceiling.
    _, info2 = lc.build_configs(_profile(["PINHOLE"], npts=10_000_000))
    assert info2["max_cap"] == lc._CAP_CEIL
    # 中間 → k*npts.
    _, info3 = lc.build_configs(_profile(["PINHOLE"], npts=200_000))
    assert info3["max_cap"] == lc._CAP_K * 200_000
    assert info3["recommended_config"] == "train_config.mrnf.json"


def test_zero_point_model_uses_random_initialization():
    configs, info = lc.build_configs(_profile(["PINHOLE"], npts=0))
    assert info["random_init"] is True
    assert all(config["random"] is True for config in configs.values())


def test_high_reproj_warns():
    _, info = lc.build_configs(_profile(["PINHOLE"], reproj=2.5))
    assert "high_reproj_error" in info["warnings"]


def test_all_presets_have_required_keys():
    required = {
        "iterations",
        "means_lr",
        "shs_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
        "lambda_dssim",
        "min_opacity",
        "refine_every",
        "start_refine",
        "stop_refine",
        "grad_threshold",
        "sh_degree",
    }
    configs, _ = lc.build_configs(_profile(["PINHOLE"]))
    for name, cfg in configs.items():
        assert required <= set(cfg), f"{name} missing required keys"
        assert cfg["strategy"] in ("mrnf", "igs+", "mcmc")


def test_binary_masks_enable_segment_mode():
    configs, _ = lc.build_configs(_profile(["OPENCV_FISHEYE"]), has_masks=True)
    for config in configs.values():
        assert config["mask_mode"] == "segment"
        assert config["invert_masks"] is False
