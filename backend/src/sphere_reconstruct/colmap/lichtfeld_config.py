"""LichtFeld-Studio 向け推奨学習 config の生成.

再構成プロファイル + 相机モデルから, 3 つの densification strategy (mrnf / igs+ / mcmc)
の config JSON を組み立てる. 公式 preset (eval/*.json, commit dee66c7 相当) を基底にし,
場面依存の `max_cap` と, 相机モデル互換に関わる `gut` / `undistort` / `random` だけを上書きする.
scene_scale は LichtFeld が読み込み時に自動計算するため config には出さない.

互換の要点 (LichtFeld 源码で確認):
- `gut` はレンダラ選択フラグ: false=fastgs (既定, PINHOLE 専用), true=gsplat (歪み/FISHEYE/
  EQUIRECTANGULAR をネイティブ描画). `gut=true` は strategy が igs+ のとき **ハードエラー**
  ("GUT and igs+ strategy cannot be used together") → GUT は mcmc / mrnf のみ.
- PINHOLE: gut 不要. FISHEYE/歪み pinhole: gut=true (mcmc/mrnf) か undistort=true (igs+ はこちら).
- EQUIRECTANGULAR: **gut=true 必須** (gsplat が native 描画; undistort は equirect に無効).
  従って igs+ は equirect を学習できない (gut 不可 + undistort 不可) → mrnf/mcmc か pinhole_rig を使う.
"""

from __future__ import annotations

from copy import deepcopy

# --- 公式 preset (eval/*.json) を基底テンプレートとして埋め込む ---------------------

_MRNF_PRESET = {
    "iterations": 30000, "sh_degree_interval": 1000, "means_lr": 0.000128,
    "means_lr_end": 0.00000016, "shs_lr": 0.005, "opacity_lr": 0.025, "scaling_lr": 0.020,
    "scaling_lr_end": 0.005, "rotation_lr": 0.0015, "lambda_dssim": 0.2, "min_opacity": 0.0039215689,
    "refine_every": 200, "start_refine": 500, "stop_refine": 28500, "grad_threshold": 0.003,
    "sh_degree": 3, "opacity_reg": 0.0, "scale_reg": 0.0, "init_opacity": 0.5, "init_scaling": 0.1,
    "max_cap": 1000000, "strategy": "mrnf", "eval_steps": [7000, 30000], "save_steps": [7000, 30000],
    "enable_eval": False, "enable_save_eval_images": True, "mip_filter": False,
    "use_bilateral_grid": False, "bg_modulation": False, "bilateral_grid_X": 16,
    "bilateral_grid_Y": 16, "bilateral_grid_W": 8, "bilateral_grid_lr": 0.002, "tv_loss_weight": 10.0,
    "revised_opacity": True, "steps_scaler": 0, "random": False, "init_num_pts": 100000,
    "init_extent": 3.0, "mask_mode": "none", "invert_masks": False,
    "mask_opacity_penalty_weight": 1.0, "mask_opacity_penalty_power": 2.0, "mask_threshold": 0.5,
    "growth_grad_threshold": 0.003, "grow_fraction": 0.07, "grow_until_iter": 15000,
    "opacity_decay": 0.004, "scale_decay": 0.002, "means_noise_weight": 50.0,
    "bounds_percentile": 0.8, "use_error_map": True, "use_edge_map": True,
}

_MCMC_PRESET = {
    "iterations": 30000, "sh_degree_interval": 1000, "means_lr": 0.000128, "shs_lr": 0.0024,
    "opacity_lr": 0.0335, "scaling_lr": 0.00475, "rotation_lr": 0.00083, "lambda_dssim": 0.2,
    "min_opacity": 0.005, "refine_every": 100, "start_refine": 500, "stop_refine": 25000,
    "grad_threshold": 0.0002, "sh_degree": 3, "opacity_reg": 0.0042, "scale_reg": 0.0042,
    "init_opacity": 0.5, "init_scaling": 0.1, "max_cap": 1000000, "strategy": "mcmc",
    "eval_steps": [7000, 30000], "save_steps": [7000, 30000], "enable_eval": False,
    "enable_save_eval_images": True, "mip_filter": False, "use_bilateral_grid": False,
    "bg_modulation": False, "bilateral_grid_X": 16, "bilateral_grid_Y": 16, "bilateral_grid_W": 8,
    "bilateral_grid_lr": 0.002, "tv_loss_weight": 10.0, "prune_opacity": 0.005, "grow_scale3d": 0.01,
    "grow_scale2d": 0.05, "prune_scale3d": 0.1, "prune_scale2d": 0.15, "reset_every": 3000,
    "pause_refine_after_reset": 0, "revised_opacity": False, "gut": False, "steps_scaler": 0,
    "random": False, "init_num_pts": 100000, "init_extent": 3.0, "mask_mode": "none",
    "invert_masks": False, "mask_opacity_penalty_weight": 1.0, "mask_opacity_penalty_power": 2.0,
    "mask_threshold": 0.5,
}

_IGSPLUS_PRESET = {
    "iterations": 30000, "sh_degree_interval": 1000, "means_lr": 0.000128, "shs_lr": 0.005,
    "opacity_lr": 0.025, "scaling_lr": 0.020, "rotation_lr": 0.0015, "lambda_dssim": 0.20,
    "min_opacity": 0.005, "refine_every": 500, "start_refine": 500, "stop_refine": 15000,
    "grad_threshold": 0.0002, "sh_degree": 3, "opacity_reg": 0.0, "scale_reg": 0.0,
    "init_opacity": 0.3, "init_scaling": 0.2, "max_cap": 1000000, "strategy": "igs+",
    "eval_steps": [7000, 30000], "save_steps": [7000, 30000], "enable_eval": False,
    "enable_save_eval_images": True, "use_bilateral_grid": False, "bg_modulation": False,
    "bilateral_grid_X": 16, "bilateral_grid_Y": 16, "bilateral_grid_W": 8, "bilateral_grid_lr": 0.002,
    "tv_loss_weight": 5.0, "prune_opacity": 0.005, "grow_scale3d": 0.01, "grow_scale2d": 0.05,
    "prune_scale3d": 0.1, "prune_scale2d": 0.15, "reset_every": 3000, "pause_refine_after_reset": 0,
    "revised_opacity": True, "steps_scaler": 0, "random": False, "init_num_pts": 100000,
    "init_extent": 3.0,
}

_PRESETS = {"mrnf": _MRNF_PRESET, "igsplus": _IGSPLUS_PRESET, "mcmc": _MCMC_PRESET}

# max_cap 導出: SfM 点数の倍率. VRAM/品質のダイヤルなので clamp する.
_CAP_K = 6
_CAP_FLOOR = 500_000
_CAP_CEIL = 3_000_000


def _camera_class(models: list[str]) -> str:
    if any(("EQUIRECTANGULAR" in m) or ("SPHERICAL" in m) for m in models):
        return "equirect"
    if any("FISHEYE" in m for m in models):
        return "fisheye"
    if any(m in ("PINHOLE", "SIMPLE_PINHOLE") for m in models):
        return "pinhole"
    if any(("OPENCV" in m) or ("RADIAL" in m) or ("THIN_PRISM" in m) for m in models):
        return "distorted_pinhole"  # gut/undistort が要る歪み pinhole 系.
    return "other"


def build_configs(profile: dict) -> tuple[dict[str, dict], dict]:
    """profile から 3 strategy の config と, 導出情報 (max_cap / 相机クラス / 警告) を返す."""
    models = profile.get("camera_models", [])
    cclass = _camera_class(models)
    npts = int(profile.get("num_points3D", 0))
    reproj = float(profile.get("mean_reprojection_error", 0.0))

    cap = int(min(_CAP_CEIL, max(_CAP_FLOOR, _CAP_K * npts)))
    sparse_init = npts < 10_000  # SfM が薄すぎる → random init に倒す.

    warnings: list[str] = []
    if reproj > 1.5:
        warnings.append("high_reproj_error")

    configs: dict[str, dict] = {}
    for name, preset in _PRESETS.items():
        cfg = deepcopy(preset)
        cfg["max_cap"] = cap
        if sparse_init:
            cfg["random"] = True
        strat = cfg["strategy"]  # mrnf / igs+ / mcmc

        # 相机モデルに応じた歪み処理. gut は igs+ で禁止.
        if cclass == "pinhole":
            cfg["gut"] = False
            cfg["undistort"] = False
        elif cclass in ("fisheye", "distorted_pinhole"):
            if strat == "igs+":
                cfg["gut"] = False
                cfg["undistort"] = True  # igs+ は GUT 不可 → on-the-fly 去畸变.
            else:
                cfg["gut"] = True
                cfg["undistort"] = False
        elif cclass == "equirect":
            if strat == "igs+":
                # equirect は gut 必須だが igs+ は gut 不可, かつ undistort も無効 → 学習不能.
                cfg["gut"] = False
                cfg["undistort"] = False
                if "equirect_igsplus_unsupported" not in warnings:
                    warnings.append("equirect_igsplus_unsupported")
            else:
                cfg["gut"] = True  # gsplat backend が equirect をネイティブ描画.
                cfg["undistort"] = False
        configs[name] = cfg

    info = {
        "camera_class": cclass,
        "camera_models": models,
        "max_cap": cap,
        "sparse_init": sparse_init,
        "num_points3D": npts,
        "num_images": int(profile.get("num_images", 0)),
        "mean_reprojection_error": round(reproj, 4),
        "scene_scale_point": profile.get("scene_scale_point"),
        "scene_scale_camera": profile.get("scene_scale_camera"),
        "warnings": warnings,
        "usage": "LichtFeld-Studio --config train_config.<strategy>.json --data-path dataset/",
    }
    return configs, info
