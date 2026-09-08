"""幾何検証済み database から sparse reconstruction を解くステージ.

特徴抽出と matching は独立ステージで完了しているため, Mapper・BA 設定だけを変更して
再実行できる. Incremental COLMAP と COLMAP 4.1 内蔵 Global Mapper を同じ契約で扱う.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
from pathlib import Path

from ..colmap import model as colmap_model
from ..colmap import quality as colmap_quality
from ..colmap import runner as colmap_runner
from ..colmap import trajectory_quality
from ..colmap.solver_diagnostics import read_solver_diagnostics
from ..colmap.input_workspace import InputSpec
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings
from . import similarity_transform
from .colmap_progress import global_mapper_progress, hidden_log, mapper_progress


@register
class Reconstruct(Stage):
    name = StageName.RECONSTRUCT
    impl_version = "2.9"

    def normalize_params(self, raw: dict) -> dict:
        mapper = str(raw.get("mapper", "incremental")).lower()
        if mapper not in {"incremental", "global"}:
            raise ValueError(f"unsupported mapper: {mapper}")
        ba_use_gpu = bool(raw.get("ba_use_gpu", False))
        max_adjacent_step_ratio = float(raw.get("max_adjacent_step_ratio", 10.0))
        if max_adjacent_step_ratio != 0.0 and max_adjacent_step_ratio <= 1.0:
            raise ValueError("max_adjacent_step_ratio must be zero or greater than one")
        triangulation_defaults = {
            "filter_max_reproj_error": 4.0,
            "filter_min_tri_angle": 1.5,
            "tri_create_max_angle_error": 2.0,
            "tri_continue_max_angle_error": 2.0,
            "tri_merge_max_reproj_error": 4.0,
            "tri_complete_max_reproj_error": 4.0,
            "tri_min_angle": 1.5,
        }
        triangulation_params = {
            key: float(raw.get(key, default))
            for key, default in triangulation_defaults.items()
        }
        invalid_triangulation = [
            key
            for key, value in triangulation_params.items()
            if not math.isfinite(value) or value < 0
        ]
        if invalid_triangulation:
            raise ValueError(
                "triangulation parameters must be finite and non-negative: "
                + ", ".join(invalid_triangulation)
            )
        return {
            "mapper": mapper,
            "view_graph_calibration": bool(raw.get("view_graph_calibration", mapper == "global")),
            "ba_use_gpu": ba_use_gpu,
            "global_positioning_use_gpu": bool(raw.get("global_positioning_use_gpu", ba_use_gpu)),
            "random_seed": int(raw.get("random_seed", 0)),
            "mapper_min_num_matches": int(raw.get("mapper_min_num_matches", 0)),
            "init_min_num_inliers": int(raw.get("init_min_num_inliers", 0)),
            "init_image_id1": int(raw.get("init_image_id1", 0)),
            "init_image_id2": int(raw.get("init_image_id2", 0)),
            "abs_pose_max_error": float(raw.get("abs_pose_max_error", 0.0)),
            **triangulation_params,
            "ba_local_max_num_iterations": int(raw.get("ba_local_max_num_iterations", 0)),
            "ba_global_max_num_iterations": int(raw.get("ba_global_max_num_iterations", 0)),
            "min_model_size": int(raw.get("min_model_size", 0)),
            "min_registered_ratio": float(raw.get("min_registered_ratio", 0.8)),
            "min_points3D": int(raw.get("min_points3D", 100)),
            "max_adjacent_step_ratio": max_adjacent_step_ratio,
            "incremental_fallback": bool(raw.get("incremental_fallback", True)),
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "match_features.json",
            ctx.project_dir / "match_features" / "database.db",
            ctx.project_dir / "extract_features" / "input_spec.json",
        ]
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.exists()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        source_database = ctx.project_dir / "match_features" / "database.db"
        spec_path = ctx.project_dir / "extract_features" / "input_spec.json"
        if not source_database.exists() or not spec_path.exists():
            raise RuntimeError("extract_features and match_features must run before reconstruction")
        spec = InputSpec.read(spec_path)
        ctx.progress.info("copying matched database", progress=0.0, key="log.recon_copy_database")
        database_path = ctx.stage_out_dir / "database.db"
        shutil.copy2(source_database, database_path)
        ctx.progress.tick(0.03, message="matched database copied", key="log.recon_database_ready")
        sparse_dir = ctx.stage_out_dir / "sparse"
        logs_dir = ctx.stage_out_dir / "logs"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        settings = get_settings()
        colmap_bin = colmap_runner.resolve_colmap_bin(settings.binaries.colmap or None)
        image_path = ctx.project_dir / "extract_features" / spec.image_path
        mapper = ctx.params["mapper"]
        effective_mapper = mapper
        video_data = any(source.media_kind == MediaKind.VIDEO for source in ctx.sources)
        require_trajectory_continuity = bool(
            ctx.primary_source is not None and ctx.primary_source.media_kind == MediaKind.VIDEO
        )
        primary_initialization: dict | None = None
        apply_view_graph_calibration = bool(
            mapper == "global" and ctx.params["view_graph_calibration"] and spec.refine_intrinsics
        )
        ctx.progress.info(
            f"sparse reconstruction: mapper={mapper}, images={spec.image_count}, frames={spec.frame_count}",
            progress=0.05,
            key="log.recon_start",
            args={"mapper": mapper, "images": spec.image_count, "frames": spec.frame_count},
        )

        if mapper == "global":
            if apply_view_graph_calibration:
                ctx.progress.info(
                    "calibrating view graph",
                    progress=0.1,
                    key="log.recon_view_graph_calibration",
                )
                colmap_runner.view_graph_calibrator(
                    colmap_bin,
                    database_path=database_path,
                    log_path=logs_dir / "view_graph_calibrator.log",
                    on_line=hidden_log(ctx, "view-graph"),
                )
                ctx.progress.tick(
                    0.12,
                    message="view graph calibration complete",
                    key="log.recon_view_graph_done",
                )
            global_error: RuntimeError | None = None
            try:
                model_dir, summary, mapper_attempts = _run_global_mapper_with_retries(
                    ctx,
                    spec,
                    colmap_bin=colmap_bin,
                    database_path=database_path,
                    image_path=image_path,
                    sparse_dir=sparse_dir,
                    logs_dir=logs_dir,
                    require_trajectory_continuity=require_trajectory_continuity,
                )
            except RuntimeError as error:
                global_error = error
                model_dir, summary = None, None
                mapper_attempts = [{"mapper": "global", "error": str(error)}]
            # Full registration でも Global pose が大きく破綻する official benchmark があるため、
            # continuity gate 後は最も robust な Incremental を original DB から試す。
            # https://github.com/colmap/colmap/pull/4590
            # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/doc/faq.rst#L94-L114
            if ctx.params["incremental_fallback"] and (
                summary is None
                or not _summary_passes(
                    summary,
                    ctx.params,
                    spec.primary_source_id,
                    require_trajectory_continuity=require_trajectory_continuity,
                )
            ):
                ctx.progress.warn(
                    "Global Mapper failed quality gates; trying Incremental Mapper from the original database",
                    key="log.recon_incremental_fallback",
                )
                fallback_database = ctx.stage_out_dir / "incremental_fallback.db"
                shutil.copy2(source_database, fallback_database)
                fallback_sparse = ctx.stage_out_dir / "incremental_fallback_sparse"
                fallback_sparse.mkdir()
                fallback_initialization = _requested_primary_initialization(
                    fallback_database, spec, ctx.params
                )
                colmap_runner.mapper(
                    colmap_bin,
                    database_path=fallback_database,
                    image_path=image_path,
                    output_path=fallback_sparse,
                    refine_intrinsics=spec.refine_intrinsics,
                    refine_rig=spec.refine_rig,
                    multiple_models=spec.multiple_models,
                    extra_args=_incremental_args(
                        ctx.params,
                        fallback_initialization,
                        video_data=video_data,
                    ),
                    log_path=logs_dir / "incremental_fallback.log",
                    on_line=mapper_progress(ctx, spec.frame_count, low=0.76, high=0.9),
                )
                fallback_model, fallback_summary = _select_largest_model(
                    fallback_sparse,
                    spec,
                    max_step_ratio=_evaluation_step_ratio(ctx.params),
                )
                fallback_passed = _summary_passes(
                    fallback_summary,
                    ctx.params,
                    spec.primary_source_id,
                    require_trajectory_continuity=require_trajectory_continuity,
                )
                mapper_attempts.append(
                    {
                        "mapper": "incremental_fallback",
                        "registered_primary": fallback_summary["source_registration"][spec.primary_source_id][
                            "registered"
                        ],
                        "registered_total": fallback_summary["num_images"],
                        "points3D": fallback_summary["num_points3D"],
                        "mean_reprojection_error": fallback_summary["mean_reprojection_error"],
                        "trajectory_continuity_passed": fallback_summary["primary_trajectory"]["passed"],
                        "quality_gates_passed": fallback_passed,
                    }
                )
                if fallback_passed:
                    canonical = sparse_dir / "0"
                    if canonical.exists():
                        shutil.rmtree(canonical)
                    shutil.copytree(fallback_model, canonical)
                    shutil.copy2(fallback_database, database_path)
                    model_dir, summary = canonical, fallback_summary
                    effective_mapper = "incremental_fallback"
                    primary_initialization = fallback_initialization
                    shutil.rmtree(fallback_sparse)
                    fallback_database.unlink()
            if summary is None or model_dir is None:
                raise global_error or RuntimeError("Global and Incremental Mapper produced no model")
        else:
            primary_initialization = _requested_primary_initialization(database_path, spec, ctx.params)
            colmap_runner.mapper(
                colmap_bin,
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_dir,
                refine_intrinsics=spec.refine_intrinsics,
                refine_rig=spec.refine_rig,
                multiple_models=spec.multiple_models,
                extra_args=_incremental_args(
                    ctx.params,
                    primary_initialization,
                    video_data=video_data,
                ),
                log_path=logs_dir / "mapper.log",
                on_line=mapper_progress(ctx, spec.frame_count, low=0.12, high=0.9),
            )
            model_dir, summary = _select_largest_model(
                sparse_dir,
                spec,
                max_step_ratio=_evaluation_step_ratio(ctx.params),
            )
            mapper_attempts = []
        if effective_mapper in {"incremental", "incremental_fallback"}:
            color_summary, color_completion = _complete_incremental_point_colors(
                ctx,
                colmap_bin=colmap_bin,
                image_path=image_path,
                model_dir=model_dir,
                logs_dir=logs_dir,
                before_summary=summary,
            )
            summary.update(color_summary)
        else:
            color_completion = {
                "applied": False,
                "reason": "global_mapper_extracts_final_colors",
                "exact_black_points_before": summary["exact_black_points"],
                "exact_black_points_after": summary["exact_black_points"],
                "resolved_points": 0,
            }
        summary["point_color_completion"] = color_completion
        if primary_initialization is not None:
            summary["primary_initialization"] = primary_initialization
        ctx.progress.tick(0.92, message="mapper complete", key="log.recon_mapper_done")
        primary = summary["source_registration"][spec.primary_source_id]
        summary["registered_ratio"] = primary["registered"] / max(1, primary["total"])
        summary["registered_total_ratio"] = summary["num_images"] / max(1, spec.image_count)
        summary["mapper"] = effective_mapper
        summary["requested_mapper"] = mapper
        summary["input_images"] = spec.image_count
        summary["view_graph_calibration"] = bool(
            apply_view_graph_calibration and effective_mapper == "global"
        )
        summary["view_graph_calibration_requested"] = bool(ctx.params["view_graph_calibration"])
        summary["ba_gpu_enabled"] = ctx.params["ba_use_gpu"]
        summary["global_positioning_gpu_requested"] = bool(
            mapper == "global" and ctx.params["global_positioning_use_gpu"]
        )
        summary["global_positioning_gpu_used"] = bool(
            effective_mapper == "global" and ctx.params["global_positioning_use_gpu"]
        )
        summary["mapper_attempts"] = mapper_attempts
        summary["solver_diagnostics"] = read_solver_diagnostics(logs_dir)
        failed_steps = summary["solver_diagnostics"]["linear_solver_failed_steps"]
        if failed_steps or summary["solver_diagnostics"]["bundle_adjustment_failures"]:
            ctx.progress.warn(
                f"COLMAP returned a model after {failed_steps} failed linear-solver steps; inspect the solver diagnostics",
                key="log.recon_solver_diagnostics",
                args={"count": failed_steps, "terminated": summary["solver_diagnostics"]["bundle_adjustment_failures"]},
            )
        _validate_summary(
            summary,
            ctx.params,
            require_trajectory_continuity=require_trajectory_continuity,
        )
        ctx.progress.info(
            "building sparse reconstruction preview",
            progress=0.94,
            key="log.recon_preview",
        )
        preview = similarity_transform.write_preview(
            ctx,
            colmap_model.read_model(model_dir),
            metadata_key="reconstruction",
            metadata=summary,
            max_points=ctx.params["max_preview_points"],
        )
        summary["preview_points"] = preview.num_points_written
        summary_path = ctx.stage_out_dir / "model_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs = [_file_ref(summary_path, ctx), _file_ref(database_path, ctx)]
        for path in model_dir.iterdir():
            if path.is_file():
                outputs.append(_file_ref(path, ctx))
        outputs.extend(
            _file_ref(ctx.stage_out_dir / "preview" / name, ctx)
            for name in ("reconstruction.json", "points.bin")
        )
        manifest.outputs = outputs
        manifest.extra = summary
        ctx.progress.info(
            f"reconstruction done: {summary['num_images']}/{spec.image_count} images, "
            f"{summary['num_points3D']} points, {summary['mean_reprojection_error']:.3f}px",
            progress=0.99,
            key="log.recon_done",
            args={
                "images": summary["num_images"],
                "total": spec.image_count,
                "points": summary["num_points3D"],
                "error": round(summary["mean_reprojection_error"], 3),
            },
        )
        return manifest


def _requested_primary_initialization(
    database_path: Path, spec: InputSpec, params: dict
) -> dict | None:
    """Validate an optional user-selected primary-source initialization pair.

    Capture order measures time, not baseline or parallax. In the default path
    COLMAP therefore chooses its own geometrically verified initial pair.
    """
    requested = (int(params["init_image_id1"]), int(params["init_image_id2"]))
    if (requested[0] > 0) != (requested[1] > 0):
        raise ValueError("init_image_id1 and init_image_id2 must be set together")
    if requested[0] == 0:
        return None

    images_by_name = {str(image["name"]): image for image in spec.images}
    with sqlite3.connect(database_path) as connection:
        id_to_image = {
            int(image_id): images_by_name[name]
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
            if name in images_by_name
        }

        primary_ids = {
            image_id
            for image_id, image in id_to_image.items()
            if image["source_id"] == spec.primary_source_id
        }
    if requested[0] not in primary_ids or requested[1] not in primary_ids:
        raise ValueError(
            "initial image IDs must both belong to the primary source; "
            "supplemental sources cannot establish the reconstruction trajectory"
        )
    return {
        "mode": "user_selected_primary_pair",
        "image_ids": list(requested),
        "inliers": None,
    }


def _incremental_args(
    params: dict,
    initialization: dict | None = None,
    *,
    video_data: bool = False,
) -> list[str]:
    args = [
        "--Mapper.random_seed",
        str(params["random_seed"]),
        "--Mapper.extract_colors",
        "0",
        *colmap_quality.mapper_extra_args(params),
    ]
    if initialization is not None:
        first_id, second_id = initialization["image_ids"]
        args += ["--Mapper.init_image_id1", str(first_id)]
        args += ["--Mapper.init_image_id2", str(second_id)]
        args += ["--Mapper.init_num_trials", "1"]
    if video_data:
        # Match COLMAP's automatic video schedule:
        # https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/controllers/option_manager.cc#L100-L105
        args += [
            "--Mapper.ba_global_frames_ratio",
            "1.4",
            "--Mapper.ba_global_points_ratio",
            "1.4",
        ]
    return args


def _complete_incremental_point_colors(
    ctx: StageContext,
    *,
    colmap_bin: str,
    image_path: Path,
    model_dir: Path,
    logs_dir: Path,
    before_summary: dict,
) -> tuple[dict, dict]:
    ctx.progress.info(
        "extracting final colors for all sparse points",
        progress=0.9,
        key="log.recon_color_start",
        args={"points": before_summary["num_points3D"]},
    )
    colored_dir = model_dir.parent / ".color-extractor"
    colmap_runner.color_extractor(
        colmap_bin,
        input_path=model_dir,
        image_path=image_path,
        output_path=colored_dir,
        num_threads=-1,
        log_path=logs_dir / "color_extractor.log",
        on_line=hidden_log(ctx, "color-extractor"),
    )
    colored_reconstruction = colmap_model.read_model(colored_dir)
    after_summary = colored_reconstruction.summary()
    _validate_color_extraction_geometry(before_summary, after_summary)

    # color_extractor は model 全体を書き直すが、LFStudio は images.bin の格納順へ test_every を
    # 適用する。Camera / validation ordering を変えず RGB だけ補完するため points3D.bin だけを採用する。
    # https://github.com/MrNeRF/LichtFeld-Studio/blob/d8c50c6a3e2273cb74130a6e9023de8d068af52d/src/training/training_setup.cpp#L493-L500
    points_path = model_dir / "points3D.bin"
    backup_points_path = model_dir / ".points3D-before-color-extractor.bin"
    points_path.rename(backup_points_path)
    (colored_dir / "points3D.bin").replace(points_path)
    shutil.rmtree(colored_dir)
    backup_points_path.unlink()

    before_black = int(before_summary["exact_black_points"])
    after_black = int(after_summary["exact_black_points"])
    result = {
        "applied": True,
        "exact_black_points_before": before_black,
        "exact_black_points_after": after_black,
        "resolved_points": max(0, before_black - after_black),
    }
    ctx.progress.tick(
        0.915,
        message=f"final point colors complete: black {before_black}->{after_black}",
        key="log.recon_color_done",
        args={"before": before_black, "after": after_black},
    )
    return after_summary, result


def _validate_color_extraction_geometry(before: dict, after: dict) -> None:
    for key in ("num_cameras", "num_images", "num_points3D", "num_observations"):
        if before[key] != after[key]:
            raise RuntimeError(
                f"color extraction changed reconstruction geometry: {key} {before[key]} -> {after[key]}"
            )
    for key in (
        "mean_reprojection_error",
        "median_reprojection_error",
        "p95_reprojection_error",
        "mean_track_length",
        "median_track_length",
        "camera_trajectory_diameter",
    ):
        if (
            key in before
            and key in after
            and not math.isclose(float(before[key]), float(after[key]), rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RuntimeError(
                f"color extraction changed reconstruction geometry: {key} {before[key]} -> {after[key]}"
            )


def _run_global_mapper_with_retries(
    ctx: StageContext,
    spec: InputSpec,
    *,
    colmap_bin: str,
    database_path: Path,
    image_path: Path,
    sparse_dir: Path,
    logs_dir: Path,
    require_trajectory_continuity: bool = False,
) -> tuple[Path, dict, list[dict]]:
    initial_seed = int(ctx.params["random_seed"])
    seeds = list(dict.fromkeys((initial_seed, initial_seed + 1, initial_seed + 2)))
    candidates: list[tuple[Path, dict]] = []
    attempts = []
    attempt_boundaries = (0.12, 0.46, 0.60, 0.72)
    for attempt_index, seed in enumerate(seeds):
        attempt_dir = sparse_dir / f"attempt_{attempt_index:02d}_seed_{seed}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_params = {**ctx.params, "random_seed": seed}
        attempt_low = attempt_boundaries[attempt_index]
        attempt_high = attempt_boundaries[attempt_index + 1]
        ctx.progress.info(
            f"Global Mapper attempt {attempt_index + 1}/{len(seeds)} seed={seed}",
            progress=attempt_low,
            key="log.recon_attempt",
            args={"cur": attempt_index + 1, "tot": len(seeds), "seed": seed},
        )
        try:
            colmap_runner.global_mapper(
                colmap_bin,
                database_path=database_path,
                image_path=image_path,
                output_path=attempt_dir,
                refine_intrinsics=spec.refine_intrinsics,
                refine_rig=spec.refine_rig,
                ba_use_gpu=ctx.params["ba_use_gpu"],
                global_positioning_use_gpu=ctx.params["global_positioning_use_gpu"],
                extra_args=colmap_quality.global_mapper_extra_args(attempt_params),
                log_path=logs_dir / f"global_mapper_seed_{seed}.log",
                on_line=global_mapper_progress(ctx, low=attempt_low, high=attempt_high),
            )
            ctx.progress.tick(
                attempt_high,
                message=f"Global Mapper seed={seed} complete",
                key="log.recon_attempt_done",
                args={"seed": seed},
            )
            model_dir, summary = _select_largest_model(
                attempt_dir,
                spec,
                max_step_ratio=_evaluation_step_ratio(ctx.params),
            )
            primary = summary["source_registration"][spec.primary_source_id]
            registered_ratio = primary["registered"] / max(1, primary["total"])
            trajectory = summary["primary_trajectory"]
            trajectory_ok = _trajectory_passes(
                trajectory,
                ctx.params,
                require_trajectory_continuity=require_trajectory_continuity,
            )
            attempt = {
                "seed": seed,
                "registered_primary": primary["registered"],
                "registered_total": summary["num_images"],
                "points3D": summary["num_points3D"],
                "mean_reprojection_error": summary["mean_reprojection_error"],
                "trajectory_continuity_passed": trajectory_ok,
                "trajectory_maximum_to_p95_ratio": trajectory.get("maximum_to_p95_ratio"),
                "trajectory_outlier_steps": trajectory.get("outlier_steps", 0),
            }
            attempts.append(attempt)
            candidates.append((model_dir, summary))
            if (
                registered_ratio >= ctx.params["min_registered_ratio"]
                and summary["num_points3D"] >= ctx.params["min_points3D"]
                and trajectory_ok
            ):
                break
            ctx.progress.warn(
                f"Global Mapper seed={seed} は quality gate 未達。別 seed で再試行します",
                key="log.recon_retry_seed",
                args={
                    "seed": seed,
                    "images": primary["registered"],
                    "points": summary["num_points3D"],
                },
            )
        except RuntimeError as error:
            attempts.append({"seed": seed, "error": str(error)})
            ctx.progress.warn(
                f"Global Mapper seed={seed} failed: {error}",
                key="log.recon_retry_error",
                args={"seed": seed, "error": str(error)},
            )
    if not candidates:
        raise RuntimeError(f"Global Mapper produced no model after seeds {seeds}")
    best_model, best_summary = max(
        candidates,
        key=lambda candidate: (
            (
                candidate[1]["source_registration"][spec.primary_source_id]["registered"]
                / max(1, candidate[1]["source_registration"][spec.primary_source_id]["total"])
                >= ctx.params["min_registered_ratio"]
                and candidate[1]["num_points3D"] >= ctx.params["min_points3D"]
                and _trajectory_passes(
                    candidate[1]["primary_trajectory"],
                    ctx.params,
                    require_trajectory_continuity=require_trajectory_continuity,
                )
            ),
            candidate[1]["source_registration"][spec.primary_source_id]["registered"],
            candidate[1]["num_points3D"],
            candidate[1]["num_images"],
        ),
    )
    canonical = sparse_dir / "0"
    shutil.copytree(best_model, canonical)
    for path in sparse_dir.iterdir():
        if path != canonical and path.is_dir():
            shutil.rmtree(path)
    return canonical, best_summary, attempts


def _validate_summary(
    summary: dict,
    params: dict,
    *,
    require_trajectory_continuity: bool = False,
) -> None:
    if summary["registered_ratio"] < params["min_registered_ratio"]:
        raise RuntimeError(
            f"reconstruction quality gate failed: registered_ratio="
            f"{summary['registered_ratio']:.3f} < {params['min_registered_ratio']:.3f}"
        )
    if summary["num_points3D"] < params["min_points3D"]:
        raise RuntimeError(
            f"reconstruction quality gate failed: points3D={summary['num_points3D']} "
            f"< {params['min_points3D']}"
        )
    if require_trajectory_continuity and params["max_adjacent_step_ratio"] > 0:
        trajectory = summary["primary_trajectory"]
        if trajectory["passed"]:
            return
        if not trajectory["available"]:
            raise RuntimeError(
                "reconstruction trajectory continuity gate unavailable: "
                f"{trajectory['reason']}; registered_captures="
                f"{trajectory['registered_captures']}/{trajectory['expected_captures']}"
            )
        cuts = ", ".join(
            f"{step['from_capture']}->{step['to_capture']} ({step['distance']:.3g})"
            for step in trajectory["largest_steps"]
            if step["distance"] > trajectory["outlier_threshold"]
        )
        raise RuntimeError(
            "reconstruction trajectory continuity gate failed: "
            f"max/p95={trajectory['maximum_to_p95_ratio']:.2f} > "
            f"{params['max_adjacent_step_ratio']:.2f}; jumps={cuts}"
        )


def _summary_passes(
    summary: dict,
    params: dict,
    primary_source_id: str,
    *,
    require_trajectory_continuity: bool,
) -> bool:
    primary = summary["source_registration"][primary_source_id]
    if primary["registered"] / max(1, primary["total"]) < params["min_registered_ratio"]:
        return False
    if summary["num_points3D"] < params["min_points3D"]:
        return False
    trajectory = summary["primary_trajectory"]
    return _trajectory_passes(
        trajectory,
        params,
        require_trajectory_continuity=require_trajectory_continuity,
    )


def _trajectory_passes(
    trajectory: dict,
    params: dict,
    *,
    require_trajectory_continuity: bool,
) -> bool:
    return bool(
        not require_trajectory_continuity or params["max_adjacent_step_ratio"] <= 0 or trajectory["passed"]
    )


def _evaluation_step_ratio(params: dict) -> float:
    configured = float(params["max_adjacent_step_ratio"])
    return configured if configured > 1.0 else 10.0


def _select_largest_model(
    sparse_dir: Path,
    spec: InputSpec,
    *,
    max_step_ratio: float,
) -> tuple[Path, dict]:
    models = [
        path for path in sorted(sparse_dir.iterdir()) if path.is_dir() and (path / "cameras.bin").is_file()
    ]
    if not models and (sparse_dir / "cameras.bin").is_file():
        direct = sparse_dir / "_direct"
        direct.mkdir()
        for source in list(sparse_dir.iterdir()):
            if source == direct or not source.is_file():
                continue
            source.rename(direct / source.name)
        models = [direct]
    if not models:
        raise RuntimeError("mapper produced no sparse model")

    reconstructions = {path: colmap_model.read_model(path) for path in models}
    source_by_name = {image["name"]: image["source_id"] for image in spec.images}
    best = max(
        models,
        key=lambda path: (
            sum(
                source_by_name.get(image.name) == spec.primary_source_id
                for image in reconstructions[path].images.values()
            ),
            len(reconstructions[path].images),
            len(reconstructions[path].points3D),
        ),
    )
    summary = reconstructions[best].summary()
    summary["primary_trajectory"] = trajectory_quality.evaluate_primary_trajectory(
        reconstructions[best],
        spec.images,
        spec.primary_source_id,
        max_step_ratio=max_step_ratio,
    )
    summary["num_models"] = len(models)
    registration = {}
    source_metadata = {source["id"]: source for source in spec.sources}
    for source_id in sorted({image["source_id"] for image in spec.images}):
        total = sum(image["source_id"] == source_id for image in spec.images)
        registered = sum(
            source_by_name.get(image.name) == source_id for image in reconstructions[best].images.values()
        )
        registration[source_id] = {
            "label": source_metadata[source_id]["label"],
            "role": source_metadata[source_id]["role"],
            "total": total,
            "registered": registered,
            "ratio": registered / max(1, total),
            "connected": registered > 0,
        }
    summary["source_registration"] = registration
    canonical = sparse_dir / "0"
    if best != canonical:
        temporary = sparse_dir / "_best"
        best.rename(temporary)
        if canonical.exists():
            shutil.rmtree(canonical)
        temporary.rename(canonical)
    return canonical, summary


def _file_ref(path: Path, ctx: StageContext) -> FileRef:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    relative = Path(final_name) / path.relative_to(ctx.stage_out_dir)
    return FileRef(
        path=str(relative),
        size=path.stat().st_size,
        sha256=sha256_file(path),
        mime="application/json" if path.suffix == ".json" else "application/octet-stream",
    )
