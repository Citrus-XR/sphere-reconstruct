"""幾何検証済み database から sparse reconstruction を解くステージ.

特徴抽出と matching は独立ステージで完了しているため, Mapper・BA 設定だけを変更して
再実行できる. Incremental COLMAP と COLMAP 4.1 内蔵 Global Mapper を同じ契約で扱う.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..colmap import model as colmap_model
from ..colmap import quality as colmap_quality
from ..colmap import runner as colmap_runner
from ..colmap import trajectory_quality
from ..colmap.input_workspace import InputSpec
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings
from .colmap_progress import global_mapper_progress, hidden_log, mapper_progress


@register
class Reconstruct(Stage):
    name = StageName.RECONSTRUCT
    impl_version = "2.5"

    def normalize_params(self, raw: dict) -> dict:
        mapper = str(raw.get("mapper", "global")).lower()
        if mapper not in {"incremental", "global"}:
            raise ValueError(f"unsupported mapper: {mapper}")
        ba_use_gpu = bool(raw.get("ba_use_gpu", False))
        max_adjacent_step_ratio = float(raw.get("max_adjacent_step_ratio", 10.0))
        if max_adjacent_step_ratio != 0.0 and max_adjacent_step_ratio <= 1.0:
            raise ValueError("max_adjacent_step_ratio must be zero or greater than one")
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
            "filter_max_reproj_error": float(raw.get("filter_max_reproj_error", 0.0)),
            "filter_min_tri_angle": float(raw.get("filter_min_tri_angle", 0.0)),
            "ba_local_max_num_iterations": int(raw.get("ba_local_max_num_iterations", 0)),
            "ba_global_max_num_iterations": int(raw.get("ba_global_max_num_iterations", 0)),
            "min_model_size": int(raw.get("min_model_size", 0)),
            "min_registered_ratio": float(raw.get("min_registered_ratio", 0.8)),
            "min_points3D": int(raw.get("min_points3D", 100)),
            "max_adjacent_step_ratio": max_adjacent_step_ratio,
            "incremental_fallback": bool(raw.get("incremental_fallback", True)),
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
        require_trajectory_continuity = bool(
            ctx.primary_source is not None and ctx.primary_source.media_kind == MediaKind.VIDEO
        )
        apply_view_graph_calibration = bool(
            mapper == "global" and ctx.params["view_graph_calibration"] and spec.refine_intrinsics
        )
        ctx.progress.info(
            f"sparse reconstruction: mapper={mapper}, images={spec.image_count}",
            progress=0.05,
            key="log.recon_start",
            args={"mapper": mapper, "images": spec.image_count},
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
                colmap_runner.mapper(
                    colmap_bin,
                    database_path=fallback_database,
                    image_path=image_path,
                    output_path=fallback_sparse,
                    refine_intrinsics=spec.refine_intrinsics,
                    refine_rig=spec.refine_rig,
                    multiple_models=spec.multiple_models,
                    extra_args=_incremental_args(ctx.params),
                    log_path=logs_dir / "incremental_fallback.log",
                    on_line=mapper_progress(ctx, spec.image_count, low=0.76, high=0.9),
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
                    shutil.rmtree(fallback_sparse)
                    fallback_database.unlink()
            if summary is None or model_dir is None:
                raise global_error or RuntimeError("Global and Incremental Mapper produced no model")
        else:
            colmap_runner.mapper(
                colmap_bin,
                database_path=database_path,
                image_path=image_path,
                output_path=sparse_dir,
                refine_intrinsics=spec.refine_intrinsics,
                refine_rig=spec.refine_rig,
                multiple_models=spec.multiple_models,
                extra_args=_incremental_args(ctx.params),
                log_path=logs_dir / "mapper.log",
                on_line=mapper_progress(ctx, spec.image_count, low=0.12, high=0.9),
            )
            model_dir, summary = _select_largest_model(
                sparse_dir,
                spec,
                max_step_ratio=_evaluation_step_ratio(ctx.params),
            )
            mapper_attempts = []
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
        _validate_summary(
            summary,
            ctx.params,
            require_trajectory_continuity=require_trajectory_continuity,
        )
        summary_path = ctx.stage_out_dir / "model_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs = [_file_ref(summary_path, ctx), _file_ref(database_path, ctx)]
        for path in model_dir.iterdir():
            if path.is_file():
                outputs.append(_file_ref(path, ctx))
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


def _incremental_args(params: dict) -> list[str]:
    args = ["--Mapper.random_seed", str(params["random_seed"]), *colmap_quality.mapper_extra_args(params)]
    if params["init_image_id1"] > 0:
        args += ["--Mapper.init_image_id1", str(params["init_image_id1"])]
    if params["init_image_id2"] > 0:
        args += ["--Mapper.init_image_id2", str(params["init_image_id2"])]
    return args


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
        not require_trajectory_continuity
        or params["max_adjacent_step_ratio"] <= 0
        or trajectory["passed"]
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
