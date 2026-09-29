"""全 track の不確実性と capture 留保予測で sparse model をフィルタする。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..colmap import model as colmap_model
from ..colmap import trajectory_quality
from ..colmap.input_workspace import InputSpec
from ..colmap.point_stability import (
    FULL_TRACK_COLUMNS,
    assess_point,
    geometry,
    retain_points,
    validate_tracks,
)
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class CleanupSparse(Stage):
    name = StageName.CLEANUP_SPARSE
    impl_version = "2.5"

    def normalize_params(self, raw: dict) -> dict:
        values = {
            "relative_error": float(raw.get("relative_error", 0.02)),
            "pixel_sigma": float(raw.get("pixel_sigma", 1.0)),
            "max_cross_error": float(raw.get("max_cross_error", 2.0)),
            "assessment_workers": int(raw.get("assessment_workers", 1)),
        }
        for key, value in values.items():
            if key == "assessment_workers":
                if value < 0:
                    raise ValueError("assessment_workers must be >= 0")
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        max_preview_points = int(raw.get("max_preview_points", 500_000))
        if max_preview_points <= 0:
            raise ValueError("max_preview_points must be positive")
        max_adjacent_step_ratio = float(raw.get("max_adjacent_step_ratio", 10.0))
        if max_adjacent_step_ratio != 0.0 and max_adjacent_step_ratio <= 1.0:
            raise ValueError("max_adjacent_step_ratio must be zero or greater than one")
        trajectory_minimum_steps = int(raw.get("trajectory_minimum_steps", 20))
        if trajectory_minimum_steps < 2:
            raise ValueError("trajectory_minimum_steps must be at least two")
        return {
            "enabled": bool(raw.get("enabled", True)),
            **values,
            "max_preview_points": max_preview_points,
            "remove_trajectory_outliers": bool(raw.get("remove_trajectory_outliers", True)),
            "max_adjacent_step_ratio": max_adjacent_step_ratio,
            "trajectory_minimum_steps": trajectory_minimum_steps,
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "extract_features" / "input_spec.json",
            ctx.project_dir / "manifests" / "scene_alignment.json",
            ctx.project_dir / "scene_alignment" / "scene_alignment.json",
            *(ctx.project_dir / "scene_alignment" / "sparse" / "0").glob("*"),
        ]
        return similarity_transform.input_refs(ctx, candidates)

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        input_model = ctx.project_dir / "scene_alignment" / "sparse" / "0"
        if not (input_model / "cameras.bin").is_file():
            raise RuntimeError("scene_alignment must run before sparse cleanup")

        ctx.progress.info("loading sparse cleanup model", progress=0.01, key="log.cleanup_sparse_start")
        reconstruction = colmap_model.read_model(input_model)
        input_points = len(reconstruction.points3D)
        result = (
            _filter_reconstruction(
                ctx,
                reconstruction,
                InputSpec.read(ctx.project_dir / "extract_features" / "input_spec.json").images,
            )
            if ctx.params["enabled"]
            else {
                "enabled": False,
                "input_points": input_points,
                "removed_points": 0,
                "output_points": input_points,
                "reason": "disabled",
            }
        )

        output_model = ctx.stage_out_dir / "sparse" / "0"
        output_model.mkdir(parents=True)
        colmap_model.write_cameras_bin(output_model / "cameras.bin", reconstruction.cameras)
        image_span = 0.76, 0.86
        colmap_model.write_images_bin(
            output_model / "images.bin",
            reconstruction.images,
            progress=lambda current, total: ctx.progress.tick(
                image_span[0] + (image_span[1] - image_span[0]) * current / max(1, total),
                message=f"rewrite cleaned sparse observations {current}/{total}",
                key="log.cleanup_sparse_images",
                args={"cur": current, "tot": total},
            ),
        )
        colmap_model.write_points3D_bin(output_model / "points3D.bin", reconstruction.points3D)
        for name in ("rigs.bin", "frames.bin", "project.ini"):
            source = input_model / name
            if source.is_file():
                if name == "frames.bin" and result.get("removed_images", 0):
                    colmap_model.filter_frames_bin(source, output_model / name, set(reconstruction.images))
                else:
                    shutil.copy2(source, output_model / name)

        result_path = ctx.stage_out_dir / "cleanup_sparse.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress.info("building sparse cleanup preview", progress=0.88, key="log.cleanup_sparse_preview")
        preview = similarity_transform.write_preview(
            ctx,
            reconstruction,
            metadata_key="sparse_cleanup",
            metadata=result,
            max_points=ctx.params["max_preview_points"],
        )
        manifest.outputs = similarity_transform.output_refs(ctx, output_model, result_path)
        assessment_path = ctx.stage_out_dir / "point_assessment.npz"
        if assessment_path.is_file():
            manifest.outputs.append(
                FileRef(
                    path=f"cleanup_sparse/{assessment_path.name}",
                    size=assessment_path.stat().st_size,
                    sha256=sha256_file(assessment_path),
                )
            )
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info(
            "sparse cleanup complete",
            progress=0.99,
            key="log.cleanup_sparse_done",
            args={"removed": result["removed_points"], "remaining": result["output_points"]},
        )
        return manifest


def _filter_reconstruction(
    ctx: StageContext, reconstruction: colmap_model.Reconstruction, image_records: list[dict]
) -> dict:
    records = {}
    for record in image_records:
        name = record["name"]
        source = record["source_id"]
        capture = record["capture_index"]
        if not isinstance(source, str) or not source or not isinstance(capture, int) or capture < 0:
            raise ValueError(f"invalid source/capture metadata: {name}")
        if name in records:
            raise ValueError(f"duplicate capture metadata: {name}")
        records[name] = record
    for image in reconstruction.images.values():
        if image.name not in records:
            raise ValueError(f"missing capture metadata: {image.name}")
    validate_tracks(reconstruction)
    input_image_count = len(reconstruction.images)
    input_observations = sum(len(point.track) for point in reconstruction.points3D.values())
    trajectory_result = _find_trajectory_outliers(ctx, reconstruction, image_records)
    removed_image_ids = _remove_trajectory_outlier_images(
        reconstruction,
        image_records,
        trajectory_result["outlier_captures_by_source"],
    )
    trajectory_removed_observations = input_observations - sum(
        len(point.track) for point in reconstruction.points3D.values()
    )
    views = geometry(reconstruction)
    points = sorted(reconstruction.points3D.values(), key=lambda point: point.point3D_id)
    if not points:
        raise ValueError("sparse cleanup input has no points")
    retained: set[int] = set()
    counts = Counter()
    metrics = np.full((len(points), len(FULL_TRACK_COLUMNS)), np.nan)
    reasons = []
    workers = _assessment_workers(ctx.params["assessment_workers"])
    batches = _assessment_batches(points, workers)
    if workers == 1:
        batches = [points]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sparse-assess") as executor:
        results = executor.map(
            _assess_batch,
            batches,
            (views for _ in batches),
            (records for _ in batches),
            (ctx.params["relative_error"] for _ in batches),
            (ctx.params["pixel_sigma"] for _ in batches),
            (ctx.params["max_cross_error"] for _ in batches),
        )
        processed = 0
        for batch_results in results:
            for point, reason, values in batch_results:
                index = processed
                metrics[index] = values
                reasons.append(reason)
                counts[reason] += 1
                if reason == "keep":
                    retained.add(point.point3D_id)
                processed += 1
            ctx.progress.tick(
                0.03 + 0.70 * processed / len(points),
                message=f"verify sparse tracks {processed}/{len(points)}",
                key="log.cleanup_sparse_assess",
                args={"cur": processed, "tot": len(points), "kept": len(retained)},
            )
    np.savez_compressed(
        ctx.stage_out_dir / "point_assessment.npz",
        point_ids=np.asarray([point.point3D_id for point in points], dtype=np.uint64),
        columns=np.asarray(FULL_TRACK_COLUMNS),
        metrics=metrics,
        reasons=np.asarray(reasons),
    )
    if not retained:
        raise ValueError(
            "no sparse points pass full-track cleanup; review the uncertainty and reprojection limits"
        )
    retain_points(reconstruction, retained)
    remaining_observations = sum(len(point.track) for point in reconstruction.points3D.values())
    output_points = len(reconstruction.points3D)
    return {
        "enabled": True,
        "policy": "full_track",
        "input_points": len(points),
        "removed_points": len(points) - output_points,
        "output_points": output_points,
        "reason_counts": dict(counts),
        "cleared_observations": input_observations - remaining_observations,
        "trajectory_removed_observations": trajectory_removed_observations,
        "images_without_points": sum(
            image.num_registered_points == 0 for image in reconstruction.images.values()
        ),
        "input_images": input_image_count,
        "output_images": len(reconstruction.images),
        "removed_images": len(removed_image_ids),
        "removed_captures": sum(
            len(captures) for captures in trajectory_result["outlier_captures_by_source"].values()
        ),
        "trajectory_outliers": trajectory_result["trajectories"],
        "minimum_captures": 3,
        "relative_error": ctx.params["relative_error"],
        "pixel_sigma": ctx.params["pixel_sigma"],
        "max_cross_error": ctx.params["max_cross_error"],
        "max_single_error": 2 * ctx.params["max_cross_error"],
        "camera_models": sorted({camera.model for camera in reconstruction.cameras.values()}),
        "original_geometry_retained": True,
        "added_points": 0,
        "assessment_workers": workers,
    }


def _assessment_workers(configured: int) -> int:
    """选择评分线程数；0 表示自动，限制上限避免小型机器过度切换。"""
    if configured < 0:
        raise ValueError("assessment_workers must be >= 0")
    if configured:
        return configured
    return min(32, max(1, os.cpu_count() or 1))


def _assessment_batches(points, workers: int):
    batch_size = max(64, min(2048, math.ceil(len(points) / max(1, workers * 8))))
    return [points[start : start + batch_size] for start in range(0, len(points), batch_size)]


def _assess_batch(points, views, records, relative_error, pixel_sigma, cross_limit):
    return [
        (
            point,
            *assess_point(
                point,
                views,
                records,
                relative_budget=relative_error,
                pixel_sigma=pixel_sigma,
                cross_limit=cross_limit,
                policy="full_track",
            ),
        )
        for point in points
    ]


_SEQUENCE_NAME = re.compile(r"(?:frame|image|img)[_-]?\d+", re.IGNORECASE)


def _find_trajectory_outliers(ctx, reconstruction, image_records: list[dict]) -> dict:
    records_by_source: dict[str, list[dict]] = defaultdict(list)
    for record in image_records:
        records_by_source[str(record["source_id"])].append(record)
    source_contexts = {source.id: source for source in getattr(ctx, "sources", ())}
    outlier_captures_by_source: dict[str, list[int]] = {}
    trajectories: dict[str, dict] = {}
    ratio = float(ctx.params.get("max_adjacent_step_ratio", 0.0))
    if not ctx.params.get("remove_trajectory_outliers", True) or ratio == 0.0:
        return {
            "outlier_captures_by_source": outlier_captures_by_source,
            "trajectories": trajectories,
        }
    for source_id, records in records_by_source.items():
        source = source_contexts.get(source_id)
        media_kind = getattr(getattr(source, "media_kind", None), "value", None)
        if media_kind != "video" and not any(_SEQUENCE_NAME.search(str(record["name"])) for record in records):
            continue
        result = trajectory_quality.evaluate_primary_trajectory(
            reconstruction,
            image_records,
            source_id,
            max_step_ratio=ratio,
            minimum_steps=int(ctx.params.get("trajectory_minimum_steps", 20)),
            step_baseline="median",
        )
        diagnostics = {"adjacent_steps": result}
        candidates = set(result.get("outlier_captures", [])) if result.get("available") else set()
        primary = getattr(ctx, "primary_source", None)
        role = getattr(getattr(source, "role", None), "value", getattr(source, "role", None))
        if (
            source is not None
            and primary is not None
            and role == "supplemental"
            and primary.id != source_id
        ):
            consistency = trajectory_quality.evaluate_source_path_consistency(
                reconstruction,
                image_records,
                primary.id,
                source_id,
                minimum_captures=int(ctx.params.get("trajectory_minimum_steps", 20)),
            )
            diagnostics["primary_path_consistency"] = consistency
            if consistency.get("available"):
                candidates.update(int(capture) for capture in consistency["outlier_captures"])
        trajectories[source_id] = diagnostics
        if candidates:
            outlier_captures_by_source[source_id] = sorted(candidates)
    return {
        "outlier_captures_by_source": outlier_captures_by_source,
        "trajectories": trajectories,
    }


def _remove_trajectory_outlier_images(
    reconstruction,
    image_records: list[dict],
    outlier_captures_by_source: dict[str, list[int]],
) -> set[int]:
    if not outlier_captures_by_source:
        return set()
    capture_keys = {
        (source_id, int(capture))
        for source_id, captures in outlier_captures_by_source.items()
        for capture in captures
    }
    records_by_name = {str(record["name"]): record for record in image_records}
    removed_image_ids = {
        image_id
        for image_id, image in reconstruction.images.items()
        if (
            str(records_by_name[image.name]["source_id"]),
            int(records_by_name[image.name]["capture_index"]),
        )
        in capture_keys
    }
    if not removed_image_ids:
        return set()
    track_keys: dict[int, set[tuple[int, int]]] = {}
    for point_id, point in reconstruction.points3D.items():
        point.track = [item for item in point.track if item[0] not in removed_image_ids]
        track_keys[point_id] = set(point.track)
    reconstruction.images = {
        image_id: image
        for image_id, image in reconstruction.images.items()
        if image_id not in removed_image_ids
    }
    for image_id, image in reconstruction.images.items():
        for index, observation in enumerate(image.points2D):
            if observation.point3D_id == 2**64 - 1:
                continue
            if (
                observation.point3D_id not in track_keys
                or (image_id, index) not in track_keys[observation.point3D_id]
            ):
                observation.point3D_id = 2**64 - 1
    validate_tracks(reconstruction)
    return removed_image_ids
