"""全 track の不確実性と capture 留保予測で sparse model をフィルタする。"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter

import numpy as np

from ..colmap import model as colmap_model
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
    impl_version = "2.0"

    def normalize_params(self, raw: dict) -> dict:
        values = {
            "relative_error": float(raw.get("relative_error", 0.02)),
            "pixel_sigma": float(raw.get("pixel_sigma", 1.0)),
            "max_cross_error": float(raw.get("max_cross_error", 2.0)),
        }
        for key, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        max_preview_points = int(raw.get("max_preview_points", 500_000))
        if max_preview_points <= 0:
            raise ValueError("max_preview_points must be positive")
        return {
            "enabled": bool(raw.get("enabled", True)),
            **values,
            "max_preview_points": max_preview_points,
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
    views = geometry(reconstruction)
    points = sorted(reconstruction.points3D.values(), key=lambda point: point.point3D_id)
    if not points:
        raise ValueError("sparse cleanup input has no points")
    retained: set[int] = set()
    counts = Counter()
    metrics = np.full((len(points), len(FULL_TRACK_COLUMNS)), np.nan)
    reasons = []
    for index, point in enumerate(points):
        reason, values = assess_point(
            point,
            views,
            records,
            relative_budget=ctx.params["relative_error"],
            pixel_sigma=ctx.params["pixel_sigma"],
            cross_limit=ctx.params["max_cross_error"],
            policy="full_track",
        )
        metrics[index] = values
        reasons.append(reason)
        counts[reason] += 1
        if reason == "keep":
            retained.add(point.point3D_id)
        ctx.progress.tick(
            0.03 + 0.70 * (index + 1) / len(points),
            message=f"verify sparse tracks {index + 1}/{len(points)}",
            key="log.cleanup_sparse_assess",
            args={"cur": index + 1, "tot": len(points), "kept": len(retained)},
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
    original_observations = sum(len(point.track) for point in points)
    retain_points(reconstruction, retained)
    return {
        "enabled": True,
        "policy": "full_track",
        "input_points": len(points),
        "removed_points": len(points) - len(retained),
        "output_points": len(retained),
        "reason_counts": dict(counts),
        "cleared_observations": original_observations
        - sum(len(point.track) for point in reconstruction.points3D.values()),
        "images_without_points": sum(
            image.num_registered_points == 0 for image in reconstruction.images.values()
        ),
        "minimum_captures": 3,
        "relative_error": ctx.params["relative_error"],
        "pixel_sigma": ctx.params["pixel_sigma"],
        "max_cross_error": ctx.params["max_cross_error"],
        "max_single_error": 2 * ctx.params["max_cross_error"],
        "camera_models": sorted({camera.model for camera in reconstruction.cameras.values()}),
        "original_geometry_retained": True,
        "added_points": 0,
    }
