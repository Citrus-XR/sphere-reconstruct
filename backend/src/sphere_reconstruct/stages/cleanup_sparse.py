"""Scene-aligned sparse model から遠距離・低視差 point だけを条件付きで除去する。"""

from __future__ import annotations

import json
import math
import shutil

import numpy as np

from ..colmap import gravity_align
from ..colmap import model as colmap_model
from ..colmap.ground_position import _nearest_trajectory_samples
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class CleanupSparse(Stage):
    name = StageName.CLEANUP_SPARSE
    impl_version = "1.0"

    def normalize_params(self, raw: dict) -> dict:
        far_distance_ratio = float(raw.get("far_distance_ratio", 0.3))
        far_min_triangulation_deg = float(raw.get("far_min_triangulation_deg", 2.0))
        if far_distance_ratio <= 0:
            raise ValueError("far_distance_ratio must be positive")
        if not 0 < far_min_triangulation_deg < 90:
            raise ValueError("far_min_triangulation_deg must be within 0..90")
        return {
            "enabled": bool(raw.get("enabled", True)),
            "far_distance_ratio": far_distance_ratio,
            "far_min_triangulation_deg": far_min_triangulation_deg,
            "max_reprojection_error": float(raw.get("max_reprojection_error", 0.0)),
            "min_track_length": int(raw.get("min_track_length", 2)),
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
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
            _filter_reconstruction(ctx, reconstruction)
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
        image_span = 0.45, 0.84
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
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info(
            "sparse cleanup complete",
            progress=0.99,
            key="log.cleanup_sparse_done",
            args={"removed": result["removed_points"], "remaining": result["output_points"]},
        )
        return manifest


def _filter_reconstruction(ctx: StageContext, reconstruction: colmap_model.Reconstruction) -> dict:
    reference_prefix = f"sources/{ctx.primary_source.id}/" if ctx.primary_source else None
    reference_images = [
        image
        for image in reconstruction.images.values()
        if reference_prefix is None or image.name.startswith(reference_prefix)
    ]
    if not reference_images:
        raise RuntimeError("sparse cleanup reference trajectory is unavailable")
    cameras = np.asarray([image.camera_center for image in reference_images], dtype=np.float64)
    diameter = gravity_align.reference_trajectory_diameter(reconstruction, reference_prefix)
    far_distance = diameter * ctx.params["far_distance_ratio"]
    points = list(reconstruction.points3D.values())
    coordinates = np.asarray([point.xyz for point in points], dtype=np.float64)
    _, nearest_camera = _nearest_trajectory_samples(coordinates[:, [0, 2]], cameras[:, [0, 2]], far_distance)
    far = nearest_camera < 0
    centers = {image.image_id: np.asarray(image.camera_center) for image in reconstruction.images.values()}
    removed_ids: set[int] = set()
    removed_far_low_angle = 0
    removed_reprojection = 0
    removed_short_track = 0
    for index, point in enumerate(points):
        reason = None
        if len(point.track) < ctx.params["min_track_length"]:
            reason = "short_track"
        elif ctx.params["max_reprojection_error"] > 0 and point.error > ctx.params["max_reprojection_error"]:
            reason = "reprojection"
        elif far[index] and not _has_triangulation_angle(
            point, centers, ctx.params["far_min_triangulation_deg"]
        ):
            reason = "far_low_angle"
        if reason is None:
            continue
        removed_ids.add(point.point3D_id)
        removed_short_track += reason == "short_track"
        removed_reprojection += reason == "reprojection"
        removed_far_low_angle += reason == "far_low_angle"

    for point_id in removed_ids:
        del reconstruction.points3D[point_id]
    cleared_observations = 0
    for image in reconstruction.images.values():
        for observation in image.points2D:
            if observation.point3D_id in removed_ids:
                observation.point3D_id = -1
                cleared_observations += 1
    return {
        "enabled": True,
        "input_points": len(points),
        "removed_points": len(removed_ids),
        "output_points": len(reconstruction.points3D),
        "removed_far_low_angle": removed_far_low_angle,
        "removed_reprojection": removed_reprojection,
        "removed_short_track": removed_short_track,
        "cleared_observations": cleared_observations,
        "reference_trajectory_diameter": diameter,
        "far_distance_threshold": far_distance,
        "far_candidate_points": int(np.count_nonzero(far)),
        "far_min_triangulation_deg": ctx.params["far_min_triangulation_deg"],
    }


def _has_triangulation_angle(
    point: colmap_model.Point3D,
    centers: dict[int, np.ndarray],
    minimum_deg: float,
) -> bool:
    directions = np.asarray([np.asarray(point.xyz) - centers[image_id] for image_id, _ in point.track])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    maximum_abs_dot = math.cos(math.radians(minimum_deg))
    for index in range(1, len(directions)):
        if np.any(np.abs(directions[:index] @ directions[index]) <= maximum_abs_dot):
            return True
    return False
