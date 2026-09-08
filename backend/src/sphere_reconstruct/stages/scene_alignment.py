"""重力方向を保ち、Manhattan 主軸と地面を dataset 座標へ揃えるステージ。"""

from __future__ import annotations

import json

import numpy as np

from ..colmap import ground_position, scene_orientation
from ..colmap import model as colmap_model
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class SceneAlignment(Stage):
    name = StageName.SCENE_ALIGNMENT
    impl_version = "1.0"

    def normalize_params(self, raw: dict) -> dict:
        method = str(raw.get("method", "auto")).lower()
        if method not in {"auto", "required", "none"}:
            raise ValueError(f"unsupported scene alignment method: {method}")
        return {
            "method": method,
            "align_manhattan_axes": bool(raw.get("align_manhattan_axes", True)),
            "min_camera_height_m": float(raw.get("min_camera_height_m", 0.5)),
            "max_camera_height_m": float(raw.get("max_camera_height_m", 4.0)),
            "histogram_bin_m": float(raw.get("histogram_bin_m", 0.05)),
            "inlier_band_m": float(raw.get("inlier_band_m", 0.15)),
            "max_path_distance_m": float(raw.get("max_path_distance_m", 4.0)),
            "min_support_ratio": float(raw.get("min_support_ratio", 0.005)),
            "min_horizontal_span_ratio": float(raw.get("min_horizontal_span_ratio", 0.25)),
            "max_analysis_points": int(raw.get("max_analysis_points", 100_000)),
            "max_orientation_points": int(raw.get("max_orientation_points", 20_000)),
            "orientation_candidate_lines": int(raw.get("orientation_candidate_lines", 2_048)),
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "restore_metric_scale.json",
            ctx.project_dir / "restore_metric_scale" / "scale_restoration.json",
            *(ctx.project_dir / "restore_metric_scale" / "sparse" / "0").glob("*"),
        ]
        return similarity_transform.input_refs(ctx, candidates)

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        input_model = ctx.project_dir / "restore_metric_scale" / "sparse" / "0"
        if not (input_model / "cameras.bin").is_file():
            raise RuntimeError("restore_metric_scale must run before scene alignment")

        ctx.progress.info("analyzing scene coordinates", progress=0.0, key="log.scene_alignment_start")
        scale_info = json.loads(
            (ctx.project_dir / "restore_metric_scale" / "scale_restoration.json").read_text(
                encoding="utf-8"
            )
        )
        reconstruction = colmap_model.read_model(input_model)
        analysis_points = ground_position.sample_analysis_points(
            reconstruction,
            maximum_points=ctx.params["max_analysis_points"],
        )
        if ctx.params["method"] == "none":
            ground = {"applied": False, "reason": "disabled", "translation": [0.0, 0.0, 0.0]}
            orientation = {"applied": False, "reason": "disabled", "yaw_deg": 0.0}
        else:
            ground = ground_position.estimate_ground_position(
                reconstruction,
                analysis_points=analysis_points,
                reference_image_prefix=(
                    f"sources/{ctx.primary_source.id}/" if ctx.primary_source is not None else None
                ),
                min_camera_height_m=ctx.params["min_camera_height_m"],
                max_camera_height_m=ctx.params["max_camera_height_m"],
                histogram_bin_m=ctx.params["histogram_bin_m"],
                inlier_band_m=ctx.params["inlier_band_m"],
                max_path_distance_m=ctx.params["max_path_distance_m"],
                min_support_ratio=ctx.params["min_support_ratio"],
                min_horizontal_span_ratio=ctx.params["min_horizontal_span_ratio"],
            )
            if "method" not in ground and ctx.params["method"] == "required":
                raise RuntimeError(f"ground prediction failed: {ground}")
            ground.setdefault("translation", [0.0, 0.0, 0.0])
            _label_ground_units(ground, metric_scale_available=bool(scale_info.get("metric")))
            orientation = (
                scene_orientation.estimate_scene_yaw(
                    analysis_points,
                    ground_y=float(ground["ground_y"]) if "ground_y" in ground else None,
                    ground_band=ctx.params["inlier_band_m"],
                    maximum_points=ctx.params["max_orientation_points"],
                    candidate_lines=ctx.params["orientation_candidate_lines"],
                )
                if ctx.params["align_manhattan_axes"]
                else {"applied": False, "reason": "disabled", "yaw_deg": 0.0}
            )

        ground["analysis_points"] = len(analysis_points)
        ground["total_points"] = len(reconstruction.points3D)
        yaw_radians = float(orientation.get("yaw_radians", 0.0)) if orientation.get("applied") else 0.0
        rotation = scene_orientation.yaw_rotation_matrix(yaw_radians)
        translation = tuple(float(value) for value in ground["translation"])
        result = {
            "applied": bool(ground.get("applied") or orientation.get("applied")),
            "method": "ground_and_manhattan_axes",
            "ground": ground,
            "orientation": orientation,
            "rotation": rotation.tolist(),
            "translation": list(translation),
            "analysis_limit": ctx.params["max_analysis_points"],
            "orientation_analysis_limit": ctx.params["max_orientation_points"],
        }

        ctx.progress.info("applying scene coordinates", progress=0.55, key="log.scene_alignment_transform")
        output_model, aligned = similarity_transform.materialize_similarity(
            ctx,
            input_model,
            rotation=rotation,
            translation=translation,
            log_label="scene-alignment",
        )
        result["model_summary"] = aligned.summary()
        result_path = ctx.stage_out_dir / "scene_alignment.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress.tick(0.75, message="scene coordinate transform complete", key="log.scene_alignment_done")
        ctx.progress.info("building scene-aligned preview", progress=0.8, key="log.scene_alignment_preview")
        preview = similarity_transform.write_preview(
            ctx,
            aligned,
            metadata_key="scene_alignment",
            metadata=result,
            max_points=ctx.params["max_preview_points"],
            additional_metadata={"metric_scale": scale_info},
        )
        manifest.outputs = similarity_transform.output_refs(ctx, output_model, result_path)
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info("scene alignment stage complete", progress=0.99, key="log.scene_alignment_complete")
        return manifest


def _label_ground_units(result: dict, *, metric_scale_available: bool) -> None:
    result["metric_scale_available"] = metric_scale_available
    if metric_scale_available:
        return
    for metric_key in (
        "camera_height_median_m",
        "path_distance_median_m",
        "ground_height_p10_m",
        "ground_height_p90_m",
        "max_path_distance_m",
    ):
        if metric_key in result:
            result[f"{metric_key.removesuffix('_m')}_model_units"] = result.pop(metric_key)
