"""重力方向を保ったまま、予測地面を dataset world の Y=0 へ移すステージ。"""

from __future__ import annotations

import json

from ..colmap import ground_position
from ..colmap import model as colmap_model
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class PositionGround(Stage):
    name = StageName.POSITION_GROUND
    impl_version = "1.1"

    def normalize_params(self, raw: dict) -> dict:
        method = str(raw.get("method", "auto")).lower()
        if method not in {"auto", "points", "none"}:
            raise ValueError(f"unsupported ground positioning method: {method}")
        return {
            "method": method,
            "min_camera_height_m": float(raw.get("min_camera_height_m", 0.5)),
            "max_camera_height_m": float(raw.get("max_camera_height_m", 4.0)),
            "histogram_bin_m": float(raw.get("histogram_bin_m", 0.05)),
            "inlier_band_m": float(raw.get("inlier_band_m", 0.15)),
            "max_path_distance_m": float(raw.get("max_path_distance_m", 4.0)),
            "min_support_ratio": float(raw.get("min_support_ratio", 0.005)),
            "min_horizontal_span_ratio": float(raw.get("min_horizontal_span_ratio", 0.25)),
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
            raise RuntimeError("restore_metric_scale must run before ground positioning")
        ctx.progress.info("predicting ground position", progress=0.0, key="log.ground_start")
        scale_info = json.loads(
            (ctx.project_dir / "restore_metric_scale" / "scale_restoration.json").read_text(encoding="utf-8")
        )
        reconstruction = colmap_model.read_model(input_model)
        if ctx.params["method"] == "none":
            result = {
                "applied": False,
                "method": "none",
                "reason": "disabled",
                "translation": [0.0, 0.0, 0.0],
            }
        elif not scale_info.get("metric"):
            if ctx.params["method"] == "points":
                raise RuntimeError("ground positioning requires a metric reconstruction")
            result = {
                "applied": False,
                "method": "none",
                "reason": "metric_scale_unavailable",
                "translation": [0.0, 0.0, 0.0],
            }
        else:
            result = ground_position.estimate_ground_position(
                reconstruction,
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
            if "method" not in result and ctx.params["method"] == "points":
                raise RuntimeError(f"ground prediction failed: {result}")
            result.setdefault("translation", [0.0, 0.0, 0.0])

        ctx.progress.info("positioning predicted ground", progress=0.45, key="log.ground_transform")
        translation = tuple(float(value) for value in result["translation"])
        output_model, positioned = similarity_transform.materialize_similarity(
            ctx,
            input_model,
            translation=translation,
            log_label="ground-position",
        )
        result["model_summary"] = positioned.summary()
        result_path = ctx.stage_out_dir / "ground_position.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress.tick(0.75, message="ground positioning complete", key="log.ground_done")
        ctx.progress.info("building grounded preview", progress=0.8, key="log.ground_preview")
        preview = similarity_transform.write_preview(
            ctx,
            positioned,
            metadata_key="ground_position",
            metadata=result,
            max_points=ctx.params["max_preview_points"],
            additional_metadata={"metric_scale": scale_info},
        )
        manifest.outputs = similarity_transform.output_refs(ctx, output_model, result_path)
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info("ground positioning stage complete", progress=0.99, key="log.ground_complete")
        return manifest
