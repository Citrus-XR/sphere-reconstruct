"""既知の物理 rig baseline から reconstruction を meter scale へ戻すステージ。"""

from __future__ import annotations

import json

from ..colmap import metric_scale
from ..colmap import model as colmap_model
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class RestoreMetricScale(Stage):
    name = StageName.RESTORE_METRIC_SCALE
    impl_version = "1.0"

    def normalize_params(self, raw: dict) -> dict:
        method = str(raw.get("method", "auto")).lower()
        if method not in {"auto", "rig", "none"}:
            raise ValueError(f"unsupported metric scale method: {method}")
        return {
            "method": method,
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "align_reconstruction.json",
            ctx.project_dir / "align_reconstruction" / "alignment.json",
            ctx.project_dir / "prepare_images" / "image_catalog.json",
            ctx.project_dir / "prepare_images" / "rig_config.json",
            *(ctx.project_dir / "align_reconstruction" / "sparse" / "0").glob("*"),
        ]
        return similarity_transform.input_refs(ctx, candidates)

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        input_model = ctx.project_dir / "align_reconstruction" / "sparse" / "0"
        if not (input_model / "cameras.bin").is_file():
            raise RuntimeError("align_reconstruction must run before metric scale restoration")

        ctx.progress.info("estimating metric scale", progress=0.0, key="log.metric_scale_start")
        reconstruction = colmap_model.read_model(input_model)
        if ctx.params["method"] == "none":
            result = {
                "applied": False,
                "metric": False,
                "method": "none",
                "reason": "disabled",
                "scale_factor": 1.0,
            }
        else:
            catalog = json.loads(
                (ctx.project_dir / "prepare_images" / "image_catalog.json").read_text(
                    encoding="utf-8"
                )
            )
            rig_config = json.loads(
                (ctx.project_dir / "prepare_images" / "rig_config.json").read_text(
                    encoding="utf-8"
                )
            )
            result = metric_scale.estimate_metric_scale(reconstruction, catalog, rig_config)
            if not result.get("metric") and ctx.params["method"] == "rig":
                raise RuntimeError(f"metric scale restoration failed: {result}")
            result.setdefault("scale_factor", 1.0)

        ctx.progress.info(
            "applying metric scale",
            progress=0.45,
            key="log.metric_scale_transform",
        )
        output_model, restored = similarity_transform.materialize_similarity(
            ctx,
            input_model,
            scale=float(result["scale_factor"]),
            log_label="metric-scale",
        )
        result["model_summary"] = restored.summary()
        result_path = ctx.stage_out_dir / "scale_restoration.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.progress.tick(0.75, message="metric scale transform complete", key="log.metric_scale_done")
        ctx.progress.info(
            "building metric-scale preview", progress=0.8, key="log.metric_scale_preview"
        )
        preview = similarity_transform.write_preview(
            ctx,
            restored,
            metadata_key="metric_scale",
            metadata=result,
            max_points=ctx.params["max_preview_points"],
        )
        manifest.outputs = similarity_transform.output_refs(ctx, output_model, result_path)
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info(
            "metric scale restoration complete", progress=0.99, key="log.metric_scale_complete"
        )
        return manifest
