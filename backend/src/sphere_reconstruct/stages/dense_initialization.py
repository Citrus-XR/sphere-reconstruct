"""RoMaV2 correspondence を native camera ray で三角化する独立 Step。"""

from __future__ import annotations

import json
import shutil

from ..colmap import model as colmap_model
from ..dense_init import DenseInitializationConfig, densify_reconstruction
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..pipeline import prepared_images
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from . import similarity_transform


@register
class DenseInitialization(Stage):
    name = StageName.DENSE_INITIALIZATION
    impl_version = "1.0"

    def normalize_params(self, raw: dict) -> dict:
        quality = str(raw.get("quality", "turbo")).lower()
        if quality not in {"turbo", "fast", "base", "high"}:
            raise ValueError(f"unsupported RoMaV2 quality: {quality}")
        return {
            "enabled": bool(raw.get("enabled", False)),
            "quality": quality,
            "reference_fraction": float(raw.get("reference_fraction", 0.25)),
            "neighbors_per_reference": int(raw.get("neighbors_per_reference", 2)),
            "matches_per_pair": int(raw.get("matches_per_pair", 2000)),
            "confidence_threshold": float(raw.get("confidence_threshold", 0.2)),
            "reprojection_threshold_px": float(raw.get("reprojection_threshold_px", 1.5)),
            "minimum_parallax_deg": float(raw.get("minimum_parallax_deg", 0.5)),
            "maximum_new_points": int(raw.get("maximum_new_points", 200_000)),
            "voxel_size_ratio": float(raw.get("voxel_size_ratio", 0.0005)),
            "use_feature_masks": bool(raw.get("use_feature_masks", True)),
            "seed": int(raw.get("seed", 0)),
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "position_ground.json",
            ctx.project_dir / "manifests" / "extract_features.json",
            prepared_images.catalog_path(ctx.project_dir),
            *(ctx.project_dir / "position_ground" / "sparse" / "0").glob("*"),
        ]
        return similarity_transform.input_refs(ctx, candidates)

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        input_model = ctx.project_dir / "position_ground" / "sparse" / "0"
        output_model = ctx.stage_out_dir / "sparse" / "0"
        if not (input_model / "cameras.bin").is_file():
            raise RuntimeError("position_ground must run before dense initialization")

        if not ctx.params["enabled"]:
            shutil.copytree(input_model, output_model)
            reconstruction = colmap_model.read_model(output_model)
            result = {
                "enabled": False,
                "method": "passthrough",
                "base_points": len(reconstruction.points3D),
                "new_points": 0,
                "total_points": len(reconstruction.points3D),
            }
        else:
            ctx.progress.info(
                "loading RoMaV2 dense matcher",
                progress=0.01,
                key="log.dense_model_start",
            )
            reconstruction = colmap_model.read_model(input_model)
            base_points = len(reconstruction.points3D)
            catalog = prepared_images.load_catalog(ctx.project_dir)
            from ..dense_init.matcher import RomaV2Matcher  # noqa: PLC0415

            matcher = RomaV2Matcher(setting=ctx.params["quality"], seed=ctx.params["seed"])
            try:
                dense_result = densify_reconstruction(
                    reconstruction,
                    catalog,
                    ctx.project_dir / "extract_features" / "images",
                    ctx.project_dir / "extract_features" / "masks",
                    matcher,
                    DenseInitializationConfig(
                        reference_fraction=ctx.params["reference_fraction"],
                        neighbors_per_reference=ctx.params["neighbors_per_reference"],
                        matches_per_pair=ctx.params["matches_per_pair"],
                        confidence_threshold=ctx.params["confidence_threshold"],
                        reprojection_threshold_px=ctx.params["reprojection_threshold_px"],
                        minimum_parallax_deg=ctx.params["minimum_parallax_deg"],
                        maximum_new_points=ctx.params["maximum_new_points"],
                        voxel_size_ratio=ctx.params["voxel_size_ratio"],
                        use_feature_masks=ctx.params["use_feature_masks"],
                        seed=ctx.params["seed"],
                    ),
                    progress=lambda current, total: ctx.progress.info(
                        f"dense matching {current}/{total}",
                        progress=0.08 + 0.82 * current / max(total, 1),
                        key="log.dense_pairs_progress",
                        args={"cur": current, "tot": total},
                    ),
                )
            finally:
                matcher.close()
            output_model.mkdir(parents=True)
            colmap_model.write_cameras_bin(output_model / "cameras.bin", reconstruction.cameras)
            colmap_model.write_images_bin(output_model / "images.bin", reconstruction.images)
            colmap_model.write_points3D_bin(output_model / "points3D.bin", reconstruction.points3D)
            for name in ("rigs.bin", "frames.bin"):
                source = input_model / name
                if source.is_file():
                    shutil.copy2(source, output_model / name)
            result = {
                "enabled": True,
                "method": "romav2_native_rays",
                "quality": ctx.params["quality"],
                "base_points": base_points,
                "new_points": dense_result.kept_points,
                "total_points": len(reconstruction.points3D),
                **dense_result.__dict__,
            }

        result_path = ctx.stage_out_dir / "dense_initialization.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        preview = similarity_transform.write_preview(
            ctx,
            reconstruction,
            metadata_key="dense_initialization",
            metadata=result,
            max_points=ctx.params["max_preview_points"],
        )
        manifest.outputs = similarity_transform.output_refs(ctx, output_model, result_path)
        manifest.extra = {**result, "preview_points": preview.num_points_written}
        ctx.progress.info(
            "dense initialization complete",
            progress=0.99,
            key="log.dense_complete",
        )
        return manifest
