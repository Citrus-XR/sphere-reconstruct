"""画像 workspace の準備と COLMAP 特徴抽出を行う独立ステージ."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..colmap import input_workspace
from ..colmap import runner as colmap_runner
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings
from .colmap_progress import counted_progress, hidden_log

_FEATURE_TYPES = {"SIFT", "ALIKED_N16ROT", "ALIKED_N32"}


@register
class ExtractFeatures(Stage):
    name = StageName.EXTRACT_FEATURES
    impl_version = "2.1"

    def normalize_params(self, raw: dict) -> dict:
        feature_type = str(raw.get("feature_type", "SIFT")).upper()
        if feature_type not in _FEATURE_TYPES:
            raise ValueError(f"unsupported feature type: {feature_type}")
        return {
            "reconstruction_mode": str(raw.get("reconstruction_mode", "native_fisheye")),
            "feature_type": feature_type,
            "use_gpu": bool(raw.get("use_gpu", True)),
            "use_masks": bool(raw.get("use_masks", True)),
            "max_image_size": int(raw.get("max_image_size", 2048)),
            "max_num_features": int(raw.get("max_num_features", 8192)),
            "sift_peak_threshold": float(raw.get("sift_peak_threshold", 0.0)),
            "sift_edge_threshold": float(raw.get("sift_edge_threshold", 0.0)),
            "sift_affine_dsp": bool(raw.get("sift_affine_dsp", False)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "prepare_images.json",
            ctx.project_dir / "prepare_images" / "image_catalog.json",
            ctx.project_dir / "manifests" / "generate_masks.json",
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
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params
        spec = input_workspace.build(ctx.project_dir, ctx.stage_out_dir, ctx.params["use_masks"])
        logs_dir = ctx.stage_out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        database_path = ctx.stage_out_dir / "database.db"
        settings = get_settings()
        colmap_bin = colmap_runner.resolve_colmap_bin(settings.binaries.colmap or None)

        ctx.progress.info(
            f"feature extraction: {ctx.params['feature_type']}, {spec.image_count} images",
            progress=0.05,
            key="log.features_start",
            args={"type": ctx.params["feature_type"], "images": spec.image_count},
        )
        for index, batch in enumerate(spec.feature_batches):
            low = 0.05 + 0.85 * index / len(spec.feature_batches)
            high = 0.05 + 0.85 * (index + 1) / len(spec.feature_batches)
            colmap_runner.feature_extractor(
                colmap_bin,
                database_path=database_path,
                image_path=ctx.stage_out_dir / spec.image_path,
                camera_model=batch.camera_model,
                single_camera=batch.single_camera,
                single_camera_per_folder=batch.single_camera_per_folder,
                camera_params=",".join(str(value) for value in batch.camera_params),
                use_gpu=ctx.params["use_gpu"],
                feature_type=ctx.params["feature_type"],
                mask_path=ctx.stage_out_dir / spec.mask_path if spec.mask_path else None,
                image_list_path=ctx.stage_out_dir / batch.image_list_path,
                extra_args=self._feature_args(ctx.params, settings),
                log_path=logs_dir / f"feature_extractor_{index:03d}.log",
                on_line=counted_progress(
                    ctx,
                    f"features:{batch.id}",
                    r"Processed file \[(\d+)/(\d+)\]",
                    low=low,
                    high=high,
                ),
            )
        if spec.rig_config_path:
            colmap_runner.rig_configurator(
                colmap_bin,
                database_path=database_path,
                rig_config_path=ctx.stage_out_dir / spec.rig_config_path,
                log_path=logs_dir / "rig_configurator.log",
                on_line=hidden_log(ctx, "rig"),
            )

        summary = _feature_summary(database_path)
        summary.update(
            {
                "feature_type": ctx.params["feature_type"],
                "gpu_enabled": ctx.params["use_gpu"],
                "masks_enabled": bool(spec.mask_path),
                "sources": spec.source_count,
                "camera_groups": len(spec.feature_batches),
                "camera_models": sorted({batch.camera_model for batch in spec.feature_batches}),
            }
        )
        summary_path = ctx.stage_out_dir / "feature_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.outputs = [
            _file_ref(database_path, ctx),
            _file_ref(ctx.stage_out_dir / "input_spec.json", ctx),
            _file_ref(summary_path, ctx),
        ]
        for directory_name in ("images", "masks"):
            directory = ctx.stage_out_dir / directory_name
            if directory.exists():
                manifest.outputs.extend(
                    _staged_file_ref(path, ctx) for path in directory.rglob("*") if path.is_file()
                )
        manifest.extra = summary
        ctx.progress.info(
            f"features done: avg={summary['average_keypoints']:.0f}/image",
            progress=1.0,
            key="log.features_done",
            args={"average": round(summary["average_keypoints"]), "images": summary["images"]},
        )
        return manifest

    @staticmethod
    def _feature_args(params: dict, settings) -> list[str]:
        args = []
        if params["max_image_size"] > 0:
            args += ["--FeatureExtraction.max_image_size", str(params["max_image_size"])]
        if params["feature_type"] == "SIFT":
            if params["max_num_features"] > 0:
                args += ["--SiftExtraction.max_num_features", str(params["max_num_features"])]
            if params["sift_peak_threshold"] > 0:
                args += ["--SiftExtraction.peak_threshold", str(params["sift_peak_threshold"])]
            if params["sift_edge_threshold"] > 0:
                args += ["--SiftExtraction.edge_threshold", str(params["sift_edge_threshold"])]
            if params["sift_affine_dsp"]:
                args += [
                    "--SiftExtraction.estimate_affine_shape",
                    "1",
                    "--SiftExtraction.domain_size_pooling",
                    "1",
                ]
        else:
            if params["max_num_features"] > 0:
                args += ["--AlikedExtraction.max_num_features", str(params["max_num_features"])]
            configured = settings.aliked.extractor_path
            if configured and params["feature_type"] == "ALIKED_N16ROT":
                args += ["--AlikedExtraction.n16rot_model_path", configured]
        return args


def _feature_summary(database_path: Path) -> dict:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*), MIN(rows), AVG(rows), MAX(rows) FROM keypoints").fetchone()
        descriptors = connection.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0]
    return {
        "images": int(row[0]),
        "minimum_keypoints": int(row[1] or 0),
        "average_keypoints": float(row[2] or 0.0),
        "maximum_keypoints": int(row[3] or 0),
        "descriptor_images": int(descriptors),
    }


def _file_ref(path: Path, ctx: StageContext) -> FileRef:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    relative = Path(final_name) / path.relative_to(ctx.stage_out_dir)
    return FileRef(
        path=str(relative),
        size=path.stat().st_size,
        sha256=sha256_file(path),
        mime="application/json" if path.suffix == ".json" else "application/octet-stream",
    )


def _staged_file_ref(path: Path, ctx: StageContext) -> FileRef:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    relative = Path(final_name) / path.relative_to(ctx.stage_out_dir)
    return FileRef(path=str(relative), size=path.stat().st_size, sha256="")
