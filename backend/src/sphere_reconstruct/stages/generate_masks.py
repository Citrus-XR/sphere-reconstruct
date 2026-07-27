"""Canonical image catalog へ独立した feature / training SAM3 mask を生成する。"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..domain.artifacts import FileRef, StageManifest
from ..domain.mask_artifact import (
    MASK_MANIFEST_VERSION,
    MaskPurpose,
    append_partial_mask_record,
    initialise_partial_mask_manifest,
    remove_partial_mask_manifest,
)
from ..domain.pipeline_state import StageName
from ..imaging import masks as mask_utils
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest
from ..settings import get_settings


class _GenerateMasks(Stage):
    purpose: MaskPurpose
    impl_version = "1.2"

    def normalize_params(self, raw: dict) -> dict:
        settings = get_settings().sam3
        default_prompt = (
            settings.feature_prompt
            if self.purpose is MaskPurpose.FEATURE
            else settings.training_prompt
        )
        return {
            "prompt": str(raw.get("prompt", default_prompt)),
            "max_inference_size": int(raw.get("max_inference_size", settings.max_inference_size)),
            "dilate_px": int(raw.get("dilate_px", 8)),
            "coverage_warn": float(raw.get("coverage_warn", 0.5)),
            "max_images": int(raw.get("max_images", 0)),
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "prepare_images.json",
            ctx.project_dir / "prepare_images" / "image_catalog.json",
        ]
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.is_file()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        import cv2  # noqa: PLC0415

        catalog_path = ctx.project_dir / "prepare_images" / "image_catalog.json"
        if not catalog_path.is_file():
            raise RuntimeError(f"prepare_images must run before {self.name.value}")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        images = catalog["images"]
        if ctx.params["max_images"] > 0:
            images = images[: ctx.params["max_images"]]
        prompts = [term.strip() for term in ctx.params["prompt"].split(",") if term.strip()]
        manifest_base = {
            "version": MASK_MANIFEST_VERSION,
            "purpose": self.purpose.value,
            "revision": uuid4().hex,
            "prompt": prompts,
            "max_inference_size": ctx.params["max_inference_size"],
            "dilate_px": ctx.params["dilate_px"],
            "total_images": len(images),
        }
        partial_records_path = initialise_partial_mask_manifest(ctx.stage_out_dir, manifest_base)

        engine = None
        if prompts:
            from ..sam3.engine import Sam3Engine  # noqa: PLC0415

            settings = get_settings().sam3
            ctx.progress.info(
                "SAM3 model を読み込み中",
                progress=0.01,
                key="log.mask_loading_model",
                args={"device": settings.device},
            )
            engine = Sam3Engine.from_settings()
            engine.load()
            ctx.progress.info("SAM3 model loaded", progress=0.02, key="log.mask_model_loaded")
        else:
            ctx.progress.info(
                "prompt 空: SAM3 をスキップし valid-region mask だけを生成",
                progress=0.02,
                key="log.mask_prompt_empty",
            )

        records = []
        outputs = []
        try:
            for number, image_record in enumerate(images, 1):
                image_span = ProgressSpan(
                    ctx.progress,
                    0.02 + 0.96 * (number - 1) / max(1, len(images)),
                    0.02 + 0.96 * number / max(1, len(images)),
                )
                image_span.tick(
                    0.02,
                    message=f"mask image {number}/{len(images)}",
                    key="log.mask_progress_image",
                    args={"cur": number, "tot": len(images), "name": image_record["name"]},
                )
                source_path = ctx.project_dir / image_record["path"]
                bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"cannot read prepared image: {source_path}")
                height, width = bgr.shape[:2]
                plan = mask_utils.plan_downsample(width, height, ctx.params["max_inference_size"])
                small_rgb = cv2.cvtColor(mask_utils.downsample_image(bgr, plan), cv2.COLOR_BGR2RGB)

                def prompt_progress(
                    current: int,
                    total: int,
                    *,
                    span: ProgressSpan = image_span,
                    name: str = image_record["name"],
                ) -> None:
                    span.tick(
                        0.05 + 0.85 * current / max(1, total),
                        message=f"prompt {current}/{total}: {name}",
                        key="log.mask_prompt_progress",
                        args={"cur": current, "tot": total, "name": name},
                    )

                detections = (
                    engine.detect(
                        small_rgb,
                        prompts,
                        progress=prompt_progress,
                    )
                    if engine is not None
                    else []
                )
                union = mask_utils.union_masks([mask for detection in detections for mask in detection.masks])
                dynamic = (
                    np.zeros((height, width), np.uint8)
                    if union is None
                    else mask_utils.upscale_mask(union, width, height)
                )
                dynamic = mask_utils.dilate_mask(dynamic, ctx.params["dilate_px"])
                valid, coverage = _compose_valid_mask(image_record["valid_region"], dynamic, width, height)
                output_path = ctx.stage_out_dir / f"{image_record['name']}.png"
                mask_utils.write_mask_png(valid, output_path, invert=False)
                warning = coverage > ctx.params["coverage_warn"]
                if warning:
                    ctx.progress.warn(
                        f"{image_record['name']}: dynamic coverage {coverage:.2f}",
                        key="log.mask_coverage_warn_image",
                        args={"name": image_record["name"], "cov": round(coverage, 2)},
                    )
                record = {
                    "name": image_record["name"],
                    "source_id": image_record["source_id"],
                    "capture_index": image_record["capture_index"],
                    "path": _final_relpath(output_path, ctx),
                    "coverage": coverage,
                    "coverage_warning": warning,
                    "detections": {
                        detection.prompt: len(detection.masks) for detection in detections
                    },
                }
                records.append(record)
                append_partial_mask_record(partial_records_path, record)
                outputs.append(_file_ref(output_path, ctx, "image/png"))
                image_span.tick(
                    1.0,
                    message=f"mask {number}/{len(images)}",
                    key="log.mask_progress_image",
                    args={"cur": number, "tot": len(images), "name": image_record["name"]},
                )
        finally:
            if engine is not None:
                engine.unload()

        mask_manifest = {
            **manifest_base,
            "complete": True,
            "generated_images": len(records),
            "images": records,
        }
        manifest_path = ctx.stage_out_dir / "manifest_masks.json"
        manifest_path.write_text(
            json.dumps(mask_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        remove_partial_mask_manifest(ctx.stage_out_dir)
        outputs.append(_file_ref(manifest_path, ctx, "application/json"))
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params
        manifest.outputs = outputs
        manifest.extra = _mask_statistics(mask_manifest)
        ctx.progress.info(f"{self.name.value} done", progress=0.99, key="log.mask_done")
        return manifest


@register
class GenerateFeatureMasks(_GenerateMasks):
    name = StageName.GENERATE_FEATURE_MASKS
    purpose = MaskPurpose.FEATURE


@register
class GenerateTrainingMasks(_GenerateMasks):
    name = StageName.GENERATE_TRAINING_MASKS
    purpose = MaskPurpose.TRAINING


def _compose_valid_mask(region: dict, dynamic: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    if region["kind"] == "circle":
        circle = (region["cx"] * width, region["cy"] * height, region["r"] * width)
        valid = mask_utils.valid_region_mask(width, height, circle, exclude=dynamic)
        circle_mask = mask_utils.circle_mask(width, height, *circle)
        valid_area = int(circle_mask.sum()) or 1
        coverage = float(((circle_mask > 0) & (dynamic > 0)).sum()) / valid_area
        return valid, coverage
    return np.where(dynamic > 0, 0, 255).astype(np.uint8), mask_utils.coverage_ratio(dynamic)


def _mask_statistics(manifest: dict) -> dict:
    records = manifest["images"]
    coverages = [float(record["coverage"]) for record in records]
    return {
        "purpose": manifest["purpose"],
        "images": len(records),
        "sources": len({record["source_id"] for record in records}),
        "prompt_terms": len(manifest["prompt"]),
        "detections": sum(
            sum(int(count) for count in record["detections"].values()) for record in records
        ),
        "average_dynamic_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "maximum_dynamic_coverage": max(coverages) if coverages else 0.0,
        "coverage_warnings": sum(record["coverage_warning"] for record in records),
        "max_inference_size": manifest["max_inference_size"],
        "dilate_px": manifest["dilate_px"],
    }


def _final_relpath(path: Path, ctx: StageContext) -> str:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_name) / path.relative_to(ctx.stage_out_dir))


def _file_ref(path: Path, ctx: StageContext, mime: str) -> FileRef:
    return FileRef(
        path=_final_relpath(path, ctx),
        size=path.stat().st_size,
        sha256=sha256_file(path) if path.suffix == ".json" else "",
        mime=mime,
    )
