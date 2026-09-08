"""export_dataset ステージ.

aligned reconstruction を LFStudio が直接選択できる dataset root として書き出す:

    <project>/export_dataset/images/...
    <project>/export_dataset/masks/...      (SAM3 または physical valid region)
    <project>/export_dataset/sparse/0/*
    <project>/export_dataset/preview/*
    <project>/export_dataset/train_configs/* (任意)

パラメータ:
  max_preview_points: int   プレビュー点群の上限 (default 500000)
  include_dataset: bool     標準データセットも書き出す (default True)
  emit_train_configs: bool  LFStudio 推奨設定を書き出す (default True)
  optimize_fisheye_training_images: bool  有効領域外を PNG pixel / JPEG MCU crop (default True)
  feature_masks_enabled: bool   training mask 無効時の export fallback
  training_masks_enabled: bool  export で優先する SAM3 mask。両方無効なら physical mask
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image as PilImage

from ..colmap import gravity_align, lichtfeld_config, train_profile, training_crop
from ..colmap import model as colmap_model
from ..colmap.input_workspace import InputSpec
from ..domain.artifacts import FileRef, StageManifest
from ..domain.mask_artifact import (
    MaskPurpose,
    export_purpose,
    load_mask_manifest,
    mask_manifest_path,
    records_by_name,
    stage_for,
)
from ..domain.pipeline_state import StageName
from ..imaging import catalog_validity
from ..infrastructure.filesystem import sha256_file
from ..pipeline import prepared_images
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "3.2"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        model_dir, model_source = _select_export_model(ctx.project_dir)
        candidates = [
            ctx.project_dir / "manifests" / f"{model_source}.json",
            ctx.project_dir / "restore_metric_scale" / "scale_restoration.json",
            ctx.project_dir / "manifests" / "extract_features.json",
            ctx.project_dir / "extract_features" / "input_spec.json",
            ctx.project_dir / "manifests" / "rectify_fisheye.json",
            prepared_images.catalog_path(ctx.project_dir),
            ctx.project_dir / "scene_alignment" / "scene_alignment.json",
            *(
                model_dir / name
                for name in ("rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin")
            ),
        ]
        purpose = export_purpose(
            feature_enabled=ctx.params["feature_masks_enabled"],
            training_enabled=ctx.params["training_masks_enabled"],
        )
        if purpose is not None:
            stage = stage_for(purpose)
            candidates.extend(
                [
                    ctx.project_dir / "manifests" / f"{stage.value}.json",
                    mask_manifest_path(ctx.project_dir, purpose),
                ]
            )
        return [
            FileRef(
                path=str(path.relative_to(ctx.project_dir)),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.exists()
        ]

    def normalize_params(self, raw: dict) -> dict:
        return {
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
            "include_dataset": bool(raw.get("include_dataset", True)),
            "emit_train_configs": bool(raw.get("emit_train_configs", True)),
            "optimize_fisheye_training_images": bool(raw.get("optimize_fisheye_training_images", True)),
            "feature_masks_enabled": bool(raw.get("feature_masks_enabled", True)),
            "training_masks_enabled": bool(raw.get("training_masks_enabled", True)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params

        model_dir, model_source = _select_export_model(ctx.project_dir)

        ctx.progress.info("reading aligned reconstruction", progress=0.0, key="log.export_start")
        recon = colmap_model.read_model(model_dir)
        spec = InputSpec.read(ctx.project_dir / "extract_features" / "input_spec.json")
        metric_scale_info = json.loads(
            (ctx.project_dir / "restore_metric_scale" / "scale_restoration.json").read_text(encoding="utf-8")
        )
        scene_alignment_info = json.loads(
            (ctx.project_dir / "scene_alignment" / "scene_alignment.json").read_text(encoding="utf-8")
        )
        catalog = prepared_images.load_catalog(ctx.project_dir)
        export_recon = recon
        crop_plan: training_crop.CropPlan | None = None
        jpegtran = training_crop.resolve_jpegtran(get_settings().binaries.jpegtran)
        crop_info: dict = {"enabled": False, "requested": ctx.params["optimize_fisheye_training_images"]}
        if ctx.params["optimize_fisheye_training_images"]:
            if catalog is not None:
                candidate = training_crop.build_plan(
                    recon,
                    catalog,
                    project_dir=ctx.project_dir,
                )
                if candidate.changed:
                    if training_crop.requires_jpegtran(candidate) and jpegtran is None:
                        crop_info["skipped_reason"] = "jpegtran_unavailable_for_jpeg"
                        ctx.progress.warn(
                            "JPEG training crop requires jpegtran; preserving original images",
                            key="log.export_crop_unavailable",
                        )
                    else:
                        crop_plan = candidate
                        export_recon = training_crop.crop_reconstruction(recon, crop_plan)
                        crop_info.update(
                            {
                                "enabled": True,
                                "lossless": True,
                                "alignment_px": crop_plan.alignment_px,
                                "source_pixels": crop_plan.source_pixels,
                                "cropped_pixels": crop_plan.cropped_pixels,
                                "pixel_reduction_ratio": 1.0
                                - crop_plan.cropped_pixels / crop_plan.source_pixels,
                                "camera_rectangles": {
                                    str(camera_id): rect.as_list()
                                    for camera_id, rect in crop_plan.cameras.items()
                                },
                            }
                        )
                else:
                    crop_info["skipped_reason"] = "no_circular_padding"
            else:
                crop_info["skipped_reason"] = "image_catalog_unavailable"
        out = ctx.stage_out_dir
        outputs: list[FileRef] = []
        train_profile_data = train_profile.compute_profile(recon)
        mask_purpose = export_purpose(
            feature_enabled=ctx.params["feature_masks_enabled"],
            training_enabled=ctx.params["training_masks_enabled"],
        )
        if mask_purpose is not None and not mask_manifest_path(ctx.project_dir, mask_purpose).is_file():
            raise RuntimeError(f"{stage_for(mask_purpose).value} must run before export_dataset")
        ctx.progress.tick(0.1, message="export inputs loaded", key="log.export_inputs_loaded")

        # 1) alignment stage が作った viewer preview を同梱する.
        preview_dir = out / "preview"
        ctx.progress.info(
            "copying web preview (reconstruction.json + points.bin)",
            progress=0.12,
            key="log.export_web_preview",
        )
        shutil.copytree(ctx.project_dir / model_source / "preview", preview_dir)
        preview_points = min(len(recon.points3D), ctx.params["max_preview_points"])
        ctx.progress.info(
            f"preview: {preview_points}/{len(recon.points3D)} points",
            progress=0.2,
            key="log.export_preview_points",
            args={"written": preview_points, "total": len(recon.points3D)},
        )
        for name in ("reconstruction.json", "points.bin"):
            p = preview_dir / name
            outputs.append(FileRef(path=_relpath(p, ctx), size=p.stat().st_size, sha256="", mime=None))

        # 2) export_dataset 自体を LFStudio が直接選択できるデータセット root にする.
        if ctx.params["include_dataset"]:
            ctx.progress.info(
                "assembling standard COLMAP dataset", progress=0.22, key="log.export_assemble_dataset"
            )
            ds = out
            ds_sparse = ds / "sparse" / "0"
            ds_images = ds / "images"
            ds_sparse.mkdir(parents=True, exist_ok=True)
            ds_images.mkdir(parents=True, exist_ok=True)
            for source in model_dir.iterdir():
                if source.is_file() and source.suffix.lower() in {".bin", ".txt", ".ini"}:
                    shutil.copy2(source, ds_sparse / source.name)
            if crop_plan is not None:
                colmap_model.write_cameras_bin(ds_sparse / "cameras.bin", export_recon.cameras)
            if crop_plan is not None:
                model_crop_span = ProgressSpan(ctx.progress, 0.22, 0.25)
                colmap_model.write_images_bin(
                    ds_sparse / "images.bin",
                    export_recon.images,
                    progress=lambda current, total: model_crop_span.tick(
                        current / max(1, total),
                        message=f"rewrite cropped COLMAP model {current}/{total}",
                        key="log.export_rewrite_model",
                        args={"cur": current, "tot": total},
                    ),
                )
            # 画像名は images.bin の相対 path をそのまま保つ.
            recon_images = ctx.project_dir / "extract_features" / "images"
            registered_names = {image.name for image in recon.images.values()}
            expected_sizes = {
                image.name: (
                    export_recon.cameras[image.camera_id].width,
                    export_recon.cameras[image.camera_id].height,
                )
                for image in export_recon.images.values()
            }
            image_copy_span = ProgressSpan(ctx.progress, 0.25, 0.45)
            if crop_plan is None:
                image_sizes = _copy_registered_images(
                    recon_images,
                    ds_images,
                    expected_sizes,
                    progress=lambda current, total: image_copy_span.tick(
                        current / max(1, total),
                        message=f"copy training image {current}/{total}",
                        key="log.export_copy_images",
                        args={"cur": current, "tot": total},
                    ),
                )
            else:
                try:
                    byte_stats, image_sizes = training_crop.crop_images(
                        recon_images,
                        ds_images,
                        crop_plan,
                        jpegtran,
                        progress=lambda current, total: image_copy_span.tick(
                            current / max(1, total),
                            message=f"crop training image {current}/{total}",
                            key="log.export_crop_images",
                            args={"cur": current, "tot": total},
                        ),
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    raise RuntimeError(f"LFStudio export image materialization failed: {error}") from error
                crop_info.update(byte_stats)
            mask_copy_span = ProgressSpan(ctx.progress, 0.45, 0.55)
            mask_files_copied, mask_sizes = _copy_masks(
                ctx.project_dir,
                mask_purpose,
                ds / "masks",
                registered_names,
                expected_sizes,
                crop_plan=crop_plan,
                progress=lambda current, total: mask_copy_span.tick(
                    current / max(1, total),
                    message=f"copy training mask {current}/{total}",
                    key="log.export_copy_masks",
                    args={"cur": current, "tot": total},
                ),
            )
            resolved_mask_source = mask_purpose.value if mask_purpose is not None else "physical"
            primary_prefix = f"sources/{spec.primary_source_id}/"
            if not any(name.startswith(primary_prefix) for name in registered_names):
                primary_prefix = None
            image_validation_span = ProgressSpan(ctx.progress, 0.55, 0.75)
            mask_validation_span = ProgressSpan(ctx.progress, 0.75, 0.9)
            validation = _validate_lf_dataset(
                ds,
                export_recon,
                image_sizes=image_sizes,
                mask_sizes=mask_sizes,
                reference_prefix=primary_prefix,
                progress=lambda phase, current, total: (
                    image_validation_span if phase == "images" else mask_validation_span
                ).tick(
                    current / max(1, total),
                    message=f"validate {phase} {current}/{total}",
                    key="log.export_validate_progress",
                    args={"phase": phase, "cur": current, "tot": total},
                ),
            )
            source_registration = _source_registration(recon, spec)
            if not validation["loadable"]:
                raise RuntimeError("LFStudio export 検証に失敗しました: " + ", ".join(validation["errors"]))
            export_manifest = {
                "format": "sphere-reconstruct-export",
                "version": 3,
                "load_in_lichtfeld_studio": ".",
                "camera_models": sorted({camera.model for camera in export_recon.cameras.values()}),
                "images": len(recon.images),
                "points3D": len(recon.points3D),
                "masks": validation["matched_mask_count"],
                "mask_files_copied": mask_files_copied,
                "mask_source": resolved_mask_source,
                "image_source": "lossless_fisheye_crop" if crop_plan is not None else "original",
                "training_crop": crop_info,
                "fisheye_rectification": catalog.get("rectification"),
                "model_source": model_source,
                "metric_scale": metric_scale_info,
                "scene_alignment": scene_alignment_info,
                "validation": validation,
                "source_registration": source_registration,
            }
            export_manifest_path = ds / "export_manifest.json"
            export_manifest_path.write_text(
                json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for required in (ds_sparse / "cameras.bin", ds_sparse / "images.bin", export_manifest_path):
                outputs.append(
                    FileRef(path=_relpath(required, ctx), size=required.stat().st_size, sha256="", mime=None)
                )
            for directory_name in ("images", "masks"):
                directory = ds / directory_name
                if directory.exists():
                    outputs.extend(
                        FileRef(
                            path=_relpath(path, ctx),
                            size=path.stat().st_size,
                            sha256="",
                        )
                        for path in directory.rglob("*")
                        if path.is_file()
                    )
        else:
            validation = None

        # 3) LichtFeld-Studio 推奨 config (任意).
        cfg_info: dict | None = None
        if ctx.params["emit_train_configs"]:
            configs, cfg_info = lichtfeld_config.build_configs(
                train_profile_data,
                has_masks=bool(validation and validation["matched_mask_count"] > 0),
            )
            cfg_info["gui_integration"] = {
                "train_configs_auto_applied": False,
                "warnings": [
                    "lfstudio_gui_does_not_auto_apply_train_configs",
                    "select_mrnf_enable_gut_and_segment_masks_manually",
                ],
                "required_settings": {
                    "strategy": cfg_info["recommended_strategy"],
                    "gut": configs["mrnf"]["gut"],
                    "undistort": configs["mrnf"]["undistort"],
                    "mask_mode": configs["mrnf"]["mask_mode"],
                    "ppisp": configs["mrnf"]["use_ppisp"],
                    "ppisp_controller": configs["mrnf"]["ppisp_use_controller"],
                    "max_width": cfg_info["recommended_max_width"],
                },
            }
            cfg_info["fisheye_rectification"] = catalog.get("rectification")
            tc_dir = out / "train_configs"
            tc_dir.mkdir(parents=True, exist_ok=True)
            for cname, cfg in configs.items():
                (tc_dir / f"train_config.{cname}.json").write_text(
                    json.dumps(cfg, indent=2), encoding="utf-8"
                )
            rec_path = tc_dir / "recommendations.json"
            rec_path.write_text(
                json.dumps({"profile": train_profile_data, **cfg_info}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs.append(
                FileRef(
                    path=_relpath(rec_path, ctx),
                    size=rec_path.stat().st_size,
                    sha256="",
                    mime="application/json",
                )
            )
            ctx.progress.info(
                f"train configs: cap_max={cfg_info['max_cap']}, camera={cfg_info['camera_class']}, "
                f"warnings={cfg_info['warnings']}",
                progress=0.95,
                key="log.export_train_configs",
                args={
                    "cap": cfg_info["max_cap"],
                    "camera": cfg_info["camera_class"],
                    "warn": str(cfg_info["warnings"]),
                },
            )

        manifest.outputs = outputs
        manifest.extra = {
            "preview_points": preview_points,
            "total_points": len(recon.points3D),
            **recon.summary(),
            "camera_models": sorted({camera.model for camera in export_recon.cameras.values()}),
            "mask_files": validation["matched_mask_count"] if validation else 0,
            "mask_source": (
                mask_purpose.value
                if mask_purpose is not None
                else ("physical" if validation and validation["matched_mask_count"] else None)
            ),
            "validation": validation,
            "training_profile": train_profile_data,
            "training_recommendation": cfg_info,
            "lfstudio_training_metrics": "external",
            "source_registration": _source_registration(recon, spec),
            "training_crop": crop_info,
            "fisheye_rectification": catalog.get("rectification"),
            "model_source": model_source,
            "metric_scale": metric_scale_info,
            "scene_alignment": scene_alignment_info,
        }
        ctx.progress.info("export_dataset done", progress=0.99, key="log.export_done")
        return manifest


def _select_export_model(project_dir: Path) -> tuple[Path, str]:
    for stage in (
        StageName.DENSE_INITIALIZATION,
        StageName.CLEANUP_SPARSE,
        StageName.SCENE_ALIGNMENT,
    ):
        model_dir = project_dir / stage.value / "sparse" / "0"
        if (model_dir / "cameras.bin").is_file():
            return model_dir, stage.value
    raise RuntimeError("cleanup_sparse or scene_alignment must run before export_dataset")


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_registered_images(
    source_root: Path,
    destination_root: Path,
    expected_sizes: dict[str, tuple[int, int]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, tuple[int, int]]:
    tasks: list[tuple[str, Path, Path, tuple[int, int]]] = []
    for name in sorted(expected_sizes):
        relative = _safe_relative_path(name)
        if relative is None:
            raise RuntimeError(f"LFStudio export image validation failed: invalid path {name}")
        tasks.append((name, source_root / relative, destination_root / relative, expected_sizes[name]))

    output_sizes: dict[str, tuple[int, int]] = {}
    workers = max(1, min(len(tasks), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_validate_and_link, source, destination, expected_size, "image"): name
            for name, source, destination, expected_size in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            output_sizes[futures[future]] = future.result()
            if progress is not None:
                progress(index, len(tasks))
    return output_sizes


def _validate_and_link(
    source: Path,
    destination: Path,
    expected_size: tuple[int, int],
    kind: str,
) -> tuple[int, int]:
    if not source.is_file():
        raise RuntimeError(f"LFStudio export {kind} validation failed: missing file {source}")
    try:
        with PilImage.open(source) as decoded:
            decoded.load()
            actual_size = decoded.size
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"LFStudio export {kind} validation failed: corrupt file {source}") from error
    if actual_size != expected_size:
        raise RuntimeError(
            f"LFStudio export {kind} validation failed: "
            f"{source} is {actual_size[0]}x{actual_size[1]}, "
            f"expected {expected_size[0]}x{expected_size[1]}"
        )
    _link_or_copy(source, destination)
    return actual_size


def _copy_masks(
    project_dir: Path,
    purpose: MaskPurpose | None,
    destination_dir: Path,
    image_names: set[str],
    expected_sizes: dict[str, tuple[int, int]],
    crop_plan: training_crop.CropPlan | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, dict[str, tuple[int, int]]]:
    sorted_names = sorted(image_names)
    output_sizes: dict[str, tuple[int, int]] = {}
    if purpose is None:
        catalog = prepared_images.load_catalog(project_dir)
        records = {record["name"]: record for record in catalog["images"]}
        missing = [name for name in sorted_names if name not in records]
        if missing:
            raise RuntimeError(f"image catalog is missing registered masks: {missing[:5]}")
        templates: dict[str, Path] = {}
        for image_number, image_name in enumerate(sorted_names, 1):
            record = records[image_name]
            rect = crop_plan.images[image_name] if crop_plan is not None else None
            key = catalog_validity.cache_key(
                record,
                int(record["width"]),
                int(record["height"]),
            ) + (f":crop={rect.as_list()}" if rect is not None else ":uncropped")
            destination = destination_dir / f"{image_name}.png"
            template = templates.get(key)
            if template is None:
                _write_physical_mask(project_dir, record, destination, rect)
                templates[key] = destination
            else:
                _link_or_copy(template, destination)
            output_sizes[image_name] = (
                (rect.width, rect.height)
                if rect is not None
                else (int(record["width"]), int(record["height"]))
            )
            if progress is not None:
                progress(image_number, len(sorted_names))
        return len(sorted_names), output_sizes

    records = records_by_name(load_mask_manifest(project_dir, purpose))
    missing = [name for name in sorted_names if name not in records]
    if missing:
        raise RuntimeError(f"{purpose.value} masks are missing registered images: {missing[:5]}")
    copies: list[
        tuple[str, Path, Path, training_crop.CropRect | None, tuple[int, int]]
    ] = []
    for image_name in sorted_names:
        record = records.get(image_name)
        if record is None:
            continue
        source = project_dir / record["path"]
        destination = destination_dir / f"{image_name}.png"
        rect = crop_plan.images[image_name] if crop_plan is not None else None
        source_size = (
            crop_plan.source_sizes[image_name]
            if crop_plan is not None
            else expected_sizes[image_name]
        )
        copies.append((image_name, source, destination, rect, source_size))
    workers = max(1, min(len(copies), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for image_name, source, destination, rect, source_size in copies:
            if rect is None:
                future = pool.submit(
                    _validate_and_link,
                    source,
                    destination,
                    source_size,
                    "mask",
                )
            else:
                future = pool.submit(
                    _crop_mask,
                    source,
                    destination,
                    rect,
                    source_size,
                )
            futures[future] = image_name
        for image_number, future in enumerate(as_completed(futures), 1):
            output_sizes[futures[future]] = future.result()
            if progress is not None:
                progress(image_number, len(copies))
    return len(copies), output_sizes


def _crop_mask(
    source: Path,
    destination: Path,
    rect: training_crop.CropRect,
    source_size: tuple[int, int],
) -> tuple[int, int]:
    try:
        training_crop.crop_mask_file(source, destination, rect, source_size)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"LFStudio export mask materialization failed: {error}") from error
    return rect.width, rect.height


def _write_physical_mask(
    project_dir: Path,
    record: dict,
    destination: Path,
    rect: training_crop.CropRect | None,
) -> None:
    mask = catalog_validity.render_mask(
        project_dir,
        record,
        int(record["width"]),
        int(record["height"]),
    )
    if rect is not None:
        mask = mask[rect.top : rect.bottom, rect.left : rect.right]
    destination.parent.mkdir(parents=True, exist_ok=True)
    PilImage.fromarray(mask * 255, mode="L").save(destination, format="PNG", compress_level=6)


def _validate_lf_dataset(
    dataset_dir: Path,
    recon: colmap_model.Reconstruction,
    *,
    image_sizes: dict[str, tuple[int, int]],
    mask_sizes: dict[str, tuple[int, int]],
    reference_prefix: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    images_root = dataset_dir / "images"
    masks_root = dataset_dir / "masks"
    missing_images: list[str] = []
    invalid_image_paths: list[str] = []
    corrupt_images: list[str] = []
    image_size_mismatches: list[dict] = []
    missing_camera_references: list[dict] = []
    valid_images: list[tuple[colmap_model.Image, Path, tuple[int, int]]] = []

    reconstruction_images = list(recon.images.values())
    for image_number, image in enumerate(reconstruction_images, 1):
        if progress is not None:
            progress("images", image_number, len(reconstruction_images))
        relative = _safe_relative_path(image.name)
        if relative is None:
            invalid_image_paths.append(image.name)
            continue
        image_path = images_root / relative
        if not image_path.is_file():
            missing_images.append(image.name)
            continue
        camera = recon.cameras.get(image.camera_id)
        if camera is None:
            missing_camera_references.append({"image": image.name, "camera_id": image.camera_id})
            continue
        size = image_sizes.get(image.name)
        if size is None:
            corrupt_images.append(image.name)
            continue
        expected = (camera.width, camera.height)
        if size != expected:
            image_size_mismatches.append(
                {"image": image.name, "actual": list(size), "expected": list(expected)}
            )
            continue
        valid_images.append((image, relative, size))

    unsupported_cameras = [
        {"camera_id": camera.camera_id, "model_id": camera.model_id, "model": camera.model}
        for camera in recon.cameras.values()
        if camera.model_id not in colmap_model.LFSTUDIO_SUPPORTED_CAMERA_MODEL_IDS
    ]

    matched_masks = 0
    missing_masks: list[str] = []
    corrupt_masks: list[str] = []
    mask_size_mismatches: list[dict] = []
    for image_number, (image, relative, expected_size) in enumerate(valid_images, 1):
        if progress is not None:
            progress("masks", image_number, len(valid_images))
        mask_path = _find_mask_path(masks_root, relative)
        if mask_path is None:
            missing_masks.append(image.name)
            continue
        size = mask_sizes.get(image.name)
        if size is None:
            corrupt_masks.append(image.name)
            continue
        if size != expected_size:
            mask_size_mismatches.append(
                {"image": image.name, "actual": list(size), "expected": list(expected_size)}
            )
            continue
        matched_masks += 1

    mask_file_count = (
        sum(1 for path in masks_root.rglob("*.png") if path.is_file()) if masks_root.exists() else 0
    )
    summary = recon.summary()
    reference_centers = gravity_align.reference_camera_centers(recon, reference_prefix)
    reference_diameter = gravity_align.reference_trajectory_diameter(recon, reference_prefix)
    reference_unique = len({tuple(round(value, 6) for value in center) for center in reference_centers})
    errors: list[str] = []
    for code, values in (
        ("unsupported_camera_models", unsupported_cameras),
        ("invalid_image_paths", invalid_image_paths),
        ("missing_camera_references", missing_camera_references),
        ("missing_images", missing_images),
        ("corrupt_images", corrupt_images),
        ("image_size_mismatches", image_size_mismatches),
        ("corrupt_masks", corrupt_masks),
        ("mask_size_mismatches", mask_size_mismatches),
    ):
        if values:
            errors.append(code)
    warnings: list[str] = []
    if reference_diameter < 1e-4 and len(reference_centers) > 1:
        warnings.append("collapsed_reference_camera_trajectory")
    if 0 < matched_masks < len(recon.images):
        warnings.append("partial_registered_masks")
    if mask_file_count > matched_masks:
        warnings.append("unmatched_mask_files")
    return {
        "loadable": not errors,
        "training_ready": not errors and "collapsed_reference_camera_trajectory" not in warnings,
        "errors": errors,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "invalid_image_path_count": len(invalid_image_paths),
        "corrupt_image_count": len(corrupt_images),
        "image_size_mismatch_count": len(image_size_mismatches),
        "unsupported_cameras": unsupported_cameras,
        "matched_mask_count": matched_masks,
        "missing_mask_count": len(missing_masks),
        "corrupt_mask_count": len(corrupt_masks),
        "mask_size_mismatch_count": len(mask_size_mismatches),
        "mask_files_copied": mask_file_count,
        "verification_mode": "materialization",
        "verified_image_count": len(image_sizes),
        "verified_mask_count": len(mask_sizes),
        "reference_camera_trajectory_diameter": reference_diameter,
        "reference_unique_camera_centers": reference_unique,
        "camera_trajectory_diameter": summary.get("camera_trajectory_diameter", 0.0),
        "unique_camera_centers": summary.get("unique_camera_centers", 0),
        "warnings": warnings,
    }


def _source_registration(recon: colmap_model.Reconstruction, spec: InputSpec) -> dict:
    source_by_name = {image["name"]: image["source_id"] for image in spec.images}
    source_metadata = {source["id"]: source for source in spec.sources}
    result = {}
    for source_id in sorted({image["source_id"] for image in spec.images}):
        total = sum(image["source_id"] == source_id for image in spec.images)
        registered = sum(source_by_name.get(image.name) == source_id for image in recon.images.values())
        result[source_id] = {
            "label": source_metadata[source_id]["label"],
            "role": source_metadata[source_id]["role"],
            "total": total,
            "registered": registered,
            "ratio": registered / max(1, total),
            "connected": registered > 0,
        }
    return result


def _safe_relative_path(name: str) -> Path | None:
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if not posix.parts or posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        return None
    if any(part in {"", "."} for part in posix.parts):
        return None
    return Path(*posix.parts)


def _find_mask_path(root: Path, image_relative: Path) -> Path | None:
    candidates = [root / image_relative]
    candidates.append(root / image_relative.with_suffix(".png"))
    candidates.append(root / image_relative.with_suffix(".mask.png"))
    candidates.append(root / f"{image_relative}.png")
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
