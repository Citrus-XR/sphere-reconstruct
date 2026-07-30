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
  optimize_fisheye_training_images: bool  円形領域外を lossless crop (default True)
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

from ..colmap import gravity_align, lfstudio_compat, lichtfeld_config, train_profile, training_crop
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
from ..imaging import valid_region
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import ProgressSpan, Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "3.0"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "dense_initialization.json",
            ctx.project_dir / "restore_metric_scale" / "scale_restoration.json",
            ctx.project_dir / "manifests" / "extract_features.json",
            ctx.project_dir / "extract_features" / "input_spec.json",
            ctx.project_dir / "prepare_images" / "image_catalog.json",
            ctx.project_dir / "position_ground" / "ground_position.json",
            ctx.project_dir / "dense_initialization" / "sparse" / "0" / "rigs.bin",
            ctx.project_dir / "dense_initialization" / "sparse" / "0" / "cameras.bin",
            ctx.project_dir / "dense_initialization" / "sparse" / "0" / "frames.bin",
            ctx.project_dir / "dense_initialization" / "sparse" / "0" / "images.bin",
            ctx.project_dir / "dense_initialization" / "sparse" / "0" / "points3D.bin",
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
            "optimize_fisheye_training_images": bool(
                raw.get("optimize_fisheye_training_images", True)
            ),
            "lfstudio_stock_thin_prism_workaround": bool(
                raw.get("lfstudio_stock_thin_prism_workaround", True)
            ),
            "feature_masks_enabled": bool(raw.get("feature_masks_enabled", True)),
            "training_masks_enabled": bool(raw.get("training_masks_enabled", True)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params

        model_dir = ctx.project_dir / "dense_initialization" / "sparse" / "0"
        if not (model_dir / "cameras.bin").exists():
            raise RuntimeError("dense_initialization must run first (sparse/0 missing)")

        ctx.progress.info("reading aligned reconstruction", progress=0.0, key="log.export_start")
        recon = colmap_model.read_model(model_dir)
        spec = InputSpec.read(ctx.project_dir / "extract_features" / "input_spec.json")
        metric_scale_info = json.loads(
            (ctx.project_dir / "restore_metric_scale" / "scale_restoration.json").read_text(
                encoding="utf-8"
            )
        )
        ground_position_info = json.loads(
            (ctx.project_dir / "position_ground" / "ground_position.json").read_text(
                encoding="utf-8"
            )
        )
        catalog_path = ctx.project_dir / "prepare_images" / "image_catalog.json"
        catalog = (
            json.loads(catalog_path.read_text(encoding="utf-8"))
            if catalog_path.is_file()
            else None
        )
        export_recon = recon
        crop_plan: training_crop.CropPlan | None = None
        crop_info: dict = {"enabled": False, "requested": ctx.params["optimize_fisheye_training_images"]}
        if ctx.params["optimize_fisheye_training_images"]:
            jpegtran = training_crop.resolve_jpegtran(get_settings().binaries.jpegtran)
            if jpegtran is None:
                crop_info["skipped_reason"] = "jpegtran_unavailable"
                ctx.progress.warn(
                    "jpegtran unavailable; preserving original training images",
                    key="log.export_crop_unavailable",
                )
            elif catalog is not None:
                candidate = training_crop.build_plan(recon, catalog)
                if candidate.changed:
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
        compatibility_info = {
            "requested": ctx.params["lfstudio_stock_thin_prism_workaround"],
            "applied": False,
            "consumer": "lichtfeld_stock",
            "camera_reports": [],
        }
        if ctx.params["lfstudio_stock_thin_prism_workaround"]:
            if catalog is None:
                raise RuntimeError("LFStudio camera compatibility には image catalog が必要です")
            export_recon, compatibility_info = lfstudio_compat.apply_stock_camera_workarounds(
                export_recon,
                catalog,
                crop_plan,
            )
            compatibility_info["requested"] = True
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
        shutil.copytree(ctx.project_dir / "position_ground" / "preview", preview_dir)
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
            if crop_plan is not None or compatibility_info["applied"]:
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
            sorted_names = sorted(registered_names)
            image_copy_span = ProgressSpan(ctx.progress, 0.25, 0.45)
            if crop_plan is None:
                for image_number, name in enumerate(sorted_names, 1):
                    source = recon_images / name
                    if not source.is_file():
                        raise RuntimeError(f"registered training image is missing: {name}")
                    _link_or_copy(source, ds_images / name)
                    image_copy_span.tick(
                        image_number / max(1, len(sorted_names)),
                        message=f"copy training image {image_number}/{len(sorted_names)}",
                        key="log.export_copy_images",
                        args={"cur": image_number, "tot": len(sorted_names)},
                    )
            else:
                byte_stats = training_crop.crop_images(
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
                crop_info.update(byte_stats)
            mask_copy_span = ProgressSpan(ctx.progress, 0.45, 0.55)
            mask_files_copied = _copy_masks(
                ctx.project_dir,
                mask_purpose,
                ds / "masks",
                registered_names,
                crop_plan=crop_plan,
                progress=lambda current, total: mask_copy_span.tick(
                    current / max(1, total),
                    message=f"copy training mask {current}/{total}",
                    key="log.export_copy_masks",
                    args={"cur": current, "tot": total},
                ),
            )
            resolved_mask_source = (
                mask_purpose.value if mask_purpose is not None else "physical"
            )
            primary_prefix = f"sources/{spec.primary_source_id}/"
            if not any(name.startswith(primary_prefix) for name in registered_names):
                primary_prefix = None
            image_validation_span = ProgressSpan(ctx.progress, 0.55, 0.75)
            mask_validation_span = ProgressSpan(ctx.progress, 0.75, 0.9)
            validation = _validate_lf_dataset(
                ds,
                export_recon,
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
                "lfstudio_camera_compatibility": compatibility_info,
                "metric_scale": metric_scale_info,
                "ground_position": ground_position_info,
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
            cfg_info["camera_compatibility"] = compatibility_info
            if compatibility_info["applied"]:
                cfg_info["gui_integration"]["warnings"].append(
                    "stock_lfstudio_thin_prism_inverse_workaround_applied"
                )
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
            "lfstudio_camera_compatibility": compatibility_info,
            "metric_scale": metric_scale_info,
            "ground_position": ground_position_info,
        }
        ctx.progress.info("export_dataset done", progress=0.99, key="log.export_done")
        return manifest


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_masks(
    project_dir: Path,
    purpose: MaskPurpose | None,
    destination_dir: Path,
    image_names: set[str],
    crop_plan: training_crop.CropPlan | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    sorted_names = sorted(image_names)
    if purpose is None:
        catalog = json.loads(
            (project_dir / "prepare_images" / "image_catalog.json").read_text(encoding="utf-8")
        )
        records = {record["name"]: record for record in catalog["images"]}
        missing = [name for name in sorted_names if name not in records]
        if missing:
            raise RuntimeError(f"image catalog is missing registered masks: {missing[:5]}")
        templates: dict[str, Path] = {}
        for image_number, image_name in enumerate(sorted_names, 1):
            record = records[image_name]
            rect = crop_plan.images[image_name] if crop_plan is not None else None
            key = valid_region.cache_key(
                record.get("valid_region", {"kind": "full"}),
                int(record["width"]),
                int(record["height"]),
            ) + (f":crop={rect.as_list()}" if rect is not None else ":uncropped")
            destination = destination_dir / f"{image_name}.png"
            template = templates.get(key)
            if template is None:
                _write_physical_mask(record, destination, rect)
                templates[key] = destination
            else:
                _link_or_copy(template, destination)
            if progress is not None:
                progress(image_number, len(sorted_names))
        return len(sorted_names)

    records = records_by_name(load_mask_manifest(project_dir, purpose))
    missing = [name for name in sorted_names if name not in records]
    if missing:
        raise RuntimeError(f"{purpose.value} masks are missing registered images: {missing[:5]}")
    copies: list[
        tuple[Path, Path, training_crop.CropRect | None, tuple[int, int] | None]
    ] = []
    for image_name in sorted_names:
        record = records.get(image_name)
        if record is None:
            continue
        source = project_dir / record["path"]
        destination = destination_dir / f"{image_name}.png"
        rect = crop_plan.images[image_name] if crop_plan is not None else None
        expected_size = crop_plan.source_sizes[image_name] if crop_plan is not None else None
        copies.append((source, destination, rect, expected_size))
    workers = min(16, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for source, destination, rect, expected_size in copies:
            if rect is None:
                futures.append(pool.submit(_link_or_copy, source, destination))
            else:
                futures.append(
                    pool.submit(
                        training_crop.crop_mask_file,
                        source,
                        destination,
                        rect,
                        expected_size,
                    )
                )
        for image_number, future in enumerate(as_completed(futures), 1):
            future.result()
            if progress is not None:
                progress(image_number, len(copies))
    return len(copies)


def _write_physical_mask(
    record: dict,
    destination: Path,
    rect: training_crop.CropRect | None,
) -> None:
    mask = valid_region.render_mask(
        record.get("valid_region", {"kind": "full"}),
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
        try:
            with PilImage.open(image_path) as decoded:
                decoded.load()
                size = decoded.size
        except (OSError, ValueError):
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
        try:
            with PilImage.open(mask_path) as decoded:
                decoded.load()
                size = decoded.size
        except (OSError, ValueError):
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
