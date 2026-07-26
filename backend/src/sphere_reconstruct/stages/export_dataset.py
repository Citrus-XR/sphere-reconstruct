"""export_dataset ステージ.

aligned reconstruction を LFStudio が直接選択できる dataset root として書き出す:

    <project>/export_dataset/images/...
    <project>/export_dataset/masks/...      (任意)
    <project>/export_dataset/sparse/0/*
    <project>/export_dataset/preview/*
    <project>/export_dataset/train_configs/* (任意)

パラメータ:
  max_preview_points: int   プレビュー点群の上限 (default 500000)
  include_dataset: bool     標準データセットも書き出す (default True)
  emit_train_configs: bool  LFStudio 推奨設定を書き出す (default True)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image as PilImage

from ..colmap import gravity_align, lichtfeld_config, train_profile
from ..colmap import model as colmap_model
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "0.7"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [
            ctx.project_dir / "manifests" / "align_reconstruction.json",
            ctx.project_dir / "manifests" / "extract_features.json",
            ctx.project_dir / "align_reconstruction" / "alignment.json",
            ctx.project_dir / "align_reconstruction" / "sparse" / "0" / "rigs.bin",
            ctx.project_dir / "align_reconstruction" / "sparse" / "0" / "cameras.bin",
            ctx.project_dir / "align_reconstruction" / "sparse" / "0" / "frames.bin",
            ctx.project_dir / "align_reconstruction" / "sparse" / "0" / "images.bin",
            ctx.project_dir / "align_reconstruction" / "sparse" / "0" / "points3D.bin",
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

    def normalize_params(self, raw: dict) -> dict:
        return {
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
            "include_dataset": bool(raw.get("include_dataset", True)),
            "emit_train_configs": bool(raw.get("emit_train_configs", True)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        model_dir = ctx.project_dir / "align_reconstruction" / "sparse" / "0"
        if not (model_dir / "cameras.bin").exists():
            raise RuntimeError("align_reconstruction must run first (sparse/0 missing)")

        recon = colmap_model.read_model(model_dir)
        out = ctx.stage_out_dir
        outputs: list[FileRef] = []
        train_profile_data = train_profile.compute_profile(recon)

        # 1) alignment stage が作った viewer preview を同梱する.
        preview_dir = out / "preview"
        ctx.progress.info(
            "building web preview (reconstruction.json + points.bin)",
            progress=0.2,
            key="log.export_web_preview",
        )
        shutil.copytree(ctx.project_dir / "align_reconstruction" / "preview", preview_dir)
        preview_points = min(len(recon.points3D), ctx.params["max_preview_points"])
        ctx.progress.info(
            f"preview: {preview_points}/{len(recon.points3D)} points",
            progress=0.5,
            key="log.export_preview_points",
            args={"written": preview_points, "total": len(recon.points3D)},
        )
        for name in ("reconstruction.json", "points.bin"):
            p = preview_dir / name
            outputs.append(FileRef(path=_relpath(p, ctx), size=p.stat().st_size, sha256="", mime=None))

        # 2) export_dataset 自体を LFStudio が直接選択できるデータセット root にする.
        if ctx.params["include_dataset"]:
            ctx.progress.info(
                "assembling standard COLMAP dataset", progress=0.6, key="log.export_assemble_dataset"
            )
            ds = out
            ds_sparse = ds / "sparse" / "0"
            ds_images = ds / "images"
            ds_sparse.mkdir(parents=True, exist_ok=True)
            ds_images.mkdir(parents=True, exist_ok=True)
            for source in model_dir.iterdir():
                if source.is_file() and source.suffix.lower() in {".bin", ".txt", ".ini"}:
                    shutil.copy2(source, ds_sparse / source.name)
            # 画像名は images.bin の相対 path をそのまま保つ.
            recon_images = ctx.project_dir / "extract_features" / "images"
            if recon_images.exists():
                image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                for img_file in recon_images.rglob("*"):
                    if not img_file.is_file() or img_file.suffix.lower() not in image_extensions:
                        continue
                    rel = img_file.relative_to(recon_images)
                    dst = ds_images / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(img_file, dst)
            mask_files_copied = _copy_masks(ctx.project_dir / "extract_features" / "masks", ds / "masks")
            validation = _validate_lf_dataset(ds, recon)
            if not validation["loadable"]:
                raise RuntimeError(
                    "LFStudio export 検証に失敗しました: " + ", ".join(validation["errors"])
                )
            export_manifest = {
                "format": "sphere-reconstruct-export",
                "version": 1,
                "load_in_lichtfeld_studio": ".",
                "camera_models": sorted({camera.model for camera in recon.cameras.values()}),
                "images": len(recon.images),
                "points3D": len(recon.points3D),
                "masks": validation["matched_mask_count"],
                "mask_files_copied": mask_files_copied,
                "image_source": "original",
                "validation": validation,
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
            cfg_info["usage"] = (
                "LichtFeld-Studio --config train_configs/train_config.mrnf.json "
                "--data-path <export_dataset>"
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
                },
            }
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
                progress=0.9,
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
            "camera_models": sorted({camera.model for camera in recon.cameras.values()}),
            "mask_files": validation["matched_mask_count"] if validation else 0,
            "validation": validation,
            "training_profile": train_profile_data,
            "training_recommendation": cfg_info,
            "lfstudio_training_metrics": "external",
        }
        ctx.progress.info("export_dataset done", progress=1.0, key="log.export_done")
        return manifest

def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_masks(source_dir: Path, destination_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    count = 0
    for source in source_dir.rglob("*.png"):
        destination = destination_dir / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(source, destination)
        count += 1
    return count


def _validate_lf_dataset(dataset_dir: Path, recon: colmap_model.Reconstruction) -> dict:
    images_root = dataset_dir / "images"
    masks_root = dataset_dir / "masks"
    missing_images: list[str] = []
    invalid_image_paths: list[str] = []
    corrupt_images: list[str] = []
    image_size_mismatches: list[dict] = []
    missing_camera_references: list[dict] = []
    valid_images: list[tuple[colmap_model.Image, Path, tuple[int, int]]] = []

    for image in recon.images.values():
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
    for image, relative, expected_size in valid_images:
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
    reference_centers = gravity_align.reference_camera_centers(recon)
    reference_diameter = gravity_align.reference_trajectory_diameter(recon)
    reference_unique = len(
        {tuple(round(value, 6) for value in center) for center in reference_centers}
    )
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
