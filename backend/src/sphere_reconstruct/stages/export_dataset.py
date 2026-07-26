"""export_dataset ステージ.

reconstruct の結果を 2 つの形で書き出す:

1. Web ビューア用プレビュー:
     <project>/preview/reconstruction.json
     <project>/preview/points.bin

2. LichtFeld Studio が直接選択できる COLMAP データセット:
     <project>/export_dataset/images/...
     <project>/export_dataset/masks/...      (任意)
     <project>/export_dataset/sparse/0/*

パラメータ:
  max_preview_points: int   プレビュー点群の上限 (default 500000)
  include_dataset: bool     標準データセットも書き出す (default True)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..colmap import lichtfeld_config, train_profile
from ..colmap import model as colmap_model
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "0.5"

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
            masks_written = _copy_masks(ctx.project_dir / "extract_features" / "masks", ds / "masks")
            validation = _validate_lf_dataset(ds, recon)
            export_manifest = {
                "format": "sphere-reconstruct-export",
                "version": 1,
                "load_in_lichtfeld_studio": ".",
                "camera_models": sorted({camera.model for camera in recon.cameras.values()}),
                "images": len(recon.images),
                "points3D": len(recon.points3D),
                "masks": masks_written,
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
            masks_written = 0

        # 3) LichtFeld-Studio 推奨 config (任意).
        if ctx.params["emit_train_configs"]:
            configs, cfg_info = lichtfeld_config.build_configs(
                train_profile_data, has_masks=masks_written > 0
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
    missing_images = [
        image.name for image in recon.images.values() if not (dataset_dir / "images" / image.name).is_file()
    ]
    summary = recon.summary()
    warnings: list[str] = []
    if summary.get("camera_trajectory_diameter", 0.0) < 1e-4 and len(recon.images) > 1:
        warnings.append("collapsed_camera_trajectory")
    if missing_images:
        warnings.append("missing_images")
    return {
        "loadable": not missing_images,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "camera_trajectory_diameter": summary.get("camera_trajectory_diameter", 0.0),
        "unique_camera_centers": summary.get("unique_camera_centers", 0),
        "warnings": warnings,
    }


def _relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
