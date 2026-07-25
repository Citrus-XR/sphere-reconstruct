"""export_dataset ステージ.

reconstruct の結果を 2 つの形で書き出す:

1. Web ビューア用プレビュー:
     <project>/preview/reconstruction.json
     <project>/preview/points.bin

2. 標準 COLMAP データセット (学習/再利用向け):
     <project>/export/images/...             (pinhole 画像 link/copy)
     <project>/export/sparse/0/*.bin         (COLMAP model)
     <project>/export/masks/...              (任意)

パラメータ:
  max_preview_points: int   プレビュー点群の上限 (default 500000)
  include_dataset: bool     標準データセットも書き出す (default True)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..colmap import model as colmap_model
from ..colmap import gravity_align, lichtfeld_config, train_profile, web_preview
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "0.4"  # equirect gut 修正

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        summary = ctx.project_dir / "reconstruct" / "model_summary.json"
        if summary.exists():
            return [
                FileRef(path=str(summary.relative_to(ctx.project_dir)), size=summary.stat().st_size, sha256=sha256_file(summary))
            ]
        return []

    def normalize_params(self, raw: dict) -> dict:
        return {
            "max_preview_points": int(raw.get("max_preview_points", 500_000)),
            "include_dataset": bool(raw.get("include_dataset", True)),
            "emit_train_configs": bool(raw.get("emit_train_configs", False)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        model_dir = ctx.project_dir / "reconstruct" / "sparse" / "0"
        if not (model_dir / "cameras.bin").exists():
            raise RuntimeError("reconstruct must run first (sparse/0 missing)")

        recon = colmap_model.read_model(model_dir)
        out = ctx.stage_out_dir
        outputs: list[FileRef] = []

        # 学習プロファイル (scene_scale / 点数 / 相机モデル / 品質) は回転不変なので align 前に取る.
        train_profile_data = (
            train_profile.compute_profile(recon) if ctx.params.get("emit_train_configs") else None
        )

        # 重力対齐: IMU 重力方向 (inspect_source) + カメラ姿勢から点群を起こす. viewer 用の
        # web preview にのみ適用 (COLMAP データセットは正準のまま). 重力が無い / 一致度が悪い
        # 場合は見送る.
        gravity_imu = _read_gravity_imu(ctx.project_dir)
        R_align, ga_info = gravity_align.compute_align_rotation(recon, gravity_imu)
        if R_align is not None:
            gravity_align.apply_alignment(recon, R_align)
            ctx.progress.info(
                f"gravity align applied: up={ga_info['up_world']} spread={ga_info['spread_deg']}deg",
                progress=0.15, key="log.export_gravity_align",
                args={"up": str(ga_info["up_world"]), "spread": ga_info["spread_deg"], "n": ga_info["front_images"]},
            )
        else:
            ctx.progress.info(
                f"gravity align skipped ({ga_info.get('reason')})",
                progress=0.15, key="log.export_gravity_skip",
                args={"reason": str(ga_info.get("reason"))},
            )

        # 1) Web preview. これは project 直下 preview/ に置きたいが, ステージ出力は
        # stage_out_dir (export_dataset/) に集約するので, preview サブフォルダに置く.
        preview_dir = out / "preview"
        ctx.progress.info(
            "building web preview (reconstruction.json + points.bin)",
            progress=0.2,
            key="log.export_web_preview",
        )
        wp = web_preview.write_web_preview(recon, preview_dir, max_points=ctx.params["max_preview_points"])
        ctx.progress.info(
            f"preview: {wp.num_points_written}/{wp.num_points_total} points",
            progress=0.5,
            key="log.export_preview_points",
            args={"written": wp.num_points_written, "total": wp.num_points_total},
        )
        for name in ("reconstruction.json", "points.bin"):
            p = preview_dir / name
            outputs.append(FileRef(path=_relpath(p, ctx), size=p.stat().st_size, sha256="", mime=None))

        # 2) 標準 COLMAP データセット.
        if ctx.params["include_dataset"]:
            ctx.progress.info(
                "assembling standard COLMAP dataset", progress=0.6, key="log.export_assemble_dataset"
            )
            ds = out / "dataset"
            ds_sparse = ds / "sparse" / "0"
            ds_images = ds / "images"
            ds_sparse.mkdir(parents=True, exist_ok=True)
            ds_images.mkdir(parents=True, exist_ok=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                shutil.copy2(model_dir / name, ds_sparse / name)
            # 画像は reconstruct/images をそのまま link/copy.
            recon_images = ctx.project_dir / "reconstruct" / "images"
            if recon_images.exists():
                for img_file in recon_images.rglob("*.jpg"):
                    rel = img_file.relative_to(recon_images)
                    dst = ds_images / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(img_file, dst)
            outputs.append(
                FileRef(path=_relpath(ds_sparse / "cameras.bin", ctx), size=(ds_sparse / "cameras.bin").stat().st_size, sha256="", mime=None)
            )

        # 3) LichtFeld-Studio 推奨 config (任意).
        if train_profile_data is not None:
            configs, cfg_info = lichtfeld_config.build_configs(train_profile_data)
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
            outputs.append(FileRef(path=_relpath(rec_path, ctx), size=rec_path.stat().st_size, sha256="", mime="application/json"))
            ctx.progress.info(
                f"train configs: cap_max={cfg_info['max_cap']}, camera={cfg_info['camera_class']}, "
                f"warnings={cfg_info['warnings']}",
                progress=0.9, key="log.export_train_configs",
                args={"cap": cfg_info["max_cap"], "camera": cfg_info["camera_class"], "warn": str(cfg_info["warnings"])},
            )

        manifest.outputs = outputs
        manifest.extra = {
            "preview_points": wp.num_points_written,
            "total_points": wp.num_points_total,
            **recon.summary(),
        }
        ctx.progress.info("export_dataset done", progress=1.0, key="log.export_done")
        return manifest


def _read_gravity_imu(project_dir: Path):
    """inspect_source/source.json から IMU 重力方向 (list[3]) を読む. 無ければ None."""
    src = project_dir / "inspect_source" / "source.json"
    if not src.exists():
        return None
    try:
        data = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    grav = data.get("gravity")
    if isinstance(grav, dict) and isinstance(grav.get("imu"), list) and len(grav["imu"]) == 3:
        return grav["imu"]
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
