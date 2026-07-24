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
from ..colmap import web_preview
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest


@register
class ExportDataset(Stage):
    name = StageName.EXPORT_DATASET
    impl_version = "0.1"

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

        # 1) Web preview. これは project 直下 preview/ に置きたいが, ステージ出力は
        # stage_out_dir (export_dataset/) に集約するので, preview サブフォルダに置く.
        preview_dir = out / "preview"
        ctx.progress.info("building web preview (reconstruction.json + points.bin)", progress=0.2)
        wp = web_preview.write_web_preview(recon, preview_dir, max_points=ctx.params["max_preview_points"])
        ctx.progress.info(
            f"preview: {wp.num_points_written}/{wp.num_points_total} points", progress=0.5
        )
        for name in ("reconstruction.json", "points.bin"):
            p = preview_dir / name
            outputs.append(FileRef(path=_relpath(p, ctx), size=p.stat().st_size, sha256="", mime=None))

        # 2) 標準 COLMAP データセット.
        if ctx.params["include_dataset"]:
            ctx.progress.info("assembling standard COLMAP dataset", progress=0.6)
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

        manifest.outputs = outputs
        manifest.extra = {
            "preview_points": wp.num_points_written,
            "total_points": wp.num_points_total,
            **recon.summary(),
        }
        ctx.progress.info("export_dataset done", progress=1.0)
        return manifest


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
