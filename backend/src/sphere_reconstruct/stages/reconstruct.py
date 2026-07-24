"""reconstruct ステージ (COLMAP).

reproject_views の pinhole 画像 (+ 任意で generate_masks の mask) から COLMAP で
sparse reconstruction を作る.

V1 固定パイプライン (spec §7):
  feature_extractor (PINHOLE, single_camera, SIFT, GPU)
  -> sequential_matcher (overlap 窓, loop detection は任意)
  -> mapper (incremental)

COLMAP 作業ツリー:
  <project>/reconstruct/images/frame_XXXXXX/<view>_lens<idx>.jpg   (link or copy)
  <project>/reconstruct/masks/ frame_XXXXXX/<view>_lens<idx>.jpg.png (任意, COLMAP 命名規則)
  <project>/reconstruct/database.db
  <project>/reconstruct/sparse/0/{cameras,images,points3D}.bin
  <project>/reconstruct/logs/*.log
  <project>/reconstruct/model_summary.json

パラメータ:
  matcher: "sequential" | "exhaustive"   (default sequential)
  overlap: int                            sequential の窓 (default 10)
  use_masks: bool                         generate_masks 出力を使う (default True)
  use_gpu: bool                           (default True)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings
from ..colmap import model as colmap_model
from ..colmap import runner as colmap_runner


@register
class Reconstruct(Stage):
    name = StageName.RECONSTRUCT
    impl_version = "0.1"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        rig = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        masks = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        out = []
        for p in (rig, masks):
            if p.exists():
                out.append(
                    FileRef(path=str(p.relative_to(ctx.project_dir)), size=p.stat().st_size, sha256=sha256_file(p))
                )
        return out

    def normalize_params(self, raw: dict) -> dict:
        return {
            "matcher": str(raw.get("matcher", "sequential")),
            "overlap": int(raw.get("overlap", 10)),
            "use_masks": bool(raw.get("use_masks", True)),
            "use_gpu": bool(raw.get("use_gpu", True)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        settings = get_settings()
        colmap_bin = colmap_runner.resolve_colmap_bin(settings.binaries.colmap or None)

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        rig_path = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        if not rig_path.exists():
            raise RuntimeError("reproject_views must run first")
        rig = json.loads(rig_path.read_text())

        out = ctx.stage_out_dir
        images_dir = out / "images"
        masks_dir = out / "masks"
        sparse_dir = out / "sparse"
        logs_dir = out / "logs"
        for d in (images_dir, sparse_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1) pinhole 画像を作業ツリーへ link/copy. mask があれば COLMAP 命名で mirror.
        masks_manifest = None
        mm_path = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        use_masks = ctx.params["use_masks"] and mm_path.exists()
        if use_masks:
            masks_manifest = json.loads(mm_path.read_text())
            masks_dir.mkdir(parents=True, exist_ok=True)

        n_imgs = 0
        for fr in rig["frames"]:
            for v in fr["views"]:
                src_img = ctx.project_dir / v["path"]
                rel = Path(v["path"]).relative_to("reproject_views")  # frame_XXXXXX/<view>_lensN.jpg
                dst_img = images_dir / rel
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(src_img, dst_img)
                n_imgs += 1

        if use_masks and masks_manifest is not None:
            for fr in masks_manifest["frames"]:
                for v in fr["views"]:
                    src_mask = ctx.project_dir / v["path"]  # generate_masks/frame_XXXXXX/<view>_lensN.png
                    rel_png = Path(v["path"]).relative_to("generate_masks")  # frame_XXXXXX/<view>_lensN.png
                    # COLMAP mask 命名: <image_name>.png. image は .jpg なので <...>.jpg.png.
                    stem = rel_png.stem  # <view>_lensN
                    dst_mask = masks_dir / rel_png.parent / f"{stem}.jpg.png"
                    dst_mask.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(src_mask, dst_mask)

        db_path = out / "database.db"
        ctx.progress.info(f"colmap feature_extractor on {n_imgs} images", progress=0.05)

        def logline(prefix: str):
            def _cb(line: str) -> None:
                if line.strip():
                    ctx.progress.info(f"[{prefix}] {line}")
            return _cb

        # 2) feature extraction (mask があれば mask_path 付き).
        colmap_runner.feature_extractor(
            colmap_bin,
            database_path=db_path,
            image_path=images_dir,
            camera_model="PINHOLE",
            single_camera=True,
            use_gpu=ctx.params["use_gpu"],
            mask_path=masks_dir if use_masks else None,
            log_path=logs_dir / "feature_extractor.log",
            on_line=logline("features"),
        )

        ctx.progress.info("colmap matching", progress=0.3)
        # 3) matching.
        if ctx.params["matcher"] == "exhaustive":
            colmap_runner.exhaustive_matcher(
                colmap_bin, database_path=db_path, use_gpu=ctx.params["use_gpu"],
                log_path=logs_dir / "matcher.log", on_line=logline("match"),
            )
        else:
            colmap_runner.sequential_matcher(
                colmap_bin, database_path=db_path, overlap=ctx.params["overlap"],
                use_gpu=ctx.params["use_gpu"],
                log_path=logs_dir / "matcher.log", on_line=logline("match"),
            )

        ctx.progress.info("colmap mapper (incremental)", progress=0.55)
        # 4) mapper.
        colmap_runner.mapper(
            colmap_bin,
            database_path=db_path,
            image_path=images_dir,
            output_path=sparse_dir,
            log_path=logs_dir / "mapper.log",
            on_line=logline("mapper"),
        )

        # 5) model summary.
        model_dir = sparse_dir / "0"
        if not (model_dir / "cameras.bin").exists():
            raise RuntimeError("mapper produced no reconstruction (sparse/0 missing)")
        recon = colmap_model.read_model(model_dir)
        summary = recon.summary()
        summary["registered_ratio"] = (
            summary["num_images"] / n_imgs if n_imgs else 0.0
        )
        (out / "model_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ctx.progress.info(
            f"reconstruction: {summary['num_images']}/{n_imgs} images, "
            f"{summary['num_points3D']} points, "
            f"mean err {summary['mean_reprojection_error']:.3f}px",
            progress=0.95,
        )

        # manifest outputs: model bin + summary (画像/mask は link なので個別追加しない).
        outputs = []
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            p = model_dir / name
            outputs.append(
                FileRef(path=_relpath(p, ctx), size=p.stat().st_size, sha256="", mime="application/octet-stream")
            )
        sp = out / "model_summary.json"
        outputs.append(FileRef(path=_relpath(sp, ctx), size=sp.stat().st_size, sha256=sha256_file(sp), mime="application/json"))
        manifest.outputs = outputs
        manifest.extra = summary

        ctx.progress.info("reconstruct done", progress=1.0)
        return manifest


def _link_or_copy(src: Path, dst: Path) -> None:
    """同一ボリュームなら hardlink, 失敗したら copy. 既存ならスキップ."""
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
