"""generate_masks ステージ.

reproject_views の出力 (pinhole rig 画像) に対して SAM3 を実行し, 各画像の
「除外すべき物体 (人 / 自撮り棒 / 三脚 / 影)」の mask を生成する.

8K 由来の pinhole が大きい場合に備え, SAM3 には縮小画像を渡し, 得た mask を
元解像度へ bilinear 拡大する (imaging/masks の plan_downsample / upscale_mask).

出力:
  <project>/generate_masks/frame_XXXXXX/<view>_lens<idx>.png   COLMAP 用 mask
                                                                (除外物体=0, 使用=255)
  <project>/generate_masks/manifest_masks.json                 coverage 等の統計

パラメータ:
  prompt: str            comma 区切り. 未指定なら settings.sam3.default_prompt.
  max_inference_size: int  未指定なら settings.sam3.max_inference_size.
  coverage_warn: float   この面積比を超えたら警告 (default 0.5).
  max_frames: int        0 = 全部.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import masks as mask_utils
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class GenerateMasks(Stage):
    name = StageName.GENERATE_MASKS
    impl_version = "0.1"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        rig = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        if not rig.exists():
            return []
        return [
            FileRef(path=str(rig.relative_to(ctx.project_dir)), size=rig.stat().st_size, sha256=sha256_file(rig))
        ]

    def normalize_params(self, raw: dict) -> dict:
        s = get_settings().sam3
        return {
            "prompt": str(raw.get("prompt", s.default_prompt)),
            "max_inference_size": int(raw.get("max_inference_size", s.max_inference_size)),
            "coverage_warn": float(raw.get("coverage_warn", 0.5)),
            "max_frames": int(raw.get("max_frames", 0)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        import cv2  # noqa: PLC0415

        from ..sam3.engine import Sam3Engine  # 遅延 import (torch を引きずるため).

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        rig_path = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        if not rig_path.exists():
            raise RuntimeError("reproject_views must run first")
        rig = json.loads(rig_path.read_text())

        prompts = [t.strip() for t in ctx.params["prompt"].split(",") if t.strip()]
        if not prompts:
            raise RuntimeError("no prompt terms; set sam3.default_prompt or pass prompt param")
        max_size = ctx.params["max_inference_size"]
        coverage_warn = ctx.params["coverage_warn"]

        ctx.progress.info(f"loading SAM3 model (device={get_settings().sam3.device})", progress=0.02)
        engine = Sam3Engine.from_settings()
        engine.load()
        ctx.progress.info("SAM3 model loaded", progress=0.08)

        frames = rig["frames"]
        if ctx.params["max_frames"] > 0:
            frames = frames[: ctx.params["max_frames"]]
        total_imgs = sum(len(f["views"]) for f in frames)
        done = 0

        mask_records = []
        outputs: list[FileRef] = []

        try:
            for fr in frames:
                frame_dir = ctx.stage_out_dir / f"frame_{fr['index']:06d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                view_records = []

                for v in fr["views"]:
                    src_path = ctx.project_dir / v["path"]
                    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"cannot read pinhole view: {src_path}")
                    h, w = bgr.shape[:2]

                    # 縮小 -> SAM3 -> union -> 元解像度へ拡大.
                    plan = mask_utils.plan_downsample(w, h, max_size)
                    small_bgr = mask_utils.downsample_image(bgr, plan)
                    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

                    detections = engine.detect(small_rgb, prompts)
                    small_masks = [m for d in detections for m in d.masks]
                    union_small = mask_utils.union_masks(small_masks)

                    if union_small is None:
                        # 何も検出されなかった -> 全面使用 (mask 全 0).
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                    else:
                        full_mask = mask_utils.upscale_mask(union_small, w, h)

                    out_name = f"{v['view']}_lens{v['lens']}.png"
                    out_path = frame_dir / out_name
                    # COLMAP 用: 検出物体を 0 (無視), それ以外 255.
                    mask_utils.write_mask_png(full_mask, out_path, invert=True)

                    cov = mask_utils.coverage_ratio(full_mask)
                    rec = {
                        "view": v["view"],
                        "lens": v["lens"],
                        "path": _final_relpath(out_path, ctx),
                        "coverage": cov,
                        "detections": {d.prompt: len(d.masks) for d in detections},
                    }
                    if cov > coverage_warn:
                        ctx.progress.warn(
                            f"frame {fr['index']} {out_name}: mask coverage {cov:.2f} > {coverage_warn}"
                        )
                        rec["coverage_warning"] = True
                    view_records.append(rec)
                    outputs.append(
                        FileRef(
                            path=_final_relpath(out_path, ctx),
                            size=out_path.stat().st_size,
                            sha256="",
                            mime="image/png",
                        )
                    )

                    done += 1
                    ctx.progress.info(
                        f"mask {done}/{total_imgs} (frame {fr['index']} {out_name}, cov={cov:.2f})",
                        progress=0.08 + 0.9 * (done / max(1, total_imgs)),
                    )

                mask_records.append({"index": fr["index"], "views": view_records})
        finally:
            engine.unload()

        masks_manifest = {
            "kind": "sam3_pinhole_masks",
            "prompt": prompts,
            "max_inference_size": max_size,
            "frames": mask_records,
        }
        mm_path = ctx.stage_out_dir / "manifest_masks.json"
        mm_path.write_text(json.dumps(masks_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(
            FileRef(
                path=_final_relpath(mm_path, ctx),
                size=mm_path.stat().st_size,
                sha256=sha256_file(mm_path),
                mime="application/json",
            )
        )

        manifest.outputs = outputs
        ctx.progress.info("generate_masks done", progress=1.0)
        return manifest


def _final_relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
