"""generate_masks ステージ.

再構成モードに応じて 3 レイアウトの SAM3 mask を作る:

- **fisheye** (native_fisheye 用): extract_frames の生前後魚眼 (lens0/lens1) に SAM3 を
  かけ, 動体 (人 / 自撮り棒 / 三脚 / 影) を膨張させて除外し, さらに魚眼の円形有効領域
  (fisheye_region) の外側も除外した COLMAP mask を作る.
- **pinhole** (pinhole_rig 用): reproject_views の pinhole rig 画像に SAM3 をかけ, 動体を
  除外する.
- **erp** (equirectangular 用): extract_frames の ERP 全天球フレームに SAM3 をかけ, 動体を
  除外する. 円形有効領域は無く, フレーム全面から動体だけを引く.

8K 由来の大きい画像に備え, SAM3 には縮小画像を渡し, 得た mask を元解像度へ bilinear
拡大する (imaging/masks). 縮小は無効化もできる (max_inference_size=0).

出力 (fisheye):
  <project>/generate_masks/lens0/frame_XXXXXX.png   COLMAP 用 (使う所=255)
  <project>/generate_masks/lens1/frame_XXXXXX.png
出力 (pinhole):
  <project>/generate_masks/frame_XXXXXX/<view>_lens<idx>.png
共通:
  <project>/generate_masks/manifest_masks.json

パラメータ:
  layout: "auto" | "pinhole" | "fisheye" | "erp"  auto は入力形式で判定.
  prompt: str            comma 区切り. 未指定なら settings.sam3.default_prompt.
  max_inference_size: int  SAM3 前の縮小長辺. 0 で縮小しない.
  dilate_px: int         動体 mask の膨張画素 (縁漏れ防止). default 0.
  coverage_warn: float   除外面積比の警告閾値 (default 0.5).
  max_frames: int        0 = 全部.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import fisheye_region
from ..imaging import masks as mask_utils
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class GenerateMasks(Stage):
    name = StageName.GENERATE_MASKS
    impl_version = "0.2"

    def _resolve_layout(self, ctx: StageContext) -> str:
        layout = ctx.params["layout"]
        if layout != "auto":
            return layout
        # reproject_views の出力があれば pinhole (INSV/ERP どちらの pinhole モードでも再投影像に掛ける).
        if (ctx.project_dir / "reproject_views" / "manifest_rig.json").exists():
            return "pinhole"
        # 再投影が無い ERP は生 ERP フレームに, それ以外は生魚眼に掛ける.
        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if frames_json.exists():
            kind = json.loads(frames_json.read_text()).get("kind")
            if kind in ("erp_video", "erp_images"):
                return "erp"
        return "fisheye"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        layout = self._resolve_layout(ctx)
        if layout == "fisheye":
            cands = [
                ctx.project_dir / "extract_frames" / "manifest_frames.json",
                fisheye_region.region_path(ctx.project_dir),
            ]
        elif layout == "erp":
            cands = [ctx.project_dir / "extract_frames" / "manifest_frames.json"]
        else:
            cands = [ctx.project_dir / "reproject_views" / "manifest_rig.json"]
        return [
            FileRef(path=str(p.relative_to(ctx.project_dir)), size=p.stat().st_size, sha256=sha256_file(p))
            for p in cands
            if p.exists()
        ]

    def normalize_params(self, raw: dict) -> dict:
        s = get_settings().sam3
        return {
            "layout": str(raw.get("layout", "auto")),
            "prompt": str(raw.get("prompt", s.default_prompt)),
            "max_inference_size": int(raw.get("max_inference_size", s.max_inference_size)),
            "dilate_px": int(raw.get("dilate_px", 0)),
            "coverage_warn": float(raw.get("coverage_warn", 0.5)),
            "max_frames": int(raw.get("max_frames", 0)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        # Torch/SAM3 の import・モデル読み込みは数十秒かかることがある. その間ログが
        # 途絶えて見えないよう, 重い import の前に開始を必ず 1 件流す.
        ctx.progress.info(
            "generate_masks 開始: SAM3 準備中 (初回はモデル読み込みに時間がかかる)",
            progress=0.01,
            key="log.mask_start",
        )
        from ..sam3.engine import Sam3Engine  # 遅延 import (torch を引きずるため).

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        prompts = [t.strip() for t in ctx.params["prompt"].split(",") if t.strip()]
        # prompt が空なら SAM3 を読み込まず, 動体マスク無し (fisheye は円のみ, pinhole は全面使用).
        engine = None
        if prompts:
            ctx.progress.info(
                f"loading SAM3 model (device={get_settings().sam3.device})",
                progress=0.02,
                key="log.mask_loading_model",
                args={"device": get_settings().sam3.device},
            )
            engine = Sam3Engine.from_settings()
            engine.load()
            ctx.progress.info("SAM3 model loaded", progress=0.08, key="log.mask_model_loaded")
        else:
            ctx.progress.info(
                "prompt 空: SAM3 をスキップ (円マスクのみ)",
                progress=0.08,
                key="log.mask_prompt_empty",
            )
        try:
            layout = self._resolve_layout(ctx)
            if layout == "fisheye":
                outputs, mask_manifest = self._run_fisheye(ctx, engine, prompts)
            elif layout == "erp":
                outputs, mask_manifest = self._run_erp(ctx, engine, prompts)
            else:
                outputs, mask_manifest = self._run_pinhole(ctx, engine, prompts)
        finally:
            if engine is not None:
                engine.unload()

        mm_path = ctx.stage_out_dir / "manifest_masks.json"
        mm_path.write_text(json.dumps(mask_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(
            FileRef(path=_final_relpath(mm_path, ctx), size=mm_path.stat().st_size, sha256=sha256_file(mm_path), mime="application/json")
        )
        manifest.outputs = outputs
        ctx.progress.info("generate_masks done", progress=1.0, key="log.mask_done")
        return manifest

    # -- fisheye (native) --------------------------------------------------------
    def _run_fisheye(self, ctx, engine, prompts) -> tuple[list[FileRef], dict]:
        import cv2  # noqa: PLC0415

        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if not frames_json.exists():
            raise RuntimeError("extract_frames must run first")
        frames_mf = json.loads(frames_json.read_text())
        if frames_mf.get("kind") != "insv_dual":
            raise RuntimeError(f"fisheye masks require dual-lens INSV, got {frames_mf.get('kind')}")

        region = fisheye_region.load_region(ctx.project_dir)
        max_size = ctx.params["max_inference_size"]
        dilate_px = ctx.params["dilate_px"]
        coverage_warn = ctx.params["coverage_warn"]

        frames = frames_mf["frames"]
        if ctx.params["max_frames"] > 0:
            frames = frames[: ctx.params["max_frames"]]
        total = len(frames) * 2
        done = 0
        outputs: list[FileRef] = []
        recs = []

        for fr in frames:
            frec = {"index": fr["index"], "lenses": []}
            for lens in (0, 1):
                src = ctx.project_dir / fr[f"lens{lens}"]
                bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"cannot read fisheye frame: {src}")
                h, w = bgr.shape[:2]

                plan = mask_utils.plan_downsample(w, h, max_size)
                small_rgb = cv2.cvtColor(mask_utils.downsample_image(bgr, plan), cv2.COLOR_BGR2RGB)
                detections = engine.detect(small_rgb, prompts) if engine is not None else []
                union_small = mask_utils.union_masks([m for d in detections for m in d.masks])
                dynamic = (
                    np.zeros((h, w), np.uint8)
                    if union_small is None
                    else mask_utils.upscale_mask(union_small, w, h)
                )
                dynamic = mask_utils.dilate_mask(dynamic, dilate_px)

                circle = fisheye_region.circle_px(region[f"lens{lens}"], w, h)
                valid = mask_utils.valid_region_mask(w, h, circle, exclude=dynamic)

                out_path = ctx.stage_out_dir / f"lens{lens}" / f"frame_{fr['index']:06d}.png"
                # 使う所=255, 除外 (動体 or 円外)=0.
                mask_utils.write_mask_png(valid, out_path, invert=False)

                # coverage = 円内で除外された動体の割合 (円外は元々無効なので除く).
                circ = mask_utils.circle_mask(w, h, *circle)
                circ_area = int(circ.sum()) or 1
                cov = float(((circ > 0) & (dynamic > 0)).sum()) / circ_area
                rec = {
                    "lens": lens,
                    "path": _final_relpath(out_path, ctx),
                    "coverage": cov,
                    "detections": {d.prompt: len(d.masks) for d in detections},
                }
                if cov > coverage_warn:
                    ctx.progress.warn(
                        f"frame {fr['index']} lens{lens}: dynamic coverage {cov:.2f} > {coverage_warn}",
                        key="log.mask_coverage_warn_lens",
                        args={"frame": fr["index"], "lens": lens, "cov": round(cov, 2), "warn": coverage_warn},
                    )
                    rec["coverage_warning"] = True
                frec["lenses"].append(rec)
                outputs.append(FileRef(path=_final_relpath(out_path, ctx), size=out_path.stat().st_size, sha256="", mime="image/png"))
                done += 1
                ctx.progress.tick(
                    progress=0.08 + 0.9 * (done / max(1, total)),
                    message=f"mask {done}/{total} (frame {fr['index']} lens{lens}, dyn={cov:.2f})",
                    key="log.mask_progress_lens",
                    args={"done": done, "total": total, "frame": fr["index"], "lens": lens, "cov": round(cov, 2)},
                )
            recs.append(frec)

        manifest = {
            "kind": "sam3_fisheye_masks",
            "prompt": prompts,
            "max_inference_size": max_size,
            "dilate_px": dilate_px,
            "frames": recs,
        }
        return outputs, manifest

    # -- pinhole (fallback) ------------------------------------------------------
    def _run_pinhole(self, ctx, engine, prompts) -> tuple[list[FileRef], dict]:
        import cv2  # noqa: PLC0415

        rig_path = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        if not rig_path.exists():
            raise RuntimeError("reproject_views must run first")
        rig = json.loads(rig_path.read_text())
        max_size = ctx.params["max_inference_size"]
        dilate_px = ctx.params["dilate_px"]
        coverage_warn = ctx.params["coverage_warn"]

        frames = rig["frames"]
        if ctx.params["max_frames"] > 0:
            frames = frames[: ctx.params["max_frames"]]
        total = sum(len(f["views"]) for f in frames)
        done = 0
        outputs: list[FileRef] = []
        recs = []

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
                plan = mask_utils.plan_downsample(w, h, max_size)
                small_rgb = cv2.cvtColor(mask_utils.downsample_image(bgr, plan), cv2.COLOR_BGR2RGB)
                detections = engine.detect(small_rgb, prompts) if engine is not None else []
                union_small = mask_utils.union_masks([m for d in detections for m in d.masks])
                dynamic = (
                    np.zeros((h, w), np.uint8)
                    if union_small is None
                    else mask_utils.upscale_mask(union_small, w, h)
                )
                dynamic = mask_utils.dilate_mask(dynamic, dilate_px)

                out_path = frame_dir / f"{v['view']}_lens{v['lens']}.png"
                # 動体=0 (無視), それ以外=255.
                mask_utils.write_mask_png(dynamic, out_path, invert=True)

                cov = mask_utils.coverage_ratio(dynamic)
                rec = {
                    "view": v["view"], "lens": v["lens"], "path": _final_relpath(out_path, ctx),
                    "coverage": cov, "detections": {d.prompt: len(d.masks) for d in detections},
                }
                if cov > coverage_warn:
                    ctx.progress.warn(
                        f"frame {fr['index']} {out_path.name}: coverage {cov:.2f} > {coverage_warn}",
                        key="log.mask_coverage_warn_view",
                        args={"frame": fr["index"], "name": out_path.name, "cov": round(cov, 2), "warn": coverage_warn},
                    )
                    rec["coverage_warning"] = True
                view_records.append(rec)
                outputs.append(FileRef(path=_final_relpath(out_path, ctx), size=out_path.stat().st_size, sha256="", mime="image/png"))
                done += 1
                ctx.progress.tick(
                    progress=0.08 + 0.9 * (done / max(1, total)),
                    message=f"mask {done}/{total} (frame {fr['index']} {out_path.name}, cov={cov:.2f})",
                    key="log.mask_progress_view",
                    args={"done": done, "total": total, "frame": fr["index"], "name": out_path.name, "cov": round(cov, 2)},
                )
            recs.append({"index": fr["index"], "views": view_records})

        manifest = {
            "kind": "sam3_pinhole_masks",
            "prompt": prompts,
            "max_inference_size": max_size,
            "dilate_px": dilate_px,
            "frames": recs,
        }
        return outputs, manifest


    # -- erp (equirectangular) ---------------------------------------------------
    def _run_erp(self, ctx, engine, prompts) -> tuple[list[FileRef], dict]:
        import cv2  # noqa: PLC0415

        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if not frames_json.exists():
            raise RuntimeError("extract_frames must run first")
        frames_mf = json.loads(frames_json.read_text())
        if frames_mf.get("kind") not in ("erp_video", "erp_images"):
            raise RuntimeError(f"erp masks require ERP frames, got {frames_mf.get('kind')}")

        max_size = ctx.params["max_inference_size"]
        dilate_px = ctx.params["dilate_px"]
        coverage_warn = ctx.params["coverage_warn"]

        frames = frames_mf["frames"]
        if ctx.params["max_frames"] > 0:
            frames = frames[: ctx.params["max_frames"]]
        total = len(frames)
        done = 0
        outputs: list[FileRef] = []
        recs = []

        for fr in frames:
            src = ctx.project_dir / fr["erp"] if "erp" in fr else Path(fr["erp_source"])
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot read ERP frame: {src}")
            h, w = bgr.shape[:2]
            plan = mask_utils.plan_downsample(w, h, max_size)
            small_rgb = cv2.cvtColor(mask_utils.downsample_image(bgr, plan), cv2.COLOR_BGR2RGB)
            detections = engine.detect(small_rgb, prompts) if engine is not None else []
            union_small = mask_utils.union_masks([m for d in detections for m in d.masks])
            dynamic = (
                np.zeros((h, w), np.uint8)
                if union_small is None
                else mask_utils.upscale_mask(union_small, w, h)
            )
            dynamic = mask_utils.dilate_mask(dynamic, dilate_px)

            out_path = ctx.stage_out_dir / f"frame_{fr['index']:06d}.png"
            # 動体=0 (無視), それ以外=255. ERP は円形有効領域が無く全面が有効.
            mask_utils.write_mask_png(dynamic, out_path, invert=True)

            cov = mask_utils.coverage_ratio(dynamic)
            rec = {
                "index": fr["index"], "path": _final_relpath(out_path, ctx),
                "coverage": cov, "detections": {d.prompt: len(d.masks) for d in detections},
            }
            if cov > coverage_warn:
                ctx.progress.warn(
                    f"frame {fr['index']}: dynamic coverage {cov:.2f} > {coverage_warn}",
                    key="log.mask_coverage_warn_erp",
                    args={"frame": fr["index"], "cov": round(cov, 2), "warn": coverage_warn},
                )
                rec["coverage_warning"] = True
            recs.append(rec)
            outputs.append(FileRef(path=_final_relpath(out_path, ctx), size=out_path.stat().st_size, sha256="", mime="image/png"))
            done += 1
            ctx.progress.tick(
                progress=0.08 + 0.9 * (done / max(1, total)),
                message=f"mask {done}/{total} (frame {fr['index']}, cov={cov:.2f})",
                key="log.mask_progress_erp",
                args={"done": done, "total": total, "frame": fr["index"], "cov": round(cov, 2)},
            )

        manifest = {
            "kind": "sam3_erp_masks",
            "prompt": prompts,
            "max_inference_size": max_size,
            "dilate_px": dilate_px,
            "frames": recs,
        }
        return outputs, manifest


def _final_relpath(p: Path, ctx: StageContext) -> str:
    rel = p.relative_to(ctx.stage_out_dir)
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)
