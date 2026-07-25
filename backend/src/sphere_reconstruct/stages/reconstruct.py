"""reconstruct ステージ (COLMAP).

3 つの再構成モードを持つ. どの入力形式 (INSV 前後魚眼 / ERP 全天球) から始めるかで選ぶ:

- **native_fisheye** (INSV 既定): extract_frames の生の前後魚眼 (lens0/lens1) をそのまま
  COLMAP に渡す. 前後 2 レンズを OPENCV_FISHEYE の 2 センサー rig とし, front を参照,
  back を「Y 軸 180deg + 物理 baseline」で強制固定する (colmap.native_rig). 前後半球に
  視覚的重なりが無くても frame 共有で必ず 1 モデルへ合流する. 事前の pinhole 再投影が
  不要で, 全 FoV / 原画素を保ったまま特徴を取れる.
- **pinhole_rig** (INSV fallback): reproject_views が作った 6-view cubemap pinhole 画像を使う.
  12 (= 6 view x 2 lens) 仮想カメラの既知相対姿勢を rig 拘束にする. 事前に「生成」
  (reproject_views) が要る唯一のモード.
- **equirectangular** (ERP 既定): 全天球 (equirectangular) 動画/画像の各フレームを COLMAP
  4.1+ の EQUIRECTANGULAR カメラモデル (球面, 焦点距離なし, params = width/height) で直接
  解く. 単一カメラ, rig なし, 再投影なし. ERP は既に平面展開されているので pinhole
  への「生成」は不要.

特徴 backend は sift | aliked を選べる. この COLMAP ビルドは SIFT のみ (deep features
非対応) なので, ALIKED は外部 onnxruntime で抽出/マッチして COLMAP DB へ書き込む
(database_creator -> keypoints -> LightGlue -> matches_importer). 魚眼は縮小すると角
分解能が落ちるため ALIKED は全解像度で抽出し, 8K で VRAM が足りなければ CPU へ自動
フォールバックする (features.aliked_lightglue).

パラメータ:
  reconstruction_mode: "native_fisheye" | "pinhole_rig" | "equirectangular"  (default native_fisheye)
  feature_backend:     "sift" | "aliked"                   (default sift)
  extraction_device:   "auto" | "cuda" | "cpu"             (ALIKED 抽出, default auto)
  matcher:             "sequential" | "exhaustive"         (default sequential)
  overlap:             sequential 窓 (default 10)
  use_gpu:             SIFT/matcher の GPU (default True)
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from ..colmap import model as colmap_model
from ..colmap import native_rig
from ..colmap import quality as colmap_quality
from ..colmap import runner as colmap_runner
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import fisheye_region
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class Reconstruct(Stage):
    name = StageName.RECONSTRUCT
    impl_version = "0.8"  # COLMAP 詳細パラメータ

    # -- inputs / params ---------------------------------------------------------
    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        mode = ctx.params.get("reconstruction_mode", "native_fisheye")
        if mode == "native_fisheye":
            cands = [
                ctx.project_dir / "inspect_source" / "source.json",
                ctx.project_dir / "extract_frames" / "manifest_frames.json",
                fisheye_region.region_path(ctx.project_dir),
                ctx.project_dir / "generate_masks" / "manifest_masks.json",
            ]
        elif mode == "equirectangular":
            cands = [
                ctx.project_dir / "extract_frames" / "manifest_frames.json",
                ctx.project_dir / "generate_masks" / "manifest_masks.json",
            ]
        else:
            cands = [
                ctx.project_dir / "reproject_views" / "manifest_rig.json",
                ctx.project_dir / "generate_masks" / "manifest_masks.json",
            ]
        return [
            FileRef(path=str(p.relative_to(ctx.project_dir)), size=p.stat().st_size, sha256=sha256_file(p))
            for p in cands
            if p.exists()
        ]

    def normalize_params(self, raw: dict) -> dict:
        mode = str(raw.get("reconstruction_mode", "native_fisheye"))
        p: dict = {
            "reconstruction_mode": mode,
            "matcher": str(raw.get("matcher", "sequential")),
            "overlap": int(raw.get("overlap", 10)),
            "use_gpu": bool(raw.get("use_gpu", True)),
            "feature_backend": str(raw.get("feature_backend", "sift")),
            # ALIKED 抽出デバイス. auto は VRAM 見積り + 実行時 OOM で CPU へ落ちる.
            "extraction_device": str(raw.get("extraction_device", "auto")),
            # ALIKED 抽出の長辺上限 (px). 0 = settings の既定 (2048) を使う. 大きいほど精度↑
            # だが 8K は VRAM/RAM を溢れさせクラッシュしやすい.
            "extract_max_size": int(raw.get("extract_max_size", 0)),
            # COLMAP 詳細パラメータ (高級選項). 0/False = COLMAP 既定 (flag を出さない).
            # プリセットはフロントで具体値へ展開して送る.
            "sift_max_num_features": int(raw.get("sift_max_num_features", 0)),
            "sift_max_image_size": int(raw.get("sift_max_image_size", 0)),
            "sift_peak_threshold": float(raw.get("sift_peak_threshold", 0.0)),
            "sift_edge_threshold": float(raw.get("sift_edge_threshold", 0.0)),
            "sift_affine_dsp": bool(raw.get("sift_affine_dsp", False)),
            "max_num_matches": int(raw.get("max_num_matches", 0)),
            "guided_matching": bool(raw.get("guided_matching", False)),
            "two_view_min_num_inliers": int(raw.get("two_view_min_num_inliers", 0)),
            "mapper_min_num_matches": int(raw.get("mapper_min_num_matches", 0)),
            "init_min_num_inliers": int(raw.get("init_min_num_inliers", 0)),
            "abs_pose_max_error": float(raw.get("abs_pose_max_error", 0.0)),
            "filter_max_reproj_error": float(raw.get("filter_max_reproj_error", 0.0)),
            "filter_min_tri_angle": float(raw.get("filter_min_tri_angle", 0.0)),
            "ba_local_max_num_iterations": int(raw.get("ba_local_max_num_iterations", 0)),
            "ba_global_max_num_iterations": int(raw.get("ba_global_max_num_iterations", 0)),
            "min_model_size": int(raw.get("min_model_size", 0)),
            # マッチ戦略拡張 / BA GPU (両 backend で mapper に効く).
            "loop_closure": bool(raw.get("loop_closure", False)),
            "ba_use_gpu": bool(raw.get("ba_use_gpu", False)),
        }
        if mode == "native_fisheye":
            # 内参は粗初期値から COLMAP に精修させる. rig 外参 (180deg+baseline) は
            # 実測値で固定 (前後に共有点が無く BA で解けないため, また metric アンカーのため).
            p["refine_intrinsics"] = bool(raw.get("refine_intrinsics", True))
            # 円形有効領域は常に適用する. use_masks は SAM3 動体マスク (generate_masks
            # fisheye 出力) を併用するかどうか. False でも円マスクは効く.
            p["use_masks"] = bool(raw.get("use_masks", True))
        elif mode == "equirectangular":
            # EQUIRECTANGULAR は球面モデルで焦点距離を持たず, params は width/height のみ
            # (画像寸法で一意). 精修する内参が無いので refine_intrinsics は無効固定.
            p["refine_intrinsics"] = False
            p["use_masks"] = bool(raw.get("use_masks", True))
        else:
            p["use_masks"] = bool(raw.get("use_masks", True))
            p["refine_intrinsics"] = bool(raw.get("refine_intrinsics", False))
            p["use_rig"] = bool(raw.get("use_rig", True))
            p["refine_rig"] = bool(raw.get("refine_rig", True))
            p["loop_closure"] = bool(raw.get("loop_closure", False))
        return p

    # -- execute -----------------------------------------------------------------
    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        p = ctx.params
        ctx.progress.info(
            f"reconstruct 開始: mode={p['reconstruction_mode']}, backend={p['feature_backend']}, "
            f"device={p.get('extraction_device', '-')}, matcher={p['matcher']}, overlap={p['overlap']}, "
            f"use_masks={p.get('use_masks')}, refine_intrinsics={p.get('refine_intrinsics')}",
            progress=0.01,
            key="log.recon_start",
            args={
                "mode": p["reconstruction_mode"],
                "backend": p["feature_backend"],
                "device": p.get("extraction_device", "-"),
                "matcher": p["matcher"],
                "overlap": p["overlap"],
                "use_masks": p.get("use_masks"),
                "refine_intrinsics": p.get("refine_intrinsics"),
            },
        )

        if ctx.params["reconstruction_mode"] == "native_fisheye":
            summary, n_total = self._execute_native_fisheye(ctx)
        elif ctx.params["reconstruction_mode"] == "equirectangular":
            summary, n_total = self._execute_equirectangular(ctx)
        else:
            summary, n_total = self._execute_pinhole_rig(ctx)

        out = ctx.stage_out_dir
        # _model_dir は内部用 (絶対パス). ファイル/manifest には残さない.
        model_dir = Path(summary.pop("_model_dir"))
        summary["registered_ratio"] = summary["num_images"] / n_total if n_total else 0.0
        (out / "model_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ctx.progress.info(
            f"reconstruction: {summary['num_images']}/{n_total} images, "
            f"{summary['num_points3D']} points, mean err {summary['mean_reprojection_error']:.3f}px",
            progress=0.95,
            key="log.recon_summary",
            args={
                "images": summary["num_images"],
                "total": n_total,
                "points": summary["num_points3D"],
                "err": round(summary["mean_reprojection_error"], 3),
            },
        )

        outputs = []
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            fp = model_dir / name
            outputs.append(
                FileRef(path=_relpath(fp, ctx), size=fp.stat().st_size, sha256="", mime="application/octet-stream")
            )
        sp = out / "model_summary.json"
        outputs.append(FileRef(path=_relpath(sp, ctx), size=sp.stat().st_size, sha256=sha256_file(sp), mime="application/json"))
        manifest.outputs = outputs
        manifest.extra = summary
        ctx.progress.info("reconstruct done", progress=1.0, key="log.recon_done")
        return manifest

    # -- native fisheye ----------------------------------------------------------
    def _execute_native_fisheye(self, ctx: StageContext) -> tuple[dict, int]:
        """生の前後魚眼 + 強制物理 rig で COLMAP を回す."""
        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        source_json = ctx.project_dir / "inspect_source" / "source.json"
        if not frames_json.exists():
            raise RuntimeError("extract_frames must run first")
        frames_mf = json.loads(frames_json.read_text())
        if frames_mf.get("kind") != "insv_dual":
            raise RuntimeError(
                f"native_fisheye requires dual-lens INSV frames, got kind={frames_mf.get('kind')}"
            )

        w = int(frames_mf["width"])
        h = int(frames_mf["height"])
        frames = frames_mf["frames"]

        # baseline は offset_v3 のレンズ中心間距離. 無ければ既定値.
        baseline = native_rig.DEFAULT_BASELINE_M
        if source_json.exists():
            ov3 = (json.loads(source_json.read_text()).get("offset_v3") or {})
            if ov3.get("valid") and ov3.get("lenses"):
                baseline = native_rig.baseline_from_lens_centers(ov3["lenses"])

        out = ctx.stage_out_dir
        images_dir = out / "images"
        logs_dir = out / "logs"
        sparse_dir = out / "sparse"
        for d in (images_dir, logs_dir, sparse_dir):
            d.mkdir(parents=True, exist_ok=True)

        # view-major レイアウト: front/frame_XXXXXX.jpg, back/frame_XXXXXX.jpg.
        # rig_configurator は接頭辞を除いた名 (frame_XXXXXX.jpg) で front/back を同一 frame
        # とみなすため, 前後で同じファイル名を使うことが必須.
        groups: dict[str, list[tuple[str, Path]]] = {"front": [], "back": []}
        for fr in frames:
            fname = f"frame_{fr['index']:06d}.jpg"
            for lens_key, prefix in (("lens0", "front"), ("lens1", "back")):
                src = ctx.project_dir / fr[lens_key]
                if not src.exists():
                    raise RuntimeError(f"missing extract frame: {src}")
                dst = images_dir / prefix / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(src, dst)
                groups[prefix].append((f"{prefix}/{fname}", dst))
        for g in groups.values():
            g.sort()
        n_total = sum(len(g) for g in groups.values())

        # 内参の主点は光学中心 (画像中心) を使う. 有効領域の円 (region) とは別物.
        # OPENCV_FISHEYE 内参の粗初期値. equidistant r=f*theta, 端 (r=w/2) を theta~100deg
        # とみなして f0=w/2/1.75. COLMAP に精修させるので厳密でなくてよい.
        f0 = w / 2.0 / 1.75
        cp_list = [f0, f0, w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 0.0]
        cp_str = ",".join(f"{v:.6f}" for v in cp_list)

        # 各画像の COLMAP マスク (使う所=255) を用意する.
        #   - use_masks かつ generate_masks(fisheye) の出力があれば, それ (円形有効領域 +
        #     膨張済み SAM3 動体を合成済み) を使う.
        #   - 無ければ region の円形有効領域のみ生成する (黒い四隅とレンズ端の反射/汚れを除外).
        # マスクは per-image (COLMAP 命名 <image_name>.png) で masks/ に置く. 前後レンズで
        # 円が異なりうるため単一 camera_mask ではなく per-image にする.
        region = fisheye_region.load_region(ctx.project_dir)
        masks_dir = out / "masks"
        sam3_masks: dict[tuple[int, int], Path] | None = None
        fmm_path = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        if ctx.params["use_masks"] and fmm_path.exists():
            fmm = json.loads(fmm_path.read_text())
            if fmm.get("kind") == "sam3_fisheye_masks":
                sam3_masks = {
                    (lr["lens"], fr["index"]): ctx.project_dir / lr["path"]
                    for fr in fmm["frames"] for lr in fr["lenses"]
                }
        image_masks: dict[str, Path] = {}
        for fr in frames:
            fname = f"frame_{fr['index']:06d}.jpg"
            for lens, prefix in ((0, "front"), (1, "back")):
                colmap_name = f"{prefix}/{fname}"
                dst = masks_dir / f"{colmap_name}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if sam3_masks is not None and (lens, fr["index"]) in sam3_masks:
                    _link_or_copy(sam3_masks[(lens, fr["index"])], dst)
                else:
                    ccx, ccy, crad = fisheye_region.circle_px(region[f"lens{lens}"], w, h)
                    _write_circle_mask(w, h, ccx, ccy, crad, dst)
                image_masks[colmap_name] = dst

        colmap_bin = colmap_runner.resolve_colmap_bin(get_settings().binaries.colmap or None)
        db_path = out / "database.db"
        ctx.progress.info(
            f"native fisheye: {n_total} images ({w}x{h}), baseline {baseline*1000:.1f}mm, "
            f"masks={'sam3+circle' if sam3_masks else 'circle'}",
            progress=0.05,
            key="log.recon_native_fisheye",
            args={
                "images": n_total,
                "w": w,
                "h": h,
                "baseline_mm": round(baseline * 1000, 1),
                "masks": "sam3+circle" if sam3_masks else "circle",
            },
        )

        backend = ctx.params["feature_backend"]
        if backend == "aliked":
            self._aliked_extract_and_match(
                ctx, colmap_bin, db_path, groups,
                model_id=5, camera_params=cp_list, overlap=ctx.params["overlap"],
                width=w, height=h, image_masks=image_masks,
            )
        else:
            colmap_runner.feature_extractor(
                colmap_bin,
                database_path=db_path,
                image_path=images_dir,
                camera_model="OPENCV_FISHEYE",
                single_camera_per_folder=True,
                camera_params=cp_str,
                mask_path=masks_dir,
                use_gpu=ctx.params["use_gpu"],
                log_path=logs_dir / "feature_extractor.log",
                extra_args=colmap_quality.feature_extra_args(ctx.params, ctx.params["feature_backend"]),
                on_line=_logline(ctx, "features"),
            )
            self._run_matcher(ctx, colmap_bin, db_path, logs_dir)

        # 強制物理 rig を DB へ設定.
        rig_cfg = native_rig.build_physical_rig_config("OPENCV_FISHEYE", cp_list, baseline)
        rig_cfg_path = out / "rig_config.json"
        rig_cfg_path.write_text(json.dumps(rig_cfg, indent=2), encoding="utf-8")
        ctx.progress.info(
            "configuring physical rig (front ref, back 180deg+baseline)",
            progress=0.5,
            key="log.recon_config_physical_rig",
        )
        colmap_runner.rig_configurator(
            colmap_bin, database_path=db_path, rig_config_path=rig_cfg_path,
            log_path=logs_dir / "rig_configurator.log", on_line=_logline(ctx, "rig"),
        )

        ctx.progress.info(
            "colmap mapper (native fisheye rig)", progress=0.55, key="log.recon_mapper_native"
        )
        colmap_runner.mapper(
            colmap_bin, database_path=db_path, image_path=images_dir, output_path=sparse_dir,
            refine_intrinsics=ctx.params["refine_intrinsics"],
            refine_rig=False,          # 実測値 (180deg+baseline) を厳密に強制する.
            multiple_models=False,     # 1 モデルへ強制合流.
            extra_args=colmap_quality.mapper_extra_args(ctx.params),
            log_path=logs_dir / "mapper.log", on_line=_mapper_progress(ctx, n_total),
        )
        return self._largest_model_summary(sparse_dir), n_total

    # -- equirectangular (ERP direct) --------------------------------------------
    def _execute_equirectangular(self, ctx: StageContext) -> tuple[dict, int]:
        """ERP 全天球フレームを EQUIRECTANGULAR カメラモデルで直接 COLMAP に解かせる.

        ERP は既に球面を平面展開したものなので, pinhole 再投影 (reproject_views) を挟まず
        フレームを 1 台の球面カメラの連続撮影として扱う. 焦点距離を持たないモデルなので
        rig も内参精修も無い.
        """
        import cv2  # noqa: PLC0415

        frames_json = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if not frames_json.exists():
            raise RuntimeError("extract_frames must run first")
        frames_mf = json.loads(frames_json.read_text())
        kind = frames_mf.get("kind")
        if kind not in ("erp_video", "erp_images"):
            raise RuntimeError(f"equirectangular requires ERP frames, got kind={kind}")
        frames = frames_mf["frames"]
        if not frames:
            raise RuntimeError("extract_frames produced no ERP frames")

        def _frame_src(fr: dict) -> Path:
            # erp_video はプロジェクト相対 (extract_frames が抽出), erp_images は元画像の絶対パス.
            return ctx.project_dir / fr["erp"] if "erp" in fr else Path(fr["erp_source"])

        # 画像寸法: erp_video は manifest にあり, erp_images は先頭画像から読む.
        if frames_mf.get("width") and frames_mf.get("height"):
            w, h = int(frames_mf["width"]), int(frames_mf["height"])
        else:
            probe = cv2.imread(str(_frame_src(frames[0])), cv2.IMREAD_COLOR)
            if probe is None:
                raise RuntimeError(f"cannot read ERP frame: {_frame_src(frames[0])}")
            h, w = probe.shape[:2]

        out = ctx.stage_out_dir
        images_dir = out / "images"
        masks_dir = out / "masks"
        logs_dir = out / "logs"
        sparse_dir = out / "sparse"
        for d in (images_dir, logs_dir, sparse_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 全フレームを単一フォルダへ link (単一カメラ). 元拡張子を保つ.
        group: list[tuple[str, Path]] = []
        for fr in frames:
            src = _frame_src(fr)
            if not src.exists():
                raise RuntimeError(f"missing ERP frame: {src}")
            name = f"frame_{fr['index']:06d}{src.suffix.lower()}"
            dst = images_dir / name
            _link_or_copy(src, dst)
            group.append((name, dst))
        group.sort()
        groups = {"pano": group}
        n_total = len(group)

        # SAM3 動体マスク (generate_masks erp 出力) があれば per-image mask を用意する.
        # ERP には円形有効領域が無いため, マスク無しならフレーム全面を使う.
        image_masks: dict[str, Path] = {}
        mm_path = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        if ctx.params["use_masks"] and mm_path.exists():
            mm = json.loads(mm_path.read_text())
            if mm.get("kind") == "sam3_erp_masks":
                by_index = {fr["index"]: ctx.project_dir / fr["path"] for fr in mm["frames"]}
                for name, _dst in group:
                    idx = int(name.split("_")[1].split(".")[0])
                    if idx in by_index:
                        mdst = masks_dir / f"{name}.png"
                        mdst.parent.mkdir(parents=True, exist_ok=True)
                        _link_or_copy(by_index[idx], mdst)
                        image_masks[name] = mdst
        have_masks = bool(image_masks)

        colmap_bin = colmap_runner.resolve_colmap_bin(get_settings().binaries.colmap or None)
        db_path = out / "database.db"
        # EQUIRECTANGULAR の params は width,height の 2 つ (COLMAP 4.1+).
        cp_list = [float(w), float(h)]
        cp_str = f"{w},{h}"
        ctx.progress.info(
            f"equirectangular: {n_total} ERP frames ({w}x{h}), "
            f"masks={'sam3' if have_masks else 'none'}",
            progress=0.05,
            key="log.recon_equirect",
            args={"frames": n_total, "w": w, "h": h, "masks": "sam3" if have_masks else "none"},
        )

        if ctx.params["feature_backend"] == "aliked":
            self._aliked_extract_and_match(
                ctx, colmap_bin, db_path, groups,
                model_id=17, camera_params=cp_list, overlap=ctx.params["overlap"],
                width=w, height=h, image_masks=image_masks if have_masks else None,
            )
        else:
            colmap_runner.feature_extractor(
                colmap_bin, database_path=db_path, image_path=images_dir,
                camera_model="EQUIRECTANGULAR", single_camera=True, camera_params=cp_str,
                use_gpu=ctx.params["use_gpu"],
                mask_path=masks_dir if have_masks else None,
                log_path=logs_dir / "feature_extractor.log", on_line=_logline(ctx, "features"),
            )
            self._run_matcher(ctx, colmap_bin, db_path, logs_dir)

        ctx.progress.info(
            "colmap mapper (equirectangular)", progress=0.55, key="log.recon_mapper_equirect"
        )
        colmap_runner.mapper(
            colmap_bin, database_path=db_path, image_path=images_dir, output_path=sparse_dir,
            refine_intrinsics=False,   # 球面モデルは精修する内参が無い.
            extra_args=colmap_quality.mapper_extra_args(ctx.params),
            log_path=logs_dir / "mapper.log", on_line=_mapper_progress(ctx, n_total),
        )
        return self._largest_model_summary(sparse_dir), n_total

    # -- pinhole rig (fallback) --------------------------------------------------
    def _execute_pinhole_rig(self, ctx: StageContext) -> tuple[dict, int]:
        rig_path = ctx.project_dir / "reproject_views" / "manifest_rig.json"
        if not rig_path.exists():
            raise RuntimeError("reproject_views must run first (pinhole_rig mode)")
        rig = json.loads(rig_path.read_text())

        out = ctx.stage_out_dir
        images_dir = out / "images"
        masks_dir = out / "masks"
        sparse_dir = out / "sparse"
        logs_dir = out / "logs"
        for d in (images_dir, sparse_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        mm_path = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        use_masks = ctx.params["use_masks"] and mm_path.exists()
        masks_manifest = json.loads(mm_path.read_text()) if use_masks else None
        if use_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)

        def _name(view: str, lens: int, index: int) -> str:
            return f"{view}_lens{lens}/frame_{index:06d}.jpg"

        # view-major レイアウトで画像を link (sequential matcher が同一視点の連続フレームを
        # 近傍にするため. frame-major だと近傍が重ならない cubemap 面同士になり mapper が
        # 初期ペアを得られず失敗する).
        groups: dict[str, list[tuple[str, Path]]] = {}
        for fr in rig["frames"]:
            for v in fr["views"]:
                dst = images_dir / _name(v["view"], v["lens"], fr["index"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(ctx.project_dir / v["path"], dst)
                groups.setdefault(f"{v['view']}_lens{v['lens']}", []).append(
                    (_name(v["view"], v["lens"], fr["index"]), dst)
                )
        for g in groups.values():
            g.sort()
        n_total = sum(len(g) for g in groups.values())

        if use_masks and masks_manifest is not None:
            for fr in masks_manifest["frames"]:
                for v in fr["views"]:
                    name = _name(v["view"], v["lens"], fr["index"])
                    dst = masks_dir / f"{name}.png"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(ctx.project_dir / v["path"], dst)

        # 既知 pinhole 内参 (cubemap 全 view 同一). f=(size/2)/tan(fov/2).
        views_meta = rig.get("views", [])
        lenses_meta = rig.get("lenses", [])
        # rig は view 間の既知相対姿勢を拘束する. INSV は 6view x 2lens, ERP は 6view x 1lens
        # (並進ゼロの純回転). どちらも view が 2 つ以上あれば有効.
        use_rig = ctx.params["use_rig"] and len(views_meta) >= 2 and len(lenses_meta) >= 1
        size = int(views_meta[0]["size"])
        fov = float(views_meta[0]["fov_deg"])
        f = (size / 2.0) / math.tan(math.radians(fov) / 2.0)
        c = size / 2.0
        cp_str = f"{f:.6f},{f:.6f},{c:.6f},{c:.6f}"
        cp_list = [f, f, c, c]

        colmap_bin = colmap_runner.resolve_colmap_bin(get_settings().binaries.colmap or None)
        db_path = out / "database.db"
        ctx.progress.info(
            f"pinhole rig: {n_total} images, intrinsics {cp_str}",
            progress=0.05,
            key="log.recon_pinhole_rig",
            args={"images": n_total, "intrinsics": cp_str},
        )

        if ctx.params["feature_backend"] == "aliked":
            self._aliked_extract_and_match(
                ctx, colmap_bin, db_path, groups,
                model_id=1, camera_params=cp_list, overlap=ctx.params["overlap"],
                width=size, height=size, image_masks=None,
            )
        else:
            colmap_runner.feature_extractor(
                colmap_bin, database_path=db_path, image_path=images_dir,
                camera_model="PINHOLE",
                single_camera=not use_rig, single_camera_per_folder=use_rig,
                camera_params=None if use_rig else cp_str,
                use_gpu=ctx.params["use_gpu"],
                mask_path=masks_dir if use_masks else None,
                log_path=logs_dir / "feature_extractor.log", on_line=_logline(ctx, "features"),
            )
            self._run_matcher(ctx, colmap_bin, db_path, logs_dir)

        if use_rig:
            from ..colmap import rig as colmap_rig

            cams = colmap_rig.compute_rig_cameras(views_meta, lenses_meta)
            rig_cfg = colmap_rig.build_rig_config(cams, cp_list)
            rig_cfg_path = out / "rig_config.json"
            rig_cfg_path.write_text(json.dumps(rig_cfg, indent=2), encoding="utf-8")
            ctx.progress.info(
                f"configuring rig: {len(cams)} cameras",
                progress=0.5,
                key="log.recon_config_rig",
                args={"cameras": len(cams)},
            )
            colmap_runner.rig_configurator(
                colmap_bin, database_path=db_path, rig_config_path=rig_cfg_path,
                log_path=logs_dir / "rig_configurator.log", on_line=_logline(ctx, "rig"),
            )

        ctx.progress.info(
            "colmap mapper (incremental)", progress=0.55, key="log.recon_mapper_incremental"
        )
        colmap_runner.mapper(
            colmap_bin, database_path=db_path, image_path=images_dir, output_path=sparse_dir,
            refine_intrinsics=ctx.params["refine_intrinsics"], refine_rig=ctx.params["refine_rig"],
            extra_args=colmap_quality.mapper_extra_args(ctx.params),
            log_path=logs_dir / "mapper.log", on_line=_mapper_progress(ctx, n_total),
        )
        return self._largest_model_summary(sparse_dir), n_total

    # -- shared matcher / model helpers ------------------------------------------
    def _run_matcher(self, ctx: StageContext, colmap_bin: str, db_path: Path, logs_dir: Path) -> None:
        ctx.progress.info("colmap matching", progress=0.3, key="log.recon_matching")
        match_extra = colmap_quality.matcher_extra_args(ctx.params, ctx.params["feature_backend"])
        matcher = ctx.params["matcher"]
        vocab = get_settings().binaries.vocab_tree or ""
        common = dict(use_gpu=ctx.params["use_gpu"], extra_args=match_extra,
                      log_path=logs_dir / "matcher.log", on_line=_logline(ctx, "match"))
        if matcher == "exhaustive":
            colmap_runner.exhaustive_matcher(colmap_bin, database_path=db_path, **common)
        elif matcher == "vocab_tree":
            if not vocab:
                raise RuntimeError("vocab_tree matcher requires binaries.vocab_tree in config.toml")
            colmap_runner.vocab_tree_matcher(colmap_bin, database_path=db_path, vocab_tree_path=Path(vocab), **common)
        else:  # sequential (+ loop closure なら vocab tree で大域ペアを追加)
            loop = bool(ctx.params.get("loop_closure")) and bool(vocab)
            if ctx.params.get("loop_closure") and not vocab:
                ctx.progress.warn(
                    "loop closure requested but binaries.vocab_tree not set; running plain sequential",
                    key="log.recon_loop_no_vocab",
                )
            colmap_runner.sequential_matcher(
                colmap_bin, database_path=db_path, overlap=ctx.params["overlap"],
                loop_detection=loop, vocab_tree_path=Path(vocab) if loop else None, **common,
            )

    def _largest_model_summary(self, sparse_dir: Path) -> dict:
        """mapper 出力から画像数が最大のサブモデルを選び, sparse/0 に据えて summary を返す.

        COLMAP は複数サブモデルを sparse/0, sparse/1, ... に吐くことがある. 下流
        (export_dataset / web_preview) は sparse/0 固定で読むため, 最大モデルが sparse/0
        以外なら sparse/0 へ据え直して契約を保つ.
        """
        models = [
            d for d in sorted(sparse_dir.iterdir())
            if d.is_dir() and (d / "cameras.bin").exists()
        ]
        if not models:
            raise RuntimeError("mapper produced no reconstruction (no sparse/*/cameras.bin)")

        recons = {d: colmap_model.read_model(d) for d in models}
        best = max(models, key=lambda d: len(recons[d].images))
        # summary はメモリ上の Reconstruction から作る (パス非依存). 後段の rename に影響されない.
        summary = recons[best].summary()
        summary["num_models"] = len(models)

        canonical = sparse_dir / "0"
        if best != canonical:
            tmp = sparse_dir / "_best.tmp"
            best.rename(tmp)
            if canonical.exists():
                shutil.rmtree(canonical)
            tmp.rename(canonical)
        summary["_model_dir"] = str(canonical)
        return summary

    # -- ALIKED backend (shared by both modes) -----------------------------------
    def _aliked_extract_and_match(
        self, ctx, colmap_bin, db_path, groups, *, model_id, camera_params, overlap, width, height, image_masks=None
    ) -> None:
        """ALIKED 全解像度抽出 + LightGlue マッチングを行い COLMAP DB へ書き込む.

        groups: prefix -> [(colmap_image_name, image_path)] (プレフィックス毎に 1 カメラ).
        width/height: DB へ登録するカメラ画像寸法 (モデル依存の camera_params とは別に明示).
        image_masks: {colmap_image_name -> mask png path}. 与えると mask=0 の画素の
          keypoint を捨てる (魚眼有効領域 + SAM3 動体除外). COLMAP の SIFT は mask_path で
          同等の処理をするが, ALIKED は自前抽出なのでここで間引く. None で無効 (pinhole).
        """
        import cv2  # noqa: PLC0415

        from ..colmap import database as colmap_db  # noqa: PLC0415
        from ..features.aliked_lightglue import AlikedLightGlue, Features  # noqa: PLC0415

        out = ctx.stage_out_dir
        logs_dir = out / "logs"

        if db_path.exists():
            db_path.unlink()
        colmap_runner.database_creator(colmap_bin, database_path=db_path)

        # extraction_device / extract_max_size をステージパラメータで上書き (UI からの選択を反映).
        updates: dict = {"extraction_device": ctx.params["extraction_device"]}
        if ctx.params.get("extract_max_size", 0) > 0:
            updates["max_extract_size"] = ctx.params["extract_max_size"]
        cfg = get_settings().aliked.model_copy(update=updates)
        engine = AlikedLightGlue(cfg)
        engine.load()
        total = sum(len(v) for v in groups.values())
        ctx.progress.info(
            f"ALIKED extract: image={width}x{height}, extract_cap={cfg.max_extract_size}px, "
            f"device_pref={cfg.extraction_device}, {total} imgs, LightGlue overlap={overlap}. "
            f"上限超の画像は縮小抽出し keypoint を原寸へ戻す (VRAM/RAM クラッシュ回避).",
            progress=0.08,
            key="log.recon_aliked_extract_start",
            args={
                "w": width,
                "h": height,
                "cap": cfg.max_extract_size,
                "device": cfg.extraction_device,
                "imgs": total,
                "overlap": overlap,
            },
        )

        feats: dict[str, Features] = {}
        total = sum(len(v) for v in groups.values())
        done = 0
        warned_fallback = False
        with colmap_db.ColmapDatabase(db_path) as db:
            for _prefix, items in groups.items():
                cam_id = db.add_camera(model_id, width, height, list(camera_params))
                for name, img_path in items:
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"cannot read {img_path}")
                    ft = engine.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    # GPU OOM で CPU へ落ちたら Console に 1 度だけ警告 (以降 CPU で遅くなる).
                    if engine.cpu_fallback_reason and not warned_fallback:
                        warned_fallback = True
                        ctx.progress.warn(
                            f"ALIKED fell back to CPU: {engine.cpu_fallback_reason}. "
                            f"抽取分辨率上限を画像長辺より小さく (例 2048) すると GPU に載る.",
                            key="log.recon_aliked_cpu_fallback",
                            args={"reason": engine.cpu_fallback_reason},
                        )
                    mp = image_masks.get(name) if image_masks else None
                    kp, ds, sc = _mask_filter(ft, mp)
                    img_id = db.add_image(name, cam_id)
                    db.add_keypoints(img_id, kp)
                    feats[name] = Features(kp, ds, sc, ft.image_size)
                    done += 1
                    if done % 8 == 0 or done == total:
                        ctx.progress.tick(
                            progress=0.08 + 0.32 * (done / max(1, total)),
                            message=f"ALIKED extract {done}/{total} (device={engine.extractor_device})",
                            key="log.recon_aliked_extract",
                            args={"done": done, "total": total, "device": engine.extractor_device},
                        )

        # 各グループ内の連続フレーム対を LightGlue でマッチ (グループ跨ぎはしない).
        pairs: list[tuple[str, str, object]] = []
        total_pairs = sum(
            min(i + 1 + overlap, len(items)) - (i + 1)
            for items in groups.values()
            for i in range(len(items))
        )
        mdone = 0
        for _prefix, items in groups.items():
            names = [n for n, _ in items]
            for i in range(len(names)):
                for j in range(i + 1, min(i + 1 + overlap, len(names))):
                    m = engine.match(feats[names[i]], feats[names[j]])
                    if len(m) > 0:
                        pairs.append((names[i], names[j], m))
                    mdone += 1
                    if mdone % 32 == 0 or mdone == total_pairs:
                        ctx.progress.tick(
                            progress=0.40 + 0.08 * (mdone / max(1, total_pairs)),
                            message=f"LightGlue match {mdone}/{total_pairs}",
                            key="log.recon_lightglue_match",
                            args={"done": mdone, "total": total_pairs},
                        )
        engine.unload()

        match_list = out / "aliked_matches.txt"
        colmap_db.write_match_list(pairs, match_list)
        ctx.progress.info(
            f"ALIKED: {total} images, {len(pairs)} pairs -> matches_importer",
            progress=0.48,
            key="log.recon_aliked_pairs",
            args={"images": total, "pairs": len(pairs)},
        )
        colmap_runner.matches_importer(
            colmap_bin, database_path=db_path, match_list_path=match_list, match_type="raw",
            min_num_inliers=(ctx.params["two_view_min_num_inliers"] or 15),
            log_path=logs_dir / "matches_importer.log", on_line=_logline(ctx, "match"),
        )


def _mask_filter(ft, mask_path):
    """ALIKED Features を mask (使う所=255) 内の keypoint だけに絞る. mask_path=None で無加工."""
    if mask_path is None:
        return ft.keypoints, ft.descriptors, ft.scores
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return ft.keypoints, ft.descriptors, ft.scores
    kp = ft.keypoints
    h, w = m.shape[:2]
    xs = np.clip(kp[:, 0].astype(int), 0, w - 1)
    ys = np.clip(kp[:, 1].astype(int), 0, h - 1)
    keep = m[ys, xs] > 0
    return kp[keep], ft.descriptors[keep], ft.scores[keep]


def _write_circle_mask(w: int, h: int, cx: float, cy: float, r: float, path: Path) -> None:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
    cv2.imwrite(str(path), mask)


def _logline(ctx: StageContext, prefix: str):
    def _cb(line: str) -> None:
        if line.strip():
            # COLMAP の逐行出力は spam なので progress イベント (Console 非表示) として流す.
            ctx.progress.tick(message=f"[{prefix}] {line}")
    return _cb


def _mapper_progress(ctx: StageContext, n_total: int, lo: float = 0.55, hi: float = 0.93):
    """COLMAP mapper の "Registering image" 行を数えて進捗 tick を出す (行自体も hidden tick).

    mapper は最長フェーズだが逐行出力しか無いため, 登録画像数 / 総数で 0.55→0.93 を進める.
    """
    import re  # noqa: PLC0415

    pat = re.compile(r"Registering image")
    state = {"n": 0}

    def _cb(line: str) -> None:
        if not line.strip():
            return
        ctx.progress.tick(message=f"[mapper] {line}")
        if pat.search(line):
            state["n"] += 1
            frac = min(1.0, state["n"] / max(1, n_total))
            ctx.progress.tick(
                progress=lo + (hi - lo) * frac,
                message=f"mapper: registered {state['n']}/{n_total}",
                key="log.recon_mapper_progress",
                args={"done": state["n"], "total": n_total},
            )

    return _cb


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
