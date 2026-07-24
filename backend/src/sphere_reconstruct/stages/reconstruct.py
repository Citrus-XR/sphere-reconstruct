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
    impl_version = "0.6"

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
            # 特徴 backend: "sift" (COLMAP 内蔵) | "aliked" (ALIKED+LightGlue ONNX).
            "feature_backend": str(raw.get("feature_backend", "sift")),
            # 仮想 pinhole は既知 intrinsics でレンダリングしているので, 既定では
            # COLMAP に再推定させず固定する.
            "refine_intrinsics": bool(raw.get("refine_intrinsics", False)),
            # rig 拘束: 前後レンズ 6 視点 = 12 カメラの既知相対姿勢を固定する.
            "use_rig": bool(raw.get("use_rig", True)),
            # rig 外参を COLMAP に精修させるか. False にすると offset_v3 の校正を厳密に
            # 信頼して sensor_from_rig を固定する (剛性 rig). 既定 True (校正の微差を吸収).
            "refine_rig": bool(raw.get("refine_rig", True)),
            # sequential matching の loop closure. 既定 False:
            #   (1) rig 拘束で既に 100% 登録できるため通常不要,
            #   (2) COLMAP 4.x は 2025-05 に vocab tree を flann -> faiss へ変更しており,
            #       binaries.vocab_tree には faiss 形式の .bin が必要 (旧 flann は読めない).
            # faiss 形式の vocab tree を用意して明示的に有効化する場合のみ使う.
            "loop_closure": bool(raw.get("loop_closure", False)),
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
        #
        # レイアウトは view-major: <view>_lensN/frame_XXXXXX.jpg.
        # 理由: sequential_matcher は画像を名前のアルファベット順で並べ, 近傍 (overlap
        # 窓) 同士をマッチングする. frame-major (frame_XXXXXX/<view>) にすると, 近傍が
        # 「同一フレームの別視点」= 重ならない cubemap 面同士になり, 初期ペアが全く
        # 得られず mapper が失敗する. view-major なら近傍が「同一視点の連続フレーム」=
        # 十分に重なるため, 逐次マッチングが正しく働く.
        masks_manifest = None
        mm_path = ctx.project_dir / "generate_masks" / "manifest_masks.json"
        use_masks = ctx.params["use_masks"] and mm_path.exists()
        if use_masks:
            masks_manifest = json.loads(mm_path.read_text())
            masks_dir.mkdir(parents=True, exist_ok=True)

        def _colmap_name(view: str, lens: int, index: int) -> str:
            return f"{view}_lens{lens}/frame_{index:06d}.jpg"

        n_imgs = 0
        for fr in rig["frames"]:
            for v in fr["views"]:
                src_img = ctx.project_dir / v["path"]
                dst_img = images_dir / _colmap_name(v["view"], v["lens"], fr["index"])
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(src_img, dst_img)
                n_imgs += 1

        if use_masks and masks_manifest is not None:
            for fr in masks_manifest["frames"]:
                for v in fr["views"]:
                    src_mask = ctx.project_dir / v["path"]
                    # COLMAP mask 命名: <image_name>.png. image は .jpg なので <...>.jpg.png.
                    name = _colmap_name(v["view"], v["lens"], fr["index"])
                    dst_mask = masks_dir / f"{name}.png"
                    dst_mask.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(src_mask, dst_mask)

        db_path = out / "database.db"
        ctx.progress.info(f"colmap feature_extractor on {n_imgs} images", progress=0.05)

        def logline(prefix: str):
            def _cb(line: str) -> None:
                if line.strip():
                    ctx.progress.info(f"[{prefix}] {line}")
            return _cb

        # 既知 intrinsics を計算する. cubemap の全 view は同一の size/fov.
        # f = (size/2) / tan(fov/2), cx=cy=size/2.
        views_meta = rig.get("views", [])
        lenses_meta = rig.get("lenses", [])
        use_rig = ctx.params["use_rig"] and len(lenses_meta) >= 2
        camera_params = None
        camera_params_list = None
        if views_meta:
            import math

            size = int(views_meta[0]["size"])
            fov = float(views_meta[0]["fov_deg"])
            f = (size / 2.0) / math.tan(math.radians(fov) / 2.0)
            c = size / 2.0
            camera_params = f"{f:.6f},{f:.6f},{c:.6f},{c:.6f}"
            camera_params_list = [f, f, c, c]
            ctx.progress.info(f"known PINHOLE intrinsics: {camera_params}", progress=0.05)

        # 2) 特徴抽出 + マッチング (backend で分岐).
        backend = ctx.params["feature_backend"]
        if backend == "aliked":
            self._run_aliked_backend(
                ctx, colmap_bin, db_path, images_dir, rig, camera_params_list,
                overlap=ctx.params["overlap"], logline=logline,
            )
        else:
            # SIFT: feature_extractor -> matcher.
            colmap_runner.feature_extractor(
                colmap_bin,
                database_path=db_path,
                image_path=images_dir,
                camera_model="PINHOLE",
                single_camera=not use_rig,
                single_camera_per_folder=use_rig,
                camera_params=None if use_rig else camera_params,
                use_gpu=ctx.params["use_gpu"],
                mask_path=masks_dir if use_masks else None,
                log_path=logs_dir / "feature_extractor.log",
                on_line=logline("features"),
            )
            ctx.progress.info("colmap matching", progress=0.3)
            if ctx.params["matcher"] == "exhaustive":
                colmap_runner.exhaustive_matcher(
                    colmap_bin, database_path=db_path, use_gpu=ctx.params["use_gpu"],
                    log_path=logs_dir / "matcher.log", on_line=logline("match"),
                )
            else:
                vocab = settings.binaries.vocab_tree
                loop = ctx.params["loop_closure"] and bool(vocab)
                colmap_runner.sequential_matcher(
                    colmap_bin, database_path=db_path, overlap=ctx.params["overlap"],
                    loop_detection=loop,
                    vocab_tree_path=Path(vocab) if loop else None,
                    use_gpu=ctx.params["use_gpu"],
                    log_path=logs_dir / "matcher.log", on_line=logline("match"),
                )

        # 3) rig 拘束を DB に設定する (既知の前後レンズ相対姿勢を固定). backend 共通.
        if use_rig and camera_params_list is not None:
            from ..colmap import rig as colmap_rig

            cams = colmap_rig.compute_rig_cameras(views_meta, lenses_meta)
            rig_config = colmap_rig.build_rig_config(cams, camera_params_list)
            rig_cfg_path = out / "rig_config.json"
            rig_cfg_path.write_text(json.dumps(rig_config, indent=2), encoding="utf-8")
            ctx.progress.info(
                f"configuring rig: {len(cams)} cameras (view x lens), ref front_lens0",
                progress=0.5,
            )
            colmap_runner.rig_configurator(
                colmap_bin,
                database_path=db_path,
                rig_config_path=rig_cfg_path,
                log_path=logs_dir / "rig_configurator.log",
                on_line=logline("rig"),
            )

        ctx.progress.info("colmap mapper (incremental)", progress=0.55)
        # 4) mapper.
        colmap_runner.mapper(
            colmap_bin,
            database_path=db_path,
            image_path=images_dir,
            output_path=sparse_dir,
            refine_intrinsics=ctx.params["refine_intrinsics"],
            refine_rig=ctx.params["refine_rig"],
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

    def _run_aliked_backend(
        self, ctx, colmap_bin, db_path, images_dir, rig, camera_params_list, *, overlap, logline
    ) -> None:
        """ALIKED 抽出 + LightGlue マッチングを行い COLMAP DB へ書き込む.

        1. database_creator で空 DB を作る.
        2. 各 <view>_lensN フォルダ = 1 カメラ. 各画像に ALIKED keypoints を書く.
        3. 同一 view の連続フレーム対を LightGlue でマッチ -> match list.
        4. matches_importer (raw) で幾何検証して two_view_geometries を埋める.
        """
        import cv2  # noqa: PLC0415

        from ..colmap import database as colmap_db  # noqa: PLC0415
        from ..colmap import runner as colmap_runner  # noqa: PLC0415
        from ..features.aliked_lightglue import AlikedLightGlue  # noqa: PLC0415

        out = ctx.stage_out_dir
        logs_dir = out / "logs"
        if camera_params_list is None:
            raise RuntimeError("ALIKED backend requires known camera params (views meta)")

        # 1) 空 DB.
        if db_path.exists():
            db_path.unlink()
        colmap_runner.database_creator(colmap_bin, database_path=db_path)

        ctx.progress.info("loading ALIKED + LightGlue (onnxruntime)", progress=0.08)
        engine = AlikedLightGlue.from_settings()
        engine.load()

        # フォルダ (view_lens) -> フレーム画像リスト.
        # rig manifest から (view, lens, frame_index) を辿り, view-major の COLMAP 名を作る.
        groups: dict[str, list[tuple[int, str]]] = {}  # prefix -> [(frame_index, colmap_name)]
        for fr in rig["frames"]:
            for v in fr["views"]:
                prefix = f"{v['view']}_lens{v['lens']}"
                name = f"{prefix}/frame_{fr['index']:06d}.jpg"
                groups.setdefault(prefix, []).append((fr["index"], name))
        for g in groups.values():
            g.sort()

        # 2) カメラ + 画像 + keypoints.
        features: dict[str, object] = {}
        image_ids: dict[str, int] = {}
        total_imgs = sum(len(g) for g in groups.values())
        done = 0
        with colmap_db.ColmapDatabase(db_path) as db:
            for prefix, items in groups.items():
                cam_id = db.add_camera(
                    colmap_db.CAMERA_MODEL_PINHOLE,
                    int(camera_params_list[2] * 2), int(camera_params_list[3] * 2),
                    list(camera_params_list),
                )
                for _frame_idx, name in items:
                    img_path = images_dir / name
                    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"cannot read {img_path}")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    feat = engine.extract(rgb)
                    img_id = db.add_image(name, cam_id)
                    db.add_keypoints(img_id, feat.keypoints)
                    features[name] = feat
                    image_ids[name] = img_id
                    done += 1
                    if done % 12 == 0 or done == total_imgs:
                        ctx.progress.info(
                            f"ALIKED extract {done}/{total_imgs}",
                            progress=0.08 + 0.32 * (done / max(1, total_imgs)),
                        )

        # 3) 同一 view の連続フレーム対を LightGlue でマッチ.
        pairs: list[tuple[str, str, object]] = []
        n_pairs = 0
        for prefix, items in groups.items():
            names = [n for _, n in items]
            for i in range(len(names)):
                for j in range(i + 1, min(i + 1 + overlap, len(names))):
                    m = engine.match(features[names[i]], features[names[j]])
                    if len(m) > 0:
                        pairs.append((names[i], names[j], m))
                    n_pairs += 1
            ctx.progress.info(f"LightGlue matched view {prefix}", progress=0.45)
        engine.unload()

        match_list = out / "aliked_matches.txt"
        colmap_db.write_match_list(pairs, match_list)
        ctx.progress.info(
            f"ALIKED: {total_imgs} images, {len(pairs)}/{n_pairs} non-empty pairs -> matches_importer",
            progress=0.48,
        )

        # 4) 幾何検証.
        colmap_runner.matches_importer(
            colmap_bin,
            database_path=db_path,
            match_list_path=match_list,
            match_type="raw",
            log_path=logs_dir / "matches_importer.log",
            on_line=logline("match"),
        )


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
