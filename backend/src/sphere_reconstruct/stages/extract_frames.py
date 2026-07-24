"""extract_frames ステージ.

パラメータ:
  interval_sec: float   何秒に 1 枚抜くか (デフォルト 1.0)
  jpeg_quality: int     ffmpeg -q:v (デフォルト 3, 低いほど高品質)
  max_frames: int       安全リミット (0 なら無制限)

出力:
  <project>/extract_frames/lens0/lens0_XXXXXX.jpg
  <project>/extract_frames/lens1/lens1_XXXXXX.jpg   (INSV のみ)
  <project>/extract_frames/manifest_frames.json     (frame_index / timestamp のマップ)

ERP video 単一 stream / 単一 stream fisheye は Phase 3 の後半で追加. 現状は INSV のみ.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import ffmpeg, ffprobe
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class ExtractFrames(Stage):
    name = StageName.EXTRACT_FRAMES
    impl_version = "0.3"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        if ctx.source_path is None:
            return []
        p = ctx.source_path
        if not p.exists() or p.is_dir():
            return [FileRef(path=str(p), size=0, sha256="")]
        return [FileRef(path=str(p), size=p.stat().st_size, sha256=sha256_file(p))]

    def normalize_params(self, raw: dict) -> dict:
        return {
            "interval_sec": float(raw.get("interval_sec", 1.0)),
            "jpeg_quality": int(raw.get("jpeg_quality", 3)),
            "max_frames": int(raw.get("max_frames", 0)),
            # 選択モード:
            #   "interval"  … 固定時間間隔 (従来)
            #   "sharpness" … 各区間で最も鮮鋭なフレーム (sharpness_candidates 個から)
            #   "spatial"   … 2 層多基準の空間抽出 (鮮鋭度+露出+特徴+光流間隔)
            "selection_mode": str(raw.get("selection_mode", "interval")),
            "sharpness_candidates": int(raw.get("sharpness_candidates", 1)),
            # spatial 用パラメータ.
            "candidate_fps": float(raw.get("candidate_fps", 3.0)),
            "min_sharpness": float(raw.get("min_sharpness", 0.0)),
            "max_clip": float(raw.get("max_clip", 0.25)),
            "min_features": int(raw.get("min_features", 0)),
            "target_motion": float(raw.get("target_motion", 1.5)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        if ctx.source_path is None or ctx.source_kind is None:
            raise RuntimeError("source not set")

        settings = get_settings()
        ffmpeg_bin = settings.binaries.ffmpeg or None
        ffprobe_bin = settings.binaries.ffprobe or None

        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        if ctx.source_kind == "erp_images":
            # 画像フォルダはそのまま参照するのでコピーしない (次段が manifest_frames.json を見る).
            return self._prepare_erp_images(ctx, manifest)

        probe = ffprobe.probe(ctx.source_path, ffprobe_bin=ffprobe_bin)
        ctx.progress.info(
            f"probe: {len(probe.video_streams)} video streams, duration={probe.duration}s",
            progress=0.05,
        )

        if ctx.source_kind == "insv":
            self._extract_insv(ctx, manifest, probe, ffmpeg_bin)
        elif ctx.source_kind == "erp_video":
            self._extract_erp_video(ctx, manifest, probe, ffmpeg_bin)
        else:
            raise ValueError(f"unsupported source kind: {ctx.source_kind}")

        return manifest

    # -- INSV (dual lens) ---------------------------------------------------------
    def _extract_insv(
        self, ctx: StageContext, manifest: StageManifest, probe: ffprobe.ProbeResult, ffmpeg_bin: str | None
    ) -> None:
        pair = probe.dual_lens_streams()
        if pair is None:
            raise RuntimeError(
                f"expected 2 matching video streams, got {len(probe.video_streams)}"
            )
        lens0, lens1 = pair
        fps = lens0.fps
        duration = probe.duration or (lens0.nb_frames or 0) / fps
        interval = ctx.params["interval_sec"]
        if interval <= 0:
            raise ValueError("interval_sec must be > 0")

        # frame indices を fps で決定的に生成.
        n = int(duration / interval)
        indices = [int(i * interval * fps) for i in range(n)]
        max_frames = ctx.params["max_frames"]
        if max_frames > 0:
            indices = indices[:max_frames]

        if not indices:
            raise RuntimeError(
                f"no frames to extract (duration={duration}s, interval={interval}s)"
            )

        mode = ctx.params["selection_mode"]
        if mode == "spatial":
            # 2 層多基準の空間抽出. 固定間隔を使わず, 品質 + 運動量で選ぶ.
            indices = self._select_spatial_indices(
                ctx, fps=fps, duration=duration, nb_frames=lens0.nb_frames,
                ffmpeg_bin=ffmpeg_bin, fallback_count=len(indices),
            )
        elif mode == "sharpness" or ctx.params["sharpness_candidates"] > 1:
            # 各区間で候補を抜き, lens0 の鮮鋭度が最大の frame へ差し替える.
            n_candidates = max(2, ctx.params["sharpness_candidates"])
            indices = self._refine_by_sharpness(
                ctx, indices, fps=fps, interval=interval, n_candidates=n_candidates,
                nb_frames=lens0.nb_frames, ffmpeg_bin=ffmpeg_bin,
            )

        ctx.progress.info(
            f"extracting {len(indices)} paired frames (mode={mode}) from {duration:.1f}s",
            progress=0.1,
        )

        out_lens0 = ctx.stage_out_dir / "lens0"
        out_lens1 = ctx.stage_out_dir / "lens1"
        total = len(indices)
        done_by_lens = {"lens0": 0, "lens1": 0}

        def prog(lens: str, cur: int, tot: int) -> None:
            done_by_lens[lens] = cur
            avg = (done_by_lens["lens0"] + done_by_lens["lens1"]) / (2 * tot)
            ctx.progress.info(f"extract {lens} {cur}/{tot}", progress=0.1 + 0.85 * avg)

        p0, p1 = ffmpeg.extract_paired_frames(
            ctx.source_path,
            fps=fps,
            frame_indices=indices,
            out_dir_lens0=out_lens0,
            out_dir_lens1=out_lens1,
            ffmpeg_bin=ffmpeg_bin,
            progress=prog,
        )

        frame_records = [
            {
                "index": i,
                "source_frame": indices[i],
                "timestamp_sec": indices[i] / fps,
                "lens0": _final_relpath(p0[i], ctx),
                "lens1": _final_relpath(p1[i], ctx),
            }
            for i in range(len(indices))
        ]
        (ctx.stage_out_dir / "manifest_frames.json").write_text(
            json.dumps(
                {
                    "kind": "insv_dual",
                    "fps": fps,
                    "width": lens0.width,
                    "height": lens0.height,
                    "count": len(indices),
                    "frames": frame_records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest.outputs = [
            _file_ref(p, ctx.project_dir, "image/jpeg", ctx) for p in (p0 + p1)
        ] + [
            _file_ref(ctx.stage_out_dir / "manifest_frames.json", ctx.project_dir, "application/json", ctx)
        ]

    def _refine_by_sharpness(
        self,
        ctx: StageContext,
        indices: list[int],
        *,
        fps: float,
        interval: float,
        n_candidates: int,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
    ) -> list[int]:
        """各区間で lens0 の候補フレームを抜き, Laplacian 分散が最大の index を選ぶ.

        候補は区間幅 (interval*fps) に均等配置する. lens0 のみで判定し (前後鏡頭は
        露光同期しているため片側で十分), 選ばれた index を返す. 判定用の候補 JPEG は
        一時ディレクトリに出して使い捨てる.
        """
        from ..imaging import sampling  # 遅延 import (cv2).

        span = max(1, int(interval * fps))
        bound = (nb_frames - 1) if nb_frames else None
        tmp_dir = ctx.stage_out_dir / "_sharpness_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        refined: list[int] = []
        for i, center in enumerate(indices):
            cands = sampling.candidate_indices(center, span, n_candidates, fps_bound=bound)
            if len(cands) == 1:
                refined.append(cands[0])
                continue
            paths = ffmpeg.extract_frames_by_index(
                ctx.source_path,
                stream_index=0,
                fps=fps,
                frame_indices=cands,
                out_dir=tmp_dir / f"interval_{i:04d}",
                out_prefix="cand",
                ffmpeg_bin=ffmpeg_bin,
            )
            scores = [sampling.sharpness_of_file(p) for p in paths]
            best = sampling.pick_sharpest(scores)
            refined.append(cands[best])
            ctx.progress.info(
                f"sharpness interval {i + 1}/{len(indices)}: picked frame {cands[best]} "
                f"(score {scores[best]:.1f})",
                progress=0.05 + 0.04 * ((i + 1) / len(indices)),
            )

        # 使い捨ての候補画像を削除.
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return refined

    def _select_spatial_indices(
        self,
        ctx: StageContext,
        *,
        fps: float,
        duration: float,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
        fallback_count: int,
    ) -> list[int]:
        """2 層多基準の空間抽出.

        快速層: 候補を密に抜き, 各フレームの sharpness (Laplacian) と exposure を測る.
        精確層: 快速層を通ったフレームに SIFT 特徴数を付ける.
        貪欲間隔: 光流中央値を運動量として, target_motion 間隔で高品質フレームを選ぶ.

        候補は lens0 のみをスコアリングに使い (前後は露光同期), 判定用に縮小グレースケール
        をメモリに保持する. 選ばれた source frame index を返す.
        """
        import cv2  # noqa: PLC0415

        from ..imaging import quality, sampling  # noqa: PLC0415

        cand_fps = ctx.params["candidate_fps"]
        bound = (nb_frames - 1) if nb_frames else None
        n_cand = max(2, int(duration * cand_fps))
        cand_indices = sorted({int(i / cand_fps * fps) for i in range(n_cand)})
        if bound is not None:
            cand_indices = sorted({min(bound, i) for i in cand_indices})

        ctx.progress.info(
            f"spatial: extracting {len(cand_indices)} candidate frames @ {cand_fps}fps",
            progress=0.06,
        )
        tmp_dir = ctx.stage_out_dir / "_spatial_tmp"
        paths = ffmpeg.extract_frames_by_index(
            ctx.source_path, stream_index=0, fps=fps, frame_indices=cand_indices,
            out_dir=tmp_dir, out_prefix="cand", ffmpeg_bin=ffmpeg_bin,
        )

        # 快速層 + 精確層のスコアリング. 光流用に縮小グレースケールを保持.
        grays: dict[int, "object"] = {}
        candidates: list[sampling.Candidate] = []
        max_clip = ctx.params["max_clip"]
        for idx, p in zip(cand_indices, paths, strict=False):
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h, w = gray.shape[:2]
            small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
            grays[idx] = small
            sharp = sampling.laplacian_sharpness(small)
            exp_ok = quality.exposure_stats(small).is_ok(max_clip)
            feats = quality.sift_feature_count(small, downscale=1) if exp_ok else 0
            candidates.append(
                sampling.Candidate(
                    index=idx, timestamp_us=int(idx / fps * 1_000_000),
                    sharpness=sharp, exposure_ok=exp_ok, feature_count=feats,
                )
            )

        def motion_fn(a: int, b: int) -> float:
            ga, gb = grays.get(a), grays.get(b)
            if ga is None or gb is None:
                return 0.0
            return quality.optical_flow_median(ga, gb, downscale=1)

        cfg = sampling.SpatialConfig(
            min_sharpness=ctx.params["min_sharpness"],
            min_features=ctx.params["min_features"],
            target_motion=ctx.params["target_motion"],
            max_frames=ctx.params["max_frames"],
        )
        result = sampling.select_spatial(candidates, motion_fn, cfg)

        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

        ctx.progress.info(
            f"spatial: selected {len(result.selected_indices)} / {len(candidates)} candidates "
            f"(rejected fast: {result.reasons})",
            progress=0.09,
        )
        if not result.selected_indices:
            # 全滅時は素朴な等間隔にフォールバック.
            ctx.progress.warn("spatial selection empty; falling back to fixed interval")
            step = max(1, len(cand_indices) // max(1, fallback_count))
            return cand_indices[::step]
        return result.selected_indices

    # -- ERP video (単一 stream) ---------------------------------------------------
    def _extract_erp_video(
        self, ctx: StageContext, manifest: StageManifest, probe: ffprobe.ProbeResult, ffmpeg_bin: str | None
    ) -> None:
        if not probe.video_streams:
            raise RuntimeError("no video streams in ERP source")
        vs = probe.video_streams[0]
        fps = vs.fps
        duration = probe.duration or (vs.nb_frames or 0) / fps
        interval = ctx.params["interval_sec"]
        n = int(duration / interval)
        indices = [int(i * interval * fps) for i in range(n)]
        max_frames = ctx.params["max_frames"]
        if max_frames > 0:
            indices = indices[:max_frames]

        ctx.progress.info(
            f"extracting {len(indices)} ERP frames @ interval={interval}s", progress=0.1
        )

        out = ctx.stage_out_dir / "erp"
        paths = ffmpeg.extract_frames_by_index(
            ctx.source_path,
            stream_index=0,
            fps=fps,
            frame_indices=indices,
            out_dir=out,
            out_prefix="erp",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda cur, tot: ctx.progress.info(
                f"extract {cur}/{tot}", progress=0.1 + 0.85 * (cur / tot)
            ),
        )

        frames = [
            {
                "index": i,
                "source_frame": indices[i],
                "timestamp_sec": indices[i] / fps,
                "erp": _final_relpath(paths[i], ctx),
            }
            for i in range(len(paths))
        ]
        (ctx.stage_out_dir / "manifest_frames.json").write_text(
            json.dumps(
                {
                    "kind": "erp_video",
                    "fps": fps,
                    "width": vs.width,
                    "height": vs.height,
                    "count": len(paths),
                    "frames": frames,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest.outputs = [
            _file_ref(p, ctx.project_dir, "image/jpeg", ctx) for p in paths
        ] + [
            _file_ref(ctx.stage_out_dir / "manifest_frames.json", ctx.project_dir, "application/json", ctx)
        ]

    # -- ERP images (フォルダ指定) --------------------------------------------------
    def _prepare_erp_images(self, ctx: StageContext, manifest: StageManifest) -> StageManifest:
        src = ctx.source_path
        assert src is not None and src.is_dir()
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
        if not imgs:
            raise RuntimeError(f"no images found under {src}")
        ctx.progress.info(f"erp_images: found {len(imgs)}", progress=0.5)

        frames = [
            {"index": i, "erp_source": str(p)}  # コピーせず絶対パスで参照.
            for i, p in enumerate(imgs)
        ]
        (ctx.stage_out_dir / "manifest_frames.json").write_text(
            json.dumps(
                {"kind": "erp_images", "count": len(imgs), "frames": frames},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest.outputs = [
            _file_ref(
                ctx.stage_out_dir / "manifest_frames.json",
                ctx.project_dir,
                "application/json",
                ctx,
            )
        ]
        return manifest


def _final_relpath(p: Path, ctx: StageContext) -> str:
    """tmp ディレクトリ内のパスを, 原子置換後の最終パスに正規化する.

    engine が `.<stage>.tmp/` → `<stage>/` に rename するため, ここで manifest に
    書き込む文字列は最終形にする.
    """
    rel = p.relative_to(ctx.stage_out_dir)
    # ctx.stage_out_dir は project_dir / ".<stage>.tmp/" 直下. ステージ名は
    # stage_out_dir の parent 側 (project_dir) からみると .<stage>.tmp というディレクトリ名.
    final_stage_dir_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_stage_dir_name) / rel)


def _file_ref(p: Path, project_dir: Path, mime: str, ctx: StageContext) -> FileRef:
    return FileRef(
        path=_final_relpath(p, ctx),
        size=p.stat().st_size,
        sha256=sha256_file(p),
        mime=mime,
    )
