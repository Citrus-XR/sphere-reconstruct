"""extract_frames ステージ.

パラメータ:
  interval_sec: float   何秒に 1 枚抜くか (デフォルト 1.0)
  max_frames: int       安全リミット (0 なら無制限)

出力:
  <project>/extract_frames/lens0/lens0_XXXXXX.jpg
  <project>/extract_frames/lens1/lens1_XXXXXX.jpg   (INSV のみ)
  <project>/extract_frames/manifest_frames.json     (frame_index / timestamp のマップ)

ERP video は単一 stream を抽出し, ERP image folder は元画像を参照する manifest を作る.
"""

from __future__ import annotations

import json
import tempfile
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
            key="log.extract_probe",
            args={"streams": len(probe.video_streams), "duration": probe.duration},
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
            raise RuntimeError(f"expected 2 matching video streams, got {len(probe.video_streams)}")
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
            raise RuntimeError(f"no frames to extract (duration={duration}s, interval={interval}s)")

        mode = ctx.params["selection_mode"]
        selection: dict = {"mode": mode}
        frame_scores: dict[int, dict] = {}
        if mode == "spatial":
            # 2 層多基準の空間抽出. 固定間隔を使わず, 品質 + 運動量で選ぶ.
            indices, spatial_stats, frame_scores = self._select_spatial_indices(
                ctx,
                fps=fps,
                duration=duration,
                nb_frames=lens0.nb_frames,
                ffmpeg_bin=ffmpeg_bin,
                fallback_count=len(indices),
            )
            selection.update(spatial_stats)
        elif mode == "sharpness" or ctx.params["sharpness_candidates"] > 1:
            # 各区間で候補を抜き, lens0 の鮮鋭度が最大の frame へ差し替える.
            n_candidates = max(2, ctx.params["sharpness_candidates"])
            indices, frame_scores = self._refine_by_sharpness(
                ctx,
                indices,
                fps=fps,
                interval=interval,
                n_candidates=n_candidates,
                nb_frames=lens0.nb_frames,
                ffmpeg_bin=ffmpeg_bin,
            )
        selection["selected"] = len(indices)

        # spatial は候補抽出+スコアリングで 0.60 まで進むため, 最終抽出はそこから継ぐ.
        # interval/sharpness は前処理が軽いので 0.1 から.
        prog_lo = 0.62 if mode == "spatial" else 0.1
        ctx.progress.info(
            f"extracting {len(indices)} paired frames (mode={mode}) from {duration:.1f}s",
            progress=prog_lo,
            key="log.extract_paired",
            args={"count": len(indices), "mode": mode, "duration": round(duration, 1)},
        )

        out_lens0 = ctx.stage_out_dir / "lens0"
        out_lens1 = ctx.stage_out_dir / "lens1"
        done_by_lens = {"lens0": 0, "lens1": 0}

        def prog(lens: str, cur: int, tot: int) -> None:
            done_by_lens[lens] = cur
            avg = (done_by_lens["lens0"] + done_by_lens["lens1"]) / (2 * tot)
            ctx.progress.tick(
                progress=prog_lo + (0.98 - prog_lo) * avg,
                message=f"extract {lens} {cur}/{tot}",
                key="log.extract_progress",
                args={"lens": lens, "cur": cur, "tot": tot},
            )

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
                **({"score": frame_scores[indices[i]]} if indices[i] in frame_scores else {}),
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
                    "selection": selection,
                    "frames": frame_records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest.outputs = [_file_ref(p, ctx.project_dir, "image/jpeg", ctx) for p in (p0 + p1)] + [
            _file_ref(ctx.stage_out_dir / "manifest_frames.json", ctx.project_dir, "application/json", ctx)
        ]
        ctx.progress.info(
            f"extract_frames done: {len(indices)} frames",
            progress=1.0,
            key="log.extract_done",
            args={"count": len(indices)},
        )

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
    ) -> tuple[list[int], dict]:
        """各区間で lens0 の候補フレームを抜き, Laplacian 分散が最大の index を選ぶ.

        候補は区間幅 (interval*fps) に均等配置する. lens0 のみで判定し (前後鏡頭は
        露光同期しているため片側で十分), 選ばれた index と per-frame スコアを返す. 判定用の
        候補 JPEG は一時ディレクトリに出して使い捨てる.
        """
        from ..imaging import sampling  # 遅延 import (cv2).

        span = max(1, int(interval * fps))
        bound = (nb_frames - 1) if nb_frames else None
        tmp_dir = Path(tempfile.mkdtemp(prefix=".extract-frames-sharpness-", dir=ctx.project_dir))

        candidate_groups = [
            sampling.candidate_indices(center, span, n_candidates, fps_bound=bound) for center in indices
        ]
        all_candidates = sorted({candidate for group in candidate_groups for candidate in group})
        paths = ffmpeg.extract_frames_sequential(
            ctx.source_path,
            stream_index=0,
            frame_indices=all_candidates,
            out_dir=tmp_dir,
            out_prefix="candidate",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda current, total: ctx.progress.tick(
                progress=0.05 + 0.02 * current / max(1, total),
                message=f"sharpness candidates {current}/{total}",
            ),
        )
        path_by_index = dict(zip(all_candidates, paths, strict=True))
        refined: list[int] = []
        pick_scores: dict[int, dict] = {}
        for i, cands in enumerate(candidate_groups):
            if len(cands) == 1:
                refined.append(cands[0])
                continue
            scores = [sampling.sharpness_of_file(path_by_index[candidate]) for candidate in cands]
            best = sampling.pick_sharpest(scores)
            refined.append(cands[best])
            pick_scores[cands[best]] = {"sharpness": round(float(scores[best]), 1)}
            ctx.progress.tick(
                progress=0.07 + 0.02 * ((i + 1) / len(indices)),
                message=f"sharpness interval {i + 1}/{len(indices)}: picked frame {cands[best]} "
                f"(score {scores[best]:.1f})",
                key="log.extract_sharpness_interval",
                args={
                    "cur": i + 1,
                    "tot": len(indices),
                    "frame": cands[best],
                    "score": round(scores[best], 1),
                },
            )

        # 使い捨ての候補画像を削除.
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return sorted(set(refined)), pick_scores

    def _select_spatial_indices(
        self,
        ctx: StageContext,
        *,
        fps: float,
        duration: float,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
        fallback_count: int,
    ) -> tuple[list[int], dict, dict]:
        """2 層多基準の空間抽出. (選ばれた source frame index, 統計, per-frame スコア) を返す.

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
            key="log.extract_spatial_candidates",
            args={"count": len(cand_indices), "fps": cand_fps},
        )
        tmp_dir = Path(tempfile.mkdtemp(prefix=".extract-frames-spatial-", dir=ctx.project_dir))
        paths = ffmpeg.extract_frames_sequential(
            ctx.source_path,
            stream_index=0,
            frame_indices=cand_indices,
            out_dir=tmp_dir,
            out_prefix="cand",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda cur, tot: ctx.progress.tick(
                progress=0.05 + 0.30 * (cur / max(1, tot)),
                message=f"spatial: extracting candidate frames {cur}/{tot}",
                key="log.extract_spatial_candidates_progress",
                args={"cur": cur, "tot": tot},
            ),
        )

        # 快速層 + 精確層のスコアリング. 光流用に縮小グレースケールを保持.
        grays: dict[int, object] = {}
        candidates: list[sampling.Candidate] = []
        max_clip = ctx.params["max_clip"]
        n_score = len(cand_indices)
        for si, (idx, p) in enumerate(zip(cand_indices, paths, strict=True)):
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
                    index=idx,
                    timestamp_us=int(idx / fps * 1_000_000),
                    sharpness=sharp,
                    exposure_ok=exp_ok,
                    feature_count=feats,
                )
            )
            if (si + 1) % 4 == 0 or si + 1 == n_score:
                ctx.progress.tick(
                    progress=0.35 + 0.22 * ((si + 1) / max(1, n_score)),
                    message=f"spatial: scoring candidate {si + 1}/{n_score}",
                    key="log.extract_spatial_scoring",
                    args={"cur": si + 1, "tot": n_score},
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
            progress=0.60,
            key="log.extract_spatial_selected",
            args={
                "selected": len(result.selected_indices),
                "total": len(candidates),
                "reasons": str(dict(result.reasons)),
            },
        )
        stats = {"candidates": len(candidates), "reasons": dict(result.reasons)}
        # 選ばれた frame ごとの品質スコア (鮮鋭度 + 特徴数).
        cand_by_index = {c.index: c for c in candidates}
        scores = {
            idx: {
                "sharpness": round(float(cand_by_index[idx].sharpness), 1),
                "features": int(cand_by_index[idx].feature_count),
            }
            for idx in result.selected_indices
            if idx in cand_by_index
        }
        if not result.selected_indices:
            # 全滅時は素朴な等間隔にフォールバック.
            ctx.progress.warn(
                "spatial selection empty; falling back to fixed interval",
                key="log.extract_spatial_fallback",
            )
            step = max(1, len(cand_indices) // max(1, fallback_count))
            return cand_indices[::step], {**stats, "fallback": True}, {}
        return result.selected_indices, stats, scores

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
            f"extracting {len(indices)} ERP frames @ interval={interval}s",
            progress=0.1,
            key="log.extract_erp",
            args={"count": len(indices), "interval": interval},
        )

        out = ctx.stage_out_dir / "erp"
        paths = ffmpeg.extract_frames_sequential(
            ctx.source_path,
            stream_index=0,
            frame_indices=indices,
            out_dir=out,
            out_prefix="erp",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda cur, tot: ctx.progress.tick(
                progress=0.1 + 0.85 * (cur / tot),
                message=f"extract {cur}/{tot}",
                key="log.extract_progress_simple",
                args={"cur": cur, "tot": tot},
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
                    "selection": {"mode": "interval", "selected": len(paths)},
                    "frames": frames,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest.outputs = [_file_ref(p, ctx.project_dir, "image/jpeg", ctx) for p in paths] + [
            _file_ref(ctx.stage_out_dir / "manifest_frames.json", ctx.project_dir, "application/json", ctx)
        ]
        ctx.progress.info(
            f"extract_frames done: {len(paths)} frames",
            progress=1.0,
            key="log.extract_done",
            args={"count": len(paths)},
        )

    # -- ERP images (フォルダ指定) --------------------------------------------------
    def _prepare_erp_images(self, ctx: StageContext, manifest: StageManifest) -> StageManifest:
        src = ctx.source_path
        assert src is not None and src.is_dir()
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
        if not imgs:
            raise RuntimeError(f"no images found under {src}")
        ctx.progress.info(
            f"erp_images: found {len(imgs)}",
            progress=0.5,
            key="log.extract_erp_images_found",
            args={"count": len(imgs)},
        )

        frames = [
            {"index": i, "erp_source": str(p)}  # コピーせず絶対パスで参照.
            for i, p in enumerate(imgs)
        ]
        (ctx.stage_out_dir / "manifest_frames.json").write_text(
            json.dumps(
                {
                    "kind": "erp_images",
                    "count": len(imgs),
                    "selection": {"mode": "folder", "selected": len(imgs)},
                    "frames": frames,
                },
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
        ctx.progress.info(
            f"extract_frames done: {len(imgs)} images",
            progress=1.0,
            key="log.extract_done",
            args={"count": len(imgs)},
        )
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
