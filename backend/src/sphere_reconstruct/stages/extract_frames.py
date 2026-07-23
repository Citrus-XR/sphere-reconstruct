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
    impl_version = "0.1"

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
        ctx.progress.info(
            f"extracting {len(indices)} paired frames @ interval={interval}s from {duration:.1f}s",
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
