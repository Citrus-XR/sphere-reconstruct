"""LFStudio 学習画像用の時系列ノイズ除去ステージ.

Camera pose は raw frame から求めた結果を保持し、この出力は export_dataset だけが選択的に
利用する。FastDVDnet は source-rate の 5 frame を直接 decode するため、疎な抽出 JPEG を
時間方向に平均して ghost を作ることはない。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from ..denoise import MODEL_SHA256, FastDvdnetEngine, ensure_model
from ..denoise.engine import DEFAULT_TILE_OVERLAP, MIN_TILE_OVERLAP
from ..denoise.weights import MODEL_ID
from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..imaging import ffmpeg, ffprobe
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.stage import Stage, StageContext, new_manifest
from ..settings import get_settings

_METHODS = {"off", "fastdvdnet", "ffmpeg_adaptive"}


@register
class DenoiseFrames(Stage):
    name = StageName.DENOISE_FRAMES
    impl_version = "1.0"

    def normalize_params(self, raw: dict) -> dict:
        method = str(raw.get("method", "off")).lower()
        if method not in _METHODS:
            raise ValueError(f"未対応のノイズ除去方式: {method}")
        sigma = float(raw.get("sigma", 10.0))
        tile_size = int(raw.get("tile_size", 512))
        tile_overlap = int(raw.get("tile_overlap", DEFAULT_TILE_OVERLAP))
        jpeg_quality = int(raw.get("jpeg_quality", 98))
        temporal_window = int(raw.get("temporal_window", 5))
        if not 0 <= sigma <= 55:
            raise ValueError("ノイズ強度 sigma は [0, 55] の範囲で指定してください")
        if tile_size < 128 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
            raise ValueError("ノイズ除去タイルの寸法が不正です")
        if method == "fastdvdnet" and tile_overlap < MIN_TILE_OVERLAP:
            raise ValueError(f"FastDVDnet tile_overlap は {MIN_TILE_OVERLAP} px 以上が必要です")
        if not 80 <= jpeg_quality <= 100:
            raise ValueError("ノイズ除去 JPEG 品質は [80, 100] の範囲で指定してください")
        if temporal_window < 5 or temporal_window > 129 or temporal_window % 2 == 0:
            raise ValueError("temporal_window は [5, 129] の奇数で指定してください")
        return {
            "method": method,
            "sigma": sigma,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "jpeg_quality": jpeg_quality,
            "temporal_window": temporal_window,
            "luma_only": bool(raw.get("luma_only", True)),
            "requested_device": get_settings().denoise.device,
            "hardware_decode": get_settings().denoise.hardware_decode,
            "model_id": MODEL_ID,
            "model_sha256": MODEL_SHA256,
        }

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        candidates = [ctx.project_dir / "manifests" / "extract_frames.json"]
        if ctx.source_path is not None:
            candidates.append(ctx.source_path)
        configured = get_settings().denoise.model_path
        if ctx.params["method"] == "fastdvdnet" and configured:
            candidates.append(Path(configured).expanduser().resolve())
        return [
            FileRef(
                path=(
                    str(path.relative_to(ctx.project_dir))
                    if path.is_relative_to(ctx.project_dir)
                    else str(path)
                ),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in candidates
            if path.is_file()
        ]

    def execute(self, ctx: StageContext) -> StageManifest:
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params
        frames_path = ctx.project_dir / "extract_frames" / "manifest_frames.json"
        if not frames_path.is_file():
            raise RuntimeError("denoise_frames より先に extract_frames を実行してください")
        source_manifest = json.loads(frames_path.read_text(encoding="utf-8"))
        method = ctx.params["method"]
        if method == "off":
            return self._write_manifest(ctx, manifest, source_manifest, frames=[], device=None)
        if ctx.source_path is None or ctx.source_kind not in {"insv", "erp_video"}:
            raise RuntimeError("時系列ノイズ除去には INSV または ERP 動画が必要です")

        settings = get_settings()
        probe = ffprobe.probe(ctx.source_path, ffprobe_bin=settings.binaries.ffprobe or None)
        streams = self._streams(ctx, source_manifest, probe)
        total_outputs = sum(len(stream["frames"]) for stream in streams)
        completed = 0
        frame_results: dict[int, dict] = {
            int(frame["index"]): {
                "index": int(frame["index"]),
                "source_frame": int(frame["source_frame"]),
                "timestamp_sec": frame.get("timestamp_sec"),
            }
            for frame in source_manifest["frames"]
        }

        engine = None
        device = None
        if method == "fastdvdnet":
            ctx.progress.info("FastDVDnet model を確認中", progress=0.02, key="log.denoise_model")

            def download_progress(written: int, total: int | None) -> None:
                ratio = written / total if total else 0.0
                ctx.progress.tick(
                    progress=0.02 + 0.04 * min(1.0, ratio),
                    message=f"FastDVDnet model {written}/{total or '?'} bytes",
                    key="log.denoise_download",
                    args={"written": written, "total": total or 0},
                )

            weights = ensure_model(settings.denoise.model_path, on_progress=download_progress)
            engine = FastDvdnetEngine(weights, settings.denoise.device)

        ctx.progress.info(
            f"時系列ノイズ除去開始: method={method}, 画像={total_outputs}",
            progress=0.08,
            key="log.denoise_start",
            args={"method": method, "images": total_outputs},
        )

        def advance() -> None:
            nonlocal completed
            completed += 1
            ctx.progress.tick(
                progress=0.08 + 0.88 * completed / max(1, total_outputs),
                message=f"ノイズ除去 {completed}/{total_outputs}",
                key="log.denoise_progress",
                args={"cur": completed, "tot": total_outputs},
            )

        last_detail_at = 0.0

        def detail(fraction: float) -> None:
            nonlocal last_detail_at
            now = time.monotonic()
            if fraction < 1.0 and now - last_detail_at < 0.5:
                return
            last_detail_at = now
            current = completed + min(1.0, max(0.0, fraction))
            ctx.progress.tick(
                progress=0.08 + 0.88 * current / max(1, total_outputs),
                message=f"ノイズ除去 {current:.2f}/{total_outputs}",
                key="log.denoise_progress",
                args={"cur": round(current, 2), "tot": total_outputs},
            )

        for stream in streams:
            output_dir = ctx.stage_out_dir / "images" / stream["folder"]
            if method == "fastdvdnet":
                assert engine is not None
                results = _run_fastdvdnet_stream(
                    ctx.source_path,
                    stream_index=stream["stream_index"],
                    frames=stream["frames"],
                    frame_count=stream["frame_count"],
                    width=stream["width"],
                    height=stream["height"],
                    output_dir=output_dir,
                    engine=engine,
                    sigma=ctx.params["sigma"],
                    tile_size=ctx.params["tile_size"],
                    tile_overlap=ctx.params["tile_overlap"],
                    jpeg_quality=ctx.params["jpeg_quality"],
                    ffmpeg_bin=settings.binaries.ffmpeg or None,
                    scratch_dir=ctx.stage_out_dir / ".decode",
                    hardware_decode=ctx.params["hardware_decode"],
                    pixel_format=stream["pixel_format"],
                    on_frame=advance,
                    on_detail=detail,
                )
                device = engine.device
            else:
                indices = [int(frame["source_frame"]) for frame in stream["frames"]]
                progress_seen = 0

                def adaptive_progress(current: int, _total: int) -> None:
                    nonlocal progress_seen
                    while progress_seen < current:
                        progress_seen += 1
                        advance()

                paths = ffmpeg.extract_adaptive_denoised_frames(
                    ctx.source_path,
                    stream_index=stream["stream_index"],
                    frame_indices=indices,
                    out_dir=output_dir,
                    out_prefix="frame",
                    temporal_window=ctx.params["temporal_window"],
                    luma_only=ctx.params["luma_only"],
                    jpeg_quality=max(2, round((100 - ctx.params["jpeg_quality"]) / 3)),
                    ffmpeg_bin=settings.binaries.ffmpeg or None,
                    hardware_decode=ctx.params["hardware_decode"],
                    pixel_format=stream["pixel_format"],
                    progress=adaptive_progress,
                )
                results = [
                    {"path": path, "mean_abs_delta": None, "mean_rgb_delta": None}
                    for path in paths
                ]
            for source_frame, denoise_result in zip(stream["frames"], results, strict=True):
                record = frame_results[int(source_frame["index"])]
                record[stream["key"]] = _final_relpath(denoise_result["path"], ctx)
                record.setdefault("metrics", {})[stream["key"]] = {
                    "mean_abs_delta": denoise_result["mean_abs_delta"],
                    "mean_rgb_delta": denoise_result["mean_rgb_delta"],
                }

        ordered = [frame_results[index] for index in sorted(frame_results)]
        stage_manifest = self._write_manifest(
            ctx, manifest, source_manifest, frames=ordered, device=device
        )
        ctx.progress.info(
            f"時系列ノイズ除去完了: method={method}, 画像={total_outputs}",
            progress=1.0,
            key="log.denoise_done",
            args={"method": method, "images": total_outputs},
        )
        return stage_manifest

    def _write_manifest(
        self,
        ctx: StageContext,
        manifest: StageManifest,
        source_manifest: dict,
        *,
        frames: list[dict],
        device: str | None,
    ) -> StageManifest:
        payload = {
            "kind": source_manifest.get("kind"),
            "method": ctx.params["method"],
            "device": device,
            "requested_device": ctx.params["requested_device"],
            "hardware_decode": ctx.params["hardware_decode"],
            "width": source_manifest.get("width"),
            "height": source_manifest.get("height"),
            "count": source_manifest.get("count", len(frames)),
            "params": ctx.params,
            "frames": frames,
        }
        result_path = ctx.stage_out_dir / "manifest_denoise.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs = [_staged_ref(result_path, ctx, mime="application/json")]
        images = ctx.stage_out_dir / "images"
        if images.exists():
            outputs.extend(_staged_ref(path, ctx, mime="image/jpeg") for path in images.rglob("*.jpg"))
        deltas = [
            float(metrics["mean_abs_delta"])
            for frame in frames
            for metrics in frame.get("metrics", {}).values()
            if metrics.get("mean_abs_delta") is not None
        ]
        manifest.outputs = outputs
        manifest.extra = {
            "method": ctx.params["method"],
            "device": device,
            "images": len(outputs) - 1,
            "sigma": ctx.params["sigma"] if ctx.params["method"] == "fastdvdnet" else None,
            "mean_abs_delta": round(float(np.mean(deltas)), 4) if deltas else None,
            "model_id": ctx.params["model_id"] if ctx.params["method"] == "fastdvdnet" else None,
        }
        return manifest

    @staticmethod
    def _streams(ctx: StageContext, source_manifest: dict, probe: ffprobe.ProbeResult) -> list[dict]:
        frames = sorted(source_manifest["frames"], key=lambda item: int(item["source_frame"]))
        if ctx.source_kind == "insv":
            pair = probe.dual_lens_streams()
            if pair is None:
                raise RuntimeError("INSV 時系列ノイズ除去には同じ寸法の動画 stream が 2 本必要です")
            return [
                {
                    "key": "lens0" if index == 0 else "lens1",
                    "folder": "front" if index == 0 else "back",
                    "stream_index": index,
                    "frames": frames,
                    "frame_count": _frame_count(stream, probe.duration, frames),
                    "width": stream.width,
                    "height": stream.height,
                    "pixel_format": stream.pix_fmt,
                }
                for index, stream in enumerate(pair)
            ]
        if not probe.video_streams:
            raise RuntimeError("ERP 時系列ノイズ除去には動画 stream が必要です")
        stream = probe.video_streams[0]
        return [
            {
                "key": "erp",
                "folder": "",
                "stream_index": 0,
                "frames": frames,
                "frame_count": _frame_count(stream, probe.duration, frames),
                "width": stream.width,
                "height": stream.height,
                "pixel_format": stream.pix_fmt,
            }
        ]


def _run_fastdvdnet_stream(
    source_path: Path,
    *,
    stream_index: int,
    frames: list[dict],
    frame_count: int,
    width: int,
    height: int,
    output_dir: Path,
    engine: FastDvdnetEngine,
    sigma: float,
    tile_size: int,
    tile_overlap: int,
    jpeg_quality: int,
    ffmpeg_bin: str | None,
    scratch_dir: Path,
    hardware_decode: str,
    pixel_format: str,
    on_frame: Callable[[], None],
    on_detail: Callable[[float], None],
) -> list[dict]:
    if frame_count <= 0:
        raise RuntimeError("source frame 数を取得できません")
    entries = sorted(frames, key=lambda item: int(item["source_frame"]))
    windows = [
        [_reflect_index(int(entry["source_frame"]) + offset, frame_count) for offset in (-2, -1, 0, 1, 2)]
        for entry in entries
    ]
    needed = sorted({index for window in windows for index in window})
    references = Counter(index for window in windows for index in set(window))
    cache: dict[int, np.ndarray] = {}
    results: list[dict] = []
    pending = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_index, raw_frame in ffmpeg.iter_selected_rgb_frames(
        source_path,
        stream_index=stream_index,
        frame_indices=needed,
        width=width,
        height=height,
        scratch_dir=scratch_dir,
        ffmpeg_bin=ffmpeg_bin,
        hardware_decode=hardware_decode,
        pixel_format=pixel_format,
        progress=lambda _current, _total: on_detail(0.0),
    ):
        cache[source_index] = raw_frame
        while pending < len(entries) and max(windows[pending]) <= source_index:
            window = windows[pending]
            if any(index not in cache for index in window):
                raise RuntimeError(f"時系列 decode window が不足しています: {window}")
            denoised = engine.denoise(
                [cache[index] for index in window],
                sigma=sigma,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                progress=lambda current, total: on_detail(current / max(1, total)),
            )
            output_path = output_dir / f"frame_{int(entries[pending]['index']):06d}.jpg"
            Image.fromarray(denoised).save(
                output_path,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
            )
            center = cache[int(entries[pending]["source_frame"])]
            delta = np.subtract(denoised, center, dtype=np.int16)
            mean_rgb_delta = delta.mean(axis=(0, 1))
            np.abs(delta, out=delta)
            results.append(
                {
                    "path": output_path,
                    "mean_abs_delta": round(float(delta.mean()), 4),
                    "mean_rgb_delta": [round(float(value), 4) for value in mean_rgb_delta],
                }
            )
            for index in set(window):
                references[index] -= 1
                if references[index] == 0:
                    del cache[index]
            pending += 1
            on_frame()
    if pending != len(entries):
        raise RuntimeError(f"FastDVDnet の処理数が不足しています: {pending}/{len(entries)}")
    return results


def _frame_count(stream, duration: float | None, frames: list[dict]) -> int:
    selected_minimum = max((int(frame["source_frame"]) for frame in frames), default=-1) + 1
    estimated = stream.nb_frames or round((duration or 0.0) * stream.fps)
    return max(int(estimated), selected_minimum)


def _reflect_index(index: int, frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count は正数でなければなりません")
    if frame_count == 1:
        return 0
    while index < 0 or index >= frame_count:
        index = -index if index < 0 else 2 * (frame_count - 1) - index
    return index


def _final_relpath(path: Path, ctx: StageContext) -> str:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_name) / path.relative_to(ctx.stage_out_dir))


def _staged_ref(path: Path, ctx: StageContext, *, mime: str) -> FileRef:
    return FileRef(path=_final_relpath(path, ctx), size=path.stat().st_size, sha256="", mime=mime)
