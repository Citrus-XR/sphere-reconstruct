"""全 source を source-qualified capture へ正規化する stage。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind, Projection, SourceAdapter
from ..imaging import ffmpeg, ffprobe
from ..infrastructure.filesystem import sha256_file
from ..insta360 import imu as insv_imu
from ..pipeline.manifest import register
from ..pipeline.source_inputs import IMAGE_EXTENSIONS, collect_source_inputs
from ..pipeline.stage import ProgressSpan, SourceContext, Stage, StageContext, new_manifest
from ..settings import get_settings


@dataclass
class _CandidateFrameCache:
    root: Path
    pairs_by_source_index: dict[int, tuple[Path, Path]]

    def materialize(
        self,
        source_indices: list[int],
        output_root: Path,
        progress: Callable[[int, int], None],
    ) -> tuple[list[Path], list[Path]]:
        destinations = (output_root / "lens0", output_root / "lens1")
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
        outputs0: list[Path] = []
        outputs1: list[Path] = []
        for output_index, source_index in enumerate(source_indices):
            source0, source1 = self.pairs_by_source_index[source_index]
            output0 = destinations[0] / f"lens0_{output_index:06d}.jpg"
            output1 = destinations[1] / f"lens1_{output_index:06d}.jpg"
            source0.replace(output0)
            source1.replace(output1)
            outputs0.append(output0)
            outputs1.append(output1)
            progress(output_index + 1, len(source_indices))
        return outputs0, outputs1

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@register
class ExtractFrames(Stage):
    name = StageName.EXTRACT_FRAMES
    impl_version = "3.0"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        return collect_source_inputs(
            ctx.sources,
            progress=lambda path, current, total: ctx.progress.tick(
                message=f"fingerprinting {path.name}: {current}/{total} bytes",
                key="log.source_hash_progress",
                args={"name": path.name, "cur": current, "tot": total},
            ),
        )

    def normalize_params(self, raw: dict) -> dict:
        extraction = get_settings().frame_extraction
        return {
            "interval_sec": float(raw.get("interval_sec", 1.0)),
            "max_frames": int(raw.get("max_frames", 0)),
            "selection_mode": str(raw.get("selection_mode", "interval")),
            "sharpness_candidates": int(raw.get("sharpness_candidates", 1)),
            "candidate_fps": float(raw.get("candidate_fps", 3.0)),
            "min_sharpness": float(raw.get("min_sharpness", 0.0)),
            "max_clip": float(raw.get("max_clip", 0.25)),
            "min_features": int(raw.get("min_features", 0)),
            "target_motion": float(raw.get("target_motion", 1.5)),
            "max_rolling_shutter_motion_deg": float(
                raw.get("max_rolling_shutter_motion_deg", 0.8)
            ),
            "hwaccel": str(raw.get("hwaccel", extraction.hwaccel)).lower(),
            "require_hwaccel": bool(raw.get("require_hwaccel", extraction.require_hwaccel)),
            "score_workers": int(raw.get("score_workers", extraction.score_workers)),
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        if not ctx.sources:
            raise RuntimeError("project has no enabled sources")
        settings = get_settings()
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = ctx.inputs_for(self)
        manifest.params = ctx.params

        source_manifests = []
        outputs: list[FileRef] = []
        next_index = 0
        for source_number, source in enumerate(ctx.sources):
            source_span = ProgressSpan(
                ctx.progress,
                0.98 * source_number / len(ctx.sources),
                0.98 * (source_number + 1) / len(ctx.sources),
            )
            ctx.progress.info(
                f"source extraction: {source.label}",
                progress=source_span.low,
                key="log.extract_source_start",
                args={"source": source.label, "cur": source_number + 1, "tot": len(ctx.sources)},
            )
            if source.adapter == SourceAdapter.INSTA360_INSV:
                source_manifest, source_outputs = self._extract_insv(
                    ctx,
                    source,
                    next_index,
                    ffmpeg_bin=settings.binaries.ffmpeg or None,
                    ffprobe_bin=settings.binaries.ffprobe or None,
                    progress_span=source_span,
                )
            elif source.media_kind == MediaKind.VIDEO:
                source_manifest, source_outputs = self._extract_video(
                    ctx,
                    source,
                    next_index,
                    ffmpeg_bin=settings.binaries.ffmpeg or None,
                    ffprobe_bin=settings.binaries.ffprobe or None,
                    progress_span=source_span,
                )
            else:
                source_manifest, source_outputs = self._prepare_images(source, next_index)
                source_span.tick(
                    1.0,
                    message=f"{source.label}: {len(source_manifest['frames'])} images collected",
                    key="log.extract_source_collected",
                    args={"source": source.label, "count": len(source_manifest["frames"])},
                )
            source_manifests.append(source_manifest)
            outputs.extend(source_outputs)
            next_index += len(source_manifest["frames"])

        flattened = [frame for source in source_manifests for frame in source["frames"]]
        frames_manifest = {
            "version": 2,
            "count": len(flattened),
            "sources": source_manifests,
            "frames": flattened,
        }
        manifest_path = ctx.stage_out_dir / "manifest_frames.json"
        manifest_path.write_text(json.dumps(frames_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(_file_ref(manifest_path, ctx, "application/json"))
        manifest.outputs = outputs
        manifest.extra = _frame_statistics(frames_manifest)
        ctx.progress.info(
            f"extract_frames done: {len(flattened)} captures from {len(source_manifests)} sources",
            progress=0.99,
            key="log.extract_done_mixed",
            args={"count": len(flattened), "sources": len(source_manifests)},
        )
        return manifest

    def _extract_insv(
        self,
        ctx: StageContext,
        source: SourceContext,
        start_index: int,
        *,
        ffmpeg_bin: str | None,
        ffprobe_bin: str | None,
        progress_span: ProgressSpan,
    ) -> tuple[dict, list[FileRef]]:
        probe = ffprobe.probe(source.path, ffprobe_bin=ffprobe_bin)
        pair = probe.dual_lens_streams()
        if pair is None:
            raise RuntimeError(
                f"source {source.label}: expected 2 matching video streams, got {len(probe.video_streams)}"
            )
        lens0, lens1 = pair
        timings_by_stream = ffprobe.frame_timings(
            source.path,
            stream_indices=(lens0.index, lens1.index),
            ffprobe_bin=ffprobe_bin,
        )
        timings0 = timings_by_stream[lens0.index]
        timings1 = timings_by_stream[lens1.index]
        lens0.validate_timing_count(timings0)
        lens1.validate_timing_count(timings1)
        maximum_pts_skew = ffprobe.validate_synchronized_timings(
            timings0,
            timings1,
            maximum_skew_sec=0.0005,
        )
        first_pts = timings0[0].pts_sec
        frame_times = [timing.pts_sec - first_pts for timing in timings0]
        duration = probe.duration or (lens0.nb_frames or 0) / lens0.fps
        recording = insv_imu.read_imu_recording(source.path)
        rolling_shutter_motion = (
            insv_imu.rolling_shutter_motion_at_times(
                recording,
                dict(enumerate(frame_times)),
            )
            if recording is not None
            else {}
        )
        decoder = self._resolve_decoder(ctx, source, ffmpeg_bin)
        progress_span.tick(
            0.02,
            message=f"{source.label}: video metadata ready",
            key="log.extract_probe_done",
            args={"source": source.label},
        )
        selection_end = _selection_end(ctx.params)
        indices, selection, scores, candidate_cache = self._select_indices(
            ctx,
            source,
            fps=lens0.fps,
            duration=duration,
            nb_frames=lens0.nb_frames,
            ffmpeg_bin=ffmpeg_bin,
            hwaccel=decoder.method,
            paired_candidates=True,
            paired_stream_ordinals=(lens0.video_ordinal, lens1.video_ordinal),
            frame_times_sec=frame_times,
            rolling_shutter_motion=rolling_shutter_motion,
            progress_span=progress_span.child(0.02, selection_end),
        )
        output_root = ctx.stage_out_dir / "sources" / source.id
        output_span = progress_span.child(selection_end, 1.0)

        def paired_progress(current: int, total: int) -> None:
            output_span.tick(
                current / max(1, total),
                message=f"{source.label}: output frame pair {current}/{total}",
                key="log.extract_paired_output_progress",
                args={"source": source.label, "cur": current, "tot": total},
            )

        try:
            if candidate_cache is not None:
                paths0, paths1 = candidate_cache.materialize(
                    indices,
                    output_root,
                    paired_progress,
                )
            else:
                paths0, paths1 = ffmpeg.extract_paired_frames(
                    source.path,
                    frame_indices=indices,
                    out_dir_lens0=output_root / "lens0",
                    out_dir_lens1=output_root / "lens1",
                    stream_ordinals=(lens0.video_ordinal, lens1.video_ordinal),
                    ffmpeg_bin=ffmpeg_bin,
                    hwaccel=decoder.method,
                    progress=paired_progress,
                )
        finally:
            if candidate_cache is not None:
                candidate_cache.cleanup()
        frames = [
            {
                "index": start_index + local_index,
                "source_id": source.id,
                "source_index": local_index,
                "source_frame": source_frame,
                "timestamp_sec": frame_times[source_frame],
                "lens0": _final_relpath(paths0[local_index], ctx),
                "lens1": _final_relpath(paths1[local_index], ctx),
                **(
                    {
                        "score": {
                            **scores.get(source_frame, {}),
                            **(
                                {
                                    "rolling_shutter_motion_deg": rolling_shutter_motion[
                                        source_frame
                                    ]
                                }
                                if source_frame in rolling_shutter_motion
                                else {}
                            ),
                        }
                    }
                    if source_frame in scores or source_frame in rolling_shutter_motion
                    else {}
                ),
            }
            for local_index, source_frame in enumerate(indices)
        ]
        outputs = [_file_ref(path, ctx, "image/jpeg") for path in [*paths0, *paths1]]
        selection["pairing"] = {
            "method": "pts",
            "maximum_skew_sec": maximum_pts_skew,
        }
        return (
            _source_manifest(
                source,
                kind="insv_dual",
                width=lens0.width,
                height=lens0.height,
                fps=lens0.fps,
                selection=selection,
                frames=frames,
            ),
            outputs,
        )

    def _extract_video(
        self,
        ctx: StageContext,
        source: SourceContext,
        start_index: int,
        *,
        ffmpeg_bin: str | None,
        ffprobe_bin: str | None,
        progress_span: ProgressSpan,
    ) -> tuple[dict, list[FileRef]]:
        probe = ffprobe.probe(source.path, ffprobe_bin=ffprobe_bin)
        if not probe.video_streams:
            raise RuntimeError(f"source {source.label}: video stream not found")
        stream = probe.video_streams[0]
        timings = ffprobe.frame_timings(
            source.path,
            stream_indices=(stream.index,),
            ffprobe_bin=ffprobe_bin,
        )[stream.index]
        stream.validate_timing_count(timings)
        first_pts = timings[0].pts_sec
        frame_times = [timing.pts_sec - first_pts for timing in timings]
        duration = probe.duration or (stream.nb_frames or 0) / stream.fps
        decoder = self._resolve_decoder(ctx, source, ffmpeg_bin)
        progress_span.tick(
            0.02,
            message=f"{source.label}: video metadata ready",
            key="log.extract_probe_done",
            args={"source": source.label},
        )
        selection_end = _selection_end(ctx.params)
        indices, selection, scores, _candidate_cache = self._select_indices(
            ctx,
            source,
            fps=stream.fps,
            duration=duration,
            nb_frames=stream.nb_frames,
            ffmpeg_bin=ffmpeg_bin,
            hwaccel=decoder.method,
            paired_candidates=False,
            paired_stream_ordinals=None,
            frame_times_sec=frame_times,
            progress_span=progress_span.child(0.02, selection_end),
        )
        output_span = progress_span.child(selection_end, 1.0)
        paths = ffmpeg.extract_frames_sequential(
            source.path,
            stream_index=0,
            frame_indices=indices,
            out_dir=ctx.stage_out_dir / "sources" / source.id / "images",
            out_prefix="frame",
            ffmpeg_bin=ffmpeg_bin,
            hwaccel=decoder.method,
            progress=lambda current, total: output_span.tick(
                current / max(1, total),
                message=f"{source.label}: frame {current}/{total}",
                key="log.extract_output_progress",
                args={"source": source.label, "cur": current, "tot": total},
            ),
        )
        frames = [
            {
                "index": start_index + local_index,
                "source_id": source.id,
                "source_index": local_index,
                "source_frame": source_frame,
                "timestamp_sec": frame_times[source_frame],
                "image": _final_relpath(paths[local_index], ctx),
                **({"score": scores[source_frame]} if source_frame in scores else {}),
            }
            for local_index, source_frame in enumerate(indices)
        ]
        return (
            _source_manifest(
                source,
                kind=f"{source.projection.value}_video",
                width=stream.width,
                height=stream.height,
                fps=stream.fps,
                selection=selection,
                frames=frames,
            ),
            [_file_ref(path, ctx, "image/jpeg") for path in paths],
        )

    def _prepare_images(self, source: SourceContext, start_index: int) -> tuple[dict, list[FileRef]]:
        images = sorted(
            item
            for item in source.path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"source {source.label}: no supported images")
        frames = [
            {
                "index": start_index + local_index,
                "source_id": source.id,
                "source_index": local_index,
                "timestamp_sec": None,
                "image_source": str(path),
            }
            for local_index, path in enumerate(images)
        ]
        return (
            _source_manifest(
                source,
                kind=f"{source.projection.value}_images",
                width=None,
                height=None,
                fps=None,
                selection={"mode": "all", "selected": len(frames)},
                frames=frames,
            ),
            [],
        )

    def _select_indices(
        self,
        ctx: StageContext,
        source: SourceContext,
        *,
        fps: float,
        duration: float,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
        hwaccel: str | None,
        paired_candidates: bool,
        paired_stream_ordinals: tuple[int, int] | None,
        frame_times_sec: list[float],
        rolling_shutter_motion: dict[int, float] | None = None,
        progress_span: ProgressSpan,
    ) -> tuple[list[int], dict, dict[int, dict], _CandidateFrameCache | None]:
        from ..imaging import sampling  # noqa: PLC0415

        interval = ctx.params["interval_sec"]
        if interval <= 0:
            raise ValueError("interval_sec must be > 0")
        if not frame_times_sec:
            raise RuntimeError(f"source {source.label}: frame PTS がありません")
        frame_times = np.asarray(frame_times_sec, dtype=np.float64)
        if np.any(np.diff(frame_times) <= 0.0):
            raise ValueError(f"source {source.label}: frame PTS が単調増加ではありません")
        targets = np.arange(0.0, duration, interval, dtype=np.float64)
        right = np.searchsorted(frame_times, targets, side="left")
        right = np.clip(right, 0, len(frame_times) - 1)
        left = np.maximum(0, right - 1)
        choose_left = np.abs(targets - frame_times[left]) <= np.abs(frame_times[right] - targets)
        indices = sorted(set(int(value) for value in np.where(choose_left, left, right)))
        if ctx.params["max_frames"] > 0:
            indices = indices[: ctx.params["max_frames"]]
        if not indices:
            raise RuntimeError(f"source {source.label}: no frames to extract")

        mode = ctx.params["selection_mode"]
        selection: dict = {"mode": mode}
        scores: dict[int, dict] = {}
        candidate_cache = None
        rolling_shutter_motion = rolling_shutter_motion or {}
        if mode == "interval" and rolling_shutter_motion:
            original_indices = indices
            adjusted_indices = sampling.prefer_low_rolling_shutter_motion(
                original_indices,
                span=max(1, int(round(interval * fps))),
                motion_by_index=rolling_shutter_motion,
                maximum_motion_deg=ctx.params["max_rolling_shutter_motion_deg"],
                frame_bound=nb_frames - 1 if nb_frames else None,
            )
            selection["rolling_shutter_adjusted"] = sum(
                current != original
                for current, original in zip(adjusted_indices, original_indices, strict=True)
            )
            indices = sorted(set(adjusted_indices))
            selection["maximum_rolling_shutter_motion_deg"] = ctx.params[
                "max_rolling_shutter_motion_deg"
            ]
        if mode == "spatial":
            indices, statistics, scores, candidate_cache = self._select_spatial_indices(
                ctx,
                source,
                fps=fps,
                duration=duration,
                nb_frames=nb_frames,
                ffmpeg_bin=ffmpeg_bin,
                hwaccel=hwaccel,
                paired_candidates=paired_candidates,
                paired_stream_ordinals=paired_stream_ordinals,
                frame_times_sec=frame_times_sec,
                rolling_shutter_motion=rolling_shutter_motion,
                fallback_count=len(indices),
                progress_span=progress_span,
            )
            selection.update(statistics)
        elif mode == "sharpness" or ctx.params["sharpness_candidates"] > 1:
            indices, scores = self._refine_by_sharpness(
                ctx,
                source,
                indices,
                fps=fps,
                interval=interval,
                candidate_count=max(2, ctx.params["sharpness_candidates"]),
                nb_frames=nb_frames,
                ffmpeg_bin=ffmpeg_bin,
                hwaccel=hwaccel,
                paired_candidates=paired_candidates,
                paired_stream_ordinals=paired_stream_ordinals,
                rolling_shutter_motion=rolling_shutter_motion,
                progress_span=progress_span,
            )
        else:
            progress_span.tick(
                1.0,
                message=f"{source.label}: {len(indices)} frame indices selected",
                key="log.extract_selection_done",
                args={"source": source.label, "count": len(indices)},
            )
        selection["selected"] = len(indices)
        return indices, selection, scores, candidate_cache

    def _refine_by_sharpness(
        self,
        ctx: StageContext,
        source: SourceContext,
        indices: list[int],
        *,
        fps: float,
        interval: float,
        candidate_count: int,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
        hwaccel: str | None,
        paired_candidates: bool,
        paired_stream_ordinals: tuple[int, int] | None,
        rolling_shutter_motion: dict[int, float],
        progress_span: ProgressSpan,
    ) -> tuple[list[int], dict[int, dict]]:
        from ..imaging import sampling  # noqa: PLC0415

        span = max(1, int(interval * fps))
        bound = nb_frames - 1 if nb_frames else None
        groups = [
            sampling.candidate_indices(index, span, candidate_count, fps_bound=bound) for index in indices
        ]
        candidates = sorted({candidate for group in groups for candidate in group})
        scratch = Path(
            tempfile.mkdtemp(prefix=f".extract-frames-sharpness-{source.id}-", dir=ctx.project_dir)
        )
        try:
            decode_span = progress_span.child(0.0, 0.75)
            def decoded_progress(current: int, total: int) -> None:
                decode_span.tick(
                    current / max(1, total),
                    message=f"{source.label}: candidate frame {current}/{total}",
                    key="log.extract_candidates_progress",
                    args={"source": source.label, "cur": current, "tot": total},
                )

            if paired_candidates:
                if paired_stream_ordinals is None:
                    raise ValueError("paired extraction に stream ordinal がありません")
                primary_paths, secondary_paths = ffmpeg.extract_paired_frames(
                    source.path,
                    frame_indices=candidates,
                    out_dir_lens0=scratch / "lens0",
                    out_dir_lens1=scratch / "lens1",
                    stream_ordinals=paired_stream_ordinals,
                    ffmpeg_bin=ffmpeg_bin,
                    hwaccel=hwaccel,
                    progress=decoded_progress,
                )
                paths_by_index = dict(
                    zip(
                        candidates,
                        zip(primary_paths, secondary_paths, strict=True),
                        strict=True,
                    )
                )
            else:
                primary_paths = ffmpeg.extract_frames_sequential(
                    source.path,
                    stream_index=0,
                    frame_indices=candidates,
                    out_dir=scratch,
                    out_prefix="candidate",
                    ffmpeg_bin=ffmpeg_bin,
                    hwaccel=hwaccel,
                    progress=decoded_progress,
                )
                paths_by_index = {
                    index: (path,) for index, path in zip(candidates, primary_paths, strict=True)
                }
            score_span = progress_span.child(0.75, 0.98)
            sharpness_by_index = {}

            def score_paths(paths: tuple[Path, ...]) -> float:
                return min(sampling.sharpness_of_file(path) for path in paths)

            with _single_threaded_opencv_workers(), ThreadPoolExecutor(
                max_workers=_candidate_score_workers(ctx.params["score_workers"])
            ) as executor:
                futures = {
                    executor.submit(score_paths, paths_by_index[index]): index
                    for index in candidates
                }
                for candidate_number, future in enumerate(as_completed(futures), 1):
                    sharpness_by_index[futures[future]] = future.result()
                    score_span.tick(
                        candidate_number / max(1, len(candidates)),
                        message=f"{source.label}: sharpness candidate {candidate_number}/{len(candidates)}",
                        key="log.extract_scoring_progress",
                        args={
                            "source": source.label,
                            "cur": candidate_number,
                            "tot": len(candidates),
                        },
                    )
            selected = []
            scores_by_index = {}
            for group in groups:
                scores = [sharpness_by_index[index] for index in group]
                safe = [
                    position
                    for position, index in enumerate(group)
                    if rolling_shutter_motion.get(index, 0.0)
                    <= ctx.params["max_rolling_shutter_motion_deg"]
                ]
                if safe:
                    best = max(safe, key=lambda position: scores[position])
                else:
                    best = min(
                        range(len(group)),
                        key=lambda position: rolling_shutter_motion.get(group[position], 0.0),
                    )
                selected.append(group[best])
                scores_by_index[group[best]] = {
                    "sharpness": round(float(scores[best]), 1),
                    "rolling_shutter_motion_deg": rolling_shutter_motion.get(group[best], 0.0),
                }
            progress_span.tick(
                1.0,
                message=f"{source.label}: sharpness selection complete",
                key="log.extract_selection_done",
                args={"source": source.label, "count": len(selected)},
            )
            return sorted(set(selected)), scores_by_index
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _select_spatial_indices(
        self,
        ctx: StageContext,
        source: SourceContext,
        *,
        fps: float,
        duration: float,
        nb_frames: int | None,
        ffmpeg_bin: str | None,
        hwaccel: str | None,
        paired_candidates: bool,
        paired_stream_ordinals: tuple[int, int] | None,
        frame_times_sec: list[float],
        fallback_count: int,
        progress_span: ProgressSpan,
        rolling_shutter_motion: dict[int, float] | None = None,
    ) -> tuple[list[int], dict, dict[int, dict], _CandidateFrameCache | None]:
        import cv2  # noqa: PLC0415

        from ..imaging import quality, sampling  # noqa: PLC0415

        rolling_shutter_motion = rolling_shutter_motion or {}
        candidate_fps = ctx.params["candidate_fps"]
        if candidate_fps <= 0:
            raise ValueError("candidate_fps must be > 0")
        bound = nb_frames - 1 if nb_frames else None
        frame_times = np.asarray(frame_times_sec, dtype=np.float64)
        targets = np.arange(0.0, duration, 1.0 / candidate_fps, dtype=np.float64)
        timing_indices = np.searchsorted(frame_times, targets, side="left")
        timing_indices = np.clip(timing_indices, 0, len(frame_times) - 1)
        candidate_indices = sorted(set(int(index) for index in timing_indices))
        if bound is not None:
            candidate_indices = sorted({min(bound, index) for index in candidate_indices})
        scratch = Path(tempfile.mkdtemp(prefix=f".extract-frames-spatial-{source.id}-", dir=ctx.project_dir))
        keep_scratch = False
        try:
            decode_span = progress_span.child(0.0, 0.65)

            def candidate_progress(current: int, total: int) -> None:
                decode_span.tick(
                    current / max(1, total),
                    message=f"{source.label}: spatial candidate {current}/{total}",
                    key="log.extract_candidates_progress",
                    args={"source": source.label, "cur": current, "tot": total},
                )
            candidate_cache = None
            if paired_candidates:
                if paired_stream_ordinals is None:
                    raise ValueError("paired extraction に stream ordinal がありません")
                paths, paired_paths = ffmpeg.extract_paired_frames(
                    source.path,
                    frame_indices=candidate_indices,
                    out_dir_lens0=scratch / "lens0",
                    out_dir_lens1=scratch / "lens1",
                    stream_ordinals=paired_stream_ordinals,
                    ffmpeg_bin=ffmpeg_bin,
                    hwaccel=hwaccel,
                    progress=candidate_progress,
                )
                candidate_cache = _CandidateFrameCache(
                    root=scratch,
                    pairs_by_source_index=dict(
                        zip(candidate_indices, zip(paths, paired_paths, strict=True), strict=True)
                    ),
                )
            else:
                paths = ffmpeg.extract_frames_sequential(
                    source.path,
                    stream_index=0,
                    frame_indices=candidate_indices,
                    out_dir=scratch,
                    out_prefix="candidate",
                    ffmpeg_bin=ffmpeg_bin,
                    hwaccel=hwaccel,
                    progress=candidate_progress,
                )
                paired_paths = []
            grays = {}
            candidates = []
            score_span = progress_span.child(0.65, 0.85)

            sensor_paths = (
                list(zip(paths, paired_paths, strict=True))
                if paired_candidates
                else [(path,) for path in paths]
            )

            def score_candidate(item: tuple[int, tuple[Path, ...]]):
                index, paths_for_capture = item
                small_images = []
                for path in paths_for_capture:
                    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        return None
                    height, width = gray.shape[:2]
                    small_images.append(
                        cv2.resize(
                            gray,
                            (max(1, width // 4), max(1, height // 4)),
                            interpolation=cv2.INTER_AREA,
                        )
                    )
                exposure_ok = all(
                    quality.exposure_stats(image).is_ok(ctx.params["max_clip"])
                    for image in small_images
                )
                candidate = sampling.Candidate(
                    index=index,
                    timestamp_us=int(frame_times_sec[index] * 1_000_000),
                    sharpness=min(
                        sampling.laplacian_sharpness(image) for image in small_images
                    ),
                    exposure_ok=exposure_ok,
                    feature_count=(
                        min(
                            quality.sift_feature_count(image, downscale=1)
                            for image in small_images
                        )
                        if exposure_ok
                        else 0
                    ),
                    rolling_shutter_motion_deg=rolling_shutter_motion.get(index, 0.0),
                )
                return index, tuple(small_images), candidate

            with _single_threaded_opencv_workers(), ThreadPoolExecutor(
                max_workers=_candidate_score_workers(ctx.params["score_workers"])
            ) as executor:
                scored = executor.map(
                    score_candidate,
                    zip(candidate_indices, sensor_paths, strict=True),
                )
                for candidate_number, result in enumerate(scored, 1):
                    if result is not None:
                        index, small_images, candidate = result
                        grays[index] = small_images
                        candidates.append(candidate)
                    score_span.tick(
                        candidate_number / max(1, len(paths)),
                        message=f"{source.label}: score candidate {candidate_number}/{len(paths)}",
                        key="log.extract_scoring_progress",
                        args={"source": source.label, "cur": candidate_number, "tot": len(paths)},
                    )

            def motion(first: int, second: int) -> float:
                return max(
                    quality.optical_flow_median(left, right, downscale=1)
                    for left, right in zip(grays[first], grays[second], strict=True)
                )

            motion_span = progress_span.child(0.85, 1.0)
            result = sampling.select_spatial(
                candidates,
                motion,
                sampling.SpatialConfig(
                    min_sharpness=ctx.params["min_sharpness"],
                    min_features=ctx.params["min_features"],
                    target_motion=ctx.params["target_motion"],
                    max_frames=ctx.params["max_frames"],
                    max_rolling_shutter_motion_deg=ctx.params[
                        "max_rolling_shutter_motion_deg"
                    ],
                ),
                progress=lambda current, total: motion_span.tick(
                    current / max(1, total),
                    message=f"{source.label}: motion pair {current}/{total}",
                    key="log.extract_motion_progress",
                    args={"source": source.label, "cur": current, "tot": total},
                ),
            )
            if len(candidates) <= 1:
                motion_span.tick(
                    1.0,
                    message=f"{source.label}: spatial selection complete",
                    key="log.extract_selection_done",
                    args={"source": source.label, "count": len(result.selected_indices)},
                )
            statistics = {"candidates": len(candidates), "reasons": dict(result.reasons)}
            if not result.selected_indices:
                step = max(1, len(candidate_indices) // max(1, fallback_count))
                keep_scratch = candidate_cache is not None
                return (
                    candidate_indices[::step],
                    {**statistics, "fallback": True},
                    {},
                    candidate_cache,
                )
            by_index = {candidate.index: candidate for candidate in candidates}
            scores = {
                index: {
                    "sharpness": round(float(by_index[index].sharpness), 1),
                    "features": int(by_index[index].feature_count),
                }
                for index in result.selected_indices
            }
            keep_scratch = candidate_cache is not None
            return result.selected_indices, statistics, scores, candidate_cache
        finally:
            if not keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)

    def _resolve_decoder(
        self,
        ctx: StageContext,
        source: SourceContext,
        ffmpeg_bin: str | None,
    ) -> ffmpeg.HardwareDecode:
        decoder = ffmpeg.resolve_hardware_decode(
            source.path,
            stream_index=0,
            preference=ctx.params["hwaccel"],
            required=ctx.params["require_hwaccel"],
            ffmpeg_bin=ffmpeg_bin,
        )
        args = {
            "source": source.label,
            "decoder": decoder.method or "software",
            "detail": decoder.detail,
        }
        if decoder.method is None and ctx.params["hwaccel"] not in {"", "none", "software"}:
            ctx.progress.warn(
                f"{source.label}: {decoder.detail}",
                key="log.extract_decoder_fallback",
                args=args,
            )
        elif decoder.method is None:
            ctx.progress.info(
                f"{source.label}: decoder=software",
                key="log.extract_decoder_software",
                args=args,
            )
        else:
            ctx.progress.info(
                f"{source.label}: decoder={decoder.method}",
                key="log.extract_decoder",
                args=args,
            )
        return decoder


def _source_manifest(
    source: SourceContext,
    *,
    kind: str,
    width: int | None,
    height: int | None,
    fps: float | None,
    selection: dict,
    frames: list[dict],
) -> dict:
    return {
        "id": source.id,
        "label": source.label,
        "role": source.role.value,
        "adapter": source.adapter,
        "media_kind": source.media_kind.value,
        "projection": source.projection.value,
        "kind": kind,
        "width": width,
        "height": height,
        "fps": fps,
        "count": len(frames),
        "selection": selection,
        "frames": frames,
    }


def _selection_end(params: dict) -> float:
    if params["selection_mode"] == "spatial":
        return 0.72
    if params["selection_mode"] == "sharpness" or params["sharpness_candidates"] > 1:
        return 0.55
    return 0.05


def _candidate_score_workers(configured: int) -> int:
    if configured < 0:
        raise ValueError("score_workers must be >= 0")
    return configured or max(1, os.cpu_count() or 1)


@contextmanager
def _single_threaded_opencv_workers():
    import cv2  # noqa: PLC0415

    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(max(1, previous))


def _frame_statistics(manifest: dict) -> dict:
    sources = manifest["sources"]
    return {
        "sources": len(sources),
        "frames": len(manifest["frames"]),
        "video_sources": sum(source["media_kind"] == "video" for source in sources),
        "image_sources": sum(source["media_kind"] == "images" for source in sources),
        "perspective_frames": sum(
            len(source["frames"])
            for source in sources
            if source["projection"] == Projection.PERSPECTIVE.value
        ),
        "spherical_frames": sum(
            len(source["frames"])
            for source in sources
            if source["projection"] != Projection.PERSPECTIVE.value
        ),
    }


def _final_relpath(path: Path, ctx: StageContext) -> str:
    final_name = ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp")
    return str(Path(final_name) / path.relative_to(ctx.stage_out_dir))


def _file_ref(path: Path, ctx: StageContext, mime: str) -> FileRef:
    return FileRef(
        path=_final_relpath(path, ctx),
        size=path.stat().st_size,
        sha256=sha256_file(path) if path.suffix == ".json" else "",
        mime=mime,
    )
