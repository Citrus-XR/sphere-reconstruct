"""全 source を source-qualified capture へ正規化する stage。"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind, Projection, SourceAdapter
from ..imaging import ffmpeg, ffprobe
from ..infrastructure.filesystem import sha256_file
from ..pipeline.manifest import register
from ..pipeline.source_inputs import IMAGE_EXTENSIONS, collect_source_inputs
from ..pipeline.stage import SourceContext, Stage, StageContext, new_manifest
from ..settings import get_settings


@register
class ExtractFrames(Stage):
    name = StageName.EXTRACT_FRAMES
    impl_version = "2.0"

    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        return collect_source_inputs(ctx.sources)

    def normalize_params(self, raw: dict) -> dict:
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
        }

    def execute(self, ctx: StageContext) -> StageManifest:
        if not ctx.sources:
            raise RuntimeError("project has no enabled sources")
        settings = get_settings()
        manifest = new_manifest(self.name, self.impl_version)
        manifest.inputs = self.collect_inputs(ctx)
        manifest.params = ctx.params

        source_manifests = []
        outputs: list[FileRef] = []
        next_index = 0
        for source_number, source in enumerate(ctx.sources):
            ctx.progress.info(
                f"source extraction: {source.label}",
                progress=source_number / len(ctx.sources),
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
                )
            elif source.media_kind == MediaKind.VIDEO:
                source_manifest, source_outputs = self._extract_video(
                    ctx,
                    source,
                    next_index,
                    ffmpeg_bin=settings.binaries.ffmpeg or None,
                    ffprobe_bin=settings.binaries.ffprobe or None,
                )
            else:
                source_manifest, source_outputs = self._prepare_images(source, next_index)
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
            progress=1.0,
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
    ) -> tuple[dict, list[FileRef]]:
        probe = ffprobe.probe(source.path, ffprobe_bin=ffprobe_bin)
        pair = probe.dual_lens_streams()
        if pair is None:
            raise RuntimeError(
                f"source {source.label}: expected 2 matching video streams, got {len(probe.video_streams)}"
            )
        lens0, lens1 = pair
        duration = probe.duration or (lens0.nb_frames or 0) / lens0.fps
        indices, selection, scores = self._select_indices(
            ctx,
            source,
            fps=lens0.fps,
            duration=duration,
            nb_frames=lens0.nb_frames,
            ffmpeg_bin=ffmpeg_bin,
        )
        output_root = ctx.stage_out_dir / "sources" / source.id
        paths0, paths1 = ffmpeg.extract_paired_frames(
            source.path,
            fps=lens0.fps,
            frame_indices=indices,
            out_dir_lens0=output_root / "lens0",
            out_dir_lens1=output_root / "lens1",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda lens, current, total: ctx.progress.tick(
                progress=current / max(1, total),
                message=f"{source.label} {lens}: paired frame {current}/{total}",
                key="log.extract_source_progress",
                args={"source": source.label, "lens": lens, "cur": current, "tot": total},
            ),
        )
        frames = [
            {
                "index": start_index + local_index,
                "source_id": source.id,
                "source_index": local_index,
                "source_frame": source_frame,
                "timestamp_sec": source_frame / lens0.fps,
                "lens0": _final_relpath(paths0[local_index], ctx),
                "lens1": _final_relpath(paths1[local_index], ctx),
                **({"score": scores[source_frame]} if source_frame in scores else {}),
            }
            for local_index, source_frame in enumerate(indices)
        ]
        outputs = [_file_ref(path, ctx, "image/jpeg") for path in [*paths0, *paths1]]
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
    ) -> tuple[dict, list[FileRef]]:
        probe = ffprobe.probe(source.path, ffprobe_bin=ffprobe_bin)
        if not probe.video_streams:
            raise RuntimeError(f"source {source.label}: video stream not found")
        stream = probe.video_streams[0]
        duration = probe.duration or (stream.nb_frames or 0) / stream.fps
        indices, selection, scores = self._select_indices(
            ctx,
            source,
            fps=stream.fps,
            duration=duration,
            nb_frames=stream.nb_frames,
            ffmpeg_bin=ffmpeg_bin,
        )
        paths = ffmpeg.extract_frames_sequential(
            source.path,
            stream_index=0,
            frame_indices=indices,
            out_dir=ctx.stage_out_dir / "sources" / source.id / "images",
            out_prefix="frame",
            ffmpeg_bin=ffmpeg_bin,
            progress=lambda current, total: ctx.progress.tick(
                progress=current / max(1, total),
                message=f"{source.label}: frame {current}/{total}",
                key="log.extract_source_progress",
                args={"source": source.label, "cur": current, "tot": total},
            ),
        )
        frames = [
            {
                "index": start_index + local_index,
                "source_id": source.id,
                "source_index": local_index,
                "source_frame": source_frame,
                "timestamp_sec": source_frame / stream.fps,
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
    ) -> tuple[list[int], dict, dict[int, dict]]:
        interval = ctx.params["interval_sec"]
        if interval <= 0:
            raise ValueError("interval_sec must be > 0")
        count = int(duration / interval)
        indices = [int(index * interval * fps) for index in range(count)]
        if ctx.params["max_frames"] > 0:
            indices = indices[: ctx.params["max_frames"]]
        if not indices:
            raise RuntimeError(f"source {source.label}: no frames to extract")

        mode = ctx.params["selection_mode"]
        selection: dict = {"mode": mode}
        scores: dict[int, dict] = {}
        if mode == "spatial":
            indices, statistics, scores = self._select_spatial_indices(
                ctx,
                source,
                fps=fps,
                duration=duration,
                nb_frames=nb_frames,
                ffmpeg_bin=ffmpeg_bin,
                fallback_count=len(indices),
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
            )
        selection["selected"] = len(indices)
        return indices, selection, scores

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
            paths = ffmpeg.extract_frames_sequential(
                source.path,
                stream_index=0,
                frame_indices=candidates,
                out_dir=scratch,
                out_prefix="candidate",
                ffmpeg_bin=ffmpeg_bin,
            )
            path_by_index = dict(zip(candidates, paths, strict=True))
            selected = []
            scores_by_index = {}
            for group in groups:
                scores = [sampling.sharpness_of_file(path_by_index[index]) for index in group]
                best = sampling.pick_sharpest(scores)
                selected.append(group[best])
                scores_by_index[group[best]] = {"sharpness": round(float(scores[best]), 1)}
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
        fallback_count: int,
    ) -> tuple[list[int], dict, dict[int, dict]]:
        import cv2  # noqa: PLC0415

        from ..imaging import quality, sampling  # noqa: PLC0415

        candidate_fps = ctx.params["candidate_fps"]
        bound = nb_frames - 1 if nb_frames else None
        candidate_count = max(2, int(duration * candidate_fps))
        candidate_indices = sorted({int(index / candidate_fps * fps) for index in range(candidate_count)})
        if bound is not None:
            candidate_indices = sorted({min(bound, index) for index in candidate_indices})
        scratch = Path(tempfile.mkdtemp(prefix=f".extract-frames-spatial-{source.id}-", dir=ctx.project_dir))
        try:
            paths = ffmpeg.extract_frames_sequential(
                source.path,
                stream_index=0,
                frame_indices=candidate_indices,
                out_dir=scratch,
                out_prefix="candidate",
                ffmpeg_bin=ffmpeg_bin,
            )
            grays = {}
            candidates = []
            for index, path in zip(candidate_indices, paths, strict=True):
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                height, width = gray.shape[:2]
                small = cv2.resize(
                    gray,
                    (max(1, width // 4), max(1, height // 4)),
                    interpolation=cv2.INTER_AREA,
                )
                grays[index] = small
                exposure_ok = quality.exposure_stats(small).is_ok(ctx.params["max_clip"])
                candidates.append(
                    sampling.Candidate(
                        index=index,
                        timestamp_us=int(index / fps * 1_000_000),
                        sharpness=sampling.laplacian_sharpness(small),
                        exposure_ok=exposure_ok,
                        feature_count=quality.sift_feature_count(small, downscale=1) if exposure_ok else 0,
                    )
                )

            def motion(first: int, second: int) -> float:
                return quality.optical_flow_median(grays[first], grays[second], downscale=1)

            result = sampling.select_spatial(
                candidates,
                motion,
                sampling.SpatialConfig(
                    min_sharpness=ctx.params["min_sharpness"],
                    min_features=ctx.params["min_features"],
                    target_motion=ctx.params["target_motion"],
                    max_frames=ctx.params["max_frames"],
                ),
            )
            statistics = {"candidates": len(candidates), "reasons": dict(result.reasons)}
            if not result.selected_indices:
                step = max(1, len(candidate_indices) // max(1, fallback_count))
                return candidate_indices[::step], {**statistics, "fallback": True}, {}
            by_index = {candidate.index: candidate for candidate in candidates}
            scores = {
                index: {
                    "sharpness": round(float(by_index[index].sharpness), 1),
                    "features": int(by_index[index].feature_count),
                }
                for index in result.selected_indices
            }
            return result.selected_indices, statistics, scores
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


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
        "adapter": source.adapter.value,
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
