"""Spatial / sharpness 抽出の sub-phase progress contract を検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.domain.source import (  # noqa: E402
    MediaKind,
    Projection,
    SourceAdapter,
    SourceRole,
)
from sphere_reconstruct.imaging import ffmpeg, quality, sampling  # noqa: E402
from sphere_reconstruct.pipeline.stage import (  # noqa: E402
    ProgressReporter,
    ProgressSpan,
    SourceContext,
    StageContext,
)
from sphere_reconstruct.stages.extract_frames import (  # noqa: E402
    ExtractFrames,
    _candidate_score_workers,
)


def _source(tmp_path: Path) -> SourceContext:
    source_path = tmp_path / "source.insv"
    source_path.write_bytes(b"video")
    return SourceContext(
        id="source",
        label="Source",
        role=SourceRole.PRIMARY,
        adapter=SourceAdapter.INSTA360_INSV,
        media_kind=MediaKind.VIDEO,
        projection=Projection.DUAL_FISHEYE,
        path=source_path,
        ordinal=0,
        enabled=True,
    )


def _fake_sequential(*_args, frame_indices, out_dir, out_prefix, progress, **_kwargs):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, _frame_index in enumerate(frame_indices, 1):
        path = out_dir / f"{out_prefix}_{number - 1:06d}.jpg"
        image = np.full((64, 64, 3), number * 20, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        paths.append(path)
        progress(number, len(frame_indices))
    return paths


def _fake_paired(*_args, frame_indices, out_dir_lens0, out_dir_lens1, progress, **_kwargs):
    outputs = []
    for lens, directory in enumerate((out_dir_lens0, out_dir_lens1)):
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for number, _frame_index in enumerate(frame_indices, 1):
            path = directory / f"lens{lens}_{number - 1:06d}.jpg"
            image = np.full((64, 64, 3), number * 20 + lens, dtype=np.uint8)
            assert cv2.imwrite(str(path), image)
            paths.append(path)
        outputs.append(paths)
    for number in range(1, len(frame_indices) + 1):
        progress(number, len(frame_indices))
    return outputs[0], outputs[1]


def test_spatial_selection_reports_decode_scoring_and_motion(tmp_path, monkeypatch):
    source = _source(tmp_path)
    stage = ExtractFrames()
    params = stage.normalize_params(
        {
            "selection_mode": "spatial",
            "candidate_fps": 2,
            "min_sharpness": 0,
            "min_features": 0,
            "target_motion": 1,
        }
    )
    events = []
    reporter = ProgressReporter(lambda *args: events.append(args), tick_min_interval=0)
    context = StageContext(
        project_id="p",
        project_dir=tmp_path,
        stage_out_dir=tmp_path / ".extract_frames.tmp",
        params=params,
        sources=(source,),
        progress=reporter,
    )
    monkeypatch.setattr(ffmpeg, "extract_frames_sequential", _fake_sequential)
    monkeypatch.setattr(quality, "optical_flow_median", lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(quality, "sift_feature_count", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(sampling, "laplacian_sharpness", lambda image: float(image.mean()))

    selected, statistics, _scores, cache = stage._select_spatial_indices(
        context,
        source,
        fps=10,
        duration=2,
        nb_frames=20,
        ffmpeg_bin=None,
        hwaccel=None,
        paired_candidates=False,
        fallback_count=2,
        progress_span=ProgressSpan(reporter, 0.0, 1.0),
    )

    assert selected
    assert cache is None
    assert statistics["candidates"] == 4
    numeric = [event[1] for event in events if event[1] is not None]
    assert numeric == sorted(numeric)
    assert numeric[-1] == 1.0
    keys = {event[3] for event in events}
    assert {
        "log.extract_candidates_progress",
        "log.extract_scoring_progress",
        "log.extract_motion_progress",
    } <= keys


def test_zero_score_workers_uses_every_logical_cpu(monkeypatch):
    monkeypatch.setattr("sphere_reconstruct.stages.extract_frames.os.cpu_count", lambda: 24)
    assert _candidate_score_workers(0) == 24
    assert _candidate_score_workers(6) == 6


def test_spatial_insv_candidates_are_reused_as_final_pairs(tmp_path, monkeypatch):
    source = _source(tmp_path)
    stage = ExtractFrames()
    params = stage.normalize_params(
        {
            "selection_mode": "spatial",
            "candidate_fps": 2,
            "min_sharpness": 0,
            "min_features": 0,
            "target_motion": 1,
        }
    )
    reporter = ProgressReporter(lambda *_args: None)
    context = StageContext(
        project_id="p",
        project_dir=tmp_path,
        stage_out_dir=tmp_path / ".extract_frames.tmp",
        params=params,
        sources=(source,),
        progress=reporter,
    )
    monkeypatch.setattr(ffmpeg, "extract_paired_frames", _fake_paired)
    monkeypatch.setattr(quality, "optical_flow_median", lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(quality, "sift_feature_count", lambda *_args, **_kwargs: 100)
    monkeypatch.setattr(sampling, "laplacian_sharpness", lambda image: float(image.mean()))

    selected, _statistics, _scores, cache = stage._select_spatial_indices(
        context,
        source,
        fps=10,
        duration=2,
        nb_frames=20,
        ffmpeg_bin=None,
        hwaccel="cuda",
        paired_candidates=True,
        fallback_count=2,
        progress_span=ProgressSpan(reporter, 0.0, 1.0),
    )

    assert cache is not None
    output0, output1 = cache.materialize(selected, context.stage_out_dir / "source", lambda *_: None)
    assert len(output0) == len(output1) == len(selected)
    assert all(path.is_file() for path in [*output0, *output1])
    cache.cleanup()
    assert not cache.root.exists()


def test_mixed_sources_extract_only_video_and_collect_still_images(tmp_path, monkeypatch):
    video_path = tmp_path / "phone.mp4"
    video_path.write_bytes(b"video")
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.jpg").write_bytes(b"b")
    video = SourceContext(
        id="video",
        label="Phone video",
        role=SourceRole.PRIMARY,
        adapter=SourceAdapter.GENERIC_VIDEO,
        media_kind=MediaKind.VIDEO,
        projection=Projection.PERSPECTIVE,
        path=video_path,
        ordinal=0,
        enabled=True,
    )
    photos = SourceContext(
        id="photos",
        label="Phone photos",
        role=SourceRole.SUPPLEMENTAL,
        adapter=SourceAdapter.GENERIC_IMAGES,
        media_kind=MediaKind.IMAGES,
        projection=Projection.PERSPECTIVE,
        path=image_dir,
        ordinal=1,
        enabled=True,
    )
    stage = ExtractFrames()
    decoded_sources = []

    def fake_video(_ctx, source, start_index, **_kwargs):
        decoded_sources.append(source.id)
        return ({
            "id": source.id,
            "label": source.label,
            "role": source.role.value,
            "adapter": source.adapter.value,
            "media_kind": source.media_kind.value,
            "projection": source.projection.value,
            "kind": "perspective_video",
            "width": 100,
            "height": 100,
            "fps": 30,
            "count": 1,
            "selection": {"mode": "interval", "selected": 1},
            "frames": [{
                "index": start_index,
                "source_id": source.id,
                "source_index": 0,
                "source_frame": 0,
                "timestamp_sec": 0.0,
                "image": "extract_frames/video/frame_000000.jpg",
            }],
        }, [])

    monkeypatch.setattr(stage, "_extract_video", fake_video)
    output = tmp_path / ".extract_frames.tmp"
    output.mkdir()
    context = StageContext(
        project_id="p",
        project_dir=tmp_path,
        stage_out_dir=output,
        params=stage.normalize_params({}),
        sources=(video, photos),
        progress=ProgressReporter(lambda *_args: None, tick_min_interval=0),
    )

    stage.execute(context)

    assert decoded_sources == ["video"]
    document = json.loads((output / "manifest_frames.json").read_text())
    photo_source = next(source for source in document["sources"] if source["id"] == "photos")
    assert photo_source["selection"] == {"mode": "all", "selected": 2}
    assert all("image_source" in frame for frame in photo_source["frames"])
