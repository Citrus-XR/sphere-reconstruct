"""imaging.ffmpeg / ffprobe の結合テスト.

lavfi (ffmpeg の internal color generator) を使って 2 stream の小さな MP4 を
その場で作り, extract_paired_frames / probe が正しく動くことを確認する.

ffmpeg / ffprobe が PATH にないとスキップする.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sphere_reconstruct.imaging import ffmpeg, ffprobe

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg / ffprobe not found in PATH",
)


def _make_dual_stream_mp4(
    dst: Path,
    duration: float = 2.0,
    fps: int = 10,
    size: str = "320x320",
    b_frames: int = 0,
) -> None:
    """testsrc / smptebars 各 duration 秒 を 2 stream として mux した MP4 を作る."""
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={size}:rate={fps}",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=duration={duration}:size={size}:rate={fps}",
        "-map",
        "0:v",
        "-map",
        "1:v",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-bf",
        str(b_frames),
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(args, check=True, capture_output=True)


def test_ffprobe_detects_dual_streams(tmp_path: Path):
    mp4 = tmp_path / "dual.mp4"
    _make_dual_stream_mp4(mp4)
    result = ffprobe.probe(mp4)
    assert len(result.video_streams) == 2
    pair = result.dual_lens_streams()
    assert pair is not None
    a, b = pair
    assert a.width == 320 and b.width == 320
    assert a.fps == 10 and b.fps == 10

    timings = ffprobe.frame_timings(mp4, stream_indices=(a.index, b.index))
    assert len(timings[a.index]) == len(timings[b.index]) == 20
    assert ffprobe.validate_synchronized_timings(
        timings[a.index], timings[b.index], maximum_skew_sec=1e-6
    ) == 0.0


def test_packet_timings_restore_b_frame_presentation_order(tmp_path: Path):
    mp4 = tmp_path / "dual-bframes.mp4"
    _make_dual_stream_mp4(mp4, duration=1.0, fps=12, size="64x64", b_frames=2)
    streams = ffprobe.probe(mp4).video_streams

    timings = ffprobe.frame_timings(
        mp4,
        stream_indices=(streams[0].index, streams[1].index),
    )

    assert len(timings[streams[0].index]) == len(timings[streams[1].index]) == 12
    assert [timing.index for timing in timings[streams[0].index]] == list(range(12))
    assert all(
        current.pts_sec > previous.pts_sec
        for previous, current in zip(
            timings[streams[0].index][:-1],
            timings[streams[0].index][1:],
            strict=True,
        )
    )


def test_extract_paired_frames_roundtrip(tmp_path: Path):
    mp4 = tmp_path / "dual.mp4"
    _make_dual_stream_mp4(mp4, duration=2.0, fps=10)
    out0 = tmp_path / "lens0"
    out1 = tmp_path / "lens1"
    indices = [0, 5, 10]  # 3 pairs
    progress = []
    p0, p1 = ffmpeg.extract_paired_frames(
        mp4,
        frame_indices=indices,
        out_dir_lens0=out0,
        out_dir_lens1=out1,
        progress=lambda current, total: progress.append((current, total)),
    )
    assert len(p0) == 3 and len(p1) == 3
    for p in p0 + p1:
        assert p.exists()
        assert p.stat().st_size > 500  # 少なくとも JPEG ヘッダはあるサイズ
    assert progress[-1] == (len(indices), len(indices))
    assert [current for current, _total in progress] == sorted(current for current, _total in progress)

    # extract できたペアが同時刻で対応することを確認するのは難しいので, ここでは
    # ファイル数と非空サイズだけを担保する.


def test_selection_expression_compresses_arithmetic_runs():
    expression = ffmpeg._selection_expression([0, 6, 12, 18, 25, 31, 37])
    assert "between(n\\,0\\,18)*not(mod(n-0\\,6))" in expression
    assert "between(n\\,25\\,37)*not(mod(n-25\\,6))" in expression


def test_large_temporal_context_is_split_into_bounded_filter_expressions():
    indices = sorted({center + offset for center in range(0, 900, 8) for offset in range(5)})
    chunks = ffmpeg._selection_chunks(indices, max_expression_chars=1000)
    assert len(chunks) > 1
    assert [index for chunk in chunks for index in chunk] == indices
    assert all(len(ffmpeg._selection_expression(chunk)) <= 1000 for chunk in chunks)


def test_chunked_jpeg_selection_preserves_exact_frames(tmp_path: Path):
    mp4 = tmp_path / "dual.mp4"
    _make_dual_stream_mp4(mp4, duration=2.0, fps=10, size="64x64")
    indices = [0, 2, 3, 4, 6, 8, 11, 13, 14, 15, 18]
    direct = ffmpeg.extract_frames_sequential(
        mp4,
        stream_index=0,
        frame_indices=indices,
        out_dir=tmp_path / "direct-jpeg",
    )
    chunked = ffmpeg.extract_frames_sequential(
        mp4,
        stream_index=0,
        frame_indices=indices,
        out_dir=tmp_path / "chunked-jpeg",
        max_filter_expression_chars=24,
    )
    assert all(
        np.array_equal(np.asarray(Image.open(left)), np.asarray(Image.open(right)))
        for left, right in zip(direct, chunked, strict=True)
    )


def test_chunked_paired_selection_preserves_both_streams(tmp_path: Path):
    mp4 = tmp_path / "dual.mp4"
    _make_dual_stream_mp4(mp4, duration=2.0, fps=10, size="64x64")
    indices = [0, 2, 3, 4, 6, 8, 11, 13, 14, 15, 18]
    paired0, paired1 = ffmpeg.extract_paired_frames(
        mp4,
        frame_indices=indices,
        out_dir_lens0=tmp_path / "paired0",
        out_dir_lens1=tmp_path / "paired1",
        max_filter_expression_chars=24,
    )
    direct0 = ffmpeg.extract_frames_sequential(
        mp4,
        stream_index=0,
        frame_indices=indices,
        out_dir=tmp_path / "direct0",
    )
    direct1 = ffmpeg.extract_frames_sequential(
        mp4,
        stream_index=1,
        frame_indices=indices,
        out_dir=tmp_path / "direct1",
    )

    assert all(
        np.array_equal(np.asarray(Image.open(paired)), np.asarray(Image.open(direct)))
        for paired, direct in zip([*paired0, *paired1], [*direct0, *direct1], strict=True)
    )
