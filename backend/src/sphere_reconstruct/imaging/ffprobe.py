"""ffprobe (subprocess) から JSON メタデータを取得する."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoStream:
    index: int
    video_ordinal: int
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    r_frame_rate: str  # "24/1"
    avg_frame_rate: str
    duration: float | None
    nb_frames: int | None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        # r_frame_rate は "num/den" 形式.
        num, _, den = self.r_frame_rate.partition("/")
        try:
            return float(num) / float(den or 1)
        except ValueError:
            return 0.0

    def validate_timing_count(self, timings: list[FrameTiming]) -> None:
        if self.nb_frames is not None and len(timings) != self.nb_frames:
            raise ValueError(
                f"stream {self.index} の packet/frame 数が一致しません: "
                f"{len(timings)} != {self.nb_frames}"
            )


@dataclass(frozen=True)
class FrameTiming:
    index: int
    pts_sec: float
    duration_sec: float | None


@dataclass
class ProbeResult:
    path: str
    format_name: str
    duration: float | None
    bit_rate: int | None
    video_streams: list[VideoStream]
    raw: dict[str, Any]

    def dual_lens_streams(self) -> tuple[VideoStream, VideoStream] | None:
        """同じ解像度 / fps の video stream が丁度 2 本あれば返す (INSV 想定).

        どちらが front / back かはここでは判定しない.
        """
        vs = [s for s in self.video_streams if s.codec_name in ("hevc", "h264")]
        if len(vs) != 2:
            return None
        a, b = vs
        if a.width == b.width and a.height == b.height and a.fps == b.fps:
            return (a, b)
        return None


def _resolve_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffprobe")
    if not found:
        raise FileNotFoundError("ffprobe not found in PATH; set binaries.ffprobe in config.toml")
    return found


def probe(path: Path, *, ffprobe_bin: str | None = None, timeout: float = 30.0) -> ProbeResult:
    """ffprobe を呼び出して映像メタデータを取得する.

    失敗時はここでは弾かず, subprocess.CalledProcessError を伝播させる.
    """
    binary = _resolve_bin(ffprobe_bin)
    args = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
    data = json.loads(proc.stdout)

    streams = []
    for s in data.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        dur_str = s.get("duration")
        try:
            dur = float(dur_str) if dur_str else None
        except ValueError:
            dur = None
        nbf_str = s.get("nb_frames")
        try:
            nbf = int(nbf_str) if nbf_str else None
        except ValueError:
            nbf = None
        streams.append(
            VideoStream(
                index=int(s.get("index", 0)),
                video_ordinal=len(streams),
                codec_name=s.get("codec_name", ""),
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                pix_fmt=s.get("pix_fmt", ""),
                r_frame_rate=s.get("r_frame_rate", "0/1"),
                avg_frame_rate=s.get("avg_frame_rate", "0/1"),
                duration=dur,
                nb_frames=nbf,
                tags=s.get("tags", {}) or {},
            )
        )

    fmt = data.get("format", {}) or {}
    return ProbeResult(
        path=str(path),
        format_name=fmt.get("format_name", ""),
        duration=float(fmt["duration"]) if fmt.get("duration") else None,
        bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        video_streams=streams,
        raw=data,
    )


def frame_timings(
    path: Path,
    *,
    stream_indices: tuple[int, ...],
    ffprobe_bin: str | None = None,
    timeout: float = 30.0,
) -> dict[int, list[FrameTiming]]:
    """Packet metadata を presentation 順へ並べ、frame PTS / duration として取得する。

    MP4/H.26x は通常一 packet が一 coded picture に対応する。`-show_frames` は巨大な raw camera
    file を software decode 相当で走査するため使わない。B-frame の demux 順は PTS sort で戻す。
    """
    if not stream_indices:
        raise ValueError("stream_indices が空です")
    binary = _resolve_bin(ffprobe_bin)
    args = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_packets",
        "-show_entries",
        "packet=stream_index,pts_time,duration_time",
        str(path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
    requested = set(stream_indices)
    grouped: dict[int, list[FrameTiming]] = {index: [] for index in stream_indices}
    for packet in json.loads(proc.stdout).get("packets", []):
        stream_index = int(packet["stream_index"])
        if stream_index not in requested:
            continue
        if "pts_time" not in packet:
            raise ValueError(f"stream {stream_index} の packet PTS がありません")
        duration_value = packet.get("duration_time")
        grouped[stream_index].append(
            FrameTiming(
                index=0,
                pts_sec=float(packet["pts_time"]),
                duration_sec=float(duration_value) if duration_value is not None else None,
            )
        )
    for stream_index, timings in grouped.items():
        if not timings:
            raise ValueError(f"stream {stream_index} の frame timing がありません")
        timings.sort(key=lambda timing: timing.pts_sec)
        grouped[stream_index] = [
            FrameTiming(
                index=index,
                pts_sec=timing.pts_sec,
                duration_sec=timing.duration_sec,
            )
            for index, timing in enumerate(timings)
        ]
        timings = grouped[stream_index]
        if any(
            current.pts_sec <= previous.pts_sec
            for previous, current in zip(timings, timings[1:], strict=False)
        ):
            raise ValueError(f"stream {stream_index} の frame PTS が単調増加ではありません")
    return grouped


def validate_synchronized_timings(
    first: list[FrameTiming],
    second: list[FrameTiming],
    *,
    maximum_skew_sec: float,
) -> float:
    """二つの sensor stream が全 frame で同期していることを検証する。"""
    if maximum_skew_sec < 0.0:
        raise ValueError("maximum_skew_sec は 0 以上でなければなりません")
    if len(first) != len(second):
        raise ValueError(f"sensor stream の frame 数が一致しません: {len(first)} != {len(second)}")
    maximum = max(
        (
            abs(left.pts_sec - right.pts_sec)
            for left, right in zip(first, second, strict=True)
        ),
        default=0.0,
    )
    if maximum > maximum_skew_sec:
        raise ValueError(
            f"sensor stream の PTS skew {maximum:.9f}s が許容値 {maximum_skew_sec:.9f}s を超えました"
        )
    return maximum
