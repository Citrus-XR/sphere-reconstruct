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
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    r_frame_rate: str      # "24/1"
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
