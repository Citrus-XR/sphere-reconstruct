"""ffmpeg (subprocess) 呼び出し.

- 指定した video stream から, 指定した timestamp または frame index の集合で
  静止画を抜き出す (JPEG).
- 進捗を stderr から拾って呼び出し側 callback へ送る.

INSV の 2 本の hevc stream は index=0 / index=1 で選択できる. `-map 0:v:N`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path


def _resolve_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg not found in PATH; set binaries.ffmpeg in config.toml")
    return found


# ffmpeg のプログレス "frame=  123" を拾う.
_FRAME_RE = re.compile(rb"frame=\s*(\d+)")


def extract_frames_by_timestamp(
    src: Path,
    *,
    stream_index: int,
    timestamps_sec: Sequence[float],
    out_dir: Path,
    out_prefix: str = "frame",
    jpeg_quality: int = 3,   # ffmpeg -q:v (2 が最高, 31 が最低, デフォルト 3)
    ffmpeg_bin: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """指定 stream から, 指定 timestamp (秒) の直近フレームを 1 枚ずつ書き出す.

    実装方針: 各 timestamp 毎に `-ss ... -i src -frames:v 1` で呼ぶ. INSV では
    seek 精度確保のため入力側 fast seek + 出力側 accurate seek のコンボを使う.
    (シーケンシャル抽出よりは遅いが実装が単純. 後で最適化する.)

    Returns: 実際に書けた出力ファイルパスのリスト.
    """
    binary = _resolve_bin(ffmpeg_bin)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    total = len(timestamps_sec)
    for i, ts in enumerate(timestamps_sec):
        out_path = out_dir / f"{out_prefix}_{i:06d}.jpg"
        args = [
            binary,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            f"{ts:.6f}",
            "-i",
            str(src),
            "-map",
            f"0:v:{stream_index}",
            "-frames:v",
            "1",
            "-q:v",
            str(jpeg_quality),
            "-y",
            str(out_path),
        ]
        proc = subprocess.run(args, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed at ts={ts:.3f}s stream={stream_index}: "
                f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
            )
        out_paths.append(out_path)
        if progress is not None:
            progress(i + 1, total)
    return out_paths


def extract_frames_by_index(
    src: Path,
    *,
    stream_index: int,
    fps: float,
    frame_indices: Sequence[int],
    out_dir: Path,
    **kwargs,
) -> list[Path]:
    """frame index (0-based) 指定版. 内部で timestamp に変換する."""
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")
    timestamps = [idx / fps for idx in frame_indices]
    return extract_frames_by_timestamp(
        src, stream_index=stream_index, timestamps_sec=timestamps, out_dir=out_dir, **kwargs
    )


def extract_paired_frames(
    src: Path,
    *,
    fps: float,
    frame_indices: Sequence[int],
    out_dir_lens0: Path,
    out_dir_lens1: Path,
    ffmpeg_bin: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Path], list[Path]]:
    """INSV 想定. stream 0 と stream 1 の同じ frame index を対で書き出す.

    ffmpeg を並列に 2 本走らせる (I/O バウンドなので効果がある).
    どちらの stream が front/back かは呼び出し側が管理.
    """
    total = len(frame_indices)
    results: dict[int, list[Path]] = {}
    errors: dict[int, BaseException] = {}

    def _run(idx: int, out_dir: Path) -> None:
        try:
            done = 0

            def local_progress(cur: int, tot: int) -> None:
                nonlocal done
                done = cur
                if progress is not None:
                    progress(f"lens{idx}", cur, tot)

            results[idx] = extract_frames_by_index(
                src,
                stream_index=idx,
                fps=fps,
                frame_indices=frame_indices,
                out_dir=out_dir,
                out_prefix=f"lens{idx}",
                ffmpeg_bin=ffmpeg_bin,
                progress=local_progress,
            )
        except BaseException as e:
            errors[idx] = e

    t0 = threading.Thread(target=_run, args=(0, out_dir_lens0), daemon=True)
    t1 = threading.Thread(target=_run, args=(1, out_dir_lens1), daemon=True)
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    if errors:
        first = next(iter(errors.values()))
        raise first
    if total and (0 not in results or 1 not in results):
        raise RuntimeError("paired extraction incomplete")
    return results.get(0, []), results.get(1, [])
