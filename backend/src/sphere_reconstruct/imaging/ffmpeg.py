"""ffmpeg (subprocess) 呼び出し.

- 指定した video stream から, 指定した timestamp または frame index の集合で
  静止画を抜き出す (JPEG).
- `-progress pipe:1` を読み、呼び出し側 callback へ送る.

INSV の 2 本の hevc stream は index=0 / index=1 で選択できる. `-map 0:v:N`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _resolve_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg not found in PATH; set binaries.ffmpeg in config.toml")
    return found


@dataclass(frozen=True)
class HardwareDecode:
    method: str | None
    detail: str


def resolve_hardware_decode(
    src: Path,
    *,
    stream_index: int,
    preference: str,
    required: bool,
    ffmpeg_bin: str | None = None,
) -> HardwareDecode:
    """実 source を decode できる hardware backend を 1 frame probe で決定する。"""
    binary = _resolve_bin(ffmpeg_bin)
    requested = preference.strip().lower()
    if requested in {"", "none", "software"}:
        if required:
            raise RuntimeError("hardware decode is required but frame_extraction.hwaccel is disabled")
        return HardwareDecode(None, "software decode configured")

    available = _hardware_acceleration_methods(binary)
    candidates = (
        [requested]
        if requested != "auto"
        else [
            method
            for method in (
                "cuda",
                "videotoolbox",
                "qsv",
                "d3d11va",
                "d3d12va",
                "dxva2",
                "vaapi",
                "vdpau",
            )
            if method in available
        ]
    )
    failures: list[str] = []
    for method in candidates:
        if method not in available:
            failures.append(f"{method}: not compiled into FFmpeg")
            continue
        try:
            result = subprocess.run(
                [
                    binary,
                    "-nostdin",
                    "-hide_banner",
                    "-v",
                    "error",
                    "-hwaccel",
                    method,
                    "-i",
                    str(src),
                    "-map",
                    f"0:v:{stream_index}",
                    "-frames:v",
                    "1",
                    "-vf",
                    "null",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{method}: source probe timed out")
            continue
        if result.returncode == 0:
            return HardwareDecode(method, f"{method} hardware decode verified")
        error = result.stderr.strip().splitlines()
        failures.append(f"{method}: {error[-1] if error else 'probe failed'}")

    detail = "; ".join(failures) or "FFmpeg reports no supported hardware acceleration method"
    if required:
        raise RuntimeError(f"hardware decode unavailable: {detail}")
    return HardwareDecode(None, f"hardware decode unavailable; using software: {detail}")


@lru_cache(maxsize=8)
def _hardware_acceleration_methods(binary: str) -> frozenset[str]:
    result = subprocess.run(
        [binary, "-nostdin", "-hide_banner", "-v", "error", "-hwaccels"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    )


def _hardware_input_args(method: str | None) -> list[str]:
    return ["-hwaccel", method] if method is not None else []


@dataclass(frozen=True)
class _ProgressProcessResult:
    returncode: int
    stderr: str
    latest_frame: int


def _run_progress_process(
    args: list[str],
    total: int,
    progress: Callable[[int, int], None] | None,
) -> _ProgressProcessResult:
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_lines: deque[str] = deque(maxlen=80)

    def drain_stderr() -> None:
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    latest = 0
    try:
        for line in process.stdout:
            if not line.startswith("frame="):
                continue
            try:
                current = min(total, int(line.split("=", 1)[1]))
            except ValueError:
                continue
            if current > latest:
                latest = current
                if progress is not None:
                    progress(current, total)
        process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        stderr_thread.join(timeout=5)
    return _ProgressProcessResult(process.returncode, "".join(stderr_lines), latest)


def _extract_one(
    binary: str,
    src: Path,
    stream_index: int,
    ts: float,
    out_path: Path,
    jpeg_quality: int,
    hwaccel: str | None,
) -> None:
    """1 timestamp 分を書き出す. 入力側 fast seek (`-ss` を `-i` の前) で高速化."""
    args = [
        binary,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-ss",
        f"{ts:.6f}",
        *_hardware_input_args(hwaccel),
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


def extract_frames_by_timestamp(
    src: Path,
    *,
    stream_index: int,
    timestamps_sec: Sequence[float],
    out_dir: Path,
    out_prefix: str = "frame",
    jpeg_quality: int = 3,  # ffmpeg -q:v (2 が最高, 31 が最低, デフォルト 3)
    ffmpeg_bin: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
    hwaccel: str | None = None,
) -> list[Path]:
    """指定 stream から, 指定 timestamp (秒) の直近フレームを 1 枚ずつ書き出す.

    各 timestamp 毎に `-ss ... -i src -frames:v 1` で呼ぶ. INSV では seek 精度確保のため
    入力側 fast seek を使う. workers>1 で ffmpeg を並列に走らせる — 8K HEVC を大量に抜く場合,
    プロセス起動 + seek + 単一フレームデコードが直列だと数分かかるため, 有界プールで短縮する.

    Returns: 出力ファイルパスのリスト (入力 timestamp と同じ順序).
    """
    binary = _resolve_bin(ffmpeg_bin)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(timestamps_sec)
    out_paths = [out_dir / f"{out_prefix}_{i:06d}.jpg" for i in range(total)]

    if workers <= 1 or total <= 1:
        for i, ts in enumerate(timestamps_sec):
            _extract_one(binary, src, stream_index, ts, out_paths[i], jpeg_quality, hwaccel)
            if progress is not None:
                progress(i + 1, total)
        return out_paths

    done = 0
    lock = threading.Lock()

    def _task(i: int, ts: float) -> None:
        nonlocal done
        _extract_one(binary, src, stream_index, ts, out_paths[i], jpeg_quality, hwaccel)
        if progress is not None:
            with lock:
                done += 1
                progress(done, total)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, i, ts) for i, ts in enumerate(timestamps_sec)]
        for f in futures:
            f.result()  # 例外はここで再送出 (最初の失敗で伝播).
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


def extract_frames_sequential(
    src: Path,
    *,
    stream_index: int,
    frame_indices: Sequence[int],
    out_dir: Path,
    out_prefix: str = "frame",
    jpeg_quality: int = 3,
    ffmpeg_bin: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    max_filter_expression_chars: int = 3000,
    hwaccel: str | None = None,
) -> list[Path]:
    """動画を 1 回だけ順次 decode し, source frame index の集合を抽出する.

    1 frame ごとの ``-ss`` は GOP を毎回再 decode するため, 8K HEVC では短い動画でも数分
    かかる. select filter なら stream 全体を 1 回だけ通過する.
    """
    indices = list(frame_indices)
    if indices != sorted(set(indices)):
        raise ValueError("frame_indices は昇順かつ重複なしでなければなりません")
    if not indices:
        return []
    binary = _resolve_bin(ffmpeg_bin)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{out_prefix}_{index:06d}.jpg" for index in range(len(indices))]
    for output in outputs:
        if output.exists():
            output.unlink()
    filter_script = out_dir / f".{out_prefix}-select.txt"
    chunks = _selection_chunks(indices, max_filter_expression_chars)
    if len(chunks) == 1:
        filter_script.write_text(f"select={_selection_expression(indices)}", encoding="utf-8")
        filter_args = ["-map", f"0:v:{stream_index}", _filter_file_option(binary), str(filter_script)]
    else:
        filter_script.write_text(_complex_selection_filter(stream_index, chunks), encoding="utf-8")
        filter_args = [_complex_filter_file_option(binary), str(filter_script), "-map", "[selected]"]
    output_pattern = out_dir / f"{out_prefix}_%06d.jpg"
    args = [
        binary,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        *_hardware_input_args(hwaccel),
        "-i",
        str(src),
        *filter_args,
        *_vfr_output_args(binary),
        "-start_number",
        "0",
        "-q:v",
        str(jpeg_quality),
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(output_pattern),
    ]
    try:
        result = _run_progress_process(args, len(outputs), progress)
    finally:
        filter_script.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg sequential extraction failed: {result.stderr[-1000:]}")
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"ffmpeg extracted {len(outputs) - len(missing)}/{len(outputs)} frames; "
            f"first missing: {missing[0]}"
        )
    if progress is not None and result.latest_frame < len(outputs):
        progress(len(outputs), len(outputs))
    return outputs


def _selection_chunks(indices: list[int], max_expression_chars: int = 3000) -> list[list[int]]:
    """平衡化済みの選択式を、指定した文字数以下の chunk に再帰分割する."""
    if len(_selection_expression(indices)) <= max_expression_chars or len(indices) == 1:
        return [indices]
    middle = len(indices) // 2
    return [
        *_selection_chunks(indices[:middle], max_expression_chars),
        *_selection_chunks(indices[middle:], max_expression_chars),
    ]


def _complex_selection_filter(
    stream_index: int,
    chunks: list[list[int]],
    *,
    pre_filter: str = "",
) -> str:
    """複数の小さな select 式を 1 decode の時系列 segment として連結する."""
    if len(chunks) < 2 or any(not chunk for chunk in chunks):
        raise ValueError("complex select には 2 個以上の非空 chunk が必要です")
    inputs = "".join(f"[chunk{index}]" for index in range(len(chunks)))
    prefix = f"{pre_filter}," if pre_filter else ""
    lines = [f"[0:v:{stream_index}]{prefix}split={len(chunks)}{inputs}"]
    selected = []
    for index, chunk in enumerate(chunks):
        start = chunk[0]
        relative = [frame - start for frame in chunk]
        name = f"selected{index}"
        lines.append(
            f"[chunk{index}]trim=start_frame={start}:end_frame={chunk[-1] + 1},"
            f"select={_selection_expression(relative)},setpts=PTS-STARTPTS[{name}]"
        )
        selected.append(f"[{name}]")
    lines.append(f"{''.join(selected)}concat=n={len(chunks)}:v=1:a=0[selected]")
    return ";\n".join(lines)


def _paired_selection_filter(
    stream_ordinal: int,
    output_index: int,
    chunks: list[list[int]],
) -> str:
    """2-stream extraction 用に衝突しない label を持つ selection graph を作る。"""
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError("paired select には非空 chunk が必要です")
    output = f"paired_selected{output_index}"
    if len(chunks) == 1:
        return f"[0:v:{stream_ordinal}]select={_selection_expression(chunks[0])}[{output}]"

    chunk_labels = [f"paired_s{output_index}_chunk{index}" for index in range(len(chunks))]
    lines = [
        f"[0:v:{stream_ordinal}]split={len(chunks)}"
        + "".join(f"[{label}]" for label in chunk_labels)
    ]
    selected_labels = []
    for index, (chunk, chunk_label) in enumerate(zip(chunks, chunk_labels, strict=True)):
        start = chunk[0]
        relative = [frame - start for frame in chunk]
        selected = f"paired_s{output_index}_selected{index}"
        lines.append(
            f"[{chunk_label}]trim=start_frame={start}:end_frame={chunk[-1] + 1},"
            f"select={_selection_expression(relative)},setpts=PTS-STARTPTS[{selected}]"
        )
        selected_labels.append(selected)
    lines.append(
        "".join(f"[{label}]" for label in selected_labels)
        + f"concat=n={len(chunks)}:v=1:a=0[{output}]"
    )
    return ";\n".join(lines)


@lru_cache(maxsize=8)
def _filter_file_option(binary: str) -> str:
    """FFmpeg build が受け付ける video filter-file option を実行 probe で決定する."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as file:
        file.write("null")
        script = Path(file.name)
    try:
        failures: list[str] = []
        for option in ("-/filter:v", "-filter_script:v"):
            result = subprocess.run(
                [
                    binary,
                    "-nostdin",
                    "-hide_banner",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=2x2:rate=1:duration=1",
                    option,
                    str(script),
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return option
            failures.append(result.stderr[-500:])
        raise RuntimeError(f"FFmpeg filter file option を利用できません: {failures}")
    finally:
        script.unlink(missing_ok=True)


@lru_cache(maxsize=8)
def _complex_filter_file_option(binary: str) -> str:
    """FFmpeg build が受け付ける complex filter-file option を実行 probe で決定する."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as file:
        file.write("[0:v]null[out]")
        script = Path(file.name)
    try:
        failures: list[str] = []
        for option in ("-/filter_complex", "-filter_complex_script"):
            result = subprocess.run(
                [
                    binary,
                    "-nostdin",
                    "-hide_banner",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=2x2:rate=1:duration=1",
                    option,
                    str(script),
                    "-map",
                    "[out]",
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return option
            failures.append(result.stderr[-500:])
        raise RuntimeError(f"FFmpeg complex filter file option を利用できません: {failures}")
    finally:
        script.unlink(missing_ok=True)


@lru_cache(maxsize=8)
def _vfr_output_args(binary: str) -> tuple[str, str]:
    """新旧 FFmpeg の VFR output option を実行 probe で選ぶ."""
    candidates = (("-fps_mode", "vfr"), ("-vsync", "vfr"))
    failures: list[str] = []
    for option in candidates:
        result = subprocess.run(
            [
                binary,
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=2x2:rate=1:duration=1",
                *option,
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return option
        failures.append(result.stderr[-500:])
    raise RuntimeError(f"FFmpeg VFR output option を利用できません: {failures}")


def _selection_expression(indices: Sequence[int]) -> str:
    terms: list[str] = []
    start = 0
    while start < len(indices):
        if start + 2 < len(indices):
            step = indices[start + 1] - indices[start]
            end = start + 2
            while end < len(indices) and indices[end] - indices[end - 1] == step:
                end += 1
            if end - start >= 3:
                first, last = indices[start], indices[end - 1]
                terms.append(
                    f"between(n\\,{first}\\,{last})"
                    if step == 1
                    else f"between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{step}))"
                )
                start = end
                continue
        terms.append(f"eq(n\\,{indices[start]})")
        start += 1
    if not terms:
        return ""

    # FFmpeg は式木の深さ超過も ENOMEM として返すため、加算を平衡化する。
    # https://github.com/FFmpeg/FFmpeg/blob/80eb9e99b9/libavutil/eval.c#L551
    def balanced_sum(first: int, last: int) -> str:
        if last - first == 1:
            return terms[first]
        middle = (first + last) // 2
        return f"({balanced_sum(first, middle)}+{balanced_sum(middle, last)})"

    return balanced_sum(0, len(terms))


def extract_paired_frames(
    src: Path,
    *,
    frame_indices: Sequence[int],
    out_dir_lens0: Path,
    out_dir_lens1: Path,
    stream_ordinals: tuple[int, int] = (0, 1),
    ffmpeg_bin: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    jpeg_quality: int = 3,
    max_filter_expression_chars: int = 3000,
    hwaccel: str | None = None,
) -> tuple[list[Path], list[Path]]:
    """二つの video stream の同じ decoded frame index を対で書き出す。

    1 process / 1 demux pass で両 stream を decode する。別 process にすると巨大な INSV を
    2 回読み、同じ storage 上では decode より重い I/O bottleneck になる。
    """
    indices = list(frame_indices)
    if indices != sorted(set(indices)):
        raise ValueError("frame_indices は昇順かつ重複なしでなければなりません")
    if not indices:
        return [], []
    if stream_ordinals[0] == stream_ordinals[1] or min(stream_ordinals) < 0:
        raise ValueError(f"stream_ordinals が不正です: {stream_ordinals}")

    binary = _resolve_bin(ffmpeg_bin)
    for directory in (out_dir_lens0, out_dir_lens1):
        directory.mkdir(parents=True, exist_ok=True)
    outputs0 = [out_dir_lens0 / f"lens0_{index:06d}.jpg" for index in range(len(indices))]
    outputs1 = [out_dir_lens1 / f"lens1_{index:06d}.jpg" for index in range(len(indices))]
    for output in (*outputs0, *outputs1):
        output.unlink(missing_ok=True)

    chunks = _selection_chunks(indices, max_filter_expression_chars)
    filter_script = out_dir_lens0.parent / ".paired-select.txt"
    filter_script.write_text(
        _paired_selection_filter(stream_ordinals[0], 0, chunks)
        + ";\n"
        + _paired_selection_filter(stream_ordinals[1], 1, chunks),
        encoding="utf-8",
    )
    args = [
        binary,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        *_hardware_input_args(hwaccel),
        "-i",
        str(src),
        _complex_filter_file_option(binary),
        str(filter_script),
        "-map",
        "[paired_selected0]",
        *_vfr_output_args(binary),
        "-start_number",
        "0",
        "-q:v",
        str(jpeg_quality),
        str(out_dir_lens0 / "lens0_%06d.jpg"),
        "-map",
        "[paired_selected1]",
        *_vfr_output_args(binary),
        "-start_number",
        "0",
        "-q:v",
        str(jpeg_quality),
        str(out_dir_lens1 / "lens1_%06d.jpg"),
    ]
    try:
        result = _run_progress_process(args, len(indices), progress)
    finally:
        filter_script.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg paired extraction failed: {result.stderr[-1000:]}")
    missing0 = [str(path) for path in outputs0 if not path.is_file()]
    missing1 = [str(path) for path in outputs1 if not path.is_file()]
    if missing0 or missing1:
        first_missing = (missing0 + missing1)[0]
        raise RuntimeError(
            f"ffmpeg extracted {len(outputs0) - len(missing0)}/{len(outputs0)} lens0 and "
            f"{len(outputs1) - len(missing1)}/{len(outputs1)} lens1 frames; "
            f"first missing: {first_missing}"
        )
    if progress is not None and result.latest_frame < len(indices):
        progress(len(indices), len(indices))
    return outputs0, outputs1
