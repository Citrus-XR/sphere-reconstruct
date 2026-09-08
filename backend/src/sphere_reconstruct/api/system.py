"""system / filesystem API.

- GET /api/system/stats       : CPU / GPU 使用率 + GPU 名 (下部ステータスバー用).
- GET /api/fs/roots           : allowed_roots 一覧 (ファイルピッカーの起点).
- GET /api/fs/browse?path=    : allowed_roots 内のディレクトリ列挙 (INSV/動画/画像を選ぶ).

FastAPI プロセスで動くため CUDA/torch は触らない. GPU 情報は nvidia-smi サブプロセスを
呼んで得る（無ければ空）。CPU/RAM は必須 dependency の psutil から取得する。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import psutil
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from ..diagnostics import diagnose
from ..infrastructure.filesystem import PathNotAllowedError, ensure_within_any
from ..settings import get_settings

router = APIRouter(tags=["system"])

# 選択可能な入力拡張子.
_SOURCE_EXTS = {
    ".insv",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".mts",
    ".m2ts",
    ".webm",
    ".wmv",
    ".3gp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@router.get("/api/system/stats")
async def system_stats() -> dict:
    cpu = _cpu_percent()
    gpus = await _gpu_stats()
    return {"cpu_percent": cpu, "ram": _ram_stats(), "gpus": gpus}


@router.get("/api/system/doctor")
async def system_doctor() -> dict:
    return await run_in_threadpool(diagnose)


def _ram_stats() -> dict:
    """システム RAM 使用率を返す。"""
    vm = psutil.virtual_memory()
    return {
        "percent": float(vm.percent),
        "used_mb": vm.used / (1024 * 1024),
        "total_mb": vm.total / (1024 * 1024),
    }


@router.get("/api/fs/roots")
async def fs_roots() -> dict:
    roots = [str(Path(r)) for r in get_settings().filesystem.allowed_roots]
    return {"roots": roots}


@router.get("/api/fs/drives")
async def fs_drives() -> dict:
    """選択可能なドライブ/ルート (実在する allowed_roots のみ). ドライブ選択 UI 用."""
    out = []
    for r in get_settings().filesystem.allowed_roots:
        p = Path(r)
        try:
            if p.exists():
                out.append(str(p))
        except OSError:
            continue
    return {"drives": out}


@router.get("/api/fs/browse")
async def fs_browse(path: str = Query(...)) -> dict:
    """path 直下のディレクトリと入力候補ファイルを返す. allowed_roots 内に限定する."""
    roots = get_settings().filesystem.allowed_roots
    try:
        p = ensure_within_any(roots, Path(path))
    except PathNotAllowedError as e:
        raise HTTPException(status_code=403, detail=str(e)) from None
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail=f"not a directory: {p}")

    dirs, files = [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry), "is_dir": True})
                elif entry.suffix.lower() in _SOURCE_EXTS:
                    files.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "is_dir": False,
                            "size": entry.stat().st_size,
                            "ext": entry.suffix.lower(),
                        }
                    )
            except OSError:
                continue  # アクセス不能なエントリは飛ばす.
    except PermissionError:
        raise HTTPException(status_code=403, detail="permission denied") from None

    parent = str(p.parent) if _within_any(roots, p.parent) and p.parent != p else None
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files}


def _within_any(roots, path: Path) -> bool:
    try:
        ensure_within_any(roots, path)
        return True
    except PathNotAllowedError:
        return False


def _cpu_percent() -> float:
    # interval=None は前回呼び出しからの平均 (非ブロッキング). 定期ポーリング前提.
    return float(psutil.cpu_percent(interval=None))


async def _gpu_stats() -> list[dict]:
    """nvidia-smi で GPU 名 / 使用率 / VRAM を取得. 失敗時は空リスト."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return []
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
    except (TimeoutError, OSError):
        await _terminate_gpu_probe(proc)
        return []
    except asyncio.CancelledError:
        await _terminate_gpu_probe(proc)
        raise
    gpus = []
    for line in out.decode(errors="replace").strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        name, util, used, total = parts[:4]
        gpus.append(
            {
                "name": name,
                "util_percent": _num(util),
                "mem_used_mb": _num(used),
                "mem_total_mb": _num(total),
            }
        )
    return gpus


async def _terminate_gpu_probe(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    await proc.communicate()


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None
