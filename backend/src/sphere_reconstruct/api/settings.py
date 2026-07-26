"""Read-only runtime settings と軽量 dependency check API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..sam3.settings import quick_check as sam3_quick_check
from ..settings import get_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def read_settings() -> dict[str, Any]:
    s = get_settings()
    return {
        "server": s.server.model_dump(),
        "workspace": {"root": str(s.workspace.root)},
        "filesystem": {"allowed_roots": [str(p) for p in s.filesystem.allowed_roots]},
        "binaries": s.binaries.model_dump(),
        "sam3": s.sam3.model_dump(),
        "aliked": s.aliked.model_dump(),
        "denoise": s.denoise.model_dump(),
        "log": s.log.model_dump(),
    }


@router.post("/test-sam3")
async def test_sam3() -> dict[str, Any]:
    """SAM3 の path だけを検証する. API process では Torch を import しない."""
    check = sam3_quick_check()
    return {
        "ok": check.ok,
        "repo_exists": check.repo_exists,
        "checkpoint_exists": check.checkpoint_exists,
        "sam3_package_present": check.sam3_package_present,
        "checkpoint_size": check.checkpoint_size,
        "message": check.message,
    }


@router.get("/env-check")
async def env_check() -> dict[str, Any]:
    """外部依存 (ffmpeg / ffprobe / colmap / SAM3) の在否を確認する.

    起動画面で「何が足りないか」を一覧するための軽量チェック. バイナリは
    -h/--version を叩かず, パス解決のみ (実行は環境によって重いので避ける).
    """
    import shutil

    s = get_settings()

    def _bin(explicit: str, name: str) -> dict[str, Any]:
        if explicit:
            p = Path(explicit)
            return {"configured": explicit, "found": p.exists(), "source": "config"}
        found = shutil.which(name)
        return {"configured": None, "found": found is not None, "resolved": found, "source": "path"}

    sam3 = sam3_quick_check()
    return {
        "ffmpeg": _bin(s.binaries.ffmpeg, "ffmpeg"),
        "ffprobe": _bin(s.binaries.ffprobe, "ffprobe"),
        "colmap": _bin(s.binaries.colmap, "colmap"),
        "sam3": {"ok": sam3.ok, "message": sam3.message},
        "workspace": str(s.workspace.root),
        "allowed_roots": [str(p) for p in s.filesystem.allowed_roots],
    }
