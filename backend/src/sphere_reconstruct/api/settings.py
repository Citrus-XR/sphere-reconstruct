"""settings API (最小構成).

現段階は read-only. 書き込みは Phase 7 で SAM3 パステスト等と合わせて実装.
"""

from __future__ import annotations

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
        "log": s.log.model_dump(),
    }


@router.post("/test-sam3")
async def test_sam3() -> dict[str, Any]:
    """SAM3 のパス設定を検証する. Torch import は行わない (それは Phase 4 で Worker に)."""
    check = sam3_quick_check()
    return {
        "ok": check.ok,
        "repo_exists": check.repo_exists,
        "checkpoint_exists": check.checkpoint_exists,
        "sam3_package_present": check.sam3_package_present,
        "checkpoint_size": check.checkpoint_size,
        "message": check.message,
    }
