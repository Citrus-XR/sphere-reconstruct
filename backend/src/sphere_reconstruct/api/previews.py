"""previews API のプレースホルダ.

Phase 3 以降で frames / masks / reconstruction / points のバイナリを配る.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["previews"])


@router.get("/api/projects/{project_id}/frames")
async def list_frames(project_id: str) -> dict:
    raise HTTPException(status_code=501, detail="frames listing not implemented until phase 3")
