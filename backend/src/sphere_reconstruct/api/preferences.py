"""Workspace 全体の UI 設定 API。ブラウザの保存領域を使用しない。"""

import json

from fastapi import APIRouter

from ..domain.ui_state import WorkspacePreferences, merge_ui_patch
from ..infrastructure.database import get_db

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


@router.get("", response_model=WorkspacePreferences)
async def get_preferences() -> WorkspacePreferences:
    row = await (await get_db().conn.execute(
        "SELECT value_json FROM workspace_preferences WHERE id=1"
    )).fetchone()
    return WorkspacePreferences.model_validate_json(row[0]) if row else WorkspacePreferences()


@router.patch("", response_model=WorkspacePreferences)
async def patch_preferences(body: WorkspacePreferences) -> WorkspacePreferences:
    async with get_db().transaction() as conn:
        row = await (await conn.execute(
            "SELECT value_json FROM workspace_preferences WHERE id=1"
        )).fetchone()
        current = json.loads(row[0]) if row else {}
        updated = WorkspacePreferences.model_validate(
            merge_ui_patch(current, body.model_dump(exclude_unset=True))
        )
        await conn.execute(
            "INSERT INTO workspace_preferences(id,value_json) VALUES(1,?) "
            "ON CONFLICT(id) DO UPDATE SET value_json=excluded.value_json",
            (updated.model_dump_json(),),
        )
    return updated
