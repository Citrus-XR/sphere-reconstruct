"""旧 cleanup の編集値を全 track cleanup の設定へ一度だけ移行する。"""

from __future__ import annotations

import json

from ..infrastructure.database import Database


async def migrate_cleanup_settings(db: Database) -> int:
    obsolete = {
        "cleanupFarDistanceRatio",
        "cleanupFarMinAngle",
        "cleanupMaxReprojection",
        "cleanupMinTrackLength",
    }
    defaults = {"cleanupRelativeError": 0.02, "cleanupPixelSigma": 1.0, "cleanupMaxCrossError": 2.0}
    migrated = 0
    async with db.transaction() as conn:
        rows = await (await conn.execute("SELECT id, metadata_json FROM project")).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            params = metadata.get("ui", {}).get("params", {})
            old_cleanup = bool(obsolete.intersection(params))
            if not old_cleanup and "colmapTriangulationPreset" not in params:
                continue
            metadata["ui"]["params"] = {
                **(defaults if old_cleanup else {}),
                **{
                    key: value
                    for key, value in params.items()
                    if key not in obsolete and key != "colmapTriangulationPreset"
                },
            }
            await conn.execute(
                "UPDATE project SET metadata_json=? WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), row["id"]),
            )
            migrated += 1
    return migrated
