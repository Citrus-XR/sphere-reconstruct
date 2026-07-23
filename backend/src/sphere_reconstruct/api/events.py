"""events WebSocket.

Worker が SQLite の event テーブルに書き込む log を, ID 昇順で follow する.
クライアントは ?since=<id> で古いのを飛ばせる. tail するのが目的なので, 未来の
event が入ってくるまで poll する.

将来的にはより効率的な pubsub (multiprocessing Queue + Manager) に置換する
候補があるが, Phase 1 では単純さ優先.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..infrastructure.database import get_db

router = APIRouter(tags=["events"])


@router.websocket("/api/events")
async def events_ws(
    ws: WebSocket,
    since: int = Query(0, description="この ID より新しいイベントだけを送る"),
    job_id: str | None = Query(None, description="指定した job のイベントだけを送る"),
    project_id: str | None = Query(None, description="指定した project のイベントだけを送る"),
) -> None:
    await ws.accept()
    db = get_db()
    last_id = since
    try:
        while True:
            rows = await _fetch(db, last_id, job_id, project_id)
            if rows:
                for r in rows:
                    payload = {
                        "id": r["id"],
                        "job_id": r["job_id"],
                        "project_id": r["project_id"],
                        "stage": r["stage"],
                        "level": r["level"],
                        "message": r["message"],
                        "progress": r["progress"],
                        "ts": r["ts"],
                    }
                    await ws.send_text(json.dumps(payload, ensure_ascii=False))
                    last_id = r["id"]
            else:
                # ハングを避けるため, 500ms 待って再ポーリング.
                await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


async def _fetch(db, last_id: int, job_id: str | None, project_id: str | None):
    where = ["id > ?"]
    args: list = [last_id]
    if job_id is not None:
        where.append("job_id = ?")
        args.append(job_id)
    if project_id is not None:
        where.append("project_id = ?")
        args.append(project_id)
    query = f"SELECT * FROM event WHERE {' AND '.join(where)} ORDER BY id ASC LIMIT 200"
    cur = await db.conn.execute(query, args)
    return await cur.fetchall()
