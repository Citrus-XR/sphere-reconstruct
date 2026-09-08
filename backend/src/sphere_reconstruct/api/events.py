"""events WebSocket.

Worker が SQLite の event テーブルに書き込む log を, ID 昇順で follow する.
クライアントは ?since=<id> で古いのを飛ばせる. tail するのが目的なので, 未来の
event が入ってくるまで poll する.

SQLite WAL を単一の event source とし, process restart 後も同じ cursor 契約を保つ.
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
    # since < 0 は「今から」= 現在の最新 id から tail する (過去ログを再送しない).
    last_id = since if since >= 0 else await _max_id(db, job_id, project_id)
    try:
        while True:
            rows = await _fetch(db, last_id, job_id, project_id)
            if rows:
                for r in rows:
                    keys = r.keys()
                    payload = {
                        "id": r["id"],
                        "job_id": r["job_id"],
                        "project_id": r["project_id"],
                        "stage": r["stage"],
                        "level": r["level"],
                        "message": r["message"],
                        "msg_key": r["msg_key"] if "msg_key" in keys else None,
                        "msg_args": r["msg_args"] if "msg_args" in keys else None,
                        "progress": r["progress"],
                        "kind": r["kind"] if "kind" in keys else "log",
                        "ts": r["ts"],
                    }
                    await ws.send_text(json.dumps(payload, ensure_ascii=False))
                    last_id = r["id"]
            else:
                # receive を timeout 付きで待つことで、idle 中も client disconnect / server shutdown
                # を検出する。sleep だけでは接続状態が更新されず Uvicorn shutdown が完了しない。
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=0.5)
                    if message["type"] == "websocket.disconnect":
                        return
                except TimeoutError:
                    pass
    except (WebSocketDisconnect, asyncio.CancelledError):
        return


async def _max_id(db, job_id: str | None, project_id: str | None) -> int:
    where = ["1=1"]
    args: list = []
    if job_id is not None:
        where.append("job_id = ?")
        args.append(job_id)
    if project_id is not None:
        where.append("project_id = ?")
        args.append(project_id)
    cur = await db.conn.execute(
        f"SELECT COALESCE(MAX(id), 0) AS m FROM event WHERE {' AND '.join(where)}", args
    )
    row = await cur.fetchone()
    return int(row["m"]) if row else 0


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
