"""JobSupervisor のキャンセル状態と非同期停止を検証する。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sphere_reconstruct.infrastructure.database import Database
from sphere_reconstruct.job_supervisor import JobSupervisor


@pytest.mark.asyncio
async def test_spawn_failure_is_terminal_and_preserves_diagnostic(tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    try:
        await database.conn.execute(
            "INSERT INTO project (id,name,created_at,updated_at,state) "
            "VALUES ('project','project','2026-09-07','2026-09-07','created')"
        )
        supervisor = JobSupervisor(database, tmp_path / "state.db")
        error = OSError("worker process unavailable")

        def fail(*_args, **_kwargs):
            raise error

        monkeypatch.setattr("sphere_reconstruct.job_supervisor.spawn_worker", fail)
        with pytest.raises(OSError) as raised:
            await supervisor.enqueue_run_pipeline(project_id="project", stage="extract_frames")
        assert raised.value is error
        row = await (await database.conn.execute("SELECT * FROM job")).fetchone()
        assert row["status"] == "failed"
        assert row["finished_at"] is not None
        assert row["started_at"] is None
        assert str(error) in row["error_text"]
        event = await (await database.conn.execute("SELECT level,message FROM event")).fetchone()
        assert event["level"] == "error"
        assert str(error) in event["message"]
        assert not supervisor._handles
        assert not supervisor._tasks
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_cancel_is_terminal_even_when_worker_exit_races(tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    now = datetime.now(UTC).isoformat()
    async with database.transaction() as connection:
        await connection.execute(
            "INSERT INTO project (id, name, created_at, updated_at, state) VALUES (?, ?, ?, ?, ?)",
            ("project", "test", now, now, "created"),
        )
        await connection.execute(
            "INSERT INTO job (id, project_id, kind, status, created_at) VALUES (?, ?, ?, ?, ?)",
            ("job", "project", "run_pipeline", "running", now),
        )
        await connection.execute(
            """
            INSERT INTO stage_run
                (id, project_id, job_id, stage, impl_version, params_hash, inputs_hash, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("stage", "project", "job", "extract_frames", "1", "", "", "running", now),
        )

    supervisor = JobSupervisor(database, tmp_path / "state.db")
    handle = object()
    supervisor._handles["job"] = handle  # type: ignore[assignment]
    stopped: list[object] = []
    monkeypatch.setattr(
        "sphere_reconstruct.job_supervisor.cancel",
        lambda target, *, grace_seconds: stopped.append(target),
    )

    assert await supervisor.cancel_job("job") is True
    assert stopped == [handle]
    row = await (await database.conn.execute("SELECT status, error_text FROM job WHERE id='job'")).fetchone()
    stage = await (await database.conn.execute("SELECT status FROM stage_run WHERE id='stage'")).fetchone()
    assert row["status"] == "cancelled" and row["error_text"] is None
    assert stage["status"] == "cancelled"

    await database.conn.execute("UPDATE job SET status='failed', error_text='worker race' WHERE id='job'")
    await database.conn.execute("UPDATE stage_run SET status='failed' WHERE id='stage'")
    await database.conn.commit()
    await supervisor._finalize_if_orphaned("job", -15)
    row = await (await database.conn.execute("SELECT status, error_text FROM job WHERE id='job'")).fetchone()
    stage = await (await database.conn.execute("SELECT status FROM stage_run WHERE id='stage'")).fetchone()
    assert row["status"] == "cancelled" and row["error_text"] is None
    assert stage["status"] == "cancelled"
    await database.close()


@pytest.mark.asyncio
async def test_cancel_cleanup_removes_stage_temporary_artifacts(tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    now = datetime.now(UTC).isoformat()
    async with database.transaction() as connection:
        await connection.execute(
            "INSERT INTO project (id, name, created_at, updated_at, state) VALUES (?, ?, ?, ?, ?)",
            ("project", "test", now, now, "created"),
        )
        await connection.execute(
            "INSERT INTO job (id, project_id, kind, status, created_at) VALUES (?, ?, ?, ?, ?)",
            ("job", "project", "rerun_stage", "cancelled", now),
        )
        await connection.execute(
            """
            INSERT INTO stage_run
                (id, project_id, job_id, stage, impl_version, params_hash, inputs_hash, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("stage", "project", "job", "generate_feature_masks", "1", "", "", "cancelled", now),
        )

    project_dir = tmp_path / "projects" / "project"
    temporary = project_dir / ".generate_feature_masks.tmp"
    temporary.mkdir(parents=True)
    (temporary / "partial.png").write_bytes(b"partial")
    final = project_dir / "generate_feature_masks"
    final.mkdir()
    (final / "manifest.json").write_text("{}")
    monkeypatch.setattr("sphere_reconstruct.job_supervisor.workspace_root", lambda: tmp_path)

    supervisor = JobSupervisor(database, tmp_path / "state.db")
    await supervisor.cleanup_scratch("job")

    assert not temporary.exists()
    assert final.is_dir()
    await database.close()
