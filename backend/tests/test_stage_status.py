"""Stage status と resumable progress snapshot の API contract。"""

import json

from sphere_reconstruct.api import stages
from sphere_reconstruct.api.stages import _event_payload, _resolve_stage_status
from sphere_reconstruct.infrastructure.database import Database


def test_running_stage_is_not_hidden_by_previous_stale_marker():
    assert _resolve_stage_status(has_output=False, stale=True, run_status="running") == "running"


def test_missing_invalidated_output_is_stale_after_previous_success():
    assert _resolve_stage_status(has_output=False, stale=True, run_status="succeeded") == "stale"


def test_activity_and_numeric_progress_are_independent_event_payloads():
    row = {
        "stage": "reconstruct",
        "job_id": "job-1",
        "activity_id": 12,
        "activity_level": "info",
        "activity_message": "global refinement",
        "activity_msg_key": "log.recon_global_refinement",
        "activity_msg_args": '{"pass":2,"done":1487,"total":1488}',
        "activity_progress": None,
        "activity_kind": "progress",
        "activity_ts": "2026-08-01T00:00:02Z",
        "numeric_id": 11,
        "numeric_level": "info",
        "numeric_message": "registered frames",
        "numeric_msg_key": "log.recon_mapper_progress",
        "numeric_msg_args": '{"done":1487,"total":1488}',
        "numeric_progress": 0.899,
        "numeric_kind": "progress",
        "numeric_ts": "2026-08-01T00:00:01Z",
    }

    activity = _event_payload(row, "activity", "project-1")
    progress = _event_payload(row, "numeric", "project-1")

    assert activity["id"] == 12
    assert activity["progress"] is None
    assert activity["msg_args"]["pass"] == 2
    assert progress["id"] == 11
    assert progress["progress"] == 0.899
    assert progress["msg_args"] == {"done": 1487, "total": 1488}


async def test_stage_api_restores_latest_activity_and_numeric_progress_separately(tmp_path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    try:
        await database.conn.execute(
            "INSERT INTO project (id, name, created_at, updated_at, state) VALUES (?, ?, ?, ?, ?)",
            ("project-1", "Project", "2026-08-01T00:00:00Z", "2026-08-01T00:00:00Z", "matched"),
        )
        await database.conn.execute(
            "INSERT INTO job (id, project_id, kind, stage, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "job-1",
                "project-1",
                "rerun_stage",
                "reconstruct",
                "running",
                "2026-08-01T00:00:00Z",
            ),
        )
        await database.conn.execute(
            "INSERT INTO job_request (job_id, params_by_stage_json, skip_json) VALUES (?, ?, ?)",
            (
                "job-1",
                json.dumps({"reconstruct": {"mapper": "incremental"}}),
                "[]",
            ),
        )
        await database.conn.execute(
            "INSERT INTO stage_run "
            "(id, project_id, job_id, stage, impl_version, params_hash, inputs_hash, status, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run-1",
                "project-1",
                "job-1",
                "reconstruct",
                "1",
                "params",
                "inputs",
                "running",
                "2026-08-01T00:00:00Z",
            ),
        )
        await database.conn.execute(
            "INSERT INTO event "
            "(job_id, project_id, stage, level, message, msg_key, msg_args, progress, kind, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "job-1",
                "project-1",
                "reconstruct",
                "info",
                "registered frames",
                "log.recon_mapper_progress",
                json.dumps({"done": 1487, "total": 1488}),
                0.899,
                "progress",
                "2026-08-01T00:00:01Z",
            ),
        )
        await database.conn.execute(
            "INSERT INTO event "
            "(job_id, project_id, stage, level, message, msg_key, msg_args, progress, kind, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "job-1",
                "project-1",
                "reconstruct",
                "info",
                "global refinement",
                "log.recon_global_refinement",
                json.dumps({"pass": 2, "done": 1487, "total": 1488}),
                None,
                "progress",
                "2026-08-01T00:00:02Z",
            ),
        )
        await database.conn.commit()
        monkeypatch.setattr(stages, "get_db", lambda: database)
        monkeypatch.setattr(stages, "_project_dir", lambda project_id: tmp_path / "projects" / project_id)

        result = await stages.list_stages("project-1")
        reconstruct = next(stage for stage in result["stages"] if stage["stage"] == "reconstruct")

        assert reconstruct["progress_event"]["id"] < reconstruct["activity_event"]["id"]
        assert reconstruct["progress_event"]["msg_args"] == {"done": 1487, "total": 1488}
        assert reconstruct["activity_event"]["msg_args"] == {
            "pass": 2,
            "done": 1487,
            "total": 1488,
        }
        assert reconstruct["active_params"] == {"mapper": "incremental"}
    finally:
        await database.close()
