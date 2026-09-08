"""成果物公開中の失敗と再実行の lifecycle を検証する。"""

import sqlite3

import pytest

from sphere_reconstruct.domain.artifacts import FileRef, StageManifest, manifest_path
from sphere_reconstruct.domain.pipeline_state import StageName
from sphere_reconstruct.infrastructure.database import _SCHEMA
from sphere_reconstruct.infrastructure.filesystem import sha256_bytes
from sphere_reconstruct.pipeline.engine import Engine
from sphere_reconstruct.pipeline.stage import Stage, new_manifest
from sphere_reconstruct.worker_entry import run_pipeline_entry


class ProbeStage(Stage):
    name = StageName.EXTRACT_FRAMES

    def collect_inputs(self, ctx):
        return []

    def execute(self, ctx):
        content = b"published output"
        (ctx.stage_out_dir / "payload.bin").write_bytes(content)
        manifest = new_manifest(self.name, self.impl_version)
        manifest.outputs = [FileRef(path="extract_frames/payload.bin", size=len(content), sha256=sha256_bytes(content))]
        return manifest


@pytest.fixture
def worker_project(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT INTO project (id,name,created_at,updated_at,state) "
            "VALUES ('project','project','2026-09-07','2026-09-07','created')"
        )
        connection.execute(
            "INSERT INTO job (id,project_id,kind,status,created_at) "
            "VALUES ('job','project','rerun_stage','queued','2026-09-07')"
        )
    monkeypatch.setattr("sphere_reconstruct.pipeline.engine.get_stage_cls", lambda _stage: ProbeStage)
    return tmp_path, path


def _run(workspace, db_path):
    run_pipeline_entry(
        db_path=str(db_path), workspace_root=str(workspace), project_id="project", job_id="job", stage="extract_frames"
    )


@pytest.mark.parametrize("failure", ["replace", "manifest", "invalidate", "state"])
def test_publication_failure_finishes_both_job_and_stage(worker_project, monkeypatch, failure):
    workspace, db_path = worker_project
    error = PermissionError(f"publication failed: {failure}")

    def fail(*_args, **_kwargs):
        raise error

    if failure == "replace":
        monkeypatch.setattr("sphere_reconstruct.pipeline.engine.atomic_replace_dir", fail)
    elif failure == "manifest":
        monkeypatch.setattr(StageManifest, "dump", fail)
    elif failure == "invalidate":
        monkeypatch.setattr(Engine, "_invalidate_downstream", fail)
    else:
        monkeypatch.setattr(Engine, "_refresh_project_state", fail)
    with pytest.raises(PermissionError) as raised:
        _run(workspace, db_path)
    assert raised.value is error
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT status FROM job").fetchone()[0] == "failed"
        status, finished_at, diagnostic = connection.execute(
            "SELECT status,finished_at,error_text FROM stage_run"
        ).fetchone()
        assert status == "failed"
        assert finished_at is not None
        assert str(error) in diagnostic
        assert connection.execute("SELECT COUNT(*) FROM event WHERE msg_key='log.stage_published'").fetchone()[0] == 0


def test_invalidation_failure_does_not_publish_a_cache_hit_manifest(worker_project, monkeypatch):
    workspace, db_path = worker_project

    def fail(*_args):
        raise PermissionError("downstream locked")

    with monkeypatch.context() as failure:
        failure.setattr(Engine, "_invalidate_downstream", fail)
        with pytest.raises(PermissionError, match="downstream locked"):
            _run(workspace, db_path)
    project_dir = workspace / "projects" / "project"
    assert not manifest_path(project_dir, "extract_frames").exists()
    _run(workspace, db_path)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT status FROM job").fetchone()[0] == "succeeded"
        assert connection.execute("SELECT COUNT(*) FROM event WHERE msg_key='log.cache_hit'").fetchone()[0] == 0


def test_partial_manifest_write_does_not_poison_retry(worker_project, monkeypatch):
    workspace, db_path = worker_project

    def partial_write(_manifest, path):
        path.write_text("{")
        raise OSError("incomplete manifest")

    with monkeypatch.context() as failure:
        failure.setattr(StageManifest, "dump", partial_write)
        with pytest.raises(OSError, match="incomplete manifest"):
            _run(workspace, db_path)
    project_dir = workspace / "projects" / "project"
    assert not manifest_path(project_dir, "extract_frames").exists()
    assert not list((project_dir / "manifests").glob("*.tmp"))
    _run(workspace, db_path)
    assert StageManifest.load(manifest_path(project_dir, "extract_frames")).stage == "extract_frames"
