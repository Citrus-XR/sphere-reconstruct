"""実行と成果物変更の project 単位の所有権を検証する。"""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from sphere_reconstruct.api import jobs, previews, project_access, projects, stages
from sphere_reconstruct.domain import project as project_domain
from sphere_reconstruct.domain import source as source_domain
from sphere_reconstruct.imaging.source_region import default_region
from sphere_reconstruct.infrastructure.database import Database
from sphere_reconstruct.infrastructure.project_lock import project_lock
from sphere_reconstruct.job_supervisor import JobSupervisor


@pytest.fixture
async def project_api(tmp_path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    monkeypatch.setattr(project_domain, "workspace_root", lambda: tmp_path)
    monkeypatch.setattr("sphere_reconstruct.job_supervisor.workspace_root", lambda: tmp_path)
    project = await project_domain.create_project(database, "ownership")
    images = tmp_path / "images"
    images.mkdir()
    source = await source_domain.add_source(
        database, project.id, label="phone", role=source_domain.SourceRole.PRIMARY,
        adapter=source_domain.SourceAdapter.GENERIC_IMAGES, media_kind=source_domain.MediaKind.IMAGES,
        projection=source_domain.Projection.PERSPECTIVE, path=str(images),
    )
    temporary = project.workspace_dir / ".extract_frames.tmp"
    temporary.mkdir()
    (temporary / "payload.bin").write_bytes(b"active")
    supervisor = JobSupervisor(database, tmp_path / "state.db")
    for module in (jobs, previews, project_access, projects, stages):
        monkeypatch.setattr(module, "get_db", lambda: database)
    monkeypatch.setattr(jobs, "get_supervisor", lambda: supervisor)
    monkeypatch.setattr(projects, "_resolve_source_path", lambda _path: images)
    monkeypatch.setattr(
        "sphere_reconstruct.job_supervisor.spawn_worker", lambda *_args, **_kwargs: SimpleNamespace(pid=12345)
    )
    finished = asyncio.Event()

    async def monitor(*_args):
        await finished.wait()
        return 0

    monkeypatch.setattr(supervisor, "_await_worker", monitor)
    app = FastAPI()
    for module in (jobs, previews, projects, stages):
        app.include_router(module.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        try:
            yield SimpleNamespace(
                client=client, database=database, project=project, source=source,
                supervisor=supervisor, temporary=temporary,
            )
        finally:
            finished.set()
            await asyncio.gather(*supervisor._tasks.values())
            await database.close()


async def _insert_active(context, status="running"):
    await context.database.conn.execute(
        "INSERT INTO job (id,project_id,kind,status,created_at) VALUES ('active',?,'rerun_stage',?,'2026-09-07')",
        (context.project.id, status),
    )


@pytest.mark.parametrize("status", ["queued", "running"])
@pytest.mark.parametrize("operation", ["clear", "clear_outputs", "delete", "add_source", "delete_source", "primary", "region", "run", "rerun"])
async def test_active_job_rejects_conflicting_mutations(project_api, status, operation):
    context = project_api
    await _insert_active(context, status)
    base = f"/api/projects/{context.project.id}"
    requests = {
        "clear": ("POST", f"{base}/stages/extract_frames/clear", None),
        "clear_outputs": ("POST", f"{base}/clear-outputs", {"stages": ["extract_frames"]}),
        "delete": ("DELETE", base, None),
        "add_source": ("POST", f"{base}/sources", {"adapter": "generic_images", "media_kind": "images", "projection": "perspective", "path": "images"}),
        "delete_source": ("DELETE", f"{base}/sources/{context.source.id}", None),
        "primary": ("POST", f"{base}/sources/{context.source.id}/make-primary", None),
        "region": ("PUT", f"{base}/source-region?source_id={context.source.id}", default_region("perspective")),
        "run": ("POST", f"{base}/run", {}),
        "rerun": ("POST", f"{base}/rerun/extract_frames", {}),
    }
    method, url, body = requests[operation]
    response = await context.client.request(method, url, json=body)
    assert response.status_code == 409, response.text
    assert (context.temporary / "payload.bin").read_bytes() == b"active"
    assert not (context.project.workspace_dir / "source_regions.json").exists()


async def test_concurrent_submissions_only_start_one_worker(project_api):
    url = f"/api/projects/{project_api.project.id}/rerun/extract_frames"
    responses = await asyncio.gather(project_api.client.post(url), project_api.client.post(url))
    assert sorted(response.status_code for response in responses) == [200, 409]
    rows = await (await project_api.database.conn.execute("SELECT id FROM job")).fetchall()
    assert len(rows) == 1


async def test_enqueue_waits_for_clear_but_preferences_and_reads_do_not(project_api, monkeypatch):
    context = project_api
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_clear(function, *args, **kwargs):
        started.set()
        await finish.wait()
        return function(*args, **kwargs)

    monkeypatch.setattr(stages, "run_in_threadpool", slow_clear)
    base = f"/api/projects/{context.project.id}"
    clear = asyncio.create_task(context.client.post(f"{base}/stages/extract_frames/clear"))
    await started.wait()
    enqueue = asyncio.create_task(context.client.post(f"{base}/rerun/extract_frames"))
    try:
        settings = await asyncio.wait_for(
            context.client.patch(f"{base}/ui-state", json={"ui": {"selectedStage": "extract_frames"}}), timeout=2
        )
        read = await asyncio.wait_for(context.client.get(base), timeout=2)
        assert settings.status_code == read.status_code == 200
        assert not enqueue.done()
    finally:
        finish.set()
        responses = await asyncio.gather(clear, enqueue)
    assert all(response.status_code == 200 for response in responses)


async def test_cancellation_keeps_artifacts_owned_until_worker_stops(project_api, monkeypatch):
    context = project_api
    await _insert_active(context)
    context.supervisor._handles["active"] = object()
    stopping = threading.Event()
    stopped = threading.Event()

    def stop(*_args, **_kwargs):
        stopping.set()
        assert stopped.wait(5)

    monkeypatch.setattr("sphere_reconstruct.job_supervisor.cancel", stop)
    cancelling = asyncio.create_task(context.client.post("/api/jobs/active/cancel"))
    assert await asyncio.to_thread(stopping.wait, 5)
    running = await (await context.database.conn.execute("SELECT status FROM job WHERE id='active'")).fetchone()
    assert running["status"] == "running"
    clear = asyncio.create_task(
        context.client.post(f"/api/projects/{context.project.id}/stages/extract_frames/clear")
    )
    try:
        await asyncio.sleep(0)
        assert not clear.done()
        assert context.temporary.exists()
    finally:
        stopped.set()
        cancelled, cleared = await asyncio.gather(cancelling, clear)
    assert cancelled.json() == {"cancelled": True}
    assert cleared.status_code == 200


async def test_missing_project_is_rejected_before_worker_start(project_api):
    response = await project_api.client.post("/api/projects/missing/run")
    assert response.status_code == 404
    assert not project_api.supervisor._handles


async def test_one_project_lock_does_not_block_another_project(project_api):
    context = project_api
    other = await project_domain.create_project(context.database, "independent")
    async with project_lock(context.project.id):
        response = await asyncio.wait_for(context.client.post(f"/api/projects/{other.id}/run"), timeout=2)
    assert response.status_code == 200


async def test_scratch_cleanup_waits_for_project_ownership(project_api):
    context = project_api
    await _insert_active(context, status="cancelled")
    await context.database.conn.execute(
        "INSERT INTO stage_run (id,project_id,job_id,stage,impl_version,params_hash,inputs_hash,status,started_at) "
        "VALUES ('stage',?,'active','extract_frames','0','','','cancelled','2026-09-07')",
        (context.project.id,),
    )
    async with project_lock(context.project.id):
        cleanup = asyncio.create_task(context.supervisor.cleanup_scratch("active"))
        await asyncio.sleep(0)
        assert not cleanup.done()
        assert context.temporary.exists()
    await cleanup
    assert not context.temporary.exists()
