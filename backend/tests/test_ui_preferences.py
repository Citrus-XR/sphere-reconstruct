"""UI 設定は server に永続化し、並行する部分更新を失わない。"""

import asyncio
import json

import httpx
import pytest

from sphere_reconstruct.api import preferences, projects
from sphere_reconstruct.domain.project import create_project, get_project, patch_ui_state
from sphere_reconstruct.infrastructure.database import Database
from sphere_reconstruct.main import app


@pytest.fixture
async def ui_client(tmp_path, monkeypatch):
    database = Database(tmp_path / "state.db")
    await database.connect()
    monkeypatch.setattr(preferences, "get_db", lambda: database)
    monkeypatch.setattr(projects, "get_db", lambda: database)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, database
    await database.close()


async def test_workspace_defaults_and_partial_updates_survive_reconnect(ui_client):
    client, database = ui_client
    defaults = (await client.get("/api/preferences")).json()
    assert defaults["theme"] == "auto"
    assert defaults["layout"] is None
    assert defaults["viewer"]["showGrid"] is True
    assert (await client.patch("/api/preferences", json={"viewer": {"pointSize": 4}})).status_code == 200
    await client.patch("/api/preferences", json={"theme": "dark", "viewer": {"showCams": False}})
    await database.close()
    await database.connect()
    saved = (await client.get("/api/preferences")).json()
    assert saved["theme"] == "dark"
    assert saved["viewer"]["pointSize"] == 4
    assert saved["viewer"]["showCams"] is False
    assert saved["viewer"]["showPoints"] is True


async def test_concurrent_workspace_updates_do_not_clobber_fields(ui_client):
    client, _ = ui_client
    responses = await asyncio.gather(
        client.patch("/api/preferences", json={"viewer": {"pointSize": 5}}),
        client.patch("/api/preferences", json={"viewer": {"showGrid": False}}),
        client.patch("/api/preferences", json={"lang": "zh"}),
    )
    assert all(response.status_code == 200 for response in responses)
    saved = (await client.get("/api/preferences")).json()
    assert saved["viewer"]["pointSize"] == 5
    assert saved["viewer"]["showGrid"] is False
    assert saved["lang"] == "zh"


@pytest.mark.parametrize("patch", [
    {"theme": "purple"}, {"viewer": {"pointSize": 100}},
    {"viewer": {"background": "javascript:bad"}}, {"console": {"search": "x" * 1001}},
    {"layout": {"layout": {"type": "row", "children": []}}}, {"unknown": True},
    {"layout": {"layout": {"type": []}}},
    {"layout": {"layout": {"type": "row", "children": [{"type": "tabset", "children": [
        {"type": "tab", "component": []},
    ]}]}}},
])
async def test_workspace_validates_input(ui_client, patch):
    client, _ = ui_client
    assert (await client.patch("/api/preferences", json=patch)).status_code == 422


async def test_project_patch_preserves_params_metadata_and_other_panels(ui_client):
    client, database = ui_client
    project = await create_project(database, "UI test")
    await database.conn.execute("UPDATE project SET metadata_json=? WHERE id=?", (
        json.dumps({"calibration": {"checked": True}, "ui": {"params": {"minSharpness": 240}}}), project.id,
    ))
    await asyncio.gather(
        patch_ui_state(database, project.id, {"params": {"fps": 1.5}}),
        patch_ui_state(database, project.id, {"selectedStage": "cleanup_sparse"}),
    )
    result = await client.patch(f"/api/projects/{project.id}/ui-state", json={"ui": {
        "cameraPose": {"position": [1, 2, 3], "quaternion": [0, 0, 0, 1]},
    }})
    assert result.status_code == 200
    ui = result.json()["ui_state"]
    assert ui["params"] == {"fps": 1.5, "minSharpness": 240}
    assert ui["selectedStage"] == "cleanup_sparse"
    assert ui["cameraPose"]["position"] == [1, 2, 3]
    saved = await get_project(database, project.id)
    assert saved.metadata["calibration"] == {"checked": True}
    assert (await client.patch(f"/api/projects/{project.id}/ui-state", json={"ui": {
        "cameraPose": {"position": [1, 2, 3], "quaternion": [0, 0, 0, 0]},
    }})).status_code == 422
    assert (await client.patch("/api/projects/missing/ui-state", json={"ui": {}})).status_code == 404
