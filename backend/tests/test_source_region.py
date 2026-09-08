"""Source 領域の独立性、旧データ移行、projection validation。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sphere_reconstruct.api import previews
from sphere_reconstruct.imaging import source_region as sr


def test_defaults_depend_on_projection(tmp_path):
    fisheye = sr.load_region(tmp_path, "fisheye", "dual_fisheye")
    assert set(fisheye["views"]) == {"lens0", "lens1"}
    assert all(view["r"] == 0.5 for view in fisheye["views"].values())
    for projection in ("perspective", "equirectangular"):
        assert sr.load_region(tmp_path, "source", projection) == {
            "views": {"main": {"kind": "full", "operations": []}}
        }


@pytest.mark.parametrize("radius", [0.01, 0.499, 0.5])
def test_circle_accepts_radius_up_to_half_image_width(radius):
    assert sr.CircleRegion(r=radius).r == radius


@pytest.mark.parametrize("radius", [0.500000001, 0.6, 0.75])
def test_circle_rejects_radius_above_half_image_width(radius):
    with pytest.raises(ValueError, match="less than or equal to 0.5"):
        sr.CircleRegion(r=radius)


def test_migration_caps_only_oversized_circle_radii(tmp_path):
    fish = sr.default_region("dual_fisheye")
    fish["views"]["lens0"]["r"] = 0.7
    fish["views"]["lens1"]["r"] = 0.45
    operation = {"mode": "subtract", "x": 0.4, "y": 0.8, "r": 0.1, "stroke_id": 3}
    fish["views"]["lens0"]["operations"] = [operation]
    phone = sr.default_region("perspective")
    phone["views"]["main"]["operations"] = [operation]
    document = {"version": 1, "sources": {
        "fish": {**fish, "needs_review": True}, "phone": {**phone, "needs_review": False},
    }}
    path = sr.region_path(tmp_path)
    path.write_text(json.dumps(document), encoding="utf-8")
    assert sr.migrate_source_regions(tmp_path) == 1
    document["sources"]["fish"]["views"]["lens0"]["r"] = 0.5
    assert json.loads(path.read_text(encoding="utf-8")) == document
    timestamp = path.stat().st_mtime_ns
    assert sr.migrate_source_regions(tmp_path) == 0
    assert path.stat().st_mtime_ns == timestamp


def test_multiple_sources_and_sensors_are_saved_independently(tmp_path):
    for source, radius in (("rig-a", 0.4), ("rig-b", 0.45)):
        region = sr.default_region("dual_fisheye")
        region["views"]["lens0"]["r"] = radius
        sr.save_region(tmp_path, source, "dual_fisheye", region)
    phone = sr.default_region("perspective")
    phone["views"]["main"]["operations"] = [
        {"mode": "subtract", "x": x, "y": 0.8, "r": 0.1, "stroke_id": 1} for x in (0.2, 0.3)
    ]
    sr.save_region(tmp_path, "phone", "perspective", phone)
    assert sr.load_region(tmp_path, "phone", "perspective") == phone
    assert sr.load_region(tmp_path, "rig-a", "dual_fisheye")["views"]["lens0"]["r"] == 0.4
    assert sr.load_region(tmp_path, "rig-b", "dual_fisheye")["views"]["lens0"]["r"] == 0.45
    assert sr.load_region(tmp_path, "rig-b", "dual_fisheye")["views"]["lens1"]["r"] == 0.5


def test_migration_preserves_radius_operations_and_stroke_groups(tmp_path):
    legacy = tmp_path / "fisheye_regions.json"
    operations = [
        {"mode": "subtract", "x": 0.3, "y": 0.4, "r": 0.05},
        {"mode": "add", "x": 0.3, "y": 0.4, "r": 0.02, "stroke_id": 7},
        {"mode": "add", "x": 0.4, "y": 0.4, "r": 0.02, "stroke_id": 7},
    ]
    legacy.write_text(json.dumps({"version": 3, "sources": {
        "old": {"lens0": {"r": 0.5005304353, "operations": operations}, "_coordinate_version": 3},
        "review": {"lens0": {"r": 0.4}},
    }}))
    assert sr.migrate_source_regions(tmp_path) == 1
    region = sr.load_region(tmp_path, "old", "dual_fisheye")
    assert region["views"]["lens0"]["r"] == 0.5
    assert sr.load_region(tmp_path, "review", "dual_fisheye")["views"]["lens0"]["r"] == 0.4
    assert [op["stroke_id"] for op in region["views"]["lens0"]["operations"]] == [8, 7, 7]
    assert not sr.region_status(tmp_path, "old")["needs_review"]
    assert sr.region_status(tmp_path, "review")["needs_review"]
    assert not legacy.exists()
    assert sr.migrate_source_regions(tmp_path) == 0
    assert sr.load_region(tmp_path, "old", "dual_fisheye") == region


@pytest.mark.parametrize("projection,region", [
    ("perspective", sr.default_region("dual_fisheye")),
    ("dual_fisheye", sr.default_region("perspective")),
    ("perspective", {"views": {"main": {"kind": "circle"}}}),
])
def test_projection_region_mismatch_is_rejected(projection, region):
    with pytest.raises(ValueError, match="requires"):
        sr.validate_region(region, projection)


@pytest.mark.parametrize("stroke_id", [None, 0, -1, True, 1.5, "1"])
def test_invalid_stroke_ids_are_rejected(stroke_id):
    with pytest.raises(ValueError):
        sr.RegionOperation(mode="subtract", x=0.5, y=0.5, r=0.1, stroke_id=stroke_id)


async def test_api_validates_source_ownership_and_perspective_shape(tmp_path, monkeypatch):
    from sphere_reconstruct.api import project_access

    project = SimpleNamespace(workspace_dir=tmp_path, sources=[
        SimpleNamespace(id="phone", projection="perspective"),
        SimpleNamespace(id="rig", projection="dual_fisheye"),
    ])
    db = SimpleNamespace(conn=SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(
        fetchone=AsyncMock(return_value={"id": "project", "active_job": None}),
    )), commit=AsyncMock()))
    monkeypatch.setattr(previews, "get_db", lambda: db)
    monkeypatch.setattr(project_access, "get_db", lambda: db)
    monkeypatch.setattr(previews.project_domain, "get_project", AsyncMock(return_value=project))
    invalidate = []
    monkeypatch.setattr(previews, "invalidate_from", lambda *args, **kwargs: invalidate.append(args[1]) or [])
    app = FastAPI()
    app.include_router(previews.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = "/api/projects/project/source-region"
        assert (await client.get(url, params={"source_id": "unrelated"})).status_code == 404
        response = await client.get(url, params={"source_id": "phone"})
        assert response.json()["views"]["main"]["kind"] == "full"
        wrong = await client.put(url, params={"source_id": "phone"}, json=sr.default_region("dual_fisheye"))
        assert wrong.status_code == 422
        region = sr.default_region("perspective")
        region["views"]["main"]["operations"].append({
            "mode": "subtract", "x": 0.5, "y": 0.5, "r": 0.1, "stroke_id": 1,
        })
        response = await client.put(url, params={"source_id": "phone"}, json=region)
        assert response.status_code == 200
        assert response.json()["views"] == region["views"]
        assert [stage.value for stage in invalidate] == ["prepare_images"]
        assert (await client.get(url, params={"source_id": "rig"})).json()["views"]["lens0"]["r"] == 0.5
        before = sr.region_path(tmp_path).read_bytes()
        oversized = sr.default_region("dual_fisheye")
        oversized["views"]["lens0"]["r"] = 0.500000001
        response = await client.put(url, params={"source_id": "rig"}, json=oversized)
        assert response.status_code == 422
        assert sr.region_path(tmp_path).read_bytes() == before
        assert len(invalidate) == 1
