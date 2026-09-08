"""SPA 配信の document root と API 境界を検証する。"""

import httpx
import pytest
from fastapi import FastAPI

from sphere_reconstruct import main


@pytest.fixture
async def static_client(tmp_path, monkeypatch):
    distribution = tmp_path / "frontend" / "dist"
    (distribution / "assets").mkdir(parents=True)
    (distribution / "index.html").write_text("<html>application</html>")
    (distribution / "public.txt").write_text("public file")
    (tmp_path / "frontend" / "private.txt").write_text("private file")
    monkeypatch.setattr(main, "__file__", str(tmp_path / "backend" / "src" / "sphere_reconstruct" / "main.py"))
    monkeypatch.setattr(main, "app", FastAPI())
    main._mount_frontend()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        yield client, distribution


@pytest.mark.parametrize("path", ["/..%2Fprivate.txt", "/%2E%2E%2Fprivate.txt", "/api/unknown", "/api"])
async def test_spa_rejects_traversal_and_unknown_api_paths(static_client, path):
    client, _ = static_client
    response = await client.get(path)
    assert response.status_code == 404
    assert "private file" not in response.text


async def test_spa_rejects_symlinks_outside_distribution(static_client):
    client, distribution = static_client
    try:
        (distribution / "escaped.txt").symlink_to(distribution.parent / "private.txt")
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    response = await client.get("/escaped.txt")
    assert response.status_code == 404


async def test_spa_serves_public_files_and_client_routes(static_client):
    client, _ = static_client
    public = await client.get("/public.txt")
    route = await client.get("/projects/project-id")
    assert public.status_code == 200
    assert public.text == "public file"
    assert route.status_code == 200
    assert route.text == "<html>application</html>"
