"""previews API.

パイプライン成果物 (frames / masks / reconstruction / points) を配信する.
全て workspace 内のファイルを FileResponse で返す. パスは project_dir 配下に
限定し, traversal を防ぐ.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..domain import project as project_domain
from ..infrastructure.database import get_db
from ..infrastructure.filesystem import PathNotAllowedError, ensure_within

router = APIRouter(tags=["previews"])


async def _project_dir(project_id: str) -> Path:
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    return p.workspace_dir


def _safe(project_dir: Path, rel: str) -> Path:
    try:
        return ensure_within(project_dir, project_dir / rel)
    except PathNotAllowedError:
        raise HTTPException(status_code=400, detail="invalid path")


@router.get("/api/projects/{project_id}/frames")
async def list_frames(project_id: str) -> dict:
    project_dir = await _project_dir(project_id)
    mf = project_dir / "extract_frames" / "manifest_frames.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="extract_frames not run yet")
    data = json.loads(mf.read_text())
    return {
        "kind": data.get("kind"),
        "count": data.get("count", len(data.get("frames", []))),
        "width": data.get("width"),
        "height": data.get("height"),
        "fps": data.get("fps"),
        "frames": [
            {"index": f["index"], "timestamp_sec": f.get("timestamp_sec")}
            for f in data.get("frames", [])
        ],
    }


@router.get("/api/projects/{project_id}/frames/{index}/image")
async def frame_image(project_id: str, index: int, lens: int = 0) -> FileResponse:
    """抽出フレーム (fisheye) を返す. INSV は lens0/lens1."""
    project_dir = await _project_dir(project_id)
    mf = project_dir / "extract_frames" / "manifest_frames.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="extract_frames not run yet")
    data = json.loads(mf.read_text())
    frame = next((f for f in data["frames"] if f["index"] == index), None)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame not found")
    key = f"lens{lens}" if f"lens{lens}" in frame else "erp"
    if key not in frame:
        raise HTTPException(status_code=404, detail=f"no {key} for this frame")
    path = _safe(project_dir, frame[key])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/api/projects/{project_id}/pinhole/{index}/{view}")
async def pinhole_view(project_id: str, index: int, view: str, lens: int = 0, mask: bool = False) -> FileResponse:
    """reproject_views の pinhole 画像, または generate_masks の mask を返す."""
    project_dir = await _project_dir(project_id)
    if mask:
        rel = f"generate_masks/frame_{index:06d}/{view}_lens{lens}.png"
        media = "image/png"
    else:
        rel = f"reproject_views/frame_{index:06d}/{view}_lens{lens}.jpg"
        media = "image/jpeg"
    path = _safe(project_dir, rel)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{'mask' if mask else 'pinhole'} not found")
    return FileResponse(path, media_type=media)


@router.get("/api/projects/{project_id}/reconstruction")
async def reconstruction(project_id: str) -> FileResponse:
    project_dir = await _project_dir(project_id)
    path = project_dir / "export_dataset" / "preview" / "reconstruction.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="reconstruction not exported yet")
    return FileResponse(path, media_type="application/json")


@router.get("/api/projects/{project_id}/reconstruction/points")
async def reconstruction_points(project_id: str) -> FileResponse:
    project_dir = await _project_dir(project_id)
    path = project_dir / "export_dataset" / "preview" / "points.bin"
    if not path.exists():
        raise HTTPException(status_code=404, detail="points not exported yet")
    return FileResponse(path, media_type="application/octet-stream")
