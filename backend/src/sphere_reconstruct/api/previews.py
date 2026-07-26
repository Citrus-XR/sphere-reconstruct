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
from pydantic import BaseModel

from ..domain import project as project_domain
from ..imaging import fisheye_region
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
        raise HTTPException(status_code=400, detail="invalid path") from None


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
        "selection": data.get("selection"),
        "frames": [
            {"index": f["index"], "timestamp_sec": f.get("timestamp_sec"), "score": f.get("score")}
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
async def pinhole_view(
    project_id: str, index: int, view: str, lens: int = 0, mask: bool = False
) -> FileResponse:
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


@router.get("/api/projects/{project_id}/masks")
async def list_masks(project_id: str) -> dict:
    """generate_masks の manifest (frame ごとの view + coverage) を返す."""
    project_dir = await _project_dir(project_id)
    mf = project_dir / "generate_masks" / "manifest_masks.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="generate_masks not run yet")
    return json.loads(mf.read_text())


@router.get("/api/projects/{project_id}/export-info")
async def export_info(project_id: str) -> dict:
    """export_dataset の出力ディレクトリ (絶対パス) を返す. Inspector で場所を表示する用."""
    project_dir = await _project_dir(project_id)
    export = project_dir / "export_dataset"
    if not export.exists():
        raise HTTPException(status_code=404, detail="export_dataset not run yet")
    train_configs = export / "train_configs"
    recommendations = train_configs / "recommendations.json"
    gui_integration = None
    if recommendations.is_file():
        recommendation_data = json.loads(recommendations.read_text(encoding="utf-8"))
        gui_integration = recommendation_data.get("gui_integration")
    return {
        "dir": str(export),
        "gui_integration": gui_integration,
    }


class FisheyeRegion(BaseModel):
    lens0: dict
    lens1: dict


@router.get("/api/projects/{project_id}/fisheye-region")
async def get_fisheye_region(project_id: str) -> dict:
    """魚眼有効領域 (円) の保存値を返す. 未保存なら既定 (中心, r=0.485).

    saved: UI から明示的に保存済みか (fisheye_region.json が存在するか). 魚眼モードでは
    有効領域の設定を必須にするため, フロントはこのフラグでゲートする.
    """
    project_dir = await _project_dir(project_id)
    saved = fisheye_region.region_path(project_dir).exists()
    region = fisheye_region.load_region(project_dir) if saved else fisheye_region.detect_region(project_dir)
    return {**region, "saved": saved, "detected": not saved}


@router.put("/api/projects/{project_id}/fisheye-region")
async def put_fisheye_region(project_id: str, region: FisheyeRegion) -> dict:
    """魚眼有効領域を保存する. 検証して正規化した内容を返す."""
    project_dir = await _project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    return fisheye_region.save_region(project_dir, region.model_dump())


@router.get("/api/projects/{project_id}/source-info")
async def source_info(project_id: str) -> dict:
    """ソースを ffprobe して尺/fps/解像度を返す (frame 抽出の予測数計算用).

    予測フレーム数 = duration_sec * fps_target (原版プラグインと同じ式) をフロントで計算する.
    """
    from fastapi.concurrency import run_in_threadpool

    from ..imaging import ffprobe
    from ..settings import get_settings

    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    if not p.source_path:
        raise HTTPException(status_code=400, detail="source not set")
    src = Path(p.source_path)
    if not src.exists() or src.is_dir():
        # 画像フォルダ等は尺が無い.
        return {"kind": p.source_kind.value if p.source_kind else None, "duration_sec": None}

    probe = await run_in_threadpool(ffprobe.probe, src, ffprobe_bin=get_settings().binaries.ffprobe or None)
    vs = probe.video_streams[0] if probe.video_streams else None
    return {
        "kind": p.source_kind.value if p.source_kind else None,
        "duration_sec": probe.duration,
        "fps": vs.fps if vs else None,
        "width": vs.width if vs else None,
        "height": vs.height if vs else None,
        "nb_frames": vs.nb_frames if vs else None,
    }


@router.get("/api/projects/{project_id}/fisheye-mask/{index}")
async def fisheye_mask(project_id: str, index: int, lens: int = 0) -> FileResponse:
    """native fisheye の生成マスク (generate_masks/lensN/frame_XXXXXX.png) を返す."""
    project_dir = await _project_dir(project_id)
    rel = f"generate_masks/lens{lens}/frame_{index:06d}.png"
    path = _safe(project_dir, rel)
    if not path.exists():
        raise HTTPException(status_code=404, detail="mask not found")
    return FileResponse(path, media_type="image/png")


@router.get("/api/projects/{project_id}/reconstruction")
async def reconstruction(project_id: str) -> FileResponse:
    project_dir = await _project_dir(project_id)
    path = project_dir / "align_reconstruction" / "preview" / "reconstruction.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="reconstruction not aligned yet")
    return FileResponse(path, media_type="application/json")


@router.get("/api/projects/{project_id}/reconstruction/points")
async def reconstruction_points(project_id: str) -> FileResponse:
    project_dir = await _project_dir(project_id)
    path = project_dir / "align_reconstruction" / "preview" / "points.bin"
    if not path.exists():
        raise HTTPException(status_code=404, detail="reconstruction not aligned yet")
    return FileResponse(path, media_type="application/octet-stream")
