"""previews API.

パイプライン成果物 (frames / masks / reconstruction / points) を配信する.
全て workspace 内のファイルを FileResponse で返す. パスは project_dir 配下に
限定し, traversal を防ぐ.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..domain import project as project_domain
from ..domain.mask_artifact import (
    MaskPurpose,
    load_mask_manifest,
    load_partial_mask_manifest,
    mask_manifest_path,
    stage_for,
)
from ..domain.pipeline_state import StageName
from ..imaging import source_region
from ..infrastructure.database import get_db
from ..infrastructure.filesystem import PathNotAllowedError, ensure_within
from ..pipeline import prepared_images
from ..pipeline.invalidation import ArtifactBusyError, derive_pipeline_state, invalidate_from
from .project_access import MutableProject

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
        "count": data.get("count", len(data.get("frames", []))),
        "sources": [
            {
                key: source.get(key)
                for key in (
                    "id",
                    "label",
                    "role",
                    "projection",
                    "kind",
                    "count",
                    "width",
                    "height",
                    "fps",
                    "selection",
                )
            }
            for source in data.get("sources", [])
        ],
        "frames": [
            {
                "index": frame["index"],
                "source_id": frame["source_id"],
                "source_index": frame["source_index"],
                "timestamp_sec": frame.get("timestamp_sec"),
                "score": frame.get("score"),
            }
            for frame in data.get("frames", [])
        ],
    }


@router.get(
    "/api/projects/{project_id}/frames/{index}/image", response_class=FileResponse, response_model=None
)
async def frame_image(project_id: str, index: int, lens: int = 0):
    """抽出フレーム (fisheye) を返す. INSV は lens0/lens1."""
    project_dir = await _project_dir(project_id)
    mf = project_dir / "extract_frames" / "manifest_frames.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail="extract_frames not run yet")
    data = json.loads(mf.read_text())
    frame = next((f for f in data["frames"] if f["index"] == index), None)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame not found")
    key = f"lens{lens}" if f"lens{lens}" in frame else "image"
    if key not in frame and "image_source" in frame:
        path = Path(frame["image_source"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="source image file missing")
        return FileResponse(path, media_type="image/jpeg")
    if key not in frame:
        raise HTTPException(status_code=404, detail=f"no {key} for this frame")
    path = _safe(project_dir, frame[key])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg")


@router.get("/api/projects/{project_id}/prepared-image", response_class=FileResponse, response_model=None)
async def prepared_image(project_id: str, name: str):
    project_dir = await _project_dir(project_id)
    record = _catalog_image(project_dir, name)
    path = _safe(project_dir, record["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="prepared image not found")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg")


@router.get("/api/projects/{project_id}/prepared-mask", response_class=FileResponse, response_model=None)
async def prepared_mask(project_id: str, name: str, purpose: MaskPurpose):
    project_dir = await _project_dir(project_id)
    document, artifact_root, partial = await _mask_artifact(project_id, project_dir, purpose)
    record = next(
        (item for item in document["images"] if item["name"] == name),
        None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="mask not found")
    if partial:
        try:
            relative = Path(record["path"]).relative_to(stage_for(purpose).value)
        except ValueError:
            raise HTTPException(status_code=500, detail="invalid partial mask path") from None
        path = _safe(project_dir, str(artifact_root.relative_to(project_dir) / relative))
    else:
        path = _safe(project_dir, record["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="mask file missing")
    return FileResponse(path, media_type="image/png")


@router.get("/api/projects/{project_id}/masks/{purpose}")
async def list_masks(project_id: str, purpose: MaskPurpose) -> dict:
    """指定用途の mask manifest を返す。"""
    project_dir = await _project_dir(project_id)
    document, _artifact_root, _partial = await _mask_artifact(project_id, project_dir, purpose)
    return document


async def _mask_artifact(
    project_id: str,
    project_dir: Path,
    purpose: MaskPurpose,
) -> tuple[dict, Path, bool]:
    stage = stage_for(purpose)
    db = get_db()
    active = await (
        await db.conn.execute(
            "SELECT 1 FROM stage_run WHERE project_id=? AND stage=? AND status='running' "
            "ORDER BY started_at DESC LIMIT 1",
            (project_id, stage.value),
        )
    ).fetchone()
    if active is not None:
        temporary_root = project_dir / f".{stage.value}.tmp"
        partial = load_partial_mask_manifest(temporary_root, purpose)
        if partial is None:
            raise HTTPException(status_code=404, detail=f"{purpose.value} mask preview not ready")
        return partial, temporary_root, True

    if not mask_manifest_path(project_dir, purpose).is_file():
        raise HTTPException(status_code=404, detail=f"{purpose.value} masks not run yet")
    return load_mask_manifest(project_dir, purpose), project_dir, False


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


async def _region_source(project_id: str, source_id: str):
    project = await project_domain.get_project(get_db(), project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found in project")
    return project, source


@router.get("/api/projects/{project_id}/source-region")
async def get_source_region(project_id: str, source_id: str) -> dict:
    project, source = await _region_source(project_id, source_id)
    region = source_region.load_region(project.workspace_dir, source_id, source.projection)
    return {**region, **source_region.region_status(project.workspace_dir, source_id)}


@router.put("/api/projects/{project_id}/source-region")
async def put_source_region(
    project_id: str, region: source_region.SourceRegion, source_id: str, project: MutableProject
) -> dict:
    """Source の projection に合う領域を保存し、変更時は画像準備以降を無効化する。"""
    db = get_db()
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found in project")
    try:
        validated = source_region.validate_region(region.model_dump(), source.projection)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    project_dir = project.workspace_dir
    project_dir.mkdir(parents=True, exist_ok=True)
    previous = source_region.load_region(project_dir, source_id, source.projection)
    invalidated = []
    if validated != previous:
        try:
            invalidated = await run_in_threadpool(
                invalidate_from,
                project_dir,
                StageName.PREPARE_IMAGES,
                include_self=True,
            )
        except ArtifactBusyError as error:
            raise HTTPException(status_code=423, detail=str(error)) from error
        state = derive_pipeline_state(project_dir)
        await db.conn.execute(
            "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
            (state.value, project_id),
        )
        await db.conn.commit()
    saved = source_region.save_region(project_dir, source_id, source.projection, validated)
    return {
        **saved,
        "saved": True,
        "needs_review": False,
        "invalidated": [stage.value for stage in invalidated],
    }


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
    if not p.sources:
        raise HTTPException(status_code=400, detail="source not set")
    source_records = []
    for source in p.sources:
        if source.media_kind.value == "video":
            probe = await run_in_threadpool(
                ffprobe.probe,
                source.filesystem_path,
                ffprobe_bin=get_settings().binaries.ffprobe or None,
            )
            stream = probe.video_streams[0] if probe.video_streams else None
            source_records.append(
                {
                    "id": source.id,
                    "duration_sec": probe.duration,
                    "fps": stream.fps if stream else None,
                    "width": stream.width if stream else None,
                    "height": stream.height if stream else None,
                    "nb_frames": stream.nb_frames if stream else None,
                }
            )
        else:
            width, height, image_count = await run_in_threadpool(
                _image_source_dimensions,
                source.filesystem_path,
            )
            source_records.append(
                {
                    "id": source.id,
                    "duration_sec": None,
                    "fps": None,
                    "width": width,
                    "height": height,
                    "nb_frames": image_count,
                }
            )
    primary = p.primary_source
    primary_info = next((record for record in source_records if primary and record["id"] == primary.id), {})
    return {
        **primary_info,
        "sources": source_records,
        "duration_sec_total": sum(record.get("duration_sec") or 0.0 for record in source_records),
    }


def _image_source_dimensions(path: Path) -> tuple[int, int, int]:
    from PIL import Image  # noqa: PLC0415

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
    if not images:
        raise RuntimeError(f"image source に対応画像がありません: {path}")
    with Image.open(images[0]) as image:
        width, height = image.size
    return width, height, len(images)


@router.get("/api/projects/{project_id}/reconstruction", response_class=FileResponse, response_model=None)
async def reconstruction(project_id: str):
    project_dir = await _project_dir(project_id)
    path = _latest_transform_preview(project_dir, "reconstruction.json")
    return FileResponse(path, media_type="application/json")


@router.get(
    "/api/projects/{project_id}/reconstruction/points",
    response_class=FileResponse,
    response_model=None,
)
async def reconstruction_points(project_id: str):
    project_dir = await _project_dir(project_id)
    path = _latest_transform_preview(project_dir, "points.bin")
    return FileResponse(path, media_type="application/octet-stream")


def _latest_transform_preview(project_dir: Path, filename: str) -> Path:
    for stage in (
        "dense_initialization",
        "cleanup_sparse",
        "scene_alignment",
        "restore_metric_scale",
        "align_reconstruction",
        "reconstruct",
    ):
        path = project_dir / stage / "preview" / filename
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail="reconstruction preview not available")


def _catalog_image(project_dir: Path, name: str) -> dict:
    catalog = prepared_images.catalog_path(project_dir)
    if not catalog.is_file():
        raise HTTPException(status_code=404, detail="rectify_fisheye not run yet")
    record = next(
        (item for item in json.loads(catalog.read_text(encoding="utf-8"))["images"] if item["name"] == name),
        None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="prepared image not found")
    return record
