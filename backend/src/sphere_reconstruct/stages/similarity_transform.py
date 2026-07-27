"""Sparse model の similarity transform と Web preview 生成を共有する。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..colmap import gravity_align, web_preview
from ..colmap import model as colmap_model
from ..colmap import runner as colmap_runner
from ..domain.artifacts import FileRef
from ..infrastructure.filesystem import sha256_file
from ..pipeline.stage import StageContext
from ..settings import get_settings
from .colmap_progress import hidden_log


def materialize_similarity(
    ctx: StageContext,
    input_model: Path,
    *,
    scale: float = 1.0,
    rotation: np.ndarray | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    log_label: str,
) -> tuple[Path, colmap_model.Reconstruction]:
    rotation = np.eye(3) if rotation is None else rotation
    output_model = ctx.stage_out_dir / "sparse" / "0"
    identity = (
        abs(scale - 1.0) <= 1e-12
        and np.allclose(rotation, np.eye(3), atol=1e-12)
        and all(abs(value) <= 1e-12 for value in translation)
    )
    if identity:
        shutil.copytree(input_model, output_model)
    else:
        quaternion = gravity_align._R_to_quat(rotation)
        transform_path = ctx.stage_out_dir / "transform.txt"
        transform_path.write_text(
            f"{scale:.17g} "
            + " ".join(f"{value:.17g}" for value in quaternion)
            + " "
            + " ".join(f"{value:.17g}" for value in translation)
            + "\n",
            encoding="utf-8",
        )
        colmap_runner.model_transformer(
            colmap_runner.resolve_colmap_bin(get_settings().binaries.colmap or None),
            input_path=input_model,
            output_path=output_model,
            transform_path=transform_path,
            log_path=ctx.stage_out_dir / "model_transformer.log",
            on_line=hidden_log(ctx, log_label),
        )
    return output_model, colmap_model.read_model(output_model)


def write_preview(
    ctx: StageContext,
    reconstruction: colmap_model.Reconstruction,
    *,
    metadata_key: str,
    metadata: dict,
    max_points: int,
    additional_metadata: dict | None = None,
) -> web_preview.WebPreview:
    gravity_align.apply_alignment(reconstruction, gravity_align.DATASET_TO_VIEWER)
    preview_dir = ctx.stage_out_dir / "preview"
    preview = web_preview.write_web_preview(reconstruction, preview_dir, max_points=max_points)
    preview_json_path = preview_dir / "reconstruction.json"
    preview_json = json.loads(preview_json_path.read_text(encoding="utf-8"))
    preview_json[metadata_key] = metadata
    if additional_metadata:
        preview_json.update(additional_metadata)
    preview_json["coordinate_system"] = {
        "dataset_up": "-Y",
        "viewer_up": "+Y",
        "dataset_to_viewer": [1, 0, 0, 0, -1, 0, 0, 0, -1],
    }
    preview_json_path.write_text(
        json.dumps(preview_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return preview


def input_refs(ctx: StageContext, paths) -> list[FileRef]:
    return [
        FileRef(
            path=str(path.relative_to(ctx.project_dir)),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
        if path.is_file()
    ]


def output_refs(ctx: StageContext, model_dir: Path, result_path: Path) -> list[FileRef]:
    paths = [
        result_path,
        ctx.stage_out_dir / "preview" / "reconstruction.json",
        ctx.stage_out_dir / "preview" / "points.bin",
        *(path for path in model_dir.iterdir() if path.is_file()),
    ]
    final_stage = Path(ctx.stage_out_dir.name.lstrip(".").removesuffix(".tmp"))
    return [
        FileRef(
            path=str(final_stage / path.relative_to(ctx.stage_out_dir)),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
    ]
