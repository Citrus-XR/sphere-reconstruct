"""SfM / mask / export が読む rectified image artifact の path contract。"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import LocalizedError

DIRECTORY = "rectify_fisheye"


def catalog_path(project_dir: Path) -> Path:
    return project_dir / DIRECTORY / "image_catalog.json"


def rig_config_path(project_dir: Path) -> Path:
    return project_dir / DIRECTORY / "rig_config.json"


def load_catalog(project_dir: Path) -> dict:
    path = catalog_path(project_dir)
    if not path.is_file():
        raise LocalizedError(
            "error.rectify_required",
            "Internal fisheye normalization must run before this step",
        )
    catalog = json.loads(path.read_text(encoding="utf-8"))
    _migrate_camera_prior_provenance(catalog)
    return catalog


def _migrate_camera_prior_provenance(catalog: dict) -> None:
    # Legacy refinable cameras did not distinguish EXIF focal lengths from guesses.
    # Only their fixed calibrated/virtual counterparts retain a trusted prior.
    # Migrate the loaded document without rewriting a hashed upstream artifact.
    for group in catalog.get("camera_groups", []):
        if "has_prior_focal_length" not in group:
            group["has_prior_focal_length"] = not group["refine_intrinsics"]
