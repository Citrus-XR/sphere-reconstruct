"""SfM / mask / export が読む rectified image artifact の path contract。"""

from __future__ import annotations

import json
from pathlib import Path

DIRECTORY = "rectify_fisheye"


def catalog_path(project_dir: Path) -> Path:
    return project_dir / DIRECTORY / "image_catalog.json"


def rig_config_path(project_dir: Path) -> Path:
    return project_dir / DIRECTORY / "rig_config.json"


def load_catalog(project_dir: Path) -> dict:
    path = catalog_path(project_dir)
    if not path.is_file():
        raise RuntimeError("rectify_fisheye を先に実行してください")
    return json.loads(path.read_text(encoding="utf-8"))
