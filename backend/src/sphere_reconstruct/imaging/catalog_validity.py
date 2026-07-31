"""Image catalog の analytic / bitmap validity を同じ API で扱う。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import valid_region


def cache_key(record: dict, width: int, height: int) -> str:
    mask_path = record.get("valid_mask_path")
    if mask_path is not None:
        return f"{width}x{height}:bitmap:{mask_path}"
    return valid_region.cache_key(record.get("valid_region", {"kind": "full"}), width, height)


def render_mask(project_dir: Path, record: dict, width: int, height: int) -> np.ndarray:
    mask_path = record.get("valid_mask_path")
    if mask_path is None:
        return valid_region.render_mask(
            record.get("valid_region", {"kind": "full"}),
            width,
            height,
        )
    path = project_dir / mask_path
    with Image.open(path) as image:
        grayscale = image.convert("L")
        if grayscale.size != (width, height):
            raise RuntimeError(
                f"validity mask size が catalog と一致しません: "
                f"{grayscale.width}x{grayscale.height} != {width}x{height}"
            )
        return (np.asarray(grayscale) > 127).astype(np.uint8)


def bounding_box(
    project_dir: Path,
    record: dict,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    mask = render_mask(project_dir, record, width, height)
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    if not len(rows) or not len(columns):
        raise ValueError(f"valid region に pixel がありません: {record['name']}")
    return (
        float(columns[0]),
        float(rows[0]),
        float(columns[-1] + 1),
        float(rows[-1] + 1),
    )
