"""Source file fingerprint を stage 間で共通化する。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..domain.artifacts import FileRef
from ..infrastructure.filesystem import sha256_file
from .stage import SourceContext

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def collect_source_inputs(
    sources: tuple[SourceContext, ...],
    progress: Callable[[Path, int, int], None] | None = None,
) -> list[FileRef]:
    references: list[FileRef] = []
    for source in sources:
        path = source.path
        if not path.exists():
            raise FileNotFoundError(f"source not found: {path}")
        if path.is_file():
            callback = (
                (lambda current, total, source_path=path: progress(source_path, current, total))
                if progress is not None
                else None
            )
            references.append(
                FileRef(
                    path=str(path),
                    size=path.stat().st_size,
                    sha256=sha256_file(path, progress=callback),
                )
            )
            continue
        files = sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"source directory contains no supported images: {path}")
        for item in files:
            callback = (
                (lambda current, total, source_path=item: progress(source_path, current, total))
                if progress is not None
                else None
            )
            references.append(
                FileRef(
                    path=str(item),
                    size=item.stat().st_size,
                    sha256=sha256_file(item, progress=callback),
                )
            )
    return references
