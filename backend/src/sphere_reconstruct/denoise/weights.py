"""FastDVDnet checkpoint の検証付き自動取得."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from ..settings import workspace_root

MODEL_ID = "fastdvdnet-clipped-noise-c8fdf61"
MODEL_SHA256 = "8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949"
MODEL_URL = (
    "https://raw.githubusercontent.com/m-tassano/fastdvdnet/"
    "c8fdf6182a0340e89dd18f5df25b47337cbede6f/model_clipped_noise.pth"
)


def default_model_path() -> Path:
    return workspace_root() / ".models" / "fastdvdnet" / "model_clipped_noise.pth"


def ensure_model(
    configured_path: str = "",
    *,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    path = Path(configured_path).expanduser().resolve() if configured_path else default_model_path()
    if path.is_file() and _sha256(path) == MODEL_SHA256:
        return path
    if configured_path:
        if not path.is_file():
            raise FileNotFoundError(f"FastDVDnet model が見つかりません: {path}")
        raise RuntimeError(f"FastDVDnet model の SHA-256 が一致しません: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "sphere-reconstruct/0.0.1"})
    try:
        for attempt in range(3):
            try:
                with (  # noqa: S310
                    urllib.request.urlopen(request, timeout=60) as response,
                    temporary.open("wb") as output,
                ):
                    total_header = response.headers.get("Content-Length")
                    total = int(total_header) if total_header else None
                    written = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        written += len(chunk)
                        if on_progress is not None:
                            on_progress(written, total)
                break
            except (OSError, urllib.error.URLError) as error:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError("FastDVDnet model の取得に 3 回失敗しました") from error
                time.sleep(attempt + 1)
        if _sha256(temporary) != MODEL_SHA256:
            raise RuntimeError("取得した FastDVDnet model の SHA-256 検証に失敗しました")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
