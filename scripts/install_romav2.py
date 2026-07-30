#!/usr/bin/env python3
"""RoMaV2 v2.0.1 weights を Torch cache へ固定 hash で導入する。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
SIZE = 1_095_883_548


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size == SIZE and sha256(destination) == SHA256:
        print(f"RoMaV2 model ready: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "sphere-reconstruct/0.0.1"})
        with urllib.request.urlopen(request) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if temporary_path.stat().st_size != SIZE:
            raise RuntimeError(
                f"RoMaV2 model size mismatch: {temporary_path.stat().st_size} != {SIZE}"
            )
        actual = sha256(temporary_path)
        if actual != SHA256:
            raise RuntimeError(f"RoMaV2 model SHA-256 mismatch: {actual}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"RoMaV2 model installed: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".runtime"
        / "torch"
        / "hub"
        / "checkpoints"
        / "romav2.0.1.pt",
    )
    args = parser.parse_args()
    install(args.destination.resolve())


if __name__ == "__main__":
    main()
