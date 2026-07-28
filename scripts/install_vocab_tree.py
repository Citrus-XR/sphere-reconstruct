"""COLMAP loop detection 用の固定 vocabulary tree installer。"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

VERSION = "colmap-3.11.1-faiss-flickr100K-words256K"
URL = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words256K.bin"
)
SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    models_dir = repository / ".runtime" / "models"
    destination = models_dir / "vocab_tree_faiss_flickr100K_words256K.bin"
    record = repository / ".runtime" / "vocab-tree-path.txt"
    if destination.is_file() and _sha256(destination) == SHA256:
        _write_record(record, destination)
        print(f"Vocabulary tree は導入済みです: {destination}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir, prefix="vocab-tree-install-") as temporary_text:
        temporary = Path(temporary_text)
        download = temporary / destination.name
        print(f"COLMAP vocabulary tree {VERSION} を取得中...")
        request = urllib.request.Request(URL, headers={"User-Agent": "sphere-reconstruct-installer"})
        with urllib.request.urlopen(request) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        digest = _sha256(download)
        if digest != SHA256:
            raise RuntimeError(f"Vocabulary tree SHA-256 mismatch: {digest}")
        download.replace(destination)

    metadata = {"version": VERSION, "sha256": SHA256, "source": URL}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_record(record, destination)
    print(f"Vocabulary tree を導入しました: {destination}")


def _write_record(record: Path, model: Path) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(str(model), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
