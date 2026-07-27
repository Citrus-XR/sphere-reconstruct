"""Windows 用 COLMAP 4.1.1 runtime installer.

独立 GLOMAP は廃止済みなので, ``global_mapper`` を含む公式 COLMAP package を固定 version・
固定 SHA-256 で取得する. 展開途中の directory は完成後にだけ置換する.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "4.1.1"
PACKAGES = {
    "cuda": {
        "url": "https://github.com/colmap/colmap/releases/download/4.1.1/colmap-x64-windows-cuda.zip",
        "sha256": "b06064e7e4bd34f5b4ef71b442d3537d95d57c666dbec5a3b475902ccd832b9b",
    },
    "cpu": {
        "url": "https://github.com/colmap/colmap/releases/download/4.1.1/colmap-x64-windows-nocuda.zip",
        "sha256": "faf1247d2ec90933aa8bd003709790abf0211cdc132cceec4c831718f2e0895a",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=PACKAGES, default="cuda")
    args = parser.parse_args()
    if platform.system() != "Windows":
        raise SystemExit("公式自動 installer は Windows のみです. Linux/macOS は system COLMAP を使用してください.")

    repository = Path(__file__).resolve().parents[1]
    tools_dir = repository / ".runtime" / "tools"
    destination = tools_dir / f"colmap-{VERSION}-{args.variant}"
    record = repository / ".runtime" / "colmap-path.txt"
    existing = next(destination.rglob("colmap.exe"), None) if destination.exists() else None
    if existing:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(str(existing), encoding="utf-8")
        print(f"COLMAP は導入済みです: {existing}")
        return

    package = PACKAGES[args.variant]
    tools_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tools_dir, prefix="colmap-install-") as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "colmap.zip"
        print(f"COLMAP {VERSION} ({args.variant}) を取得中...")
        _download(package["url"], archive)
        digest = _sha256(archive)
        if digest != package["sha256"]:
            raise RuntimeError(f"COLMAP SHA-256 mismatch: {digest}")
        extracted = temporary / "extracted"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        executable = next(extracted.rglob("colmap.exe"), None)
        if executable is None:
            raise RuntimeError("COLMAP archive に colmap.exe がありません")
        payload_root = executable.parent.parent if executable.parent.name.lower() == "bin" else executable.parent
        staged = tools_dir / f".{destination.name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(payload_root, staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)

    executable = next(destination.rglob("colmap.exe"))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(str(executable), encoding="utf-8")
    metadata = {"version": VERSION, "variant": args.variant, "sha256": package["sha256"]}
    (destination / "sphere-install.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"COLMAP を導入しました: {executable}")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "sphere-reconstruct-installer"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
