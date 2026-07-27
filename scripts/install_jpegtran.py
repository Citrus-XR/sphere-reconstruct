"""Windows 用の固定 libjpeg-turbo jpegtran runtime installer。"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

VERSION = "3.2.0"
URL = (
    "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/"
    "libjpeg-turbo-3.2.0-vc-x64.exe"
)
SHA256 = "662761d8ba8dae04aec74023ebaeceb856c2b56b9b59cfd180759d26300dda42"


def main() -> None:
    if platform.system() != "Windows":
        raise SystemExit("自動 jpegtran installer は Windows のみです。Linux/macOS は PATH から検出します。")

    repository = Path(__file__).resolve().parents[1]
    tools_dir = repository / ".runtime" / "tools"
    destination = tools_dir / f"libjpeg-turbo-{VERSION}-windows"
    record = repository / ".runtime" / "jpegtran-path.txt"
    executable = destination / "bin" / "jpegtran.exe"
    if executable.is_file():
        _write_record(record, executable)
        print(f"jpegtran は導入済みです: {executable}")
        return

    tools_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tools_dir, prefix="jpegtran-install-") as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "libjpeg-turbo-installer.exe"
        print(f"libjpeg-turbo {VERSION} jpegtran を取得中...")
        request = urllib.request.Request(URL, headers={"User-Agent": "sphere-reconstruct-installer"})
        with urllib.request.urlopen(request) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        digest = _sha256(archive)
        if digest != SHA256:
            raise RuntimeError(f"jpegtran SHA-256 mismatch: {digest}")
        staged = tools_dir / f".{destination.name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        subprocess.run(
            [str(archive), "/S", f"/D={staged}"],
            check=True,
        )
        if not (staged / "bin" / "jpegtran.exe").is_file() or not (
            staged / "bin" / "jpeg62.dll"
        ).is_file():
            raise RuntimeError("libjpeg-turbo installer の runtime が不完全です")
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)

    metadata = {"version": VERSION, "sha256": SHA256, "source": URL}
    (destination / "sphere-install.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_record(record, executable)
    print(f"jpegtran を導入しました: {executable}")


def _write_record(record: Path, executable: Path) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(str(executable), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
