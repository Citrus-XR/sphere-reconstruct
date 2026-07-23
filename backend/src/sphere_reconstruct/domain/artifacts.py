"""Artifact / Manifest ドメイン.

各ステージ実行結果はディレクトリ + manifest.json で表す. manifest は:
- inputs: ステージへの入力ファイル (path, size, sha256)
- params: ステージパラメータの正規化 dict
- outputs: 生成物ファイル一覧 (path, size, sha256, mime)
- impl_version: ステージ実装バージョン
- started_at / finished_at
- inputs_hash / params_hash: 冪等判定に使う派生ハッシュ

manifest_path は project/<pid>/manifests/<stage>.json.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FileRef(BaseModel):
    path: str  # workspace 相対パスで持つ (portable にするため)
    size: int
    sha256: str
    mime: str | None = None


class StageManifest(BaseModel):
    stage: str
    impl_version: str
    started_at: datetime
    finished_at: datetime | None = None
    inputs: list[FileRef] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: list[FileRef] = Field(default_factory=list)
    inputs_hash: str = ""
    params_hash: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    def compute_hashes(self) -> None:
        self.inputs_hash = _stable_hash(
            [{"path": f.path, "sha256": f.sha256} for f in self.inputs]
        )
        self.params_hash = _stable_hash(self.params)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StageManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    """dict / list を含む任意の JSON 化可能値のハッシュ. key ソートで正規化する."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_path(project_dir: Path, stage: str) -> Path:
    return project_dir / "manifests" / f"{stage}.json"
