"""SAM3 設定の検証.

- repo_path: `sam3` Python パッケージが置いてあるディレクトリ (`sam3/` サブフォルダを含む).
- checkpoint_path: `.pt` ファイル.
- config_path: モデル設定 YAML. 現状の SAM3 リリースでは build_sam3_image_model が
  自動で解決するため必須ではない. 空でも可.

Health check は Worker プロセス側 (engine.py 相当) から呼ぶ. FastAPI プロセスで
呼ぶと Torch import が走ってしまうため, 明確に分離する.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..settings import get_settings


@dataclass
class Sam3Paths:
    repo_path: Path
    checkpoint_path: Path
    config_path: Path | None
    device: str
    dtype: str


@dataclass
class Sam3PathCheck:
    ok: bool
    repo_exists: bool
    checkpoint_exists: bool
    sam3_package_present: bool  # repo_path/sam3/__init__.py があるか
    checkpoint_size: int
    message: str


def resolve() -> Sam3Paths | None:
    """設定から Sam3Paths を返す. 未設定なら None."""
    s = get_settings().sam3
    if not s.repo_path or not s.checkpoint_path:
        return None
    return Sam3Paths(
        repo_path=Path(s.repo_path).expanduser().resolve(),
        checkpoint_path=Path(s.checkpoint_path).expanduser().resolve(),
        config_path=Path(s.config_path).expanduser().resolve() if s.config_path else None,
        device=s.device,
        dtype=s.dtype,
    )


def quick_check() -> Sam3PathCheck:
    """FastAPI プロセスから安全に呼べる path のみのチェック. Torch を import しない."""
    paths = resolve()
    if paths is None:
        return Sam3PathCheck(
            ok=False,
            repo_exists=False,
            checkpoint_exists=False,
            sam3_package_present=False,
            checkpoint_size=0,
            message="sam3.repo_path / checkpoint_path が未設定",
        )

    repo_exists = paths.repo_path.is_dir()
    checkpoint_exists = paths.checkpoint_path.is_file()
    pkg_present = (paths.repo_path / "sam3" / "__init__.py").is_file()
    size = paths.checkpoint_path.stat().st_size if checkpoint_exists else 0

    if not repo_exists:
        msg = f"repo_path が存在しない: {paths.repo_path}"
    elif not pkg_present:
        msg = f"sam3 パッケージが repo_path/sam3/__init__.py に見つからない"
    elif not checkpoint_exists:
        msg = f"checkpoint が存在しない: {paths.checkpoint_path}"
    elif size < 1024 * 1024:
        msg = f"checkpoint が小さすぎる ({size} bytes)"
    else:
        msg = "ok"

    return Sam3PathCheck(
        ok=(repo_exists and pkg_present and checkpoint_exists and size >= 1024 * 1024),
        repo_exists=repo_exists,
        checkpoint_exists=checkpoint_exists,
        sam3_package_present=pkg_present,
        checkpoint_size=size,
        message=msg,
    )
