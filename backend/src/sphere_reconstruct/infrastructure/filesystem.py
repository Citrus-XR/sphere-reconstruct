"""ファイルシステムユーティリティ.

- 許可された root 外のパスへのアクセスを禁じる (path traversal 防止).
- 原子リプレース (書き途中のディレクトリが正式に見えないようにする).
- 単純なハッシュ.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


class PathNotAllowedError(Exception):
    """許可された root の外を触ろうとした."""


def ensure_within(root: Path, target: Path) -> Path:
    """target が root のサブパスであることを確認して, 絶対化した Path を返す.

    シンボリックリンクや `..` を含んでいても最終的な実体で判定する.
    """
    root_resolved = root.expanduser().resolve()
    target_resolved = target.expanduser().resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as e:
        raise PathNotAllowedError(f"{target} is outside {root}") from e
    return target_resolved


def ensure_within_any(allowed_roots: list[Path], target: Path) -> Path:
    """allowed_roots のいずれかの下にあることを確認する.

    サーバサイド file browser が呼ぶ想定. 空リストなら常に禁止.
    """
    if not allowed_roots:
        raise PathNotAllowedError("No allowed roots are configured.")
    target_resolved = target.expanduser().resolve()
    for root in allowed_roots:
        try:
            root_resolved = root.expanduser().resolve()
            target_resolved.relative_to(root_resolved)
            return target_resolved
        except ValueError:
            continue
    raise PathNotAllowedError(f"{target} is not under any allowed root")


def atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    """tmp_dir を final_dir に原子的に置き換える.

    final_dir 既存なら一度 `.old-<pid>` にリネームしてから tmp を rename, その後 old を削除.
    途中でクラッシュしても中途半端な final_dir は残らない.
    """
    tmp_dir = tmp_dir.resolve()
    final_dir = final_dir.resolve()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        old = final_dir.with_name(final_dir.name + f".old-{os.getpid()}")
        # old も残っていたら念のため消しておく.
        if old.exists():
            shutil.rmtree(old)
        os.rename(final_dir, old)
        try:
            os.rename(tmp_dir, final_dir)
        except BaseException:
            # rename 失敗したら元に戻す.
            os.rename(old, final_dir)
            raise
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.rename(tmp_dir, final_dir)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """ファイルの SHA-256 を hex で返す. 大きなファイル向けにストリーミング."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
