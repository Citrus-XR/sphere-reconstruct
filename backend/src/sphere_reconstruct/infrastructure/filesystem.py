"""ファイルシステムユーティリティ.

- 許可された root 外のパスへのアクセスを禁じる (path traversal 防止).
- 原子リプレース (書き途中のディレクトリが正式に見えないようにする).
- 単純なハッシュ.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
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
    if os.name == "nt":
        _atomic_replace_windows(tmp_dir, final_dir)
        return
    if final_dir.exists():
        old = final_dir.with_name(final_dir.name + f".old-{os.getpid()}")
        # old も残っていたら念のため消しておく.
        if old.exists():
            shutil.rmtree(old)
        _rename(final_dir, old)
        try:
            _rename(tmp_dir, final_dir)
        except BaseException:
            # rename 失敗したら元に戻す.
            _rename(old, final_dir)
            raise
        shutil.rmtree(old, ignore_errors=True)
    else:
        _rename(tmp_dir, final_dir)


def _atomic_replace_windows(tmp_dir: Path, final_dir: Path) -> None:
    """Native runtime が tmp handle を保持していても atomic publish できる Windows 経路."""
    ready = final_dir.with_name(final_dir.name + f".ready-{os.getpid()}")
    old = final_dir.with_name(final_dir.name + f".old-{os.getpid()}")
    for stale in (ready, old):
        if stale.exists():
            shutil.rmtree(stale)
    shutil.copytree(tmp_dir, ready, copy_function=_hardlink_or_copy)
    if final_dir.exists():
        _rename(final_dir, old)
    try:
        _rename(ready, final_dir)
    except BaseException:
        if old.exists():
            _rename(old, final_dir)
        raise
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _hardlink_or_copy(source: str, destination: str):
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _rename(source: Path, destination: Path) -> None:
    attempts = 30 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.rename(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            # Windows Defender / ONNX Runtime が process 終了直後だけ directory handle を
            # 保持することがある. 対象と例外を限定し, 最大 6 秒で元の例外を再送出する.
            time.sleep(0.2)


def sha256_file(
    path: Path,
    chunk_size: int = 1 << 20,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """ファイルの SHA-256 を hex で返す. 大きなファイル向けにストリーミング."""
    h = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    last_percent = -1
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            completed += len(chunk)
            percent = min(100, int(completed * 100 / max(1, total)))
            if progress is not None and percent > last_percent:
                last_percent = percent
                progress(completed, total)
    if progress is not None and last_percent < 100:
        progress(total, total)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
