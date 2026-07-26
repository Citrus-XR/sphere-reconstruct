"""Worker サブプロセス管理.

FastAPI プロセスは CUDA も Torch も pycolmap も触らない. これらは全部 Worker で動かす.
理由:
- CUDA OOM やドライバクラッシュを API から隔離する.
- キャンセルは Worker プロセスを SIGTERM/kill するだけで済む (VRAM 解放も OS が担保).
- Torch import に数秒かかるため, HTTP レイテンシへの影響を切り離す.

現状は multiprocessing.spawn を使う. ステージ実行は
`worker_entry.run_pipeline(project_id, job_id, stage=None)` に集約.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil

# フォーク由来のライブラリ (CUDA など) の問題を避けるため, 常に spawn.
_CTX: mp.context.BaseContext = mp.get_context("spawn")


@dataclass
class WorkerHandle:
    process: mp.Process
    job_id: str

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def is_alive(self) -> bool:
        return self.process.is_alive()

    def terminate(self) -> None:
        if self.process.is_alive():
            self.process.terminate()

    def kill(self) -> None:
        if self.process.is_alive():
            # Windows でも Python 3.9+ で kill() が使える.
            self.process.kill()


def spawn_worker(
    target: Callable[..., Any],
    *,
    job_id: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> WorkerHandle:
    """Worker プロセスを起こす. join は呼び出し側で await 監視する."""
    kwargs = kwargs or {}
    proc = _CTX.Process(target=target, args=args, kwargs=kwargs, daemon=False)
    proc.start()
    return WorkerHandle(process=proc, job_id=job_id)


async def wait_for(handle: WorkerHandle, poll_interval: float = 0.5) -> int:
    """プロセス終了を await で待つ. exit code を返す.

    asyncio 側は event loop を塞がないよう poll する.
    """
    while handle.is_alive:
        await asyncio.sleep(poll_interval)
    # exit code は join 後にしか confirm できない.
    handle.process.join(timeout=0.1)
    return int(handle.process.exitcode or 0)


def cancel(handle: WorkerHandle, grace_seconds: float = 5.0) -> None:
    """SIGTERM -> grace -> SIGKILL のフォールバック. 子孫プロセスも巻き込んで殺す.

    SAM3/Torch/CUDA は worker の下にさらに子プロセス (dataloader, CUDA サービス等) を
    起こすことがあり, 親だけ terminate すると子が残って GPU/VRAM を掴んだまま「止まらない」.
    psutil でプロセスツリー全体へ signal を送る.
    """
    _signal_tree(handle.pid, kill=False)
    handle.terminate()
    # 同期的に短時間だけ待つ. 呼び出し元は API リクエスト側なので長くしない.
    handle.process.join(timeout=grace_seconds)
    if handle.process.is_alive():
        _signal_tree(handle.pid, kill=True)
        handle.kill()
        handle.process.join(timeout=1.0)


def _signal_tree(pid: int | None, *, kill: bool) -> None:
    """psutil で pid の子孫プロセスへ terminate/kill を送る (親は呼び出し側が処理する)."""
    if pid is None:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    for child in parent.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            child.kill() if kill else child.terminate()


def is_windows() -> bool:
    return os.name == "nt"


def send_soft_signal(pid: int) -> None:
    """POSIX: SIGINT を送って KeyboardInterrupt に変換.

    Worker 側で `except KeyboardInterrupt` で cleanup できるようにする.
    Windows では CTRL_BREAK 相当がプロセスグループ必須で扱いにくいため, no-op.
    """
    if is_windows():
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGINT)
