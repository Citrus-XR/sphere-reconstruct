"""Worker cancellation が子 subprocess まで停止することを検証する。"""

from __future__ import annotations

import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil

from sphere_reconstruct.infrastructure.processes import cancel, spawn_worker


def _worker_with_child(pid_file: str) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(pid_file).write_text(str(child.pid), encoding="utf-8")
    child.wait()


def test_cancel_stops_worker_descendants(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    handle = spawn_worker(
        _worker_with_child,
        job_id="test",
        args=(str(pid_file),),
    )
    deadline = time.monotonic() + 10
    while not pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.is_file()
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    try:
        cancel(handle, grace_seconds=1.0)
        assert not handle.is_alive
        child = psutil.Process(child_pid) if psutil.pid_exists(child_pid) else None
        if child is not None:
            _gone, alive = psutil.wait_procs([child], timeout=5.0)
            assert not alive
    finally:
        if handle.is_alive:
            handle.kill()
            handle.process.join(timeout=1)
        if psutil.pid_exists(child_pid):
            with suppress(psutil.Error):
                psutil.Process(child_pid).kill()
