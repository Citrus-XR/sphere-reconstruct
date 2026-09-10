#!/usr/bin/env python3
"""開発 checkout の backend を platform 非依存で start / stop / status 管理する。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import psutil

REPOSITORY = Path(__file__).resolve().parents[1]
BACKEND = REPOSITORY / "backend"
RUNTIME = Path(os.environ.get("SPHERE_SERVICE_RUNTIME", REPOSITORY / "runtime")).resolve()
PID_FILE = RUNTIME / "backend.pid.json"
STDOUT_LOG = RUNTIME / "logs" / "backend.stdout.log"
STDERR_LOG = RUNTIME / "logs" / "backend.stderr.log"
APP_TARGET = "sphere_reconstruct.main:app"


def _python_executable() -> Path:
    candidate = BACKEND / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not candidate.is_file():
        raise RuntimeError(f"backend virtual environment がありません: {candidate}")
    return candidate


def _read_record() -> dict | None:
    if not PID_FILE.is_file():
        return None
    try:
        value = json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"PID file を解釈できません: {PID_FILE}") from error
    if not isinstance(value, dict) or not isinstance(value.get("pid"), int):
        raise RuntimeError(f"PID file の形式が不正です: {PID_FILE}")
    return value


def _managed_process(record: dict | None) -> psutil.Process | None:
    if record is None:
        return None
    try:
        process = psutil.Process(record["pid"])
        command = process.cmdline()
    except (psutil.Error, OSError):
        return None
    if APP_TARGET not in command or command != record["command"]:
        return None
    return process


def _health(port: int, timeout: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _popen_options() -> dict:
    if os.name == "nt":
        return {
            "creationflags": (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
                | 0x01000000  # CREATE_BREAKAWAY_FROM_JOB: SSH / CI の親 Job から明示的に分離する。
            )
        }
    return {"start_new_session": True}


def start(port: int, wait_seconds: float) -> int:
    record = _read_record()
    process = _managed_process(record)
    if process is not None:
        if int(record["port"]) != port:
            raise RuntimeError(
                f"既存 backend は port={record['port']} で実行中です。変更前に stop してください"
            )
        if _health(int(record.get("port", port))):
            print(f"backend は実行中です: pid={process.pid}")
            return 0
        raise RuntimeError(f"管理対象 pid={process.pid} は存在しますが health check に応答しません")
    # 他 process の health 応答を成功扱いせず、重複 server の DB 初期化も防ぐ。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"port {port} を使用できません。既存 process または SPHERE_PORT を確認してください: {error}"
            ) from error
    if PID_FILE.exists():
        PID_FILE.unlink()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(_python_executable()),
        "-u",
        "-m",
        "uvicorn",
        APP_TARGET,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    stderr_offset = STDERR_LOG.stat().st_size if STDERR_LOG.exists() else 0
    with STDOUT_LOG.open("ab", buffering=0) as stdout, STDERR_LOG.open("ab", buffering=0) as stderr:
        process_handle = subprocess.Popen(
            command,
            cwd=BACKEND,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            **_popen_options(),
        )
    record = {
        "pid": process_handle.pid,
        "port": port,
        "started_at": datetime.now(UTC).isoformat(),
        "command": command,
    }
    temporary = PID_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(PID_FILE)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if process_handle.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            raise RuntimeError(
                f"backend が起動直後に終了しました (exit={process_handle.returncode}): {STDERR_LOG}\n"
                f"{_stderr_tail(stderr_offset)}"
            )
        if _health(port):
            print(f"backend を起動しました: pid={process_handle.pid} http://127.0.0.1:{port}")
            return 0
        time.sleep(0.1)
    stop(wait_seconds)
    raise RuntimeError(
        f"backend health check が {wait_seconds:.1f}s 以内に成功しませんでした。起動 process は停止しました: {STDERR_LOG}\n"
        f"{_stderr_tail(stderr_offset)}"
    )


def _stderr_tail(start: int = 0) -> str:
    with STDERR_LOG.open("rb") as stream:
        stream.seek(max(start, stream.seek(0, 2) - 4096))
        return stream.read().decode("utf-8", errors="replace")


def stop(wait_seconds: float) -> int:
    record = _read_record()
    process = _managed_process(record)
    if process is None:
        PID_FILE.unlink(missing_ok=True)
        print("backend は停止済みです")
        return 0

    descendants = process.children(recursive=True)
    for target in reversed(descendants):
        with contextlib.suppress(psutil.Error):
            target.terminate()
    with contextlib.suppress(psutil.Error):
        process.terminate()
    _, alive = psutil.wait_procs([*descendants, process], timeout=wait_seconds)
    for target in alive:
        with contextlib.suppress(psutil.Error):
            target.kill()
    psutil.wait_procs(alive, timeout=1.0)
    PID_FILE.unlink(missing_ok=True)
    print(f"backend を停止しました: pid={process.pid}")
    return 0


def status(*, as_json: bool = False) -> int:
    record = _read_record()
    process = _managed_process(record)
    if process is None:
        print(json.dumps({"state": "stopped"}) if as_json else "stopped")
        return 1
    port = int(record.get("port", 8787))
    state = "healthy" if _health(port) else "unhealthy"
    result = {"state": state, "pid": process.pid, "port": port, "url": f"http://127.0.0.1:{port}"}
    print(json.dumps(result) if as_json else f"{state} pid={process.pid} {result['url']}")
    return 0 if state == "healthy" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SPHERE_PORT", "8787")))
    parser.add_argument("--wait", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", help="status を JSON で返す")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.wait <= 0:
        parser.error("port は 1–65535、wait は正数で指定してください")
    if args.action == "start":
        return start(args.port, args.wait)
    if args.action == "stop":
        return stop(args.wait)
    if args.action == "restart":
        stop(args.wait)
        return start(args.port, args.wait)
    return status(as_json=args.json)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
