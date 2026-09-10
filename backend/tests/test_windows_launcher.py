"""Windows の実 CMD / PowerShell で launcher の境界を検証する。"""

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher integration")


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "checkout with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[2]
    for name in ("start-windows.ps1", "start-windows.cmd"):
        shutil.copyfile(original / name, repo / name)
    backend = repo / "backend"
    venv.EnvBuilder(with_pip=False).create(backend / ".venv")
    package = backend / "sphere_reconstruct"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "settings.py").write_text(
        "import json\n"
        "class Settings:\n"
        "    def model_dump_json(self):\n"
        '        return json.dumps({"server": {"port": 8787}, "filesystem": {"allowed_roots": []}, '
        '"binaries": {"colmap": "configured", "jpegtran": "configured", "vocab_tree": "configured"}})\n'
        "def get_settings(): return Settings()\n",
        encoding="utf-8",
    )
    (package / "cli.py").write_text(
        'import sys\nprint("doctor stdout")\nprint("doctor stderr", file=sys.stderr)\n',
        encoding="utf-8",
    )
    (scripts / "server_service.py").write_text(
        "import json, os, sys\n"
        'state = os.environ.get("LAUNCH_TEST_STATE", "stopped")\n'
        'if sys.argv[1] == "status":\n'
        '    print(json.dumps({"state": state, "pid": 123, "port": 8787, "url": "http://127.0.0.1:8787"}))\n'
        '    sys.exit({"healthy": 0, "stopped": 1, "unhealthy": 2}[state])\n'
        'print("service start")\n',
        encoding="utf-8",
    )
    frontend = repo / "frontend/dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text("test", encoding="utf-8")
    commands = repo / "commands"
    commands.mkdir()
    (commands / "pnpm.cmd").write_text(
        "@echo off\necho pnpm %*\nif defined LAUNCH_TEST_FAILURE exit /b 23\nexit /b 0\n",
        encoding="ascii",
    )
    (commands / "uv.cmd").write_text("@echo off\necho uv %*\nexit /b 0\n", encoding="ascii")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPHERE_")}
    env.update(
        PATH=f"{commands}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=str(backend),
        SPHERE_LAUNCHER_NO_PAUSE="1",
    )

    def run(*args, **overrides):
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(repo / "start-windows.cmd"), "-NoBrowser", *args],
            cwd=tmp_path,
            env={**env, **overrides},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        logs = list((repo / "runtime/logs").glob("launcher-*.log"))
        transcript = "\n".join(path.read_text(encoding="utf-8-sig") for path in logs)
        return result, transcript

    return run


def test_success_captures_native_streams_and_preserves_extras(launcher):
    result, log = launcher()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "uv sync --locked --inexact --extra imaging --extra aliked" in log
    assert "doctor stdout" in log and "doctor stderr" in log
    assert "service start" in log
    assert "http://127.0.0.1:8787" in log


def test_native_failure_retains_exit_code_and_stops_later_steps(launcher):
    result, log = launcher(LAUNCH_TEST_FAILURE="1")
    assert result.returncode == 23, result.stdout + result.stderr
    assert "Frontend dependencies" in log and "exit code 23" in log
    assert "service start" not in log
    assert "startup failed" in result.stdout


def test_repeated_start_uses_existing_service_without_setup(launcher):
    result, log = launcher(LAUNCH_TEST_STATE="healthy", LAUNCH_TEST_FAILURE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pid=123" in log
    assert "pnpm" not in log and "service start" not in log


def test_unhealthy_server_is_reported_without_restarting_it(launcher):
    result, log = launcher(LAUNCH_TEST_STATE="unhealthy")
    assert result.returncode != 0
    assert "Existing server status" in log
    assert "service start" not in log


def test_skip_setup_still_runs_diagnosis(launcher):
    result, log = launcher("-SkipSetup", LAUNCH_TEST_FAILURE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "doctor stdout" in log and "service start" in log
    assert "Frontend dependencies" not in log
