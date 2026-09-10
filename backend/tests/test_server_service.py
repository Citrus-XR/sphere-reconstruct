"""起動失敗、port 衝突、background process の状態を検証する。"""

import importlib.util
import json
import socket
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def service(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/server_service.py"
    spec = importlib.util.spec_from_file_location("server_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RUNTIME", tmp_path)
    monkeypatch.setattr(module, "PID_FILE", tmp_path / "backend.pid.json")
    monkeypatch.setattr(module, "STDOUT_LOG", tmp_path / "backend.stdout.log")
    monkeypatch.setattr(module, "STDERR_LOG", tmp_path / "backend.stderr.log")
    return module


def test_occupied_port_does_not_spawn_or_check_another_servers_health(service, monkeypatch):
    spawn = Mock()
    health = Mock(return_value=True)
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(service, "_health", health)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="port"):
            service.start(listener.getsockname()[1], 1)
    spawn.assert_not_called()
    health.assert_not_called()


def test_repeated_start_reuses_healthy_process_and_rejects_port_change(service, monkeypatch):
    monkeypatch.setattr(service, "_read_record", lambda: {"port": 8787})
    monkeypatch.setattr(service, "_managed_process", lambda _: Mock(pid=123))
    monkeypatch.setattr(service, "_health", lambda _: True)
    spawn = Mock()
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    assert service.start(8787, 1) == 0
    with pytest.raises(RuntimeError, match="8787"):
        service.start(8788, 1)
    spawn.assert_not_called()


@pytest.mark.parametrize("exit_code", [7, None])
def test_start_failure_reports_stderr_and_reaps_timed_out_process(service, monkeypatch, exit_code):
    process = Mock(pid=12345)
    process.poll.return_value = exit_code
    monkeypatch.setattr(service, "_python_executable", lambda: Path(sys.executable))
    monkeypatch.setattr(service, "_health", lambda _: False)
    monkeypatch.setattr(service, "_popen_options", lambda: {})
    monkeypatch.setattr(service.time, "monotonic", Mock(side_effect=[0, 0.1, 2]))
    monkeypatch.setattr(service.time, "sleep", lambda _: None)
    stop = Mock()
    monkeypatch.setattr(service, "stop", stop)

    def spawn(*args, **kwargs):
        kwargs["stderr"].write(b"ModuleNotFoundError: missing_dependency\n")
        return process

    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with pytest.raises(RuntimeError, match="missing_dependency"):
        service.start(port, 1)
    if exit_code is None:
        stop.assert_called_once_with(1)
    else:
        stop.assert_not_called()
        assert not service.PID_FILE.exists()


def test_status_json_distinguishes_stopped_and_unhealthy(service, monkeypatch, capsys):
    assert service.status(as_json=True) == 1
    assert json.loads(capsys.readouterr().out) == {"state": "stopped"}
    monkeypatch.setattr(service, "_read_record", lambda: {"port": 8787})
    monkeypatch.setattr(service, "_managed_process", lambda _: Mock(pid=123))
    monkeypatch.setattr(service, "_health", lambda _: False)
    assert service.status(as_json=True) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "unhealthy"
