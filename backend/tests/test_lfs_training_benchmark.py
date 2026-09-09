"""長時間 training の成功判定と telemetry 障害の分離を検証する。"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def setup_run(tmp_path, monkeypatch, *, exit_code=0, complete=True):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("lfs_benchmark", scripts / "benchmark_lfs_training.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    dataset.mkdir()
    output.mkdir()
    (dataset / "export_manifest.json").write_text("{}")
    args = SimpleNamespace(executable=Path(sys.executable), dataset=dataset, output=output,
                           iterations=30000, max_cap=1000000, timeout_seconds=0, lfs_args=[])
    if complete:
        (output / "training").mkdir()
        for name in ["project.licht", "splat_30000.ply", "perf_bench.json"]:
            (output / "training" / name).write_text("completed output")
    process = Mock(pid=123)
    process.poll.side_effect = [None, exit_code, exit_code]
    process.wait.return_value = exit_code
    monkeypatch.setattr(script.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(script.subprocess, "run", Mock(side_effect=script.subprocess.TimeoutExpired("nvidia-smi", 30)))
    monkeypatch.setattr(script.time, "sleep", lambda duration: None)
    return script, args, process


def test_telemetry_failure_does_not_cancel_successful_training(tmp_path, monkeypatch):
    script, args, process = setup_run(tmp_path, monkeypatch)
    script.execute(args)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    assert "TimeoutExpired" in (args.output / "gpu.jsonl").read_text()
    assert json.loads((args.output / "status.json").read_text())["status"] == "succeeded"


def test_nonzero_exit_preserves_failure_even_if_output_files_exist(tmp_path, monkeypatch):
    script, args, _ = setup_run(tmp_path, monkeypatch, exit_code=7)
    with pytest.raises(RuntimeError, match="code 7"):
        script.execute(args)
    state = json.loads((args.output / "status.json").read_text())
    assert state["status"] == "failed" and state["exit_code"] == 7


def test_zero_exit_without_training_outputs_is_not_success(tmp_path, monkeypatch):
    script, args, _ = setup_run(tmp_path, monkeypatch, complete=False)
    with pytest.raises(RuntimeError, match="required training outputs"):
        script.execute(args)
