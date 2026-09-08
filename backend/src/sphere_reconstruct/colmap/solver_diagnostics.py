"""Ceres の失敗した step と CLI 全体の終了を区別して記録する。"""

from __future__ import annotations

import re
from pathlib import Path

_REGISTRATION = re.compile(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)")


def read_solver_diagnostics(logs_dir: Path) -> dict:
    runs = []
    terminated = []
    for path in sorted(logs_dir.glob("*.log")):
        context: dict = {"phase": "initialization"}
        failures = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                registration = _REGISTRATION.search(line)
                if registration:
                    context = {"phase": "local", "image_id": int(registration[1]),
                               "registered_frames": int(registration[2])}
                elif "Global bundle adjustment" in line:
                    context = {"phase": "global"}
                if "Linear solver failure" in line:
                    if failures and failures[-1]["context"] == context:
                        failures[-1]["failed_steps"] += 1
                    else:
                        failures.append({"context": dict(context), "failed_steps": 1})
                if "Bundle adjustment failed:" in line:
                    terminated.append({"log_file": path.name, "context": dict(context), "message": line.strip()})
        if failures:
            runs.append({"log_file": path.name, "failure_contexts": failures})
    return {
        "linear_solver_failed_steps": sum(item["failed_steps"] for run in runs for item in run["failure_contexts"]),
        "runs_with_failed_steps": runs,
        "bundle_adjustment_failures": len(terminated),
        "terminated_bundle_adjustments": terminated,
    }
