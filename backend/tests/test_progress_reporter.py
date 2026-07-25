"""ProgressReporter の kind ルーティング (info/warn/error=log, tick=progress) を検証する."""

from __future__ import annotations

from sphere_reconstruct.pipeline.stage import ProgressReporter


def _reporter() -> tuple[ProgressReporter, list[tuple]]:
    calls: list[tuple] = []
    # emit シグネチャ: (level, progress, message, key, args, kind)
    return ProgressReporter(_emit=lambda *a: calls.append(a)), calls


def test_info_warn_error_are_log_kind() -> None:
    r, calls = _reporter()
    r.info("start", progress=0.1, key="log.x")
    r.warn("careful")
    r.error("boom", progress=0.5)
    kinds = [c[5] for c in calls]
    assert kinds == ["log", "log", "log"]
    assert calls[0][0] == "info" and calls[1][0] == "warn" and calls[2][0] == "error"


def test_tick_is_progress_kind() -> None:
    r, calls = _reporter()
    r.tick(progress=0.42, message="mid", key="log.p", args={"cur": 3})
    assert len(calls) == 1
    level, prog, msg, key, args, kind = calls[0]
    assert kind == "progress"
    assert level == "info"
    assert prog == 0.42
    assert args == {"cur": 3}
