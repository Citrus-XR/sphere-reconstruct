"""ProgressReporter の kind ルーティング (info/warn/error=log, tick=progress) を検証する."""

from __future__ import annotations

import pytest

from sphere_reconstruct.pipeline.stage import ProgressReporter, ProgressSpan


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


def test_progress_span_maps_nested_local_progress() -> None:
    reporter, calls = _reporter()
    span = ProgressSpan(reporter, 0.2, 0.8).child(0.25, 0.75)

    span.tick(0.5, message="half")

    assert calls[0][1] == pytest.approx(0.5)


@pytest.mark.parametrize(("low", "high"), [(-0.1, 0.5), (0.8, 0.2), (0.2, 1.1)])
def test_progress_span_rejects_invalid_bounds(low: float, high: float) -> None:
    reporter, _calls = _reporter()
    with pytest.raises(ValueError, match="invalid progress span"):
        ProgressSpan(reporter, low, high)


def test_progress_reporter_rejects_numeric_regression() -> None:
    reporter, _calls = _reporter()
    reporter.tick(0.5)
    with pytest.raises(ValueError, match="moved backwards"):
        reporter.info("invalid phase", progress=0.4)


def test_progress_reporter_throttles_small_tick_updates_but_keeps_terminal() -> None:
    calls: list[tuple] = []
    reporter = ProgressReporter(
        _emit=lambda *args: calls.append(args),
        tick_min_interval=0,
        tick_min_progress=0.1,
    )
    reporter.tick(0.01)
    reporter.tick(0.05)
    reporter.tick(0.12)
    reporter.tick(1.0)

    assert [call[1] for call in calls] == [0.01, 0.12, 1.0]
