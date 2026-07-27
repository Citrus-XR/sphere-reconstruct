"""Stale marker と現在の stage run status の優先順位。"""

from sphere_reconstruct.api.stages import _resolve_stage_status


def test_running_stage_is_not_hidden_by_previous_stale_marker():
    assert _resolve_stage_status(has_output=False, stale=True, run_status="running") == "running"


def test_missing_invalidated_output_is_stale_after_previous_success():
    assert _resolve_stage_status(has_output=False, stale=True, run_status="succeeded") == "stale"
