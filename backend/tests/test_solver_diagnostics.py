from sphere_reconstruct.colmap.solver_diagnostics import read_solver_diagnostics


def test_rejected_steps_are_separate_from_failed_bundle_adjustments(tmp_path):
    (tmp_path / "mapper.log").write_text(
        "Registering image #271 (num_reg_frames=196)\n"
        "Linear solver failure. Failed to compute a finite step.\n"
        "Linear solver failure. Failed to compute a finite step.\n"
        "Registering image #269 (num_reg_frames=197)\n"
        "Retriangulation and Global bundle adjustment\n"
        "Bundle adjustment failed: Too many consecutive invalid steps\n",
        encoding="utf-8",
    )
    report = read_solver_diagnostics(tmp_path)
    assert report["linear_solver_failed_steps"] == 2
    assert report["bundle_adjustment_failures"] == 1
    assert report["runs_with_failed_steps"][0]["failure_contexts"] == [{
        "context": {"phase": "local", "image_id": 271, "registered_frames": 196}, "failed_steps": 2,
    }]
    assert report["terminated_bundle_adjustments"][0]["context"] == {"phase": "global"}


def test_successful_cli_completion_does_not_erase_solver_diagnostics(tmp_path):
    (tmp_path / "mapper.log").write_text(
        "Global bundle adjustment\n"
        "Linear solver failure. Failed to compute a finite step.\n"
        "Keeping successful reconstruction\n", encoding="utf-8",
    )
    report = read_solver_diagnostics(tmp_path)
    assert report["linear_solver_failed_steps"] == 1
    assert report["bundle_adjustment_failures"] == 0
