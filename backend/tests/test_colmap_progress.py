"""COLMAP log parser が実際の counter/phase 形式を単調 progress へ変換することを検証する。"""

from types import SimpleNamespace

from sphere_reconstruct.pipeline.stage import ProgressReporter
from sphere_reconstruct.stages.colmap_progress import (
    global_mapper_progress,
    mapper_progress,
    matching_progress,
)


def _context():
    calls = []
    reporter = ProgressReporter(lambda *args: calls.append(args), tick_min_interval=0)
    return SimpleNamespace(progress=reporter), calls


def test_matching_progress_parses_image_and_two_dimensional_block_formats():
    context, calls = _context()
    callback = matching_progress(context, low=0.1, high=0.9)

    callback("Indexing image [1/4]")
    callback("Indexing image [4/4]")
    callback("Processing block [1/4, 1/4]")
    callback("Processing block [2/4, 3/4]")
    callback("Processing image [4/4]")

    numeric = [call[1] for call in calls if call[1] is not None]
    assert numeric == sorted(numeric)
    assert abs(numeric[1] - 0.34) < 1e-12
    assert numeric[-1] == 0.9


def test_global_mapper_repeated_phases_never_move_backwards():
    context, calls = _context()
    callback = global_mapper_progress(context, low=0.2, high=0.8)

    callback("rotation averaging")
    callback("bundle adjustment")
    callback("retriangulation")
    callback("bundle adjustment")
    callback("Extracting colors")

    numeric = [call[1] for call in calls if call[1] is not None]
    assert numeric == sorted(numeric)
    assert numeric[-1] == 0.8


def test_transitive_iterations_and_batches_cover_the_full_progress_range():
    context, calls = _context()
    callback = matching_progress(context, low=0.7, high=0.9)

    callback("Iteration [1/3]")
    callback("Processing batch [10/10]")
    callback("Iteration [2/3]")
    callback("Processing batch [5/10]")
    callback("Iteration [3/3]")
    callback("Processing batch [10/10]")

    numeric = [call[1] for call in calls if call[1] is not None]
    assert numeric == sorted(numeric)
    assert numeric[-1] == 0.9


def test_incremental_mapper_uses_rig_frame_counter_and_preserves_global_phase_detail():
    context, calls = _context()
    callback = mapper_progress(context, frame_count=1488, low=0.12, high=0.9)

    callback("Registering image #19 (num_reg_frames=1392)")
    callback("=> Image sees 1016 / 3749 points")
    callback("Retriangulation and Global bundle adjustment")

    numeric = [call for call in calls if call[1] is not None]
    assert len(numeric) == 1
    assert numeric[0][3] == "log.recon_mapper_progress"
    assert numeric[0][4] == {"done": 1392, "total": 1488}
    assert abs(numeric[0][1] - (0.12 + 0.78 * 1392 / 1488)) < 1e-12
    assert calls[-2][3] == "log.recon_mapper_observations"
    assert calls[-2][4] == {
        "done": 1392,
        "frames": 1488,
        "visible": 1016,
        "points": 3749,
    }
    assert calls[-1][3] == "log.recon_global_refinement"
    assert calls[-1][4] == {"pass": 1, "done": 1393, "total": 1488}


def test_incremental_mapper_does_not_count_failed_registration():
    context, calls = _context()
    callback = mapper_progress(context, frame_count=10)

    callback("Registering image #9 (num_reg_frames=8)")
    callback("=> Could not register, trying another image.")
    callback("Retriangulation and Global bundle adjustment")

    global_refinement = next(call for call in calls if call[3] == "log.recon_global_refinement")
    assert global_refinement[4] == {"pass": 1, "done": 8, "total": 10}


def test_solver_warning_is_visible_once_without_advancing_registration():
    context, calls = _context()
    callback = mapper_progress(context, frame_count=10)
    callback("Registering image #9 (num_reg_frames=8)")
    callback("Linear solver failure. Failed to compute a finite step.")
    callback("Linear solver failure. Failed to compute a finite step.")
    warnings = [call for call in calls if call[3] == "log.recon_solver_step_warning"]
    assert len(warnings) == 1
    assert warnings[0][0] == "warn"
