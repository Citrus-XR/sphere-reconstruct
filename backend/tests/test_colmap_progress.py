"""COLMAP log parser が実際の counter/phase 形式を単調 progress へ変換することを検証する。"""

from types import SimpleNamespace

from sphere_reconstruct.pipeline.stage import ProgressReporter
from sphere_reconstruct.stages.colmap_progress import global_mapper_progress, matching_progress


def _context():
    calls = []
    reporter = ProgressReporter(lambda *args: calls.append(args), tick_min_interval=0)
    return SimpleNamespace(progress=reporter), calls


def test_matching_progress_parses_image_and_two_dimensional_block_formats():
    context, calls = _context()
    callback = matching_progress(context, low=0.1, high=0.9)

    callback("Processing block [1/4, 1/4]")
    callback("Processing block [2/4, 3/4]")
    callback("Processing image [4/4]")

    numeric = [call[1] for call in calls if call[1] is not None]
    assert numeric == sorted(numeric)
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
