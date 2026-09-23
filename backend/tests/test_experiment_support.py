from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from sphere_reconstruct.colmap.model import Camera, Image, Reconstruction

_PATH = Path(__file__).resolve().parents[2] / "scripts/experiment_support.py"
_SPEC = importlib.util.spec_from_file_location("experiment_support", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
experiment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experiment)


def _model():
    return Reconstruction(
        {1: Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)},
        {1: Image(1, (1.0, 0, 0, 0), (0, 0, 0), 1, "primary.jpg")}, {},
    )


def test_pose_validation_accepts_quaternion_sign_equivalence():
    reference, candidate = _model(), _model()
    candidate.images[1] = replace(candidate.images[1], qvec=(-1.0, 0, 0, 0))
    assert experiment.validate_primary_preserved(reference, candidate)["maximum_center_displacement"] == 0


@pytest.mark.parametrize("changes", [
    {"tvec": (float("nan"), 0, 0)},
    {"qvec": (float("nan"), 0, 0, 0)},
    {"qvec": (0, 0, 0, 0)},
    {"tvec": (0.01, 0, 0)},
    {"qvec": (0.0, 1.0, 0, 0)},
])
def test_pose_validation_rejects_invalid_or_changed_poses(changes):
    reference, candidate = _model(), _model()
    candidate.images[1] = replace(candidate.images[1], **changes)
    with pytest.raises(RuntimeError, match="pose"):
        experiment.validate_primary_preserved(reference, candidate)


def test_pose_validation_rejects_changed_calibration():
    reference, candidate = _model(), _model()
    candidate.cameras[1] = replace(candidate.cameras[1], params=[21, 20, 32, 32])
    with pytest.raises(RuntimeError, match="intrinsics"):
        experiment.validate_primary_preserved(reference, candidate)


def test_unused_supplemental_calibration_is_not_a_primary_constraint():
    reference, candidate = _model(), _model()
    reference.cameras[2] = Camera(2, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    candidate.cameras[2] = replace(reference.cameras[2], params=[25, 25, 32, 32])
    assert experiment.validate_primary_preserved(reference, candidate)["images"] == 1
