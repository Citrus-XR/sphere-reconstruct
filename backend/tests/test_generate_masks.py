"""Canonical image catalog に対する mask 生成を fake SAM3 で検証する。"""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext  # noqa: E402
from sphere_reconstruct.sam3 import engine as sam3_engine  # noqa: E402
from sphere_reconstruct.stages.generate_masks import (  # noqa: E402
    GenerateFeatureMasks,
    GenerateTrainingMasks,
    _compose_valid_mask,
    _semantic_exclusion,
)

_TRAINING_PROMPT = "person,camera operator,person's shadow"
_FEATURE_PROMPT = f"{_TRAINING_PROMPT},animal,sky,vehicle,water"


class _FakeEngine:
    @classmethod
    def from_settings(cls):
        return cls()

    def load(self):
        pass

    def unload(self):
        pass

    def detect(self, image_rgb, prompts, progress=None):
        height, width = image_rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:, : width // 2] = 1
        detections = []
        for index, prompt in enumerate(prompts):
            detection = sam3_engine.Sam3Detection(prompt=prompt)
            if index == 0:
                detection.masks = [mask]
                detection.scores = [0.9]
            detections.append(detection)
            if progress is not None:
                progress(index + 1, len(prompts))
        return detections


def test_mask_steps_have_distinct_runtime_default_prompts():
    training = GenerateTrainingMasks().normalize_params({})
    feature = GenerateFeatureMasks().normalize_params({})
    assert training["prompt"] == _TRAINING_PROMPT
    assert feature["prompt"] == _FEATURE_PROMPT
    assert training["max_inference_size"] == 2048
    assert feature["max_inference_size"] == 2048


def _make_catalog(project_dir, *, circle: bool) -> list[str]:
    prepared = project_dir / "rectify_fisheye"
    names = ["sources/source-a/main/frame_000000.jpg", "sources/source-a/main/frame_000001.jpg"]
    images = []
    for index, name in enumerate(names):
        path = prepared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.full((256, 256, 3), 128, dtype=np.uint8))
        images.append(
            {
                "name": name,
                "path": str(path.relative_to(project_dir)),
                "source_id": "source-a",
                "capture_index": index,
                "width": 256,
                "height": 256,
                "valid_region": (
                    {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.459} if circle else {"kind": "full"}
                ),
            }
        )
    (prepared / "image_catalog.json").write_text(
        json.dumps({"version": 1, "images": images}), encoding="utf-8"
    )
    return names


def _execute(
    project_dir,
    monkeypatch,
    stage_type,
    *,
    circle: bool,
    coverage_warn: float = 0.9,
    events: list[tuple] | None = None,
):
    names = _make_catalog(project_dir, circle=circle)
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)
    stage = stage_type()
    output = project_dir / f".{stage.name.value}.tmp"
    output.mkdir()
    context = StageContext(
        project_id="p",
        project_dir=project_dir,
        stage_out_dir=output,
        params=stage.normalize_params(
            {"prompt": "person,animal", "max_inference_size": 128, "coverage_warn": coverage_warn}
        ),
        sources=(),
        progress=ProgressReporter(
            lambda *args: events.append(args) if events is not None else None,
            tick_min_interval=0,
            tick_min_progress=0,
        ),
    )
    manifest = stage.execute(context)
    return output, names, manifest


@pytest.mark.parametrize(
    ("stage_type", "purpose"),
    [(GenerateFeatureMasks, "feature"), (GenerateTrainingMasks, "training")],
)
def test_generate_masks_for_perspective_images(tmp_path, monkeypatch, stage_type, purpose):
    project = tmp_path / "project"
    project.mkdir()
    output, names, manifest = _execute(project, monkeypatch, stage_type, circle=False)

    for name in names:
        mask = cv2.imread(str(output / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (256, 256)
        assert mask[128, 10] == 0
        assert mask[128, 245] == 255
    document = json.loads((output / "manifest_masks.json").read_text())
    assert document["version"] == 3
    assert document["purpose"] == purpose
    assert document["complete"] is True
    assert document["generated_images"] == len(names)
    assert document["total_images"] == len(names)
    assert document["revision"]
    assert 0.45 < document["images"][0]["coverage"] < 0.55
    assert document["prompt"] == ["person", "animal"]
    assert document["images"][0]["width"] == 256
    assert document["images"][0]["height"] == 256
    assert len(document["images"][0]["sha256"]) == 64
    assert len(manifest.outputs) == len(names) + 1
    assert all(len(output.sha256) == 64 for output in manifest.outputs)
    assert not (output / ".preview-mask-header.json").exists()
    assert not (output / ".preview-mask-records.jsonl").exists()


def test_generate_masks_combines_fisheye_valid_circle(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output, names, _manifest = _execute(project, monkeypatch, GenerateFeatureMasks, circle=True)

    mask = cv2.imread(str(output / f"{names[0]}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[128, 245] == 255
    assert mask[128, 10] == 0
    assert mask[2, 2] == 0


def test_high_coverage_is_recorded_without_per_image_warning_spam(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    events: list[tuple] = []

    output, _names, _manifest = _execute(
        project,
        monkeypatch,
        GenerateFeatureMasks,
        circle=False,
        coverage_warn=0.1,
        events=events,
    )

    document = json.loads((output / "manifest_masks.json").read_text())
    assert all(record["coverage_warning"] for record in document["images"])
    assert not [event for event in events if event[5] == "log" and event[3] == "log.mask_coverage_warn_image"]
    progress = [event for event in events if event[5] == "progress"]
    assert progress
    assert progress[-1][1] == pytest.approx(0.98)


def test_semantic_coverage_preserves_mask_and_separates_overlapping_sky_and_water():
    sky = np.zeros((8, 8), dtype=np.uint8)
    sky[:6] = 1
    water = np.zeros_like(sky)
    water[4:] = 1
    valid = np.ones_like(sky)
    valid[:, :2] = 0
    detections = [sam3_engine.Sam3Detection(prompt="sky", masks=[sky]),
                  sam3_engine.Sam3Detection(prompt="water", masks=[water])]
    excluded, coverage, without_sky = _semantic_exclusion(detections, 8, 8, 0, valid)
    assert np.array_equal(excluded, valid)
    assert coverage == {"sky": 0.75, "water": 0.5}
    assert without_sky == 0.5


def test_feature_sky_coverage_is_not_a_coverage_warning(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    _make_catalog(project, circle=False)
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)
    stage = GenerateFeatureMasks()
    output = project / ".generate_feature_masks.tmp"
    output.mkdir()
    manifest = stage.execute(StageContext(
        project_id="p", project_dir=project, stage_out_dir=output, sources=(),
        params=stage.normalize_params({"prompt": "sky,water", "coverage_warn": 0.1}),
        progress=ProgressReporter(lambda *_args: None),
    ))
    assert manifest.extra["average_dynamic_coverage"] > 0.5
    assert manifest.extra["average_coverage_by_prompt"]["sky"] > 0.5
    assert manifest.extra["average_coverage_without_sky"] == 0
    assert manifest.extra["coverage_warnings"] == 0


def test_prompt_statistics_do_not_open_seams_during_bilinear_mask_resize():
    first = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    detections = [sam3_engine.Sam3Detection(prompt="sky", masks=[first]),
                  sam3_engine.Sam3Detection(prompt="water", masks=[1 - first])]
    valid = np.ones((17, 17), dtype=np.uint8)
    excluded, _, _ = _semantic_exclusion(detections, 17, 17, 0, valid)
    assert np.array_equal(excluded, valid)


def test_generate_masks_combines_calibrated_fisheye_hemisphere():
    dynamic = np.zeros((100, 100), dtype=np.uint8)
    region = {
        "kind": "fisheye",
        "camera_model": "OPENCV_FISHEYE",
        "params": [25.0, 25.0, 55.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        "max_theta_rad": 1.4,
        "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.49},
    }

    mask, coverage = _compose_valid_mask(region, dynamic, 100, 100)

    assert mask[50, 55] == 1
    assert mask[50, 1] == 0
    assert mask[0, 55] == 0
    assert coverage == 0.0


def test_generate_masks_rejects_image_dimensions_changed_after_prepare(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    names = _make_catalog(project, circle=False)
    image_path = project / "rectify_fisheye" / names[0]
    assert cv2.imwrite(str(image_path), np.zeros((128, 128, 3), dtype=np.uint8))
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)
    stage = GenerateTrainingMasks()
    output = project / ".generate_training_masks.tmp"
    output.mkdir()
    context = StageContext(
        project_id="p",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params({"prompt": "person"}),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )

    with pytest.raises(RuntimeError, match="dimensions changed"):
        stage.execute(context)
