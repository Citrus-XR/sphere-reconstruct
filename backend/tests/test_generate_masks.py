"""Canonical image catalog に対する mask 生成を fake SAM3 で検証する。"""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext  # noqa: E402
from sphere_reconstruct.sam3 import engine as sam3_engine  # noqa: E402
from sphere_reconstruct.stages.generate_masks import GenerateMasks  # noqa: E402


class _FakeEngine:
    @classmethod
    def from_settings(cls):
        return cls()

    def load(self):
        pass

    def unload(self):
        pass

    def detect(self, image_rgb, prompts):
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
        return detections


def _make_catalog(project_dir, *, circle: bool) -> list[str]:
    prepared = project_dir / "prepare_images"
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
                "valid_region": (
                    {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.459} if circle else {"kind": "full"}
                ),
            }
        )
    (prepared / "image_catalog.json").write_text(
        json.dumps({"version": 1, "images": images}), encoding="utf-8"
    )
    return names


def _execute(project_dir, monkeypatch, *, circle: bool):
    names = _make_catalog(project_dir, circle=circle)
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)
    output = project_dir / ".generate_masks.tmp"
    output.mkdir()
    stage = GenerateMasks()
    context = StageContext(
        project_id="p",
        project_dir=project_dir,
        stage_out_dir=output,
        params=stage.normalize_params(
            {"prompt": "person,tripod", "max_inference_size": 128, "coverage_warn": 0.9}
        ),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )
    manifest = stage.execute(context)
    return output, names, manifest


def test_generate_masks_for_perspective_images(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output, names, manifest = _execute(project, monkeypatch, circle=False)

    for name in names:
        mask = cv2.imread(str(output / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (256, 256)
        assert mask[128, 10] == 0
        assert mask[128, 245] == 255
    document = json.loads((output / "manifest_masks.json").read_text())
    assert 0.45 < document["images"][0]["coverage"] < 0.55
    assert document["prompt"] == ["person", "tripod"]
    assert len(manifest.outputs) == len(names) + 1


def test_generate_masks_combines_fisheye_valid_circle(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output, names, _manifest = _execute(project, monkeypatch, circle=True)

    mask = cv2.imread(str(output / f"{names[0]}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[128, 245] == 255
    assert mask[128, 10] == 0
    assert mask[2, 2] == 0
