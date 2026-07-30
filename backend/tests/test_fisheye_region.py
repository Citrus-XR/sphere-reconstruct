"""imaging.fisheye_region の単体テスト (json/clamp のみ, cv2 不要)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from sphere_reconstruct.imaging import fisheye_region as fr


def test_default_region_uses_default_r():
    d = fr.default_region()
    assert d["lens0"]["r"] == fr.DEFAULT_R_NORM == 0.459
    assert d["lens1"]["r"] == 0.459
    assert d["lens0"]["cx"] == 0.5 and d["lens0"]["cy"] == 0.5
    assert d["lens0"]["operations"] == d["lens1"]["operations"] == []


def test_load_missing_returns_default(tmp_path):
    assert fr.load_region(tmp_path, "source-a") == fr.default_region()


def test_load_fills_missing_lens(tmp_path):
    fr.region_path(tmp_path).write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {"source-a": {"lens0": {"cx": 0.4, "cy": 0.6, "r": 0.5}}},
            }
        )
    )
    out = fr.load_region(tmp_path, "source-a")
    assert out["lens0"] == {"cx": 0.5, "cy": 0.5, "r": 0.5, "operations": []}
    assert out["lens1"] == fr.default_region()["lens1"]


def test_save_clamps_and_roundtrips(tmp_path):
    out = fr.save_region(
        tmp_path,
        "source-a",
        {
            "lens0": {
                "cx": 2.0,
                "cy": -1.0,
                "r": 5.0,
                "operations": [
                    {"mode": "add", "x": 2.0, "y": -1.0, "r": 0.1},
                    {"mode": "subtract", "x": 0.2, "y": 0.3, "r": 0.05},
                ],
            },
            "lens1": {"r": 0.001},
        },
    )
    assert out["lens0"]["cx"] == 0.5
    assert out["lens0"]["cy"] == 0.5
    assert out["lens0"]["r"] == 0.75  # [0.01, 0.75]
    assert out["lens1"]["r"] == 0.01
    assert out["lens0"]["operations"][0] == {
        "mode": "add",
        "x": 1.0,
        "y": 0.0,
        "r": 0.1,
    }
    assert fr.load_region(tmp_path, "source-a")["lens0"]["cx"] == 0.5
    document = json.loads(fr.region_path(tmp_path).read_text())
    assert document["version"] == fr.REGION_VERSION
    assert document["sources"]["source-a"]["_coordinate_version"] == fr.REGION_VERSION


def test_circle_px_uses_width_for_radius():
    cx, cy, r = fr.circle_px({"cx": 0.5, "cy": 0.5, "r": 0.4}, 1000, 800)
    assert (cx, cy, r) == (500.0, 400.0, 400.0)


def test_invalid_custom_operation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        fr.save_region(
            tmp_path,
            "source",
            {
                "lens0": {
                    "operations": [{"mode": "replace", "x": 0.5, "y": 0.5, "r": 0.1}]
                },
                "lens1": {},
            },
        )


def test_detect_lens_region_from_black_border(tmp_path):
    image = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(image, (205, 195), 185, 180, thickness=-1)
    path = tmp_path / "lens.jpg"
    assert cv2.imwrite(str(path), image)
    region = fr.detect_lens_region(path)
    assert region["cx"] == region["cy"] == 0.5
    assert 0.43 < region["r"] < 0.47


def test_detect_clipped_circle_uses_conservative_inner_radius(tmp_path):
    image = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(image, (200, 200), 210, 180, thickness=-1)
    path = tmp_path / "clipped.jpg"
    assert cv2.imwrite(str(path), image)
    region = fr.detect_lens_region(path)
    assert 0.45 < region["r"] < 0.49
