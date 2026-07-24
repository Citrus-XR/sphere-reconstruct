"""generate_masks ステージのロジックテスト (SAM3 engine を差し替える).

torch を使わずに, 「pinhole 画像を読む -> 縮小 -> (fake) SAM3 -> union -> 元解像度へ
拡大 -> COLMAP mask PNG 書き出し」の一連が正しいかを検証する.

Sam3Engine を fake に差し替えて, 画像左半分を常に「検出物体」として返す.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sphere_reconstruct.domain.pipeline_state import StageName  # noqa: E402
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext  # noqa: E402
from sphere_reconstruct.sam3 import engine as sam3_engine  # noqa: E402
from sphere_reconstruct.stages.generate_masks import GenerateMasks  # noqa: E402


class _FakeEngine:
    """detect() が「縮小画像の左半分」を 1 つの mask として返す fake."""

    @classmethod
    def from_settings(cls):
        return cls()

    def load(self):
        pass

    def unload(self):
        pass

    def detect(self, image_rgb, prompts):
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, : w // 2] = 1
        # 最初の prompt term だけ検出, 残りは空.
        out = []
        for i, term in enumerate(prompts):
            det = sam3_engine.Sam3Detection(prompt=term)
            if i == 0:
                det.masks = [mask]
                det.scores = [0.9]
            out.append(det)
        return out


def _make_reproject_output(project_dir: Path) -> None:
    """2 frame x 2 view x 1 lens の pinhole 画像 + manifest_rig.json を捏造する."""
    rv = project_dir / "reproject_views"
    frames = []
    for fi in range(2):
        fd = rv / f"frame_{fi:06d}"
        fd.mkdir(parents=True, exist_ok=True)
        views = []
        for name in ("front", "back"):
            img = np.full((256, 256, 3), 128, dtype=np.uint8)
            p = fd / f"{name}_lens0.jpg"
            cv2.imwrite(str(p), img)
            views.append(
                {
                    "view": name,
                    "lens": 0,
                    "path": str(p.relative_to(project_dir)),
                    "valid_ratio": 0.9,
                }
            )
        frames.append({"index": fi, "views": views})
    (rv / "manifest_rig.json").write_text(
        json.dumps({"kind": "insv_pinhole_cubemap", "view_count": 2, "frames": frames}),
        encoding="utf-8",
    )


def test_generate_masks_with_fake_engine(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _make_reproject_output(project_dir)

    # Sam3Engine を fake に差し替え.
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)

    out_dir = project_dir / ".generate_masks.tmp"
    out_dir.mkdir()

    events = []
    ctx = StageContext(
        project_id="p",
        project_dir=project_dir,
        stage_out_dir=out_dir,
        params={
            "prompt": "person,tripod",
            "max_inference_size": 128,  # 256 -> 128 に縮小させる
            "coverage_warn": 0.9,
            "max_frames": 0,
        },
        source_path=None,
        source_kind="insv",
        progress=ProgressReporter(_emit=lambda level, prog, msg: events.append((level, msg))),
    )

    stage = GenerateMasks()
    manifest = stage.execute(ctx)

    # 各 frame/view に mask PNG が出ているか.
    for fi in range(2):
        for name in ("front", "back"):
            mask_png = out_dir / f"frame_{fi:06d}" / f"{name}_lens0.png"
            assert mask_png.exists(), f"missing {mask_png}"
            m = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
            assert m.shape == (256, 256)  # 元解像度へ拡大されている
            # COLMAP mask: 検出物体 (左半分) = 0, 使用 (右半分) = 255.
            assert m[128, 10] == 0, "left half should be masked out (0)"
            assert m[128, 245] == 255, "right half should be kept (255)"

    # manifest_masks.json の coverage が概ね 0.5.
    mm = json.loads((out_dir / "manifest_masks.json").read_text())
    cov = mm["frames"][0]["views"][0]["coverage"]
    assert 0.45 < cov < 0.55, f"coverage {cov} expected ~0.5"
    assert mm["prompt"] == ["person", "tripod"]

    # manifest outputs は mask PNG 4 枚 + manifest_masks.json.
    assert len(manifest.outputs) == 2 * 2 + 1
