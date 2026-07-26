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
    stage = GenerateMasks()
    ctx = StageContext(
        project_id="p",
        project_dir=project_dir,
        stage_out_dir=out_dir,
        params=stage.normalize_params(
            {
                "prompt": "person,tripod",
                "max_inference_size": 128,  # 256 -> 128 に縮小させる
                "coverage_warn": 0.9,
                "max_frames": 0,
            }
        ),
        source_path=None,
        source_kind="insv",
        progress=ProgressReporter(
            _emit=lambda level, prog, msg, key=None, args=None, kind="log": events.append((level, msg))
        ),
    )

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


def _make_extract_output(project_dir: Path) -> None:
    """2 frame x (lens0/lens1) の魚眼画像 + manifest_frames.json を捏造する."""
    ef = project_dir / "extract_frames"
    frames = []
    for fi in range(2):
        rec = {"index": fi, "timestamp_sec": float(fi)}
        for lens in (0, 1):
            d = ef / f"lens{lens}"
            d.mkdir(parents=True, exist_ok=True)
            img = np.full((256, 256, 3), 128, dtype=np.uint8)
            p = d / f"lens{lens}_{fi:06d}.jpg"
            cv2.imwrite(str(p), img)
            rec[f"lens{lens}"] = str(p.relative_to(project_dir))
        frames.append(rec)
    (ef / "manifest_frames.json").write_text(
        json.dumps(
            {"kind": "insv_dual", "fps": 30, "width": 256, "height": 256, "count": 2, "frames": frames}
        ),
        encoding="utf-8",
    )


def test_generate_masks_fisheye(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    _make_extract_output(project_dir)  # reproject_views は無し -> auto で fisheye
    monkeypatch.setattr(sam3_engine, "Sam3Engine", _FakeEngine)

    out_dir = project_dir / ".generate_masks.tmp"
    out_dir.mkdir()

    events = []
    stage = GenerateMasks()
    ctx = StageContext(
        project_id="p",
        project_dir=project_dir,
        stage_out_dir=out_dir,
        params=stage.normalize_params(
            {"prompt": "person", "max_inference_size": 128, "coverage_warn": 0.9, "dilate_px": 2}
        ),
        source_path=None,
        source_kind="insv",
        progress=ProgressReporter(
            _emit=lambda level, prog, msg, key=None, args=None, kind="log": events.append((level, msg))
        ),
    )
    manifest = stage.execute(ctx)

    mm = json.loads((out_dir / "manifest_masks.json").read_text())
    assert mm["kind"] == "sam3_fisheye_masks"
    assert mm["dilate_px"] == 2
    for fi in range(2):
        for lens in (0, 1):
            mask_png = out_dir / f"lens{lens}" / f"frame_{fi:06d}.png"
            assert mask_png.exists(), f"missing {mask_png}"
            m = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
            assert m.shape == (256, 256)
            # 円内・右半分 (動体外) = 255, 円内・左半分 (動体) = 0, 四隅 (円外) = 0.
            assert m[128, 245] == 255, "inside circle, non-dynamic -> keep"
            assert m[128, 10] == 0, "inside circle, dynamic -> excluded"
            assert m[2, 2] == 0, "outside circle -> excluded"
    # PNG 4 枚 + manifest.
    assert len(manifest.outputs) == 2 * 2 + 1
