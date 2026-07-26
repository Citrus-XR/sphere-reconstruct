"""colmap.rig の単体テスト.

cam_from_rig の計算と JSON フォーマットを検証する. 参照センサ (front_lens0) が
identity で pose を持たないこと, 前後レンズのベースラインが並進に反映されることを確認.
"""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.colmap import rig

_VIEWS = [
    {"name": "front", "yaw_deg": 0.0, "pitch_deg": 0.0},
    {"name": "right", "yaw_deg": 90.0, "pitch_deg": 0.0},
    {"name": "back", "yaw_deg": 180.0, "pitch_deg": 0.0},
]
_LENSES = [
    {"index": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0},
    {"index": 1, "tx": 0.0, "ty": 0.0, "tz": -0.032273},
]


def test_rotmat_to_quat_identity():
    q = rig.rotmat_to_quat_wxyz(np.eye(3))
    assert abs(q[0] - 1.0) < 1e-9
    assert abs(q[1]) < 1e-9 and abs(q[2]) < 1e-9 and abs(q[3]) < 1e-9


def test_rotmat_to_quat_roundtrip():
    # 90° about Y.
    rotation = rig.yaw_pitch_rotation(90.0, 0.0)
    q = rig.rotmat_to_quat_wxyz(rotation)
    # ノルム 1.
    n = sum(c * c for c in q)
    assert abs(n - 1.0) < 1e-9


def test_compute_rig_cameras_ref_is_front_lens0():
    cams = rig.compute_rig_cameras(_VIEWS, _LENSES)
    assert len(cams) == 3 * 2
    refs = [c for c in cams if c.is_ref]
    assert len(refs) == 1
    ref = refs[0]
    assert ref.image_prefix == "front_lens0/"
    # front (identity) + lens0 (origin) -> quat identity, translation 0.
    assert abs(ref.quat_wxyz[0] - 1.0) < 1e-9
    assert all(abs(t) < 1e-9 for t in ref.translation)


def test_compute_rig_cameras_baseline_in_front_lens1():
    cams = rig.compute_rig_cameras(_VIEWS, _LENSES)
    front_lens1 = next(c for c in cams if c.image_prefix == "front_lens1/")
    # front は R_view=I なので t = -I @ (0,0,-0.032273) = (0,0,0.032273).
    assert abs(front_lens1.translation[2] - 0.032273) < 1e-6
    assert not front_lens1.is_ref


def test_build_rig_config_format():
    cams = rig.compute_rig_cameras(_VIEWS, _LENSES)
    cfg = rig.build_rig_config(cams, [320.0, 320.0, 320.0, 320.0])
    # トップレベルは配列, 1 rig.
    assert isinstance(cfg, list)
    assert len(cfg) == 1
    cameras = cfg[0]["cameras"]
    assert len(cameras) == 6
    # ref は cam_from_rig を持たず ref_sensor=True.
    ref = next(c for c in cameras if c.get("ref_sensor"))
    assert "cam_from_rig_rotation" not in ref
    assert ref["image_prefix"] == "front_lens0/"
    assert ref["camera_model_name"] == "PINHOLE"
    # 非 ref は cam_from_rig_rotation(4) + translation(3).
    non_ref = next(c for c in cameras if not c.get("ref_sensor"))
    assert len(non_ref["cam_from_rig_rotation"]) == 4
    assert len(non_ref["cam_from_rig_translation"]) == 3
