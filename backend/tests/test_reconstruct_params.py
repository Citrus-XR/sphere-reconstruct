"""stages.reconstruct.normalize_params の単体テスト (COLMAP 不要, 純 dict ロジック).

3 モード (native_fisheye / equirectangular / pinhole_rig) のパラメータ契約を固定する.
"""

from __future__ import annotations

from sphere_reconstruct.stages.reconstruct import Reconstruct

# normalize_params は self を使わないので __new__ で十分 (COLMAP 依存を避ける).
_R = Reconstruct.__new__(Reconstruct)


def test_native_fisheye_defaults():
    p = _R.normalize_params({"reconstruction_mode": "native_fisheye"})
    assert p["reconstruction_mode"] == "native_fisheye"
    assert p["refine_intrinsics"] is True
    assert p["use_masks"] is True
    assert "use_rig" not in p


def test_equirectangular_forces_no_refine():
    p = _R.normalize_params({"reconstruction_mode": "equirectangular"})
    assert p["refine_intrinsics"] is False   # 球面モデルは精修する内参が無い
    assert p["use_masks"] is True
    assert "use_rig" not in p


def test_pinhole_rig_has_rig_flags():
    p = _R.normalize_params({"reconstruction_mode": "pinhole_rig"})
    assert p["use_rig"] is True
    assert p["refine_rig"] is True
    assert p["loop_closure"] is False
    assert p["refine_intrinsics"] is False


def test_default_mode_and_extract_cap():
    p = _R.normalize_params({})
    assert p["reconstruction_mode"] == "native_fisheye"
    assert p["feature_backend"] == "sift"
    assert p["extraction_device"] == "auto"
    assert p["extract_max_size"] == 0   # 0 = settings 既定 (2048) を使う
