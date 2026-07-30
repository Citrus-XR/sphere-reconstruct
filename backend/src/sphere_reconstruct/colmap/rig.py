"""COLMAP rig 拘束の生成.

仮想 pinhole rig は全カメラの相対姿勢が既知 (レンダリング時に決まっている) なので,
COLMAP に「1 スナップショット (frame) 内の 12 カメラは固定相対姿勢」と教えて拘束を
掛ける. これにより per-image で姿勢を解く代わりに rig 軌跡 (frame ごと 1 姿勢) を解き,
前後レンズ間の 32mm ベースラインを含む幾何が一貫する.

カメラ (view V, lens L) の姿勢:
  - rig→cam 回転 R = R_view(V) = yaw_pitch_rotation(V.yaw, V.pitch)
    (cubemap の向きのみで決まり, レンズには依らない. レンダリングで
     rays_rig = rays @ R_view -> d_cam = R_view @ d_rig)
  - カメラ中心 C = camera system が正規化した sensor L の光学中心
  - cam_from_rig: 回転 q = quat(R_view), 並進 t = -R_view @ C

reference sensor は front_lens0 (R_view=I, C=t_lens0=0 -> cam_from_rig=identity).

rig config JSON フォーマットは COLMAP 4.x の rig_configurator が読む形式に合わせる:
  [ { "cameras": [ {image_prefix, ref_sensor, camera_model_name, camera_params,
                    cam_from_rig_rotation:[qw,qx,qy,qz], cam_from_rig_translation:[x,y,z]}, ... ] } ]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..imaging.projection import yaw_pitch_rotation


def rotmat_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """回転行列 -> クォータニオン (qw, qx, qy, qz). COLMAP と同じ w-first 順."""
    tr = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    # 正規化.
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


@dataclass
class RigCamera:
    image_prefix: str  # COLMAP image name の接頭辞 (例 "front_lens0/")
    is_ref: bool
    quat_wxyz: tuple[float, float, float, float]
    translation: tuple[float, float, float]


def compute_rig_cameras(
    views: list[dict],
    lenses: list[dict],
    *,
    ref_view: str = "front",
    ref_lens: int = 0,
    prefix: str = "",
) -> list[RigCamera]:
    """rig manifest の views + lenses から, 12 (= views x lenses) 個の RigCamera を計算する.

    views: [{name, yaw_deg, pitch_deg, ...}]
    lenses: [{index, tx, ty, tz}]
    """
    cams: list[RigCamera] = []
    for v in views:
        view_rotation = yaw_pitch_rotation(v["yaw_deg"], v["pitch_deg"])
        q = rotmat_to_quat_wxyz(view_rotation)
        for lens in lenses:
            center = np.array([lens["tx"], lens["ty"], lens["tz"]])
            t = -view_rotation @ center
            is_ref = v["name"] == ref_view and lens["index"] == ref_lens
            cams.append(
                RigCamera(
                    image_prefix=f"{prefix}{v['name']}_lens{lens['index']}/",
                    is_ref=is_ref,
                    quat_wxyz=q,
                    translation=(float(t[0]), float(t[1]), float(t[2])),
                )
            )
    return cams


def build_rig_config(cams: list[RigCamera], camera_params: list[float]) -> list[dict]:
    """rig_configurator が読む JSON 構造 (1 rig, 複数カメラ) を組み立てる.

    camera_params は PINHOLE の [fx, fy, cx, cy]. 全カメラ同一 intrinsics.
    """
    cam_entries = []
    for c in cams:
        entry: dict = {
            "image_prefix": c.image_prefix,
            "camera_model_name": "PINHOLE",
            "camera_params": list(camera_params),
        }
        if c.is_ref:
            entry["ref_sensor"] = True
        else:
            entry["cam_from_rig_rotation"] = list(c.quat_wxyz)
            entry["cam_from_rig_translation"] = list(c.translation)
        cam_entries.append(entry)
    return [{"cameras": cam_entries}]
