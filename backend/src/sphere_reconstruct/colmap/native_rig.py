"""Native fisheye 物理 rig の COLMAP 設定生成.

X5 の前後 2 レンズを 2 センサーの rig として拘束する. front を参照センサー, back を
「Y 軸まわり 180deg 回転 + 物理ベースライン」に固定する. これにより視覚的な重なりが
無い前後半球でも, frame 共有 (front/NNNNNN と back/NNNNNN を同一 frame とみなす) によって
必ず 1 つの再構成へ束ねられる. 前後が別モデルに分裂する事故を根絶するのが狙い.

back の相対姿勢は正常素材の合流結果から実測した: front->back の回転は Y 軸 180deg
(quat [w,x,y,z]=[0,0,1,0], 12 フレーム平均で std 0.16deg) と極めて安定. 並進は
offset_v3 のレンズ中心間距離 (X5 で ~32mm) から与える. これは metric スケールの
アンカーにもなる (SfM 単体ではスケール不定なため).

cam_from_rig の導出: back レンズ中心を rig 系 (=front レンズ系) で C=[0,0,-b] とすると,
R_back = Ry(180deg) = diag(-1,1,-1) なので
  t = -R_back @ C = -diag(-1,1,-1) @ [0,0,-b] = [0,0,-b].
よって translation = [0,0,-b]. front は cam_from_rig = identity (参照センサー).
"""

from __future__ import annotations

import numpy as np

FRONT_PREFIX = "front/"
BACK_PREFIX = "back/"

# front->back 回転 = Y 軸 180deg. quat (w, x, y, z) = (0, 0, 1, 0). 実測由来 (std 0.16deg).
BACK_QUAT_WXYZ = (0.0, 0.0, 1.0, 0.0)

# offset_v3 が無い / 異常なときのフォールバック baseline (m). X5 実測 tz.
DEFAULT_BASELINE_M = 0.032273


def baseline_from_lens_centers(lenses: list[dict]) -> float:
    """offset_v3 の lens 辞書 (tx/ty/tz) から前後レンズ中心間距離 (m) を求める.

    lenses[0]=front, lenses[1]=back を想定. 2 個未満なら DEFAULT を返す.
    """
    if len(lenses) < 2:
        return DEFAULT_BASELINE_M
    c0 = np.array([lenses[0]["tx"], lenses[0]["ty"], lenses[0]["tz"]], dtype=float)
    c1 = np.array([lenses[1]["tx"], lenses[1]["ty"], lenses[1]["tz"]], dtype=float)
    d = float(np.linalg.norm(c1 - c0))
    return d if d > 1e-4 else DEFAULT_BASELINE_M


def build_physical_rig_config(
    camera_model_name: str, camera_params: list[float], baseline_m: float
) -> list[dict]:
    """rig_configurator が読む 2 センサー rig 設定 (front 参照 + back 180deg+baseline)."""
    b = abs(baseline_m)
    return [
        {
            "cameras": [
                {
                    "image_prefix": FRONT_PREFIX,
                    "ref_sensor": True,
                    "camera_model_name": camera_model_name,
                    "camera_params": list(camera_params),
                },
                {
                    "image_prefix": BACK_PREFIX,
                    "camera_model_name": camera_model_name,
                    "camera_params": list(camera_params),
                    "cam_from_rig_rotation": list(BACK_QUAT_WXYZ),
                    "cam_from_rig_translation": [0.0, 0.0, -b],
                },
            ]
        }
    ]
