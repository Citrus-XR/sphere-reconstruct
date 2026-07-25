"""重力対齐: IMU 重力方向 + 再構成のカメラ姿勢から, 点群を「上=重力の逆」に揃える回転を求める.

背景: COLMAP の再構成はゲージ自由度により全体の回転が任意 (最初に登録された画像の姿勢に
world 座標が固定される) ため, 毎回「上」がバラバラになる. INSV の IMU 加速度計は重力方向を
持つので, それを使って 1 つの大域回転を掛け, 点群を起こす.

原理:
- inspect_source が IMU 加速度計の平均方向 g_imu (IMU 座標, 単位) を出す (静止時は上向き).
- IMU はカメラ (native rig の ref = front センサ) に固定. mounting = R_cam_from_imu.
- 各 front 画像の world->cam 回転 R_wc から, world での上方向を
      up_world_i = R_wc^T @ (mounting @ g_imu)
  で得る. カメラが概ね直立 (yaw 主体) なら up_world_i は全画像でほぼ一致する.
- それらを平均して up_world を求め, これを viewer の上 (+Y) に写す回転 R_align を作る.

mounting (IMU->camera 取付回転) は機種/個体で未校正のため, 既定は単位行列 (IMU 軸 ≒ front
カメラ軸と仮定). ずれる場合は 1 度の実測で定数を差し替える. 一致度 (consistency) が悪い
(=モデル or 重力が信用できない) ときは対齐しない.
"""

from __future__ import annotations

import numpy as np

from .model import Reconstruction

# viewer (three.js) の上方向. 重力の逆 (up) をここに写す.
TARGET_UP = np.array([0.0, 1.0, 0.0])

# IMU -> front camera の取付回転 (未校正; 既定は単位). 実測後にここを差し替える.
MOUNTING_R = np.eye(3)


def _quat_to_R(qvec) -> np.ndarray:
    """world->cam クォータニオン (qw,qx,qy,qz) -> 回転行列."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """回転行列 -> (qw,qx,qy,qz). 数値安定な分岐版."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
    return (float(qw / n), float(qx / n), float(qy / n), float(qz / n))


def _rotation_aligning(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """単位ベクトル u を t に写す最小回転 (Rodrigues)."""
    u = u / (np.linalg.norm(u) + 1e-12)
    t = t / (np.linalg.norm(t) + 1e-12)
    v = np.cross(u, t)
    s = np.linalg.norm(v)
    c = float(np.dot(u, t))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 反対向き: u に直交する任意軸で 180 度.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, axis)
        axis /= np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3) + 2 * K @ K  # 180 度回転.
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _is_front(name: str) -> bool:
    return name.startswith("front/") or "lens0" in name


def compute_align_rotation(
    recon: Reconstruction, gravity_imu, *, mounting: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict]:
    """重力対齐の大域回転 R_align (3x3) と診断情報を返す.

    front 画像が無い / 一致度が悪い場合は (None, info) を返し, 対齐を見送る.
    """
    if gravity_imu is None:
        return None, {"reason": "no_gravity"}
    mount = MOUNTING_R if mounting is None else mounting
    g_cam = mount @ np.asarray(gravity_imu, dtype=float)

    ups = []
    for img in recon.images.values():
        if not _is_front(img.name):
            continue
        R_wc = _quat_to_R(img.qvec)
        up = R_wc.T @ g_cam
        n = np.linalg.norm(up)
        if n > 1e-9:
            ups.append(up / n)
    if len(ups) < 2:
        return None, {"reason": "too_few_front_images", "count": len(ups)}

    mean_up = np.mean(ups, axis=0)
    mn = np.linalg.norm(mean_up)
    if mn < 1e-6:
        return None, {"reason": "inconsistent_up"}
    mean_up /= mn
    # 一致度: 各 up と平均のなす角の中央値 (deg). 大きいとモデル/重力が信用できない.
    angles = [np.degrees(np.arccos(np.clip(float(np.dot(u, mean_up)), -1, 1))) for u in ups]
    spread = float(np.median(angles))

    R_align = _rotation_aligning(mean_up, TARGET_UP)
    info = {
        "up_world": [round(float(x), 4) for x in mean_up],
        "spread_deg": round(spread, 2),
        "front_images": len(ups),
    }
    if spread > 20.0:
        # ばらつきが大きい: 対齐すると却って歪む恐れ. 見送り.
        return None, {**info, "reason": "up_spread_too_large"}
    return R_align, info


def apply_alignment(recon: Reconstruction, R_align: np.ndarray) -> None:
    """R_align を再構成に in-place 適用する. 点 p'=R p, カメラ R_wc'=R_wc R^T, t 不変.

    web_preview 用の軽量変換 (COLMAP バイナリは書き換えない). qvec/tvec/xyz を更新する.
    """
    Rt = R_align.T
    for p in recon.points3D.values():
        p.xyz = tuple(float(x) for x in (R_align @ np.asarray(p.xyz)))
    for img in recon.images.values():
        R_wc = _quat_to_R(img.qvec)
        img.qvec = _R_to_quat(R_wc @ Rt)
        # tvec は不変 (x_cam = R_wc p + t = (R_wc R^T)(R p) + t).
