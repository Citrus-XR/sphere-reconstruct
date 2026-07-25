"""gravity_align: 合成再構成で up 復元と大域回転を検証する."""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.colmap import gravity_align as ga
from sphere_reconstruct.colmap.model import Image, Point3D, Reconstruction


def _R_to_quat(R):
    return ga._R_to_quat(R)


def _Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # world +Z (=上) 軸まわり yaw


def test_recovers_up_and_aligns_to_plus_y():
    # world 上 = +Z. カメラは直立で world +Z 軸まわりに yaw だけ変わる (典型的な手持ち撮影).
    # cam->world (yaw=0): cam_x->world_y, cam_y->world_-z, cam_z->world_x. よって上(world+Z)は
    # cam の -Y 方向 => mounting=I 前提の g_imu = (0,-1,0).
    R_cw0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # 正規回転 (det=+1).
    images = {}
    for i, yaw in enumerate([0.0, 40.0, -70.0, 150.0]):
        R_cw = _Rz(yaw) @ R_cw0
        R_wc = R_cw.T
        images[i] = Image(image_id=i, qvec=_R_to_quat(R_wc), tvec=(0.0, 0.0, 0.0),
                          camera_id=0, name=f"front/frame_{i:06d}.jpg")
    g_imu = [0.0, -1.0, 0.0]  # cam 座標での上方向.

    recon = Reconstruction(cameras={}, images=images, points3D={
        0: Point3D(0, (0.0, 0.0, 5.0), (0, 0, 0), 0.0),  # world 上 (+Z) の点.
    })
    R_align, info = ga.compute_align_rotation(recon, g_imu)
    assert R_align is not None, info
    assert info["spread_deg"] < 1.0  # 全画像で up がほぼ一致.
    assert np.allclose(info["up_world"], [0.0, 0.0, 1.0], atol=1e-3)
    ga.apply_alignment(recon, R_align)
    # +Z にあった点が +Y (viewer 上) に写る.
    p = np.array(recon.points3D[0].xyz)
    assert np.allclose(p / np.linalg.norm(p), [0.0, 1.0, 0.0], atol=1e-6)


def test_skips_without_gravity():
    recon = Reconstruction(cameras={}, images={}, points3D={})
    R_align, info = ga.compute_align_rotation(recon, None)
    assert R_align is None and info["reason"] == "no_gravity"


def test_apply_preserves_camera_projection():
    # p' = R p, R_wc' = R_wc R^T, t 不変 => x_cam 不変 を確認.
    R_align = ga._rotation_aligning(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    R_wc = np.eye(3)
    img = Image(image_id=0, qvec=ga._R_to_quat(R_wc), tvec=(0.5, 0.0, 0.0), camera_id=0, name="front/frame_000000.jpg")
    pt = Point3D(0, (1.0, 2.0, 3.0), (0, 0, 0), 0.0)
    recon = Reconstruction(cameras={}, images={0: img}, points3D={0: pt})
    p = np.array(pt.xyz)
    x_cam_before = R_wc @ p + np.array(img.tvec)
    ga.apply_alignment(recon, R_align)
    R_wc2 = ga._quat_to_R(recon.images[0].qvec)
    p2 = np.array(recon.points3D[0].xyz)
    x_cam_after = R_wc2 @ p2 + np.array(recon.images[0].tvec)
    assert np.allclose(x_cam_before, x_cam_after, atol=1e-9)
