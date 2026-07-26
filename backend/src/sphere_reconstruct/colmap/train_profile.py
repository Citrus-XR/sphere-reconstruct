"""再構成統計から「学習プロファイル」を作る. LichtFeld-Studio 等の 3DGS トレーナ向けに,
場面スケール / 点数 / 密度 / 相机モデル / 品質信号をまとめ, 推奨ハイパラ導出の材料にする.

方法非依存の量だけを計算する (strategy 固有の preset 合成は別レイヤ). scene_scale は 2 系統:
- camera: 1.1 * max‖cam_center - centroid‖  (原版 3DGS / MCMC の spatial_lr_scale)
- point:  median‖point - center‖            (LichtFeld の scene_scale)
"""

from __future__ import annotations

import numpy as np

from .model import Reconstruction


def compute_profile(recon: Reconstruction) -> dict:
    prof: dict = dict(recon.summary())

    cams = np.array([im.camera_center for im in recon.images.values()]) if recon.images else np.zeros((0, 3))
    pts = (
        np.array([p.xyz for p in recon.points3D.values()], dtype=float)
        if recon.points3D
        else np.zeros((0, 3))
    )

    if len(cams) >= 1:
        centroid = cams.mean(axis=0)
        d = np.linalg.norm(cams - centroid, axis=1)
        # 原版 3DGS の cameras_extent (spatial_lr_scale).
        prof["scene_scale_camera"] = float(1.1 * d.max()) if d.size else 0.0
        if len(cams) >= 2:
            # 隣接カメラ間隔の中央値 (baseline の目安).
            gaps = [
                float(np.min(np.linalg.norm(np.delete(cams, i, axis=0) - c, axis=1)))
                for i, c in enumerate(cams)
            ]
            prof["camera_spacing_median"] = float(np.median(gaps))

    if len(pts) >= 3:
        center = np.median(pts, axis=0)
        dp = np.linalg.norm(pts - center, axis=1)
        # LichtFeld の scene_scale (点→中心の中央値).
        prof["scene_scale_point"] = float(np.median(dp))
        # 外れ点に強い bbox (2.5–97.5 パーセンタイル).
        lo = np.percentile(pts, 2.5, axis=0)
        hi = np.percentile(pts, 97.5, axis=0)
        prof["bbox_min"] = [round(float(x), 4) for x in lo]
        prof["bbox_max"] = [round(float(x), 4) for x in hi]
        prof["bbox_radius"] = float(np.linalg.norm(hi - lo) / 2.0)
        vol = float(np.prod(np.maximum(hi - lo, 1e-6)))
        prof["point_density"] = float(len(pts) / vol) if vol > 0 else 0.0

    prof["camera_models"] = sorted({c.model for c in recon.cameras.values()})
    return prof
