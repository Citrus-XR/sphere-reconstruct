"""COLMAP モデル -> Web ビューア用フォーマット変換.

three.js + R3F で表示するために, sparse model を軽量な JSON + バイナリに落とす.

reconstruction.json:
  cameras: [{id, model, width, height, params}]
  images:  [{id, name, camera_id, qvec, tvec, position(world), num_points}]
  stats:   {num_images, num_points3D, mean_reprojection_error, ...}

points.bin (float32/uint8 のインターリーブ列):
  各点 = xyz (3 * float32) + rgb (3 * uint8, 4byte 境界に padding 1) + error (float32)
  -> 1 点 = 12 + 4 + 4 = 20 bytes
  ヘッダ: uint32 num_points, uint32 stride(=20)

点が多すぎる場合は max_points まで決定的に間引く.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .model import Reconstruction


@dataclass
class WebPreview:
    reconstruction_json: dict
    points_bytes: bytes
    num_points_written: int
    num_points_total: int


def build_web_preview(recon: Reconstruction, *, max_points: int = 500_000) -> WebPreview:
    cameras_json = [
        {
            "id": c.camera_id,
            "model": c.model,
            "width": c.width,
            "height": c.height,
            "params": c.params,
        }
        for c in recon.cameras.values()
    ]
    images_json = []
    for img in recon.images.values():
        pos = img.camera_center
        images_json.append(
            {
                "id": img.image_id,
                "name": img.name,
                "camera_id": img.camera_id,
                "qvec": list(img.qvec),
                "tvec": list(img.tvec),
                "position": list(pos),
                "num_points": img.num_registered_points,
            }
        )

    pts = list(recon.points3D.values())
    total = len(pts)
    if total > max_points:
        # 均等間引き (決定的).
        step = total / max_points
        idx = [int(i * step) for i in range(max_points)]
        pts = [pts[i] for i in idx]

    stride = 20
    buf = bytearray()
    buf += struct.pack("<II", len(pts), stride)
    for p in pts:
        buf += struct.pack("<3f", p.xyz[0], p.xyz[1], p.xyz[2])
        buf += struct.pack("<3B", p.rgb[0], p.rgb[1], p.rgb[2])
        buf += b"\x00"  # padding to 4-byte boundary
        buf += struct.pack("<f", p.error)

    recon_json = {
        "cameras": cameras_json,
        "images": images_json,
        "stats": recon.summary(),
        "points_file": "points.bin",
        "points_stride": stride,
    }
    return WebPreview(
        reconstruction_json=recon_json,
        points_bytes=bytes(buf),
        num_points_written=len(pts),
        num_points_total=total,
    )


def write_web_preview(recon: Reconstruction, out_dir: Path, *, max_points: int = 500_000) -> WebPreview:
    out_dir.mkdir(parents=True, exist_ok=True)
    wp = build_web_preview(recon, max_points=max_points)
    (out_dir / "reconstruction.json").write_text(
        json.dumps(wp.reconstruction_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "points.bin").write_bytes(wp.points_bytes)
    return wp
