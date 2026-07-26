"""imaging.rendering.render_perspective_from_equirect の単体テスト.

equirect -> perspective の経度/緯度マッピングと, 縦方向を巻き込まない (BORDER_WRAP を
横のみに効かせる) 修正を検証する. cv2/numpy のみ, GPU 不要.
"""

from __future__ import annotations

import numpy as np
import pytest

from sphere_reconstruct.imaging import projection, rendering

cv2 = pytest.importorskip("cv2")


def _views():
    return {v.name: v for v in projection.cubemap_views(size=64, fov_deg=90.0)}


def _gradient_erp(tmp_path, w=256, h=128):
    """R チャンネル=経度(u), G チャンネル=緯度(v) の勾配 ERP を書き出しパスを返す."""
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    erp = np.zeros((h, w, 3), np.uint8)
    erp[..., 2] = (uu / w * 255).astype(np.uint8)  # BGR の R
    erp[..., 1] = (vv / h * 255).astype(np.uint8)  # G
    p = tmp_path / "erp.png"
    cv2.imwrite(str(p), erp)
    return p


def test_front_center_maps_to_erp_center(tmp_path):
    img, st = rendering.render_perspective_from_equirect(_gradient_erp(tmp_path), _views()["front"])
    assert st.valid_ratio == 1.0
    assert st.dst_size == (64, 64)
    c = img[32, 32]
    assert abs(int(c[2]) - 128) <= 4  # 経度 0 -> u=W/2
    assert abs(int(c[1]) - 128) <= 4  # 緯度 0 -> v=H/2


def test_up_and_down_do_not_wrap_vertically(tmp_path):
    p = _gradient_erp(tmp_path)
    up, _ = rendering.render_perspective_from_equirect(p, _views()["up"])
    down, _ = rendering.render_perspective_from_equirect(p, _views()["down"])
    # up の中心は ERP 上端 (G~0), down の中心は下端 (G~255). BORDER_WRAP を縦に効かせると
    # down が上端へ巻き込まれ G~0 になってしまう (修正前のバグ) ため, ここで検出する.
    assert int(up[32, 32][1]) < 20
    assert int(down[32, 32][1]) > 235


def test_front_horizontal_gradient_increases_rightward(tmp_path):
    img, _ = rendering.render_perspective_from_equirect(_gradient_erp(tmp_path), _views()["front"])
    left = int(img[32, 8][2])
    right = int(img[32, 56][2])
    assert right > left  # 右へ行くほど経度(u)大
