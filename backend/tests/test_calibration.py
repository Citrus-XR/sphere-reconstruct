"""calibration.parse_offset_v3_ascii の単体テスト.

実 X5 INSV から観測した ASCII キャリブレーション文字列を丸ごとリテラルとして
持ち込み, パース結果が期待通りかを検証する.
"""

from __future__ import annotations

from sphere_reconstruct.insta360 import calibration as calib


# 実 INSV (VID_20260724_021825_00_001.insv, X5) の inst box 内で観測された
# offset_v3 ASCII 文字列 (40 items, 1 + 19*2 + 1).
_REAL_OFFSET_V3 = (
    "2_"
    "2.000000_4278.300_4277.330_2694.630_2681.840_"
    "0.615_0.016_89.937_"
    "0.000000_0.000000_0.000000_"
    "0.18366432_2.07332635_-3.27984834_"
    "-0.00005305_0.00065176_"
    "10752_5376_113_"
    "2.000000_4296.810_4298.540_8064.920_2686.410_"
    "-0.718_0.211_89.840_"
    "-0.000048_0.000131_-0.032273_"
    "0.18302010_2.05338216_-3.26668859_"
    "0.00187136_0.00038193_"
    "10752_5376_113_"
    "197632"
)


def test_find_and_pick_returns_40_item_candidate():
    inst = b"\x00\x00\x00" + _REAL_OFFSET_V3.encode("ascii") + b"\x00\x00"
    candidates = calib.find_ascii_calibrations(inst)
    assert len(candidates) >= 1
    chosen = calib.pick_offset_v3(candidates)
    assert chosen is not None
    assert len(chosen.values) == calib.TOTAL_ITEMS


def test_parse_offset_v3_ascii_lens_a():
    cand = calib.OffsetV3Ascii(
        inst_offset=0,
        text=_REAL_OFFSET_V3,
        values=[float(x) for x in _REAL_OFFSET_V3.split("_")],
    )
    result = calib.parse_offset_v3_ascii(cand)
    assert result.is_valid()
    assert result.source == calib.CalibSource.OFFSET_V3
    assert result.raw["count"] == 2
    assert result.raw["calibration_id"] == 197632

    a = result.lenses[0]
    assert a.xi == 2.0
    assert a.fx == 4278.3
    assert a.fy == 4277.33
    assert a.cx == 2694.63
    assert a.cy == 2681.84
    assert a.yaw == 0.615
    assert a.pitch == 0.016
    assert a.roll == 89.937
    assert a.tx == a.ty == a.tz == 0.0
    assert a.k1 == 0.18366432
    assert a.ref_image_width == 10752
    assert a.ref_image_height == 5376
    assert a.lens_flags == 113


def test_parse_offset_v3_ascii_lens_b_baseline():
    cand = calib.OffsetV3Ascii(
        inst_offset=0,
        text=_REAL_OFFSET_V3,
        values=[float(x) for x in _REAL_OFFSET_V3.split("_")],
    )
    b = calib.parse_offset_v3_ascii(cand).lenses[1]
    # X5 の物理ベースライン ~30mm を確認. tz は m 単位.
    assert -0.05 < b.tz < -0.02, f"expected ~-32mm baseline, got tz={b.tz}"
    # cx は横並び配置なので 2694 + 5376 ≈ 8064 (実測 8064.92; 各レンズ主点微差あり)
    assert abs(b.cx - (2694.63 + 5376.0)) < 10.0


def test_short_candidate_returns_invalid_but_survives():
    """40 items 未満の短い候補が来ても例外にせず, invalid な結果を返す."""
    cand = calib.OffsetV3Ascii(
        inst_offset=0,
        text="1_2_3_4_5_6",
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    result = calib.parse_offset_v3_ascii(cand)
    assert not result.is_valid()
    assert result.raw.get("note", "").startswith("expected")
