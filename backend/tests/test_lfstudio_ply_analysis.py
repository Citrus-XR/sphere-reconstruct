"""LFStudio Gaussian PLY の統計解析を小さい fixture で検証する。"""

import math
import struct

import pytest
from scripts.analyze_lfstudio_ply import analyze


def test_analyze_binary_gaussian_ply(tmp_path):
    path = tmp_path / "gaussians.ply"
    properties = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    ).encode("ascii")
    records = b"".join(
        struct.pack("<7f", *values)
        for values in (
            (0.0, 0.0, 0.0, math.log(0.1), math.log(0.2), math.log(0.3), 0.0),
            (2.0, 0.0, 0.0, math.log(0.2), math.log(0.4), math.log(0.1), 2.0),
        )
    )
    path.write_bytes(header + records)

    result = analyze(path)

    assert result["vertices"] == result["finite_vertices"] == 2
    assert result["record_bytes"] == 28
    assert result["all_finite"] is True
    assert result["maximum_scale"]["maximum"] == pytest.approx(0.4)
    assert result["opacity"]["median"] > 0.5
