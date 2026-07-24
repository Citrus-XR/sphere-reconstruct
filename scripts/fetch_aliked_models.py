#!/usr/bin/env python3
"""ALIKED + LightGlue の ONNX モデルを取得する.

SAM3 と同様「手動配置 or スクリプトで取得」方式. モデルは fabio-sim/LightGlue-ONNX
(Apache-2.0) の release から取る. ALIKED (BSD-3) / LightGlue (Apache-2.0) の重みを
ONNX へ変換したもの.

使い方:
    python scripts/fetch_aliked_models.py --out D:/Models/aliked

取得後, runtime/config.toml の [aliked] に:
    extractor_path = ".../aliked-n16.onnx"
    matcher_path   = ".../aliked_lightglue.onnx"
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# fabio-sim/LightGlue-ONNX v1.0 release assets (Apache-2.0).
# バージョン/URL は release ページで確認して更新する.
_MODELS = {
    "aliked-n16.onnx": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/aliked-n16.onnx",
    "aliked_lightglue.onnx": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/aliked_lightglue.onnx",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="モデル保存先ディレクトリ")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in _MODELS.items():
        dst = out / name
        if dst.exists():
            print(f"skip (exists): {dst}")
            continue
        print(f"downloading {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)  # noqa: S310
        print(f"  {dst.stat().st_size} bytes")
    print("done. set [aliked] extractor_path / matcher_path in runtime/config.toml")


if __name__ == "__main__":
    main()
