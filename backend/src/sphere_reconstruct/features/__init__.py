"""features パッケージ.

ALIKED (学習特徴抽出) + LightGlue (学習マッチング) を ONNX Runtime で動かす.
COLMAP の feature_backend="aliked" で使う. SIFT が苦手な弱テクスチャ / 大視差 /
繰り返し模様のシーンで有利.

モデルは SAM3 同様の手動配置方式 (settings.aliked.*). onnxruntime は Worker
プロセスでのみ import する.
"""
