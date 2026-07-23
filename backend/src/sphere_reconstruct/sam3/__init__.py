"""sam3 パッケージ.

Phase 4 で本格実装. 現状は path 検証 + 遅延 import ヘルパのみ.

方針 (spec より):
- HuggingFace Token 不要, 自動 DL 一切なし.
- 手動配置: settings.sam3.repo_path / checkpoint_path / config_path.
- 実際の推論は Worker サブプロセス内で `sam3` パッケージを import する.
  FastAPI プロセスでは一切 Torch を触らない.
"""
