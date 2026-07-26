"""SAM3 path 検証と Worker 専用 inference wrapper.

方針:
- HuggingFace Token 不要, 自動 DL 一切なし.
- 手動配置: settings.sam3.repo_path / checkpoint_path / config_path.
- 実際の推論は Worker サブプロセス内で `sam3` パッケージを import する.
  FastAPI プロセスでは一切 Torch を触らない.
"""
