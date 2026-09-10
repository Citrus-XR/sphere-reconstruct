# Sphere Reconstruct backend

FastAPI API、background worker、画像処理 pipeline を提供する Python package。
全体の機能と起動方法は [project README](../README.md)、runtime 設定は [setup guide](../docs/setup-gpu.md) を参照する。

開発環境はこの directory で `uv sync --locked --inexact --extra imaging --extra aliked` を実行して作成する。
SAM3 は `--extra sam3`、RoMaV2 は `--extra dense`、test / lint は `--extra dev` を追加する。
`--inexact` は、今回選択していない optional feature の導入済み依存関係を保持する。

環境診断は作成済み `.venv` の Python から `-m sphere_reconstruct.cli` を実行する。
Package metadata の README はこの directory 内で完結させ、wheel / sdist の build 時に親 directory の file を要求しない。
