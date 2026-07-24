# GPU マシンのセットアップ (SAM3 / COLMAP)

CUDA GPU を持つマシンで SAM3 推論と COLMAP を動かすための手順. 実機 (Windows 10,
RTX 4070 Ti, CUDA 12.x) で検証した内容をまとめる.

## 1. uv の導入

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール先は `%USERPROFILE%\.local\bin\uv.exe`.

## 2. リポジトリ取得と依存同期

```
cd backend
uv sync --extra sam3 --extra imaging
```

`--extra sam3` で torch (cu128 index) + SAM3 実行時依存が入る. CUDA が無いマシンでは
このオプションを付けない (base sync では torch は入らない).

## 3. SAM3 モデルの配置 (手動)

HuggingFace token 経由の自動 DL は行わない. 以下を手動配置する:

- `sam3-main/`  … SAM3 の Python パッケージ (`sam3/` サブフォルダを含む)
- `sam3.pt`     … チェックポイント (約 3.4 GB)

`runtime/config.toml` (または `SPHERE_CONFIG` で差す別ファイル) に絶対パスを書く:

```toml
[sam3]
repo_path = "D:/path/to/sam3-main"
checkpoint_path = "D:/path/to/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
```

### SAM3 の依存について

SAM3 の `pyproject.toml` が宣言する依存は不完全で, `model_builder` が tracker /
train パスを無条件 import するため, 画像推論だけでも以下が追加で必要:

```
einops, decord, pandas, scipy, scikit-image, scikit-learn, pycocotools,
torchmetrics, submitit, matplotlib, psutil, omegaconf, hydra-core, open-clip-torch
```

Windows では triton の代わりに `triton-windows` を使う. これらは `--extra sam3`
に含めてある.

### device の注意

`build_sam3_image_model` の内部 (`_setup_device_and_mode`) は `device == "cuda"` の
完全一致でしか `model.cuda()` を呼ばない. `"cuda:0"` を渡すと重みが CPU に残り,
入力 (cuda) と型が食い違ってエラーになる. 本実装 (`sam3/engine.py`) は cuda 系
デバイスを `"cuda"` に正規化してこれを回避している.

## 4. ffmpeg / ffprobe

INSV デコードとフレーム抽出に必要. static build を配置し, PATH に通すか
`runtime/config.toml` の `[binaries]` にパスを書く.

```toml
[binaries]
ffmpeg = "D:/path/to/ffmpeg.exe"
ffprobe = "D:/path/to/ffprobe.exe"
```

## 5. COLMAP

Phase 5 で追加予定. `colmap` バイナリを配置し `[binaries].colmap` に書く.

## 動作確認

```
uv run python -c "from sphere_reconstruct.sam3.settings import quick_check; print(quick_check())"
```

`ok=True` なら SAM3 パスは正しい. 実際の推論は Worker (または直接 Engine) から行う.
