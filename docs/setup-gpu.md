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

`colmap` バイナリ (CUDA ビルド推奨) を配置し `[binaries].colmap` に書く.

## 6. ALIKED + LightGlue (任意, feature_backend="aliked")

SIFT が苦手な弱テクスチャ / 大視差 / 繰り返し模様のシーン向けの学習特徴. この
COLMAP ビルドが SIFT のみの場合でも, 外部 onnxruntime で抽出/マッチして COLMAP DB に
書き込む方式で使える.

```
uv sync --extra aliked            # onnxruntime-gpu
python scripts/fetch_aliked_models.py --out D:/Models/aliked   # モデル取得
```

`runtime/config.toml`:

```toml
[aliked]
extractor_path = "D:/Models/aliked/aliked-n16.onnx"
matcher_path   = "D:/Models/aliked/aliked_lightglue.onnx"
device = "cuda"            # LightGlue マッチングの provider
extraction_device = "auto" # ALIKED 抽出: auto | cuda | cpu
```

reconstruct ステージのパラメータ `feature_backend = "aliked"` で有効化する. 空 or
"sift" なら COLMAP 内蔵 SIFT を使う.

### 抽出デバイスと VRAM (魚眼で重要)

魚眼は 180deg+ を円内へ圧縮するため角分解能が元々低く, 縮小抽出すると暗所/弱テク
スチャで特徴が消える. よって ALIKED は**全解像度**で抽出する. ただし ALIKED は稠密な
特徴マップを作るため, 8K 級 (3840^2) を GPU で流すと VRAM を使い切って OOM する
(4070Ti 12GB で 2880^2 でも OOM を確認).

`extraction_device`:
- `auto` (既定): 空き VRAM と画素数から GPU/CPU を選び, 実行時に OOM が出たら CPU へ
  フォールバックして以降も CPU を使う. LightGlue マッチングは疎な keypoint のみで
  軽いため `device` (既定 GPU) のまま.
- `cuda`: 常に GPU. 小さい画像で速度を優先する場合のみ.
- `cpu`: 常に CPU. 全解像度でも OOM しないが低速.

reconstruct の `extraction_device` パラメータで stage ごとに上書きできる (UI の
「ALIKED 抽出」セレクタ). UI は GPU 固定 + 大画像で OOM リスクを警告する.

## 動作確認

```
uv run python -c "from sphere_reconstruct.sam3.settings import quick_check; print(quick_check())"
```

`ok=True` なら SAM3 パスは正しい. 実際の推論は Worker (または直接 Engine) から行う.
