# GPU / native runtime setup

本アプリケーションが利用する native runtime と GPU 機能を platform 別にまとめる。通常は start
script に dependency 同期と診断を任せ、手作業で system Python と project environment を混在
させない。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は workspace、filesystem root、FFmpeg / FFprobe、COLMAP capability、optional SAM3、
CUDA / cuDNN、cuDSS を個別に表示する。Optional capability が利用不能でも core pipeline の
`ready` とは分離する。

## Windows

```powershell
.\scripts\start-windows.ps1
```

この script は frontend build、backend extra 同期、COLMAP 検出または導入、Doctor、FastAPI
起動を順に行う。`imaging` と `aliked` extra を同期し、SAM3 だけを環境変数で追加する。
`filesystem.allowed_roots` が未指定なら、検出した filesystem drive を file browser root として
process environment に設定する。

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

COLMAP が無ければ `nvidia-smi` の有無から CUDA / CPU package を選び、公式 4.1.1 archive を固定
SHA-256 で検証して `.runtime/tools/` へ atomic install する。自動導入を禁止する場合:

```powershell
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
.\scripts\start-windows.ps1
```

CPU package を明示導入する場合:

```powershell
backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cpu
```

独立した `glomap.exe` は使用しない。Global Mapper は COLMAP 4.1 の `global_mapper` command として
同梱される。

## Linux

Distribution package または source build の COLMAP 4.1+ と FFmpeg を用意する。

```bash
./scripts/start-linux.sh
```

`filesystem.allowed_roots` が未指定なら user home を file browser root とする。外付け drive や
mount point は config または `SPHERE_FILESYSTEM__ALLOWED_ROOTS` で明示する。

NVIDIA 環境では uv が CUDA 12.8 build の Torch を SAM3 用に選ぶ。ALIKED CUDA provider と
cuDNN の整合は Doctor で確認する。

## macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

macOS は CUDA を前提にしない。SIFT、CPU Mapper、CPU BA を基本経路とし、SAM3 は利用する
runtime に合わせて optional に導入する。

## COLMAP CUDA と Bundle Adjustment

Bundle Adjustment は、全 camera pose、camera intrinsics、3D point を同時に調整し、観測した 2D
keypoint への reprojection error を最小化する最終最適化である。CPU / GPU の選択は主に速度と
memory の違いで、目的関数は同じ。

`COLMAP ... with CUDA` は feature extraction や ONNX CUDA support を示すが、Ceres の
CUDA / cuDSS Bundle Adjustment を保証しない。公式 Windows 4.1.1 package が次を出す環境では
BA を CPU のまま使う。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

Doctor の `colmap.capabilities.gpu_bundle_adjustment=false` に連動して UI の BA GPU option は無効に
なる。SIFT / ALIKED feature GPU とは別 capability。

## Native ALIKED

COLMAP 4.1.1 は `ALIKED_N16ROT`、`ALIKED_N32`、ALIKED Brute-force、LightGlue を内蔵する。
既定 option に model URL、filename、SHA-256 が含まれるため、未配置なら COLMAP が取得・検証
する。任意の model を固定する場合だけ config を使う。

```toml
[aliked]
extractor_path = "D:/models/aliked-n16rot.onnx"
matcher_path = "D:/models/aliked-lightglue.onnx"
```

Windows の ONNX CUDA provider は cuDNN 9 を必要とする。Runner は Torch が存在する場合、その
`site-packages/torch/lib` を COLMAP subprocess の `PATH` に加える。Doctor の
`cuda_runtime.cudnn` が空なら feature GPU を無効にするか対応 runtime を導入する。CUDA 失敗を
黙って CPU 成功として扱わない。

## SAM3

SAM3 は optional。Repository と checkpoint を runtime config に指定する。

```toml
[sam3]
repo_path = "D:/models/sam3-main"
checkpoint_path = "D:/models/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 1024
```

SAM3 / Torch は worker process だけで import する。単一 GPU の `cuda:0` は upstream builder の
制約に合わせ内部で `cuda` へ正規化する。Inference 前の既定長辺は 1024 px。

## FFmpeg

INSV 内の 2 本の HEVC stream と `select` filter を利用できる build が必要。

```toml
[binaries]
ffmpeg = "D:/tools/ffmpeg/bin/ffmpeg.exe"
ffprobe = "D:/tools/ffmpeg/bin/ffprobe.exe"
```

Frame extraction は stream ごとに一度だけ decode する。長い selection 式は複数の小さな filter
branch に分け、同じ FFmpeg filter graph 内で時系列順に連結する。1 frame ごとの process 起動、
random seek、chunk ごとの先頭からの再 decode は行わない。

## LFStudio

LFStudio は本アプリケーションの dependency ではなく、export 後の外部 trainer。Export Inspector
に表示される `export_dataset/` を dataset root として選ぶ。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json --data-path <dataset>
```

Training output directory は LFStudio 側で選択する。本アプリケーションは output path を生成、
表示、変更、移動しない。Dataset root 内に管理外 item がある状態で export を再生成または消去
すると、その item を保護するため stage は error で停止する。

LFStudio v0.5.3 の folder import は `train_configs/` を自動適用しない。GUI だけで開く場合は MRNF、
GUT、Segment mask を手動設定する。Training loss / PSNR / SSIM は外部 process の statistic なので
本 UI では取得不可と表示する。

## Service deployment

Service manager から起動するときは先に config を指定する。

```text
SPHERE_CONFIG=D:/path/to/config.remote.toml
```

更新時は listener PID とその親 process だけを終了し、machine 上の Python process を一括停止
しない。Running worker を止めると atomic publish 前の temporary output だけが破棄される。

## 検証

```bash
cd backend
uv sync --extra dev --extra imaging
uv run pytest -q
uv run ruff check src tests
uv run sphere-doctor
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 pnpm test:e2e
```
