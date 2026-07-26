# GPU / native runtime setup

本アプリケーションが利用する native runtime と GPU 機能を platform 別にまとめる。通常は
start script に dependency 同期と診断を任せ、手作業で package を混在させない。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は workspace、FFmpeg/FFprobe、COLMAP capability、SAM3、FastDVDnet、CUDA/cuDNN、
cuDSS を個別に表示する。Optional 機能が利用不能でも core pipeline の `ready` とは分離する。

## Windows

```powershell
.\scripts\start-windows.ps1
```

この script は frontend build、backend extra の同期、COLMAP の検出または導入、Doctor、
FastAPI 起動を順に行う。Backend は `imaging`、`aliked`、`denoise` extra を常に同期し、SAM3
だけは環境変数で追加する。`filesystem.allowed_roots` が未指定なら検出した filesystem drive を
file browser の root として process 環境へ設定する。

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

COLMAP が無ければ `nvidia-smi` の有無から CUDA / CPU package を選び、公式 4.1.1 archive を
固定 SHA-256 で検証して `.runtime/tools/` へ atomic install する。自動導入を禁止する場合:

```powershell
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
.\scripts\start-windows.ps1
```

CPU package を明示導入する場合:

```powershell
backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cpu
```

独立した `glomap.exe` は使用しない。Global Mapper は COLMAP 4.1 の `global_mapper` command
として同梱される。

## COLMAP CUDA と Bundle Adjustment

Bundle Adjustment は、全 camera pose・camera intrinsics・3D point を同時に調整し、観測した
2D keypoint への reprojection error を最小化する最終最適化である。Mapper の結果を幾何的に
締め直す処理であり、無効化する品質上の理由はない。CPU / GPU の選択は主に速度と memory の
違いで、目的関数は同じである。

`COLMAP ... with CUDA` は feature extraction や ONNX CUDA support を示すが、Ceres の
CUDA/cuDSS Bundle Adjustment を保証しない。公式 Windows 4.1.1 package が次を出す環境では
BA を CPU のまま使う。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

Doctor の `colmap.capabilities.gpu_bundle_adjustment=false` に連動して UI の BA GPU は無効に
なる。SIFT/ALIKED の GPU 利用とは別機能である。

## Native ALIKED

COLMAP 4.1.1 は `ALIKED_N16ROT`、`ALIKED_N32`、ALIKED Brute-force、LightGlue を内蔵する。
既定 option に model URL、filename、SHA-256 が含まれるため、未配置なら COLMAP が取得・検証
する。任意の model を固定する場合だけ config を使う。

```toml
[aliked]
extractor_path = "D:/models/aliked-n16rot.onnx"
matcher_path = "D:/models/aliked-lightglue.onnx"
```

Windows の ONNX CUDA provider は cuDNN 9 を必要とする。Runner は Torch の
`site-packages/torch/lib` を COLMAP subprocess の `PATH` に加える。Doctor の
`cuda_runtime.cudnn` が空なら feature GPU を無効にするか対応 runtime を導入する。CUDA
失敗を黙って CPU 成功として扱わない。

## FastDVDnet 時系列ノイズ除去

`denoise_frames` の FastDVDnet は 5 枚の source-rate frame を使う。抽出済みの疎な JPEG を
平均せず、元動画を single-pass decode して中心 frame の前後 2 枚を直接渡す。固定タイルと
80 px halo（network の 74 px receptive-field radius より大きい）で 3840² image を処理し、
camera pose や sparse model は変更しない。

Checkpoint は次の immutable source から workspace cache へ初回だけ取得する。

```text
model: model_clipped_noise.pth
SHA-256: 8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949
cache: <workspace>/.models/fastdvdnet/
```

取得は最大 3 回再試行し、SHA-256 が一致した file だけを publish する。固定 file を使う場合:

```toml
[denoise]
model_path = "D:/models/model_clipped_noise.pth"
device = "cuda"
hardware_decode = "auto"
```

Torch/CUDA が無い環境では FFmpeg `atadenoise` を source-rate で掛ける CPU fallback を使える。
`hqdn3d` や spatial BM3D は既定にしない。前者は移動中の trail、後者は 4K での処理時間と
view 間 flicker が問題になるためである。

`hardware_decode="auto"` は `yuv420p` 実動画の先頭 frame で CUDA/NVDEC を probe する。利用できる場合は
NV12 hardware frame を planar `yuv420p` へ lossless に並べ直してから RGB 化する。X5 実写 5 枚
では software decode 24.4 秒に対して 2.8 秒で、最終 RGB の SHA-256 は全 image で一致した。
未検証 pixel format の `auto` は software を維持する。`cuda` は利用不能または未検証 format を
error にし、`none` は software decode を強制する。

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

SAM3/Torch は worker process だけで import する。単一 GPU の `cuda:0` は upstream builder の
制約に合わせ内部で `cuda` へ正規化する。Inference 前の既定長辺は 1024 px。

## FFmpeg

INSV 内の 2 本の HEVC stream、`select`、`atadenoise` を利用できる build が必要。

```toml
[binaries]
ffmpeg = "D:/tools/ffmpeg/bin/ffmpeg.exe"
ffprobe = "D:/tools/ffmpeg/bin/ffprobe.exe"
```

Frame extraction と FastDVDnet context は stream ごとに一度だけ decode する。長い selection
式は複数の小さな filter branch に分け、同一 FFmpeg filter graph 内で時系列順に連結する。
1 frame ごとの process 起動、random seek、chunk ごとの先頭からの再 decode は行わない。

## Linux

Distribution package または source build の COLMAP 4.1+ と FFmpeg を用意する。

```bash
./scripts/start-linux.sh
```

`filesystem.allowed_roots` が未指定なら user home を file browser の root とする。外付け drive
や mount point は config または `SPHERE_FILESYSTEM__ALLOWED_ROOTS` で明示する。

NVIDIA 環境では uv が CUDA 12.8 build の Torch を選ぶ。ALIKED CUDA provider と cuDNN の
整合は Doctor で確認する。

## macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

macOS は PyPI の native Torch build を使う。CUDA を前提にせず、SIFT、CPU Mapper、
FastDVDnet CPU または FFmpeg fallback を利用する。3840² FastDVDnet の CPU 実行は遅いため、
必要な場合だけ有効にする。

## LFStudio

LFStudio は本アプリケーションの dependency ではなく、export 後の trainer である。CLI は
dataset と別の output directory を指定する。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --output-path <project>/training_outputs/<run>
```

LFStudio が dataset 内へ既定 `output/` を作った場合も、export の再生成前に unmanaged item を
`training_outputs/` へ移して保護する。使用中で移動できない file は削除せず stage error として
露出する。

LFStudio v0.5.3 の folder import は `train_configs/` を自動適用しない。GUI だけで開く場合は
MRNF、GUT、Segment mask を手動設定する必要があるため、再現可能な学習には上記 CLI を使う。
各実行は必ず異なる `<run>` を指定し、checkpoint / PLY の上書きを避ける。

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
uv sync --extra dev --extra imaging --extra denoise
uv run pytest -q
uv run ruff check
uv run sphere-doctor
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 pnpm test:e2e
```
