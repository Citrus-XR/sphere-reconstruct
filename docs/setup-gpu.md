# GPU / native runtime setup

Platform ごとの native runtime、GPU capability、mixed-source 入力の注意点をまとめる。通常は start
script に dependency sync と Doctor を任せ、system Python と project venv を混在させない。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は workspace、filesystem roots、FFmpeg / FFprobe、COLMAP 4.1 capability、optional SAM3、
CUDA / cuDNN、cuDSS を検査する。

## Windows

```powershell
.\scripts\start-windows.ps1
```

Frontend build、`imaging` / `aliked` extra、COLMAP 検出/導入、Doctor、FastAPI を順に実行する。
SAM3 は明示的に追加する。

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

COLMAP が無ければ `nvidia-smi` に応じて official CUDA / CPU archive を選び、固定 SHA-256 で
`.runtime/tools/` へ atomic install する。禁止する場合:

```powershell
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
.\scripts\start-windows.ps1
```

独立 `glomap.exe` は使わず、COLMAP 4.1 `global_mapper` を使う。

## Linux

COLMAP 4.1+ と FFmpeg を用意する。

```bash
./scripts/start-linux.sh
```

`filesystem.allowed_roots` が空なら user home を file browser root にする。Mount / external drive は
config または `SPHERE_FILESYSTEM__ALLOWED_ROOTS` で指定する。

## macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

CUDA を前提にせず SIFT、CPU Mapper、CPU BA を基本経路とする。SAM3 は利用 runtime に合わせて
optional install する。

## COLMAP camera / rig requirements

Mixed source は 1 database 内で複数 camera model を使う。必要 capability:

- `OPENCV_FISHEYE`、`SIMPLE_RADIAL`、`PINHOLE`、`EQUIRECTANGULAR`
- `feature_extractor --image_list_path`
- Multiple rig config
- Global Mapper

Feature extraction は camera group ごとに別 invocation を行い、同じ database に追記する。Phone
still は EXIF orientation を pixel へ適用し、model / resolution / 35mm focal signature で grouping
する。Extracted phone video は 1 calibration group。

Raw dual-fisheye は calibration adapter が必要。現在は Insta360 INSV。別 camera は stitched ERP
または新 adapter で追加する。

## Matching and scale

Mixed source では temporal adjacency だけでは cross-source edge が作れない。500 images 以下の
Auto は Exhaustive。大規模 SIFT dataset は `binaries.vocab_tree` を指定する。

Global Mapper は悪い focal prior / outlier match に敏感。Phone EXIF focal を維持し、動画に EXIF
が無い場合は初期 focal を refinement する。Primary physical rig があっても SfM world scale は
gauge freedom を持つため、alignment stage は primary reference trajectory を 1 に正規化する。

## Bundle Adjustment

BA は pose、intrinsics、3D points の reprojection error を同時最適化する。`COLMAP ... with CUDA`
は Ceres CUDA/cuDSS BA を保証しない。次の build は CPU BA を使う。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

Doctor の `colmap.capabilities.gpu_bundle_adjustment` と UI option が連動する。

## Native ALIKED

COLMAP 4.1.1 は ALIKED N16ROT / N32、Brute-force、LightGlue を内蔵する。Model URL と SHA-256
が option に含まれるため、通常は自動 download/cache される。固定 model を使う場合:

```toml
[aliked]
extractor_path = "D:/models/aliked-n16rot.onnx"
matcher_path = "D:/models/aliked-lightglue.onnx"
```

Windows ONNX CUDA provider は cuDNN 9 を必要とする。Doctor の `cuda_runtime.cudnn` を確認する。

## SAM3

SAM3 は prepared canonical image 全てに同じ処理を行う。Video frame、phone still、ERP、pinhole
view は full-image valid region、native fisheye は source-specific circle と dynamic mask を合成する。

```toml
[sam3]
repo_path = "D:/models/sam3-main"
checkpoint_path = "D:/models/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 1024
```

Mask は image relative path を mirror し、white=keep / black=ignore。Prepared image と同じ pixel
orientation / dimensions を保証する。

## FFmpeg

INSV dual HEVC stream と通常 video を sequential decode できる build が必要。

```toml
[binaries]
ffmpeg = "D:/tools/ffmpeg/bin/ffmpeg.exe"
ffprobe = "D:/tools/ffmpeg/bin/ffprobe.exe"
```

各 video は同じ frame-selection parameter を使う。長い selection は 1 filter graph 内で bounded
branch に分け、frame ごとの random seek を行わない。

## LichtFeld Studio

Mixed camera COLMAP loader は image ごとに projection を読む。Supported path:

- PINHOLE / SIMPLE_PINHOLE
- SIMPLE_RADIAL / RADIAL / OPENCV / FULL_OPENCV
- OPENCV_FISHEYE / supported fisheye variants
- EQUIRECTANGULAR

Distorted / fisheye / equirectangular を含む mixed dataset は MRNF/MCMC + GUT を使う。IGS+ は GUT
と併用できず、equirectangular を扱えない。最も conservative な fallback は 360° source を
pinhole cubemap にし、phone camera を undistort する。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json --data-path <dataset>
```

Training output directory は LFStudio 側で選ぶ。本 application は管理しない。Mixed-camera training
は loader / GUT 上は対応するが upstream end-to-end test が無いため experimental とする。

## Service deployment

```text
SPHERE_CONFIG=D:/path/to/config.remote.toml
```

更新時は listener の scheduled service とその worker tree だけを停止し、machine の Python process
を一括停止しない。Source mutation は active job 中 409 を返す。

## Verification

```bash
cd backend
uv sync --extra dev --extra imaging
uv run ruff check src tests
uv run pytest -q
uv run sphere-doctor
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
pnpm test:e2e
```
