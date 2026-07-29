# GPU・native runtime setup

本書は platform ごとの起動、dependency diagnosis、hardware decode、COLMAP CUDA Bundle
Adjustment、SAM3、LichtFeld Studio の実行条件をまとめる。System Python へ package を直接追加せず、
repository の start script と `backend` の uv environment を使う。

## Start scripts

### Windows

```powershell
.\scripts\start-windows.ps1
```

Command Prompt からは `scripts\start-windows.cmd` を使う。Script は次を順番に行い、native command の
終了 code が非 0 ならその場で停止する。

1. pnpm dependency install と production frontend build
2. uv dependency sync
3. GPU に合う pinned COLMAP runtime の選択・atomic install
4. jpegtran と FAISS vocabulary tree の atomic install
5. Doctor
6. `http://127.0.0.1:8787` で FastAPI を起動

SAM3 dependency も導入する場合:

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

Machine 固有 config:

```powershell
$env:SPHERE_CONFIG="D:/path/to/config.toml"
.\scripts\start-windows.ps1
```

Auto install を禁止する場合:

```powershell
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
$env:SPHERE_SKIP_AUTO_INSTALL_JPEGTRAN="1"
.\scripts\start-windows.ps1
```

### Linux

COLMAP 4.1+、FFmpeg、FFprobe を distribution package または独自 build で用意する。

```bash
./scripts/start-linux.sh
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

`filesystem.allowed_roots` が空なら user home を file browser root にする。External drive は config または
`SPHERE_FILESYSTEM__ALLOWED_ROOTS` へ追加する。

### macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

VideoToolbox decoder を probe する。CUDA BA は使わず、SIFT、Global Mapper、CPU BA が基本経路になる。

## Doctor

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は次を独立した capability として表示する。

- Workspace と file-browser roots の read/write 可否
- FFmpeg build の hardware acceleration method と actual source decoder probe
- COLMAP 4.1、Global Mapper、ALIKED、ERP camera support
- Ceres dense CUDA solver と cuDSS sparse CUDA solver
- NVIDIA GPU name、driver、compute capability
- FAISS SIFT vocabulary tree header
- COLMAP native ALIKED 用 ONNX CUDA / cuDNN runtime
- Optional SAM3 と jpegtran

`COLMAP ... with CUDA` は GPU SIFT を示すだけで、Ceres CUDA / cuDSS BA を保証しない。

## Windows COLMAP variants

`scripts/install_colmap.py` は URL、version、SHA-256 を固定し、temporary directory で検証してから
`.runtime/tools/` を atomic replace する。Explicit config、auto-installed record、system PATH の順に binary
を解決するため、一度導入した pinned runtime が古い PATH binary に隠れない。

| Variant | Selection | Feature GPU | BA GPU |
|---|---|---:|---:|
| `cuda-ba` | Driver 580+、compute capability 7.5+ | Yes | Ceres CUDA + cuDSS |
| `cuda` | 上記条件を満たさない NVIDIA GPU | Yes | No |
| `cpu` | NVIDIA GPU なし | No | No |

`SPHERE_COLMAP_VARIANT=cpu|cuda|cuda-ba` で auto selection を上書きできる。Manual install:

```powershell
backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cuda-ba
```

Pinned CUDA-BA runtime:

- COLMAP 4.1.1
- Ceres 2.3 development commit `bac1127f9ef672405bd0d2d9c84e809ae89bd239`
- CUDA 12.8
- cuDSS 0.8.0.10
- cuDNN 9.20.0.48、cuFFT、NVRTC、ONNX Runtime CUDA provider
- vcpkg commit `6d9d7df564a1ccdaa994e4ad39ccd4a32360867b`
- Native SASS: sm75 / sm80 / sm86 / sm89 / sm90、forward PTX: compute75
- CLI only、GUI / MVS / CGAL / OpenGL off
- COLMAP PR #4591 の folder-major rig sequential-pairing fix

CUDA 12.8 を選ぶ理由は、現行 Windows driver で native sm89 を実行でき、CUDA 13.2 PTX が要求する
R595 driver に依存しないため。Ceres の pinned commit は caller の `CMAKE_CUDA_ARCHITECTURES` を
`75;80;90` へ上書きするため、build script は non-empty caller value を保持する最小 patch を適用する。

Official COLMAP 4.1.1 Windows CUDA archive は COLMAP 自体を CUDA enabled で build する一方、同梱
Ceres は CUDA / cuDSS 無効で、次の warning 後に CPU BA へ fallback する。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

このため UI は version string ではなく `sphere-colmap-capabilities.json` と runtime library を検査して
`ba_use_gpu` を有効化する。

## Reproducible CUDA-BA build

```powershell
.\scripts\build_colmap_cuda_ba.ps1 -OutputDirectory dist
```

Developer machine で失敗後の同一 build root を明示的に再利用する場合:

```powershell
.\scripts\build_colmap_cuda_ba.ps1 `
  -OutputDirectory dist `
  -ReuseBuildRoot
```

Script は利用可能な最も新しい CMake 3.30+ を選び、Release try-compile を固定し、すべての native command
終了 code を検査する。Source は pinned revision だけを shallow fetch する。Vcpkg manifest の CPU Ceres を
除外し、custom Ceres を CUDA / CHOLMOD / SPQR / LAPACK / cuDSS 付きで 1 回だけ build する。ONNX CUDA の
transitive dependency である cuFFT / cuDNN / NVRTC と、cuSPARSE が必要とする nvJitLink も app-local にする。
Runtime archive には executable / DLL、capability metadata、COLMAP / Ceres / NVIDIA license だけを入れ、
development header や static library は含めない。

GitHub Actions definition は `.github/workflows/build-colmap-cuda-ba.yml`。Build job は全 PE の通常・delay-load
import を再帰検査し、Ceres と cuDSS を明示的に `LoadLibraryExW` する。ONNX CUDA provider は NVIDIA driver
がある環境で load smoke を行い、GPU のない GitHub hosted runner では PE dependency closure までを検証する。
次の clean Windows job が archive だけを展開して同じ検査と `colmap version` を再実行し、両方通った artifact
だけを release する。CUDA 12.8 Windows installer に存在しない CUDA 13 専用 component 名 `crt` / `nvvm`
は指定しない。

## FFmpeg hardware decode

```toml
[frame_extraction]
hwaccel = "auto"
require_hwaccel = false
score_workers = 0
```

Dedicated NVIDIA machine:

```toml
[frame_extraction]
hwaccel = "cuda"
require_hwaccel = true
score_workers = 0
```

Candidate priority は CUDA、VideoToolbox、QSV、D3D11VA、D3D12VA、DXVA2、VAAPI、VDPAU。FFmpeg の
`-hwaccels` は build capability だけなので、実 input の 1 frame decode と software filter transfer を
probe する。成功した method だけを本処理の `-i` より前へ渡す。`require_hwaccel=true` は probe failure を
Step error にし、software fallback を隠さない。

INSV は 2 lens stream を 1 process / 1 demux pass で処理する。Spatial selection は candidate JPEG を
final quality で生成し、採用時に再 encode しない。RTX 4070 Ti、3840² HEVC の実測は CUDA 8.98×
realtime、software 0.85×。同じ frame の output は pixel MAE 0、maximum difference 0。

`score_workers=0` は logical CPU 全数を sharpness / exposure / SIFT score に使う。各 worker は独立 SIFT
instance と OpenCV internal thread 1 を持ち、nested oversubscription を避ける。Optical-flow selection の
decision loop は前回採用 frame に依存するため sequential のままにする。

## COLMAP camera・matching requirements

必要 capability:

- `OPENCV_FISHEYE`、`SIMPLE_RADIAL`、`PINHOLE`、`EQUIRECTANGULAR`
- `feature_extractor --image_list_path`
- Multi-camera rig と generalized rig verification
- Global Mapper
- Optional native ALIKED / LightGlue

INSV native は `offset_v3` の front / back MEI calibration を別々の `OPENCV_FISHEYE` へ fit する。
COLMAP perspective fisheye は forward hemisphere 専用なので、valid radius は calibrated 180° radius の
scalar clamp にしない。Principal point、fx / fy、k1–k4 から pixel ごとの ray angle を評価し、89.55° 未満と
physical image circle の積集合を mask にする。Reliable physical intrinsics / rig extrinsics は固定し、全
camera が固定済みなら view-graph calibration を skip する。Physical circle schema v2 は image-edge
coordinates を使う。旧 schema の saved circle は移動せず UI で review / resave を要求する。

Single-source video は Sequential + loop closure + transitive matching。COLMAP 4.1.1 は folder-major sensor
境界を誤った temporal pair として展開するため、runtime は upstream
[PR #4591](https://github.com/colmap/colmap/pull/4591) の same-camera guard を backport する。修正後は旧 match
row を再利用せず database を clean rebuild する。

### FAISS vocabulary tree

COLMAP 4.1 は 2025 年に FLANN から FAISS へ移行した。旧 `vocab_tree_flickr...bin` は存在していても
matching 時に crash するため、installer と Doctor は file header `{version: 1|2, desc: 128, embedding: 64}`
を検証する。

既定は COLMAP 4.1.1 自身が参照する official 256K FAISS tree:

```text
https://github.com/colmap/colmap/releases/download/3.11.1/
vocab_tree_faiss_flickr100K_words256K.bin
SHA-256 96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc
```

3,388 images 級では 32K より識別力の高い 256K tier を使う。Indexing は COLMAP の逐次外側 loop と
FAISS CPU search であり、この phase の GPU utilization が低いのは正常。GPU SIFT matching は indexing
完了後に始まる。Rig verification は小さい sequential / loop graph に 1 回だけ適用し、その後の transitive
extension では individual two-view geometry を使って generalized RANSAC の爆発を避ける。

## SAM3

```toml
[sam3]
repo_path = "D:/models/sam3"
checkpoint_path = "D:/models/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 2048
training_prompt = "person,camera operator,person's shadow"
feature_prompt = "person,camera operator,person's shadow,animal,sky,tree,vehicle,airplane,water"
```

Feature / Training masks は別 Step、別 artifact、別 invalidation。SAM3 inference は native fisheye circle 外を
含む full image context で行い、最後に physical valid region と合成する。完成した PNG は atomic write 後
すぐ running preview index へ追加する。両 SAM3 Step を無効にした export でも physical valid-region mask は
必ず出力し、fisheye padding を training target に含めない。

Video propagation は prompt ごとに再実行するため multi-prompt で遅く、segmented bidirectional でも独立
detection より frame miss が残った。従って default には使わない。

## jpegtran lossless crop

Windows は pinned libjpeg-turbo installer を固定 SHA-256 で導入する。

```powershell
backend\.venv\Scripts\python.exe scripts\install_jpegtran.py
```

Export は fisheye valid circle を含む JPEG MCU boundary で lossless crop し、camera principal point、2D
observation、mask を同じ offset で更新する。jpegtran が無ければ original JPEG を保持し、warning を出す。

## LichtFeld Studio GPU memory

Native distorted / fisheye / ERP dataset は MRNF または MCMC + GUT。IGS+ は GUT と併用できず、ERP を
扱えない。Recommended MRNF は LFStudio UI の安定した default を保ち、segment mask と GUT だけを
追加する。PPISP と novel-view controller は opt-in で、既定では無効。

`undistort=false` では original camera model と distortion coefficient を GUT rasterizer が直接使う。Dataset
load 時に出る `Undistort: source -> destination` は split-view 用 metadata の事前計算であり、training image
を destination size へ変換した意味ではない。Actual training size は image loader の resize log で確認する。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --max-width 2048
```

GUT backward temporary memory は visible splat と image size に比例する。12 GB RTX 4070 Ti の parktest
では 2.4M / 2304 px が controller on / off の両方で OOM、2M / 2048 px が 30,000 iterations 完走した。
従って general memory default は cap 2M / max width 2048。

完走は品質合格を意味しない。旧 `eval/mrnf_optimization_params.json` preset は LFStudio UI default より
means LR 6.4 倍、scaling LR 約 2.86 倍で、parktest の scene scale 22.895 では巨大な空色 splat を生成した。
2M / 2048 の不合格 PLY は 10 m 超 28,600 個、50 m 超 3,238 個を含んだ。PPISP を無効化しても再現する
ため、config generator は v0.5.3 `mrnf_defaults()` を基準にする。

同じ dataset を UI default MRNF + GUT + segment mask、PPISP off、2M / 2048 で再学習すると 30,000 step を
61分18秒で完走した。10 m 超 splat は 12,434、10 m 超かつ opacity 0.5 超は 1,091 へ減少した。ただし
50 m 超が 1,404 残るため、runtime success と数値改善の後にも sky / ground separation の visual acceptance を行う。

疎点群は連続 surface ではない。Export statistics の sparse-point radius median / P95 / P99 / maximum で裾を
確認し、P99 / median が 5 を超える場合は遠景・小視差点が広いことを示す warning として扱う。真の遠景まで
機械的に消さないため、この warning は point を変更しない。

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

Windows runtime は Doctor の次の値をすべて確認する。

```text
colmap.capabilities.gpu_bundle_adjustment = true
colmap.capabilities.gpu_bundle_adjustment_dense = true
colmap.capabilities.gpu_bundle_adjustment_sparse = true
colmap.ceres_cuda = true
colmap.cudss = 0.8.0.10
colmap.cudnn = 9.20.0.48
colmap.capabilities.onnx_cuda_runtime = true
```

Remote deployment は scheduled listener とその worker process tree だけを停止し、machine 上の無関係な
Python process を終了しない。Active artifact publish が完了してから source / frontend を入れ替える。
