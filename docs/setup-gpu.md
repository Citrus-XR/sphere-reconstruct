# GPU / native runtime setup

Platform ごとの native runtime、FFmpeg hardware decode、COLMAP GPU capability、SAM3、LichtFeld Studio の
要件をまとめる。通常は start script に dependency sync と Doctor を任せ、system Python と project venv
を混在させない。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は workspace、filesystem roots、FFmpeg / FFprobe、hardware acceleration method、COLMAP 4.1
capability、optional SAM3、CUDA / cuDNN、cuDSS を検査する。FFmpeg の method list は build capability で、
実 source の decode 可否は Frame extraction 開始時の 1-frame probe で確定する。

## Windows

```powershell
.\scripts\start-windows.ps1
```

Frontend build、Python extras、COLMAP detection / install、Doctor、FastAPI を順に実行する。SAM3 は
明示的に追加する。

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

COLMAP が無ければ `nvidia-smi` に応じて official CUDA / CPU archive を選び、固定 SHA-256 で
`.runtime/tools/` へ atomic install する。自動導入を禁止する場合:

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

NVIDIA は `cuda`、Intel は `qsv`、AMD / Intel Linux は `vaapi` を candidate にできる。VAAPI device
selection が特殊な machine は `hwaccel="vaapi"` を明示し、Doctor と actual source probe の両方を確認する。

## macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

Frame extraction は `videotoolbox` を probe する。COLMAP は CUDA を前提にせず SIFT、CPU Mapper、CPU BA
を基本経路とする。SAM3 は利用 runtime に合わせて optional install する。

## FFmpeg hardware decode

Default:

```toml
[frame_extraction]
hwaccel = "auto"
require_hwaccel = false
score_workers = 0
```

`auto` の candidate priority は CUDA、VideoToolbox、QSV、D3D11VA、D3D12VA、DXVA2、VAAPI、VDPAU。
FFmpeg が method を表示しても driver / codec / pixel format が使えない場合があるため、actual input の
1 frame と software filter transfer を probe する。成功した method だけを本番 command の `-i` 前へ
`-hwaccel <method>` として渡す。

Dedicated NVIDIA machine:

```toml
[frame_extraction]
hwaccel = "cuda"
require_hwaccel = true
score_workers = 0
```

`require_hwaccel=true` の時は probe failure を error にする。Optional mode の software fallback も Console
へ理由を表示し、無表示では切り替えない。

INSV spatial extraction は 2 video stream を 1 FFmpeg process / 1 demux pass で candidate JPEG pair にする。
採用 pair は再 encode せず正式 output へ移す。Interval path も 2 stream を同じ process で出力し、巨大
container を lens ごとに重複読みしない。

RTX 4070 Ti、3840² HEVC sample の測定は CUDA 8.98× realtime、software 0.85× realtime。両 path の
同一 frame JPEG は全 pixel 一致した。Hardware decoder は frame selection、JPEG quality、COLMAP input
resolution を変更しない。

## CPU saturation during frame selection

`score_workers=0` は logical CPU 全数を sharpness / exposure / SIFT candidate scoring に使う。正数なら
worker 数を固定できる。OpenCV の process-global internal threads は worker pool 中だけ 1 にし、各 worker
が独立 SIFT instance を持つ。Pool 後に元の OpenCV thread count を復元するため、sequential optical flow
は library の parallel path を利用できる。

Optical flow selection 自体は前回採用 frame に依存するため、pair decision の外側 loop は sequential。
候補 decode と quality scoring は並列 / hardware 化されるが、この dependency を壊す speculative selection
は結果を変えるため行わない。

## COLMAP camera and rig requirements

Mixed source は 1 database 内で複数 camera model を使う。必要 capability:

- `OPENCV_FISHEYE`、`SIMPLE_RADIAL`、`PINHOLE`、`EQUIRECTANGULAR`
- `feature_extractor --image_list_path`
- Multiple rig configuration
- Global Mapper

Feature extraction は camera group ごとに別 invocation を行い、同じ database に追記する。Phone still は
EXIF orientation を pixel へ適用し、model / resolution / 35 mm focal signature で grouping する。Phone
video は同一 lens/zoom を 1 calibration group にする。

Raw dual-fisheye は calibration adapter が必要。現在は Insta360 INSV。別 camera は stitched ERP または
新 adapter で追加する。

## Matching and scale

Mixed source では temporal adjacency だけでは cross-source edge を作れない。500 images 以下の Auto は
Exhaustive。大規模 SIFT dataset は `binaries.vocab_tree` を指定する。

Global Mapper は悪い focal prior / outlier match に敏感。Phone EXIF focal を維持し、video に EXIF が
無い場合は初期 focal を refinement する。SfM world scale は gauge freedom を持つため、alignment Step は
primary reference trajectory diameter を 1 model unit に正規化する。

## Bundle Adjustment

BA は pose、intrinsics、3D points の reprojection error を同時最適化する。`COLMAP ... with CUDA` は
Ceres CUDA/cuDSS BA を保証しない。次の build は CPU BA を使う。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

Doctor の `colmap.capabilities.gpu_bundle_adjustment` と UI option が連動する。

COLMAP official Windows CUDA archive は COLMAP の CUDA feature を含むが、vcpkg の Ceres dependency に
`cuda` feature を指定していないため、Ceres BA 自体は CPU build になる。`with CUDA` という version
表示だけで BA capability を有効にしない。

Project の pinned CUDA-BA runtime は Ceres 2.3 development commit
[`bac1127`](https://github.com/ceres-solver/ceres-solver/commit/bac1127f9ef672405bd0d2d9c84e809ae89bd239)、
CUDA 13.2、cuDSS 0.8.0.10 を使う。Build は `.github/workflows/build-colmap-cuda-ba.yml`、再現用 script は
`scripts/build_colmap_cuda_ba.ps1`。Runtime metadata と Ceres DLL dependency の両方を Doctor が検査し、
sparse CUDA BA を確認できた package だけ `ba_use_gpu` を有効にする。

## Native ALIKED

COLMAP 4.1.1 は ALIKED N16ROT / N32、Brute-force、LightGlue を内蔵する。Model URL と SHA-256 が
option に含まれるため、通常は自動 download/cache される。固定 model を使う場合:

```toml
[aliked]
extractor_path = "D:/models/aliked-n16rot.onnx"
matcher_path = "D:/models/aliked-lightglue.onnx"
```

Windows ONNX CUDA provider は cuDNN 9 を必要とする。Doctor の `cuda_runtime.cudnn` を確認する。

## SAM3 dual-mask runtime

Feature masks と Training masks は別 Step、別 artifact、別 parameter set。どちらも prepared canonical
image 全てへ同じ projection / orientation で適用する。

```toml
[sam3]
repo_path = "D:/models/sam3-main"
checkpoint_path = "D:/models/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 2048
training_prompt = "person,camera operator,selfie stick,tripod,person shadow,selfie stick shadow,tripod shadow"
feature_prompt = "person,camera operator,selfie stick,tripod,person shadow,selfie stick shadow,tripod shadow,animal,sky,tree,vehicle,airplane,water"
```

Feature masks は COLMAP `ImageReader.mask_path` に渡す。Training masks は SfM に入れず final export で優先
する。Training が無効なら Feature を export fallback にし、両方無効なら mask directory を出力しない。

Native fisheye circle は camera valid-region であり SAM3 channel ではない。Feature masks が無効でも
feature extraction には circle を使う。2 Step を同時に生成すると SAM3 model を各 process で 1 回ずつ
load する。独立性を優先し、Training mask の再生成が feature / matching / reconstruction を invalidate
しない。

各 mask PNG は write 完了後に running-stage preview index へ追加する。Photo Inspector はこの index を
poll し、未完成 file を公開せずに生成済み画像だけを即時 preview する。Cancel / failure 時の temporary
index と PNG は supervisor が削除し、以前の final artifact と混在させない。

## LichtFeld Studio

Mixed-camera COLMAP loader は image ごとに projection を読む。Supported path:

- `PINHOLE` / `SIMPLE_PINHOLE`
- `SIMPLE_RADIAL` / `RADIAL` / `OPENCV` / `FULL_OPENCV`
- `OPENCV_FISHEYE` と supported fisheye variants
- `EQUIRECTANGULAR`

Distorted / fisheye / equirectangular dataset は MRNF/MCMC + GUT を使う。IGS+ は GUT と併用できず、
equirectangular を扱えない。Conservative fallback は 360° source を pinhole cubemap にし、phone camera
を undistort する。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json --data-path <dataset>
```

Recommended MRNF は GUT、resolved segment mask、PPISP、novel-view controller を有効にする。PPISP は
appearance compensation であり denoiser ではない。Final model は `.ply` と同名 `.ppisp` sidecar を
一緒に load する。Training output directory は LFStudio 側で管理し、本 application は制御しない。

## Service deployment

```text
SPHERE_CONFIG=D:/path/to/config.remote.toml
```

更新時は listener の scheduled service と worker tree だけを停止し、machine の Python process を一括
停止しない。Active job 中は deployment を待ち、Source mutation は active job 中に 409 を返す。

## Verification

```bash
cd backend
uv run ruff check src tests
uv run pytest -q
uv run sphere-doctor
```

```bash
cd frontend
pnpm build
pnpm test:e2e
```
