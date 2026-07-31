# GPU・native runtime setup

Platform ごとの起動、dependency diagnosis、hardware decode、COLMAP CUDA Bundle Adjustment、SAM3、
RoMaV2、LichtFeld Studio の実行条件をまとめる。System Python へ package を追加せず、start script と
`backend` の uv environment を使う。

## Start scripts

### Windows

```powershell
.\scripts\start-windows.ps1
```

Command Prompt は `scripts\start-windows.cmd` を使う。Script は frontend build、uv sync、pinned native
tool install、Doctor、FastAPI の順に実行し、native command が失敗した時点で停止する。起動 URL は
`http://127.0.0.1:8787`。

SAM3 / RoMaV2 を同時に導入する場合:

```powershell
$env:SPHERE_WITH_SAM3="1"
$env:SPHERE_WITH_DENSE="1"
.\scripts\start-windows.ps1
```

RoMaV2 v2.0.1 weights は `.runtime/torch/hub/checkpoints` へ temporary download し、固定 SHA-256 を検証後に
publish する。Torch / model は heavy worker だけが import し、API process に CUDA context を持ち込まない。

Machine 固有 config と auto-install override:

```powershell
$env:SPHERE_CONFIG="D:/path/to/config.toml"
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
$env:SPHERE_SKIP_AUTO_INSTALL_JPEGTRAN="1"
.\scripts\start-windows.ps1
```

### Linux

COLMAP 4.1+、FFmpeg、FFprobe を system package または独自 build で用意する。

```bash
./scripts/start-linux.sh
SPHERE_WITH_SAM3=1 SPHERE_WITH_DENSE=1 ./scripts/start-linux.sh
```

`filesystem.allowed_roots` が空なら user home が file-browser root になる。External drive は config の
`filesystem.allowed_roots` へ明示する。

### macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

VideoToolbox decoder を actual input で probe する。CUDA BA は使わず、GPU/CPU SIFT capability と CPU BA を
Doctor が個別表示する。

## Doctor

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は次を独立 capability として調べる。

- Workspace と file-browser root の read/write
- FFmpeg hardware method と actual source decoder
- COLMAP 4.1、Global Mapper、ALIKED、ERP camera
- Ceres dense CUDA solver と cuDSS sparse CUDA solver
- NVIDIA GPU、driver、compute capability
- FAISS SIFT vocabulary-tree header
- Native ALIKED 用 ONNX CUDA / cuDNN runtime
- Optional SAM3、RoMaV2、jpegtran

`COLMAP ... with CUDA` は GPU feature を示すだけで、Ceres CUDA / cuDSS BA を保証しない。

## Windows COLMAP runtime

`scripts/install_colmap.py` は URL、version、SHA-256 を固定し、temporary directory で検査してから
`.runtime/tools/` を atomic replace する。Binary resolution は explicit config、auto-installed record、PATH
の順なので、導入済み runtime が古い PATH binary に隠れない。

| Variant | Selection | Feature GPU | BA GPU |
|---|---|---:|---:|
| `cuda-ba` | Driver 580+、compute capability 7.5+ | Yes | Ceres CUDA + cuDSS |
| `cuda` | 上記条件を満たさない NVIDIA GPU | Yes | No |
| `cpu` | NVIDIA GPU なし | No | No |

Override / manual install:

```powershell
$env:SPHERE_COLMAP_VARIANT="cuda-ba"
backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cuda-ba
```

Pinned CUDA-BA runtime:

- COLMAP 4.1.1
- Ceres commit `bac1127f9ef672405bd0d2d9c84e809ae89bd239`
- CUDA 12.8、cuDSS 0.8.0.10、cuDNN 9.20.0.48、cuFFT、NVRTC
- ONNX Runtime CUDA provider
- vcpkg commit `6d9d7df564a1ccdaa994e4ad39ccd4a32360867b`
- Native SASS sm75 / sm80 / sm86 / sm89 / sm90、forward PTX compute75
- CLI only、GUI / MVS / CGAL / OpenGL off
- COLMAP PR #4591 の folder-major rig sequential-pairing fix

CUDA 12.8 は current Windows driver で native sm89 を使え、CUDA 13.2 PTX の R595 requirement に依存しない。
Ceres の pinned source が caller の `CMAKE_CUDA_ARCHITECTURES` を上書きするため、build script は repository の
patch で non-empty caller value を保持する。

Official COLMAP 4.1.1 Windows CUDA archive の Ceres は CUDA / cuDSS 無効で、次の warning 後に CPU BA へ
fallback する。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

UI の `ba_use_gpu` は version string ではなく capability metadata、DLL closure、actual mapper log で決める。

### Reproducible local build

```powershell
.\scripts\build_colmap_cuda_ba.ps1 -OutputDirectory dist
```

失敗後に同じ build root を明示的に再利用する時だけ:

```powershell
.\scripts\build_colmap_cuda_ba.ps1 -OutputDirectory dist -ReuseBuildRoot
```

Script は CMake 3.30+、Release try-compile、pinned shallow source を使う。Vcpkg の CPU Ceres を除外し、custom
Ceres を CUDA / CHOLMOD / SPQR / LAPACK / cuDSS 付きで一回 build する。cuFFT / cuDNN / NVRTC / nvJitLink
も app-local に配置し、PE normal / delay-load imports と `LoadLibraryExW` を検査する。Runtime archive は
executable、DLL、capability metadata、license だけを含める。

Repository は CI workflow を持たない。Runtime release は local clean-machine smoke を通した archive だけを
manual publish する。

## FFmpeg decode、PTS、frame selection

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

Priority は CUDA、VideoToolbox、QSV、D3D11VA、D3D12VA、DXVA2、VAAPI、VDPAU。`ffmpeg -hwaccels` は build
capability だけなので、actual input の一 frame decode と software filter transfer を probe する。Required
hardware が失敗した場合は software fallback を隠さず Step error にする。

Raw dual-fisheye は二 stream を一 process / 一 demux pass で decodeする。Extractor は container packet の
PTS / duration を presentation 順へ並べ、二 stream の全 frame count / PTS を 0.5 ms tolerance で検証する。
Selected capture の時刻は `frame_index / fps` ではなく実 PTS。VFR、non-zero start、gap、B-frame を想定する。

Sharpness、exposure、feature count は全 required sensor の worst value、optical-flow motion は maximum を使う。
一方の lens だけ blurred / clipped の capture を選ばない。Spatial candidate JPEG は final quality で生成し、採用時に
再 encode しない。

`score_workers=0` は logical CPU 全数を使う。各 worker の OpenCV internal thread は 1 に固定し、nested
oversubscription を避ける。Decision が前回採用 frame に依存する optical-flow selection 自体は sequential。

RTX 4070 Ti、3840² HEVC の実測は CUDA 8.98× realtime、software 0.85×。同じ frame の output は pixel
MAE 0、maximum difference 0。

## Camera、rig、rolling shutter

Core requirements:

- `THIN_PRISM_FISHEYE`、`OPENCV_FISHEYE`、`SIMPLE_RADIAL`、`PINHOLE`、`EQUIRECTANGULAR`
- `feature_extractor --image_list_path`
- Multi-camera rig と generalized rig verification
- Global Mapper
- Optional native ALIKED / LightGlue

Raw camera metadata は adapter が `camera_system.json` へ正規化する。Core projection / rig code は
`offset_v3`、メーカー名、合成 calibration canvas を知らない。Current Insta360 adapter は per-sensor MEI と
full 6DoF relative extrinsic を出力し、MEI を `THIN_PRISM_FISHEYE` へ近似する。

Raw fisheye は mask / feature より前の mandatory `rectify_fisheye` Step で、同解像度 OPENCV_FISHEYE PNG へ
一回だけ backward resample する。Target pixel → OPENCV ray → source MEI pixel を使い、validity bitmap も同じ
map で変換する。COLMAP と LFStudio は rectified image / camera / rig の一つの contract だけを読む。
192×3840² の実測は 74秒、PNG 1.89 GB。二回補間 roundtrip は 40.47–42.02 dB PSNR、MAE
0.39–0.45 / 255。実 pipeline は一回だけ補間する。PNG encode が主な CPU / disk cost になる。

X5 metadata field 27 の window crop は 5376²→5312² centered crop。各辺 32 px を引いてから decoded size へ
scale する。これを省くと focal が 1.204819% 小さくなり、principal point はほぼ同じまま single-lens shape が
曲がる。Crop と rig extrinsics は独立で、crop は `cam_from_rig` を変更しない。

COLMAP と LFStudio v0.5.3 は同名 THIN_PRISM model の tangential / prism 適用位置が異なる。Joint fit の実測:

| Sensor | COLMAP RMS | LFStudio RMS | Combined maximum |
|---|---:|---:|---:|
| lens0 | 0.0396 px | 0.0421 px | 0.247 px |
| lens1 | 0.1029 px | 0.1112 px | 0.590 px |

Consumer ごとの residual を保存し、1 px を超える近似は native default にしない。Valid region は user physical
circle と calibrated forward ray (`theta < 89.55°`) の積であり、scalar radius だけで hemisphere を切らない。
基準円中心は `(0.5,0.5)` 固定、sensor ごとの ordered add/subtract brush operation で任意領域を調整する。

Current X5 readout は 21.244001 ms。現在の frame selection は `|omega| × readout` で high-risk frame を避ける
だけで、rolling-shutter correction ではない。Correction には sensor ごとの scan direction、frame timestamp
reference、encoded crop、`R_rig_from_imu`、gyro bias、video↔IMU offset / drift が必要。Unknown value を仮定して
top-to-bottom remap を実行しない。詳細は [adding-360-camera-formats.md](adding-360-camera-formats.md)。

Opaque stitched ERP は元 sensor / scan row の二次元 time-map が無いため、単一 ERP-row rolling-shutter model を
適用しない。

## Matching と FAISS vocabulary tree

Single-source video は Sequential + loop closure + one-pass transitive。COLMAP 4.1.1 の folder-major sensor
boundary bug は upstream [PR #4591](https://github.com/colmap/colmap/pull/4591) の same-camera guard を
backport する。Runtime を切り替えた後は旧 match row を再利用せず database を clean rebuild する。

COLMAP 4.1 は FLANN から FAISS へ移行した。旧 tree は file が存在しても crash するので、installer / Doctor は
header `{version: 1|2, desc: 128, embedding: 64}` を検証する。既定 tree:

```text
https://github.com/colmap/colmap/releases/download/3.11.1/
vocab_tree_faiss_flickr100K_words256K.bin
SHA-256 96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc
```

FAISS indexing は CPU / sequential outer loop なので GPU utilization が低いのは正常。GPU SIFT pair matching は
indexing 後に始まる。Rig verification は小さい sequential / loop graph に一回適用し、transitive extension 後の
巨大 graph に generalized RANSAC を再実行しない。

## Metric scale と ground

Fixed `cam_from_rig` の sensor spacing が設定 baseline と一致することは、metric scale の独立 evidence ではない。
同 capture の複数 sensor が十分な共有 3D point / parallax を持つ場合だけ baseline が scale を拘束する。背面
dual-fisheye のように overlap が弱い source は VIO、control point、overlapping stereo なしで meter と表示しない。

Gravity Step は rotation だけ、Ground Step は Y translation だけを適用する。IMU acceleration 二重積分を scale
restoration に使わない。

## SAM3 と jpegtran

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

Feature / Training mask は別 Step / artifact。Inference は physical circle 外を含む full image context で行い、
出力だけを geometric valid region と交差する。PNG は atomic publish 後すぐ running preview index に追加する。
Propagation は multi-prompt で遅く independent detection より miss が残ったため default にしない。

Windows jpegtran installer:

```powershell
backend\.venv\Scripts\python.exe scripts\install_jpegtran.py
```

Rectified fisheye PNG は validity bitmap bounds で lossless pixel crop し、principal point、2D observation、mask
を同じ offset で更新する。Perspective JPEG の MCU crop には jpegtran を使う。

## LichtFeld Studio

Native distorted / fisheye / ERP dataset は MRNF または MCMC + GUT を使う。Recommended default:

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --max-width 2048 --headless --train
```

- MRNF v0.5.3 UI defaults
- GUT on、`undistort=false`
- resolved segment mask
- PPISP / controller off
- Mandatory OPENCV_FISHEYE rectification completed before SfM
- max width 2048
- general cap 2M、30,000 iterations

`Undistort: source -> destination` log は loader の hypothetical crop metadata であり、training tensor resize では
ない。Actual size は `Image info` log を見る。

Stock LFStudio の THIN_PRISM inverse は non-radial delta を前回 UV から五回減算し、forward と inverse が
一致しない。Lens1 は 2048 scale で maximum 約 11.4 px。Mandatory rectification が RGB / validity / camera を
feature extraction 前に一緒に OPENCV grid へ変換するため、stock LFStudio でもこの inverse を通らない。
Legacy THIN dataset を直接使う場合だけ source patch が必要。`undistort=true` は別の prism packing bug がある。

12 GB RTX 4070 Ti の Parktest は 2.4M / 2304 で OOM、2M / 2048 で完走。旧 eval preset は UI default より
means LR 6.4 倍、scaling LR 約 2.86 倍で巨大な sky splat を生成した。PPISP off でも再現したため、config は
`mrnf_defaults()` を基準にする。PPISP は appearance model で denoiser ではなく opt-in。

RoMaV2 dense seed は training-camera PSNR / SSIM を改善したが free-view sharpness を安定して改善せず、runtime
も増えたため default off。Sparse point cloud は surface ではなく feature track sample なので、地面が連続面に
見えないこと自体は異常ではない。

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

Windows CUDA-BA runtime は少なくとも次を確認する。

```text
colmap.capabilities.gpu_bundle_adjustment = true
colmap.capabilities.gpu_bundle_adjustment_dense = true
colmap.capabilities.gpu_bundle_adjustment_sparse = true
colmap.ceres_cuda = true
colmap.cudss = 0.8.0.10
colmap.cudnn = 9.20.0.48
colmap.capabilities.onnx_cuda_runtime = true
```

Remote deployment は listener とその worker process tree だけを停止し、無関係な process を終了しない。Active
artifact publish 完了後に source / frontend を入れ替える。
