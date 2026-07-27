# sphere-reconstruct

Insta360 INSV、equirectangular 360° video/image、perspective video、phone photo を同じ
COLMAP reconstruction に統合し、LichtFeld Studio が直接読める dataset を生成する local Web
application。

Project は 1 個の primary source と任意個の detail source を持つ。Container、media、projection、
camera group、mask purpose を別 field として保持するため、異なる camera model を 1 個の COLMAP
database と reconstruction に混在できる。

## Supported sources

| Input | Media | Projection / COLMAP | Role |
|---|---|---|---|
| Insta360 `.insv` | Video | 2× `OPENCV_FISHEYE` + physical rig | Primary / detail |
| Stitched 360° video | Video | `EQUIRECTANGULAR` | Primary / detail |
| Stitched 360° image folder | Images | `EQUIRECTANGULAR` | Primary / detail |
| Perspective / phone video | Video | `SIMPLE_RADIAL` | Primary / detail |
| Perspective / phone image folder | Images | EXIF group ごとの `SIMPLE_RADIAL` | Primary / detail |

別メーカーの 360 camera は stitched ERP として追加できる。Raw dual-fisheye container は camera 固有の
calibration adapter が必要で、native adapter は現在 Insta360 INSV に対応する。

離散 phone photo に時間同期は不要だが、primary source と重なる texture と parallax が必要。Detail が
primary component へ接続しない場合は source registration statistics に表示し、primary の quality gate
とは分離する。

## Install and start

### Windows

```powershell
.\scripts\start-windows.ps1
```

または:

```cmd
scripts\start-windows.cmd
```

### Linux

```bash
./scripts/start-linux.sh
```

### macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

`http://127.0.0.1:8787` を開く。Start script は frontend build、uv dependency sync、Doctor、server
起動を行う。Windows で COLMAP が無ければ official archive を固定 SHA-256 で導入する。独立した
`glomap` executable は使わず、COLMAP 4.1 の `global_mapper` を使う。

SAM3 を使う場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

Platform / GPU の詳細は [docs/setup-gpu.md](docs/setup-gpu.md) を参照する。

## Workflow and independent Steps

Source Inspector で source type と projection を確認して path を選ぶ。拡張子だけで普通の MP4 を
360° と判定しない。最初の source は primary、追加 source は detail になり、primary は後から
切り替えられる。

```text
Inspect every source
  -> extract video frames / collect still images
  -> confirm source-specific fisheye region
  -> prepare canonical images and camera groups
       ├─> Feature-mask Step ─> feature extraction / matching / SfM
       └─> Training-mask Step ───────────────────────────────┐
  -> primary-source IMU gravity alignment                   │
  -> export registered images + exactly one resolved mask <-┘
```

各 Step は個別に生成、再生成、クリアできる。Source 構成を変更すると source branch と aggregate
reconstruction を dependency graph に従って invalidate する。Scene Hierarchy の photo は source ごとに
独立して折り畳める。Step statistics は Inspector で既定折り畳み。

Frame extraction は video source だけを decode する。離散画像 folder は original file を capture として
直接 catalog し、混合 project でも video source だけに frame-selection parameter を適用する。

## Progress contract

Progress は process memory ではなく SQLite event stream を正とする。Page refresh、WebSocket reconnect、
別 browser から開始した job でも `/stages` snapshot から現在値と activity message を復元する。総量が
まだ分からない処理は 0% と偽装せず indeterminate ring を表示する。数値 progress は Step 全体で単調、
artifact の atomic publish が完了するまでは 99%、manifest / state 更新後だけ 100% になる。

| Step | Reported phases |
|---|---|
| Inspect | source fingerprint activity、MP4/footer、calibration、IMU、source completion |
| Frame extraction | decoder probe、candidate decode、candidate score、optical flow、final output |
| Image preparation | source、image、cubemap view rendering |
| Feature / Training masks | model load、image、prompt、mask write、生成済み画像の live preview |
| Feature extraction | input workspace、camera-group weighted COLMAP extraction、rig setup |
| Matching | database copy、image/block counters、summary |
| Reconstruction | database copy、view-graph calibration、Mapper phase / retry、quality gate |
| Alignment | model / IMU load、coarse/fine time-offset search、transform、preview |
| Export | registered image/mask copy、image/mask validation、config、atomic publish |

細かい counter は Console を埋めない `progress` event、開始・完了・警告・失敗は残る `log` event とする。
Late event は job ID で分離し、前の run の percentage を新しい run へ混ぜない。

## Fast frame extraction

`frame_extraction.hwaccel=auto` は FFmpeg が列挙した backend を信用するだけでなく、実 source の 1 frame
を decode して成功した backend を選ぶ。専用 NVIDIA machine では次を推奨する。

```toml
[frame_extraction]
hwaccel = "cuda"
require_hwaccel = true
score_workers = 0
```

`require_hwaccel=true` は hardware probe failure を明示的な Step error にし、software へ silently fallback
しない。`score_workers=0` は logical CPU 全数を candidate scoring に使う。OpenCV worker は各 1 thread、
SIFT instance は worker-local とし、nested oversubscription と thread-unsafe sharing を避ける。Optical flow
phase では OpenCV の machine-wide thread count を復元する。

INSV spatial mode は candidate decode の時点で 2 stream を同じ FFmpeg / demux pass から出力する。
Selection 後は採用 pair を rename して正式 output にするため、同じ巨大 INSV を final output 用に再走査
しない。Interval / non-cached path も 2 個の FFmpeg process ではなく、1 process / 1 file read で両 lens
を出力する。

### Measured decode evidence

RTX 4070 Ti、3840×3840 HEVC stream、35.96 s sample の同条件 decode:

| Decoder | Speed |
|---|---:|
| CUDA | **8.98× realtime** |
| FFmpeg software HEVC | 0.85× realtime |

同じ source frame を CUDA / software で JPEG (`-q:v 3`) にした逐 pixel 比較は 3840×3840×3 全値が一致し、
MAE 0、maximum error 0。Hardware decode によるこの sample の quality loss は無い。

旧実装の 1133.43 s / 26.804 GB INSV spatial run は 1700 candidates から 1694 pairs を選び、candidate
decode 後に lens ごとの FFmpeg が同じ file をもう 2 回走査したため 58 min 30 s かかった。この値は
software decode + duplicate I/O の baseline であり、現在の実装の expected behavior ではない。

## Image preparation

- INSV native: front/back を source 固有の 2-camera rig にする
- ERP native: `EQUIRECTANGULAR` を維持する
- Pinhole mode: 360° source を cubemap rig にする
- Phone still: EXIF orientation を pixel に適用して orientation tag を除去する
- Phone still: model、resolution、35 mm-equivalent focal が異なる場合は camera group を分ける
- Phone video: 同一 lens/zoom の frame を 1 camera group にし、focal prior を refinement する

Canonical image name は `sources/<source-id>/...`。Source、capture、view、camera group、mask の identity を
filename parsing に依存させない。

Native fisheye の valid-region editor は抽出済み frame 全体を slider / previous / next で切り替えられる。
Front/back lens も独立して選び、複数 frame の edge glare / dirt を確認してから source 固有の円を保存する。

## Two independent SAM3 mask Steps

### Feature masks

Feature extraction / matching の前に適用し、動体や view ごとに変わる領域を広く除外する。Default:

```text
person,camera operator,selfie stick,tripod,person shadow,selfie stick shadow,tripod shadow,
animal,sky,tree,vehicle,airplane,water
```

Feature-mask の再生成は feature、matching、reconstruction、alignment、export を invalidate するが、
Training-mask artifact は保持する。Native fisheye の物理的円形領域は SAM3 とは別 contract で、Feature
mask を無効にしても COLMAP feature extraction へ適用する。

### Training masks

Final LFStudio training 用。Feature mask より少ない対象を除外し、細部を多く残す。Default:

```text
person,camera operator,selfie stick,tripod,person shadow,selfie stick shadow,tripod shadow
```

Training-mask の再生成は Export だけを invalidate し、feature / SfM を再実行しない。両 Step の default
long-edge inference limit は 2048 px。

| Feature masks | Training masks | Feature extraction | Export / LFStudio |
|---|---|---|---|
| On | On | Feature masks | Training masks |
| On | Off | Feature masks | Feature masks |
| Off | On | Physical valid region only | Training masks |
| Off | Off | Physical valid region only | No mask |

Export は常に 0 または 1 mask channel だけを書く。Training が有効なら優先し、無効時だけ Feature を
fallback とする。Photo / Dataset Camera Inspector は両 artifact がある場合に channel を切り替えて
mask / overlay を比較できる。White=keep / black=ignore、image と mask の orientation / size は一致する。

Mask Step は各 PNG を完全に書いた直後、append-only preview index へ record を公開する。Photo Inspector
は running Step を 1 s 間隔で更新するため、全画像の完了を待たず、生成済みの選択画像を mask / overlay
で確認できる。Preview index は running stage だけから読み、成功時の atomic publish 後は final manifest
へ切り替える。両 Step の downsample default は backend / runtime / UI とも 2048 px。

## Feature matching and reconstruction

Pairing `Auto`:

- Single source: Sequential
- Mixed source、500 images 以下: Exhaustive
- 500 images 超: SIFT Vocab-tree。Tree 未設定時は曖昧な fallback をせず error

Matching result は verified pair、cross-source matrix、connected component を記録する。同じ database 上で
複数 matching mode を累積でき、既処理 pair を再計算しない。

- Largest model は primary registered image 数を最優先に選ぶ
- `min_registered_ratio` は primary source に適用する
- Global Mapper が camera-only / quality gate 未達の場合は deterministic seed を最大 3 回試す
- Retry ごとに独立 progress span を割り当て、percentage を巻き戻さない
- Detail source は total / registered / ratio / connected を個別記録する
- IMU alignment は primary INSV の exposure timestamp と accelerometer だけを使う
- Dataset は `-Y up`、Web Scene View は表示時に `+Y up` へ変換する
- GPS の無い SfM scale は primary reference trajectory diameter を 1 model unit に正規化する

## Recommended defaults

| Item | Default |
|---|---|
| Frame selection | Sharpness-first、1 fps、5 candidates |
| Feature masks | On、2048 px、dilation 8 px、expanded prompt |
| Training masks | On、2048 px、dilation 8 px、training prompt |
| Feature | SIFT、long edge 2048 px、maximum 8192 |
| Matcher | Brute-force、pairing Auto |
| Pair validation | maximum 16,384 matches、minimum 15 inliers、guided off |
| Mapper | Global Mapper + view-graph calibration |
| Bundle Adjustment | CPU。Doctor が cuDSS を確認した場合だけ GPU |
| Alignment | Primary IMU auto、reference trajectory normalization |
| LFStudio | MRNF + GUT + resolved mask + PPISP/controller、max cap 1M |

Standard / High preset は実写比較で良かった SIFT + Brute-force を基礎にする。High は 3072 px、
16,384 features、32,768 matches。ALIKED / LightGlue は弱 texture の alternative。

主な parameter:

- `max_image_size`: feature image の長辺上限。0 は COLMAP default
- `max_num_features`: image ごとの feature 上限
- `peak_threshold` / `edge_threshold`: SIFT response / edge filtering
- `affine_shape + DSP`: viewpoint / scale robustness。CPU cost が大きい
- `max_num_matches`: pair ごとの match 上限
- `min_num_inliers`: verified pair を残す最小 inlier
- `guided_matching`: geometry-guided second pass
- `view_graph_calibration`: Global Mapper 前の rotation / intrinsics calibration
- `ba_use_gpu`: CUDA/cuDSS BA。Feature GPU とは別 capability

旧 `SiftExtration.max_image_size` / `SiftExtraction.max_image_size` は無効。COLMAP 4.1 の正しい flag は
`FeatureExtraction.max_image_size`。

## Measured mixed-camera reconstruction

35.96 s X5 recording（35 rig captures / 70 fisheye images）と同 scene の iPhone 16 Pro 10 stills:

| Result | Primary only | + phone photos |
|---|---:|---:|
| Registered primary | 70 / 70 | 70 / 70 |
| Registered phone | — | **10 / 10** |
| Verified cross-source pairs | — | **168** |
| Cameras | 2 | 3 |
| Points3D | 10,544 | **11,268** |
| Observations | 61,140 | **64,133** |
| Mean reprojection | 1.003 px | 1.010 px |
| Median reprojection | 0.937 px | 0.940 px |
| P95 reprojection | 1.880 px | 1.918 px |

Phone images は primary component へ全て接続し、primary registration は低下しなかった。Repeat run では
Global Mapper seed 0 が camera-only model になり、quality-gated seed 1 が 80/80、11,312 points、
64,248 observations、mean 1.012 px を復元した。

38.17 s X5、104 rig captures / 208 images の feature comparison:

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| old Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | **27,202** | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s | **9.7 s** | **17.9 s** | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |

## LFStudio export and training

Export Inspector の 1 directory を dataset root として選ぶ。

```text
export_dataset/
├── images/sources/<source-id>/...
├── masks/sources/<source-id>/...    # resolved channel がある場合だけ
├── sparse/0/{rigs,cameras,frames,images,points3D}.bin
├── preview/
├── train_configs/
└── export_manifest.json
```

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json --data-path <dataset>
```

Training output path は LFStudio 側で管理する。本 application は output directory を作成、表示、移動、
削除しない。Dataset root 内に unmanaged item がある場合は export 再生成を停止する。

Recommended MRNF は GUT、resolved segment mask、PPISP、novel-view controller を有効にする。PPISP は
exposure、vignetting、white balance / color、camera response function の appearance model であり denoiser
ではない。Final PLY を開くときは同名 `.ppisp` sidecar も load する。

RTX 4070 Ti 12 GB、LFStudio v0.5.3 の measured mixed run:

| Metric | Result |
|---|---:|
| Physical cameras / frames / masks | 3 / 80 / 80 |
| Initial SfM points | 11,312 |
| Iterations | 30,000 |
| Runtime / speed | 3,773.552 s / 8.0 iter/s |
| Internal / reloadable Gaussians | 525,433 / 517,387 |
| PLY / PPISP / checkpoint | 128.3 MB / 2.9 MB / 220.7 MB |

Camera icon の green / brown / red は relative photometric-loss heatmap で、red は camera disable を意味しない。
Transform Inspector の 0 は generic SceneNode 値の場合があり、実 pose は `Camera.R/T` と `C=-Rᵀt` にある。

## Training-image denoise decision

同じ夜間素材で FastDVDnet training と raw training を LFStudio 30k まで比較した。

| Training / target | Pose | PSNR | SSIM |
|---|---:|---:|---:|
| FastDVDnet / raw | ALIKED | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **24.920** | 0.8056 |

同じ clean target の SIFT masked PSNR は denoised training 24.478、raw training 25.028。Denoise は約
0.55 dB 悪化したため denoise Step、model download、runtime dependency、UI は削除済み。Export は
original image を使う。

## Runtime config

Default は `runtime/config.toml`。Machine 固有 file は `SPHERE_CONFIG` で指定する。

```toml
[workspace]
root = "./workspace"

[filesystem]
allowed_roots = []

[binaries]
ffmpeg = ""
ffprobe = ""
colmap = ""
vocab_tree = ""

[frame_extraction]
hwaccel = "auto"
require_hwaccel = false
score_workers = 0

[sam3]
repo_path = ""
checkpoint_path = ""
device = "cuda:0"
max_inference_size = 2048
```

## Architecture

```text
project_source table
  ├─ primary source
  └─ N supplemental sources
          ↓
source-scoped inspect / extract / prepare
          ├─ feature-mask artifact -> feature workspace -> matching -> reconstruction -> alignment
          └─ training-mask artifact ----------------------------------------------┐
                                                                                   ↓
                                                       registered-only resolved export
```

Source identity、capture index、camera group、mask purpose、mask path は別 field。新 raw camera を追加する
場合は adapter、calibration、prepare 処理を追加し、global mode branch を増やさない。Stage output は
temporary directory へ書き、成功時だけ atomic replace する。

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

実素材 browser E2E は `SPHERE_E2E_SOURCE` を指定する。

## License

Project 全体は [GNU General Public License v3.0 or later](LICENSE)、SPDX identifier
`GPL-3.0-or-later`。Downloaded third-party component は各 component 自身の license に従う。

## References / legacy plugin

- [COLMAP](https://github.com/colmap/colmap): mixed cameras、multi-rig、Global Mapper、ALIKED、EQUIRECTANGULAR
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): COLMAP loader、GUT、MRNF、mask、PPISP
- [PPISP](https://github.com/nv-tlabs/ppisp): appearance compensation / controller
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata / IMU timestamp
- [Gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container research
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): legacy plugin workflow reference
