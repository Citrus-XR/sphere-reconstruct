# sphere-reconstruct

Insta360 INSV、equirectangular 360° video/image、perspective video、phone photo を同じ
COLMAP reconstruction に統合し、LichtFeld Studio が直接読める dataset を生成する local Web
application。

Project は 1 個の primary source と任意個の detail source を持つ。Container、media、projection、
camera group を独立した field として保持し、異なる camera model の feature を 1 個の COLMAP
database と reconstruction に統合する。

## 対応 source

| Input | Media | Projection / COLMAP | Role |
|---|---|---|---|
| Insta360 `.insv` | Video | 2× `OPENCV_FISHEYE` + physical rig | Primary / detail |
| Stitched 360° video | Video | `EQUIRECTANGULAR` | Primary / detail |
| Stitched 360° image folder | Images | `EQUIRECTANGULAR` | Primary / detail |
| Perspective / phone video | Video | `SIMPLE_RADIAL` | Primary / detail |
| Perspective / phone image folder | Images | EXIF group ごとの `SIMPLE_RADIAL` | Primary / detail |

別メーカーの 360 camera は stitched ERP として追加できる。Raw dual-fisheye container は camera
固有 calibration が必要なので、現在の native adapter は Insta360 INSV のみ。

離散 phone photo に時間同期は不要だが、primary source と重なる texture と parallax が必要。
Detail が primary component へ接続しない場合は registration statistics に明示し、primary の成功判定
には含めない。

## Install / start

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
`glomap` executable は不要で、COLMAP 4.1 の `global_mapper` を使う。

SAM3 を使う場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

Platform / GPU の詳細は [docs/setup-gpu.md](docs/setup-gpu.md) を参照する。

## Workflow

Source Inspector で source type と projection を確認して path を選ぶ。拡張子だけで普通の MP4 を
360° と判定しない。最初の source は primary、追加 source は detail になり、primary は後から
切り替えられる。

```text
Inspect every source
  -> extract video frames / collect still images
  -> confirm source-specific fisheye region
  -> prepare canonical images and camera groups
  -> generate SAM3 masks
  -> extract features into one COLMAP database
  -> create within-source and cross-source matches
  -> sparse reconstruction
  -> primary-source IMU gravity alignment
  -> registered-only LFStudio export
```

各 stage は個別に生成、再生成、クリアできる。Source 構成を変更すると affected source branch と
aggregate reconstruction だけを invalidation する。Stage statistics は Inspector で既定折り畳み。

### Image preparation

- INSV native: front/back を source 固有の 2-camera rig にする
- ERP native: `EQUIRECTANGULAR` を維持する
- Pinhole mode: 360° source を cubemap rig にする
- Phone still: EXIF orientation を pixel に適用して orientation tag を除去する
- Phone still の model、resolution、35 mm-equivalent focal が異なる場合は camera group を分ける
- Phone video: 同一 lens/zoom の frame を 1 camera group にし、focal prior を refinement する

Canonical image name は `sources/<source-id>/...`。Source、capture、view、camera group、mask の
identity を filename parsing に依存させない。

### SAM3 mask

SAM3 は prepared canonical image ごとに実行する。Perspective、ERP、cubemap は full valid region、
native fisheye は source-specific circle と dynamic mask を合成する。Mask は image relative path を
mirror し、white=keep / black=ignore。Image と mask の orientation / size は常に一致させる。

### Feature matching

既定の pairing `Auto`:

- Single source: Sequential
- Mixed source、500 images 以下: Exhaustive
- 500 images 超: SIFT Vocab-tree。Tree 未設定時は曖昧な fallback を行わず error

Matching result は verified pair、cross-source pair matrix、connected component を記録する。
COLMAP は同じ database 上で複数 matching mode を累積でき、既処理 pair を再計算しない。

### Reconstruction / alignment

- Largest model は primary registered image 数を最優先に選ぶ
- `min_registered_ratio` quality gate は primary source に適用する
- Global Mapper が camera-only / quality gate 未達の場合だけ deterministic seed を最大 3 回試す
- Detail source は total / registered / ratio / connected を個別に記録する
- IMU alignment は primary INSV の exposure timestamp と accelerometer だけを使う
- Dataset は `-Y up`、Web Scene View は表示時に `+Y up` へ変換する
- GPS の無い SfM scale は primary reference trajectory diameter を 1 model unit に正規化する

## 推奨既定値

| 項目 | 既定 |
|---|---|
| Frame selection | Sharpness-first、1 fps、候補 5 |
| SAM3 | 長辺 1024 px、dilation 8 px |
| Feature | SIFT、長辺 2048 px、最大 8192 |
| Matcher | Brute-force、pairing Auto |
| Pair validation | 最大 16,384 matches、最小 15 inliers、guided off |
| Mapper | Global Mapper + view-graph calibration |
| Bundle Adjustment | CPU。Doctor が cuDSS を確認した場合だけ GPU |
| Alignment | Primary IMU auto、reference trajectory normalization |
| LFStudio | MRNF + GUT + masks + PPISP/controller、max cap 1M |

Standard / High preset は実写比較で良かった SIFT + Brute-force を基礎にする。High は長辺 3072、
最大 16,384 features、最大 32,768 matches。ALIKED / LightGlue は弱 texture の alternative。

主な parameter:

- `max_image_size`: feature image の長辺上限。0 は COLMAP 既定
- `max_num_features`: image ごとの feature 上限
- `peak_threshold` / `edge_threshold`: SIFT response / edge filtering
- `affine_shape + DSP`: viewpoint / scale robustness。CPU cost が大きい
- `max_num_matches`: pair ごとの match 上限
- `min_num_inliers`: verified pair を残す最小 inlier
- `guided_matching`: geometry-guided second pass
- `view_graph_calibration`: Global Mapper 前の rotation / intrinsics calibration
- `ba_use_gpu`: CUDA/cuDSS BA。Feature GPU とは別 capability

旧 `SiftExtration.max_image_size` / `SiftExtraction.max_image_size` は無効。COLMAP 4.1 の正しい flag は
`FeatureExtraction.max_image_size`。旧 error が残る project は Feature stage から再生成する。

## Real mixed-camera reconstruction

35.96 秒の Insta360 X5 recording（35 rig captures / 70 fisheye images）と同じ scene の iPhone 16 Pro
10 still images（4224×2376、25 mm equivalent）を native mixed mode で比較した。

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

Phone images は独立 model ではなく primary component へ全て接続した。Primary registration は低下せず、
mean reprojection の増加は 0.007 px。

SAM3 は 80 images 全てを処理し、123 detections、平均 dynamic coverage 7.14%、最大 20.89%、
coverage warning 0。IMU alignment は primary の 35 exposure を使い、median / P90 residual は
1.465° / 2.341°、time offset は -0.03 s。

Feature database を再生成した repeat run では Global Mapper seed 0 が camera-only model になり、
quality-gated seed 1 retry が 80/80、11,312 points、64,248 observations、mean reprojection 1.012 px
を復元した。最終 LFStudio training dataset はこの repeat run を使う。

## LFStudio export and training

Export Inspector の 1 directory を dataset root として選ぶ。

```text
export_dataset/
├── images/sources/<source-id>/...
├── masks/sources/<source-id>/...
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

Mixed distorted camera の推奨 MRNF config は次を有効にする。

```json
{
  "strategy": "mrnf",
  "gut": true,
  "undistort": false,
  "mask_mode": "segment",
  "max_cap": 1000000,
  "use_ppisp": true,
  "ppisp_warmup_steps": 500,
  "ppisp_use_controller": true,
  "ppisp_controller_activation_step": -1,
  "ppisp_freeze_gaussians_on_distill": true
}
```

`-1` activation は 30k training の最後 5,000 step、つまり step 25,000 で controller を有効にする。
Controller distillation 中は Gaussian と per-frame PPISP parameter を固定し、novel view 用 controller
だけを学習する。MCMC / IGS+ config は PPISP 無しの comparison path として維持する。

PPISP は exposure、vignetting、white balance / color、camera response function の差を分離する appearance
model であり、denoiser ではない。PLY だけでは PPISP correction が無いので、結果を開くときは同名
`.ppisp` sidecar も選ぶ。

### Measured PPISP full run

LichtFeld Studio v0.5.3 (`d8c50c6a`)、RTX 4070 Ti 12 GB、repeat mixed dataset を使った headless
30k run:

| Metric | Result |
|---|---:|
| Physical cameras / frames / masks | 3 / 80 / 80 |
| Initial SfM points | 11,312 |
| Iterations | 30,000 |
| Runtime | 3,773.552 s |
| Average speed | 8.0 iter/s |
| Internal final Gaussians | 525,433 |
| Reloadable PLY Gaussians | 517,387 |
| PLY / PPISP / checkpoint size | 128.3 MB / 2.9 MB / 220.7 MB |

Training は `resize_factor=1`、`max_width=3840`。`Undistort: 3840x3840 -> 768x768` log は optional
undistortion target の precompute であり、`undistort=false + GUT` の native fisheye training image を
768 px へ縮小した意味ではない。

Reload smoke は final PLY、PPISP sidecar、元 dataset を別 process で読み、517,387 Gaussians、3 cameras、
80 frames、metadata mapping、3 controller を確認して 1 iteration を error 無しで完了した。

この run は全 80 images を training に使うため evaluation split を無効にしている。従って PSNR / SSIM
は生成されず、この 1 run だけでは PPISP 無しより良いと断定しない。成果は PPISP integration と
reloadability の end-to-end validation であり、photometric quality の比較には同一 split / seed の A/B
run が必要。

Mixed COLMAP camera は LFStudio loader / GUT が image ごとに dispatch できるが、upstream に
mixed-model end-to-end integration test は無い。Conservative fallback は 360° source を pinhole cubemap
へ変換し、phone camera を undistort して homogeneous pinhole dataset にする。

Camera icon の green / brown / red は相対 photometric loss heatmap であり、red は camera disable を
意味しない。Transform Inspector の 0 は generic SceneNode 値の場合があり、実 pose は `Camera.R/T`
と `C=-Rᵀt` にある。

## Previous reconstruction benchmark

38.17 秒の X5、104 rig captures / 208 images、同じ mask / rig:

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | **27,202** | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s | **9.7 s** | **17.9 s** | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |

Final SIFT run は 208/208、27,244 points、mean / median / P95 = 0.913 / 0.846 / 1.732 px、
139,839 observations。

## Training-image denoise decision

同じ夜間素材で FastDVDnet training と raw training を LFStudio 30k まで比較した。

| Training / target | Pose | PSNR | SSIM |
|---|---:|---:|---:|
| FastDVDnet / raw | ALIKED | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **24.920** | 0.8056 |

同じ clean target へ再計算した SIFT masked PSNR は denoised training 24.478、raw training 25.028。
Denoise は約 0.55 dB 悪化したため denoise stage、model download、runtime dependency、UI は削除済み。
Export は original image を使う。

## Runtime config

既定は `runtime/config.toml`。別 file は `SPHERE_CONFIG` で指定する。

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

[sam3]
repo_path = ""
checkpoint_path = ""
device = "cuda:0"
dtype = "bfloat16"
```

## Architecture

```text
project_source table
  ├─ primary source
  └─ N supplemental sources
          ↓
source-scoped inspect / extract / prepare / masks
          ↓
camera-group feature batches → one COLMAP database
          ↓
matching → primary-aware reconstruction → alignment → registered-only export
```

Source identity、capture index、camera group、mask path は別 field で保持する。新しい raw camera を追加する
場合は source adapter、calibration、prepare 処理を追加し、global mode branch を増やさない。

## Test

```bash
cd backend
uv sync --extra dev --extra imaging
uv run ruff check src tests
uv run pytest -q
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
pnpm test:e2e
```

実素材 browser E2E は `SPHERE_E2E_SOURCE` を指定する。

## License

Project license は未確定。外部公開前に決定する。

## References / legacy plugin

- [COLMAP](https://github.com/colmap/colmap): mixed cameras、multi-rig、Global Mapper、ALIKED、EQUIRECTANGULAR
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): COLMAP loader、GUT、MRNF、mask、PPISP
- [PPISP](https://github.com/nv-tlabs/ppisp): physically plausible appearance compensation / controller
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata / IMU timestamp
- [Gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container 調査
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): 旧 plugin workflow 比較対象（GPL-3.0-or-later）
