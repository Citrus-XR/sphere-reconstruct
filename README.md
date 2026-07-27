# sphere-reconstruct

Insta360 INSV、360° equirectangular 動画/画像、通常の camera/phone 動画、離散写真を同じ
COLMAP reconstruction に統合し、LichtFeld Studio が直接読める dataset を生成する local Web
application。

Project は 1 個の primary source と任意個の detail source を持つ。各 source の container、media、
projection を分離して扱い、camera model が異なる場合も source ごとの feature batch を同じ COLMAP
database に追加する。

## 対応 source

| 入力 | Media | Projection / COLMAP | 用途 |
|---|---|---|---|
| Insta360 `.insv` | Video | 2× `OPENCV_FISHEYE` + physical rig | Primary / detail |
| Stitched 360° video | Video | `EQUIRECTANGULAR` | Primary / detail |
| Stitched 360° image folder | Images | `EQUIRECTANGULAR` | Primary / detail |
| Phone / perspective video | Video | `SIMPLE_RADIAL` | Detail / primary |
| Phone / perspective image folder | Images | EXIF group ごとの `SIMPLE_RADIAL` | Detail / primary |

別メーカーの 360 camera は stitched ERP video/image として追加できる。Raw dual-fisheye container は
camera 固有 calibration が必要なため、現在の native adapter は Insta360 INSV のみ。

Phone photo は時間連続でなくてもよい。主 reconstruction と視覚的に重なる texture と parallax が
必要で、各 detail はできれば 3 view 以上で観測する。接続できなかった source は registration 統計で
明示され、primary reconstruction の成功判定には含めない。

## Quick start

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

`http://127.0.0.1:8787` を開く。Start script は frontend build、uv dependency sync、Doctor、
server 起動を行う。Windows で COLMAP が無ければ公式 package を固定 SHA-256 で導入する。独立
GLOMAP executable は不要で、COLMAP 内蔵 `global_mapper` を使う。

SAM3 を有効にする場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"; .\scripts\start-windows.ps1
```

Platform / GPU 詳細は [docs/setup-gpu.md](docs/setup-gpu.md) を参照する。

## Mixed-source workflow

Source Inspector で type を明示して path を選ぶ。拡張子だけで普通の MP4 を 360° と推測しない。
最初の source は primary、以後は detail になる。Primary は後から切り替えられる。

```text
Source inspection
  -> Frame extraction for every video / collect every still
  -> Fisheye region confirmation       (primary native dual-fisheye)
  -> Prepare images and camera groups
       - preserve native fisheye / ERP, or
       - reproject every 360° source to pinhole
       - EXIF-orient phone photos
  -> SAM3 masks for every prepared image (optional)
  -> Feature extraction per camera group into one database
  -> Cross-source feature matching
  -> Sparse reconstruction
  -> Primary-source IMU gravity alignment
  -> LFStudio export (registered images only)
```

各重い stage は個別に生成・クリアできる。Source 構成を変更すると aggregate reconstruction は
明示的に invalidation される。Stage statistics は Inspector で既定折り畳み。

### Camera preparation

- INSV native: front/back を source 固有の 2-camera rig にする
- ERP native: `EQUIRECTANGULAR` を維持する
- Pinhole mode: すべての 360° source を cubemap rig にする
- Phone photo: EXIF orientation を pixel に適用して orientation tag を除去する
- 同じ phone folder 内でも model、resolution、35mm-equivalent focal が異なる画像は別 camera group
- Phone video: 抽出 frame は同じ camera group とし、EXIF が無ければ conservative focal prior を使う

Canonical image name は `sources/<source-id>/...`。同名 frame が複数 source にあっても capture、mask、
camera、registration が衝突しない。

### Matching

既定 `Auto`:

- Single source: Sequential
- Mixed source、500 images 以下: Exhaustive
- 500 images 超: SIFT Vocab-tree。未設定なら曖昧な fallback をせず error

Matching statistics は verified cross-source pair と source-pair matrix を記録する。複数 matching mode
は同じ database に累積でき、COLMAP は既処理 pair を再計算しない。

### Reconstruction / alignment / export

- Largest model は primary registered image 数を最優先に選ぶ
- `min_registered_ratio` quality gate は primary に適用する
- Global Mapper が camera-only / quality gate 未達の場合だけ、連続する deterministic seed を最大 3 回試す
- Detail source は source ごとに total / registered / ratio / connected を記録する
- IMU alignment は primary INSV だけを使い、別 recording の IMU を混ぜない
- Export は registered image と対応 mask だけをコピーする
- LFStudio validation は camera model、image/mask size、trajectory、unique center を再検査する

## Real mixed-camera result

35.96 秒の Insta360 X5 recording（35 rig captures / 70 fisheye images）と、同じ scene を撮った
iPhone 16 Pro の 10 枚の離散写真（4224×2376、25 mm equivalent）を Native mixed mode で比較。

| Result | Primary only | + 10 phone photos |
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

Phone images は独立 model ではなく primary component に全て接続した。Point は 724 増え、primary
registration は低下せず、mean reprojection の増加は 0.007 px。

SAM3 は 80 images 全てを処理し、123 detections、平均 dynamic coverage 7.14%、最大 20.89%、
coverage warning 0。IMU alignment は primary の 35 exposure のみを使い、median / P90 residual =
1.465° / 2.341°、time offset = -0.03 s。

Export は 80 images / 80 masks、camera models = `OPENCV_FISHEYE + SIMPLE_RADIAL`、loadable = true、
training_ready = true、warning 0。MRNF 推奨 config は mixed distortion に合わせ GUT を有効にする。
LichtFeld Studio v0.5.3 の headless smoke training は 3 camera / 80 images / 80 masks / 11,268
initial Gaussians を実際に load し、MRNF + GUT + segment mask の 1 iteration と checkpoint / PLY
保存を error 無しで完了した。

Feature database を再生成した repeat run では Global Mapper seed 0 が 80 cameras / 0 points の
camera-only model になった。Quality-gated retry が自動で seed 1 を実行し、80/80、11,312 points、
64,248 observations、mean reprojection 1.012 px を復元した。これにより mapper の確率的失敗を
手動再実行へ委ねず、成功 run の追加 cost は増やさない。

この実測から、今回の離散 phone photo は detail source として有効と判断する。ただし一般素材では
overlap が無い写真は登録されず、detail が必ず品質を改善するわけではない。

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
| Alignment | Primary IMU auto、primary reference trajectory を正規化 |
| LFStudio | MRNF + GUT（distorted / mixed camera）、max cap 1M |

標準・高品質 preset は実写比較で良かった SIFT + Brute-force を基礎にする。High は長辺 3072、
最大 16,384 features、最大 32,768 matches。ALIKED / LightGlue は弱 texture の alternative。

主な parameter:

- `max_image_size`: feature 抽出の長辺上限。0 は COLMAP 既定
- `max_num_features`: image ごとの feature 上限
- `peak_threshold` / `edge_threshold`: SIFT response / edge filtering
- `affine_shape + DSP`: viewpoint/scale robustness。CPU cost が大きい
- `max_num_matches`: pair ごとの match 上限
- `min_num_inliers`: verified pair を残す最小 inlier
- `guided_matching`: geometry-guided second pass
- `view_graph_calibration`: Global Mapper 前の rotation/intrinsics calibration
- `ba_use_gpu`: CUDA/cuDSS BA。Feature GPU とは別 capability

`Failed to parse options - SiftExtration.max_image_size` / `SiftExtraction.max_image_size` の旧 run は
Feature stage だけを再生成する。COLMAP 4.1 の正しい flag は
`FeatureExtraction.max_image_size`。

## Previous reconstruction benchmark

38.17 秒の X5、104 rig captures / 208 images、同じ mask / rig:

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | **27,202** | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s | **9.7 s** | **17.9 s** | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |

最終 SIFT run は 208/208、27,244 points、mean / median / P95 = 0.913 / 0.846 /
1.732 px、139,839 observations。

## Training image denoise decision

同じ夜間素材で FastDVDnet training と raw training を LFStudio 30k まで比較した。

| Training / target | Pose | PSNR | SSIM |
|---|---:|---:|---:|
| FastDVDnet / raw | ALIKED | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **24.920** | 0.8056 |

同じ clean target へ再計算した SIFT masked PSNR は denoised training 24.478、raw training
25.028。Denoise は約 0.55 dB 悪化したため、denoise stage、model download、runtime dependency、
UI は削除済み。Export は原画を使う。

## Gravity alignment

Primary INSV の exposure timestamp と accelerometer を対応させ、24 個の right-handed signed-axis
mapping から robust consensus を選ぶ。`model_transformer` で rig / frame / camera / point 全体を
変換する。Dataset は `-Y up`、Web viewer は `+Y up`。GPS が無い scale は primary reference
trajectory diameter を 1 model unit に正規化する。

## LFStudio

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

Training output path は LFStudio 側で管理する。本 application は作成、表示、移動しない。
`export_dataset/` 内に unmanaged item がある場合は再生成前に停止し、外部結果を削除しない。

Mixed COLMAP camera は LFStudio loader / GUT が image ごとに dispatch でき、今回の native mixed
dataset は validation を通過した。一方 upstream に mixed-model end-to-end integration test は無く、
heterogeneous training は experimental。最も conservative な fallback は全 360° source を pinhole
cubemap にし、phone camera を undistort して homogeneous pinhole dataset にする。

Camera icon の green / brown / red は相対 photometric loss heatmap で、red は camera disable を
意味しない。Transform Inspector の 0 は generic SceneNode 値の場合があり、実 pose は `Camera.R/T`
と `C=-Rᵀt` にある。本 UI Camera Inspector は実 center と source を表示する。

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
inspect / extract / prepare canonical image catalog / masks
          ↓
camera-group feature batches → one COLMAP database
          ↓
matching → primary-aware reconstruction → alignment → registered-only export
```

Source identity、capture index、camera group、mask path は別 field で保持し、filename parsing を
pipeline contract にしない。Raw adapter を増やす場合は source adapter と prepare 処理を追加する。

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
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): COLMAP loader、GUT、MRNF、mask
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata / IMU timestamp
- [Gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container 調査
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): 旧 plugin workflow 比較対象（GPL-3.0-or-later）
