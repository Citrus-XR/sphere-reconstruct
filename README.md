# sphere-reconstruct

Insta360 dual-fisheye INSV、equirectangular 動画、ERP 画像列から、COLMAP sparse model と
LichtFeld Studio が直接読み込める dataset を生成するローカル Web application。

Source 検査、frame 抽出、再投影、mask、feature、matching、Mapper、重力整列、export を独立
stage として扱う。各 stage は個別に生成・消去でき、入力または parameter が変わった consumer
だけを artifact DAG に従って無効化する。

## 主な機能

- 生の前後魚眼を `OPENCV_FISHEYE` 2-camera rig として解く Native fisheye mode
- Dual fisheye / ERP を cubemap へ変換する Pinhole rig fallback
- COLMAP 4.1 `EQUIRECTANGULAR` camera による ERP direct reconstruction
- COLMAP native SIFT / ALIKED N16ROT / ALIKED N32
- Brute-force / LightGlue と Sequential / Exhaustive / Vocab-tree の独立選択
- COLMAP Global Mapper / Incremental Mapper、view-graph calibration、CPU/GPU BA
- SAM3 dynamic mask と画像から推定する fisheye valid circle
- Exposure timestamp と同期した per-frame IMU gravity alignment
- LFStudio が直接選択できる `export_dataset/`
- Stage cache、transitive invalidation、atomic publish、worker isolation
- 各 stage の実行統計、3D viewer、photo/camera Inspector
- 日本語・中国語・英語 UI と dockable layout
- Windows / Linux / macOS start script と dependency Doctor

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

起動後に `http://127.0.0.1:8787` を開く。Start script は frontend build、uv dependency sync、
Doctor の順に実行してから server を開始する。Windows で COLMAP が無い場合は公式 4.1.1
package を固定 SHA-256 で取得する。独立した GLOMAP package は不要で、COLMAP 内蔵
`global_mapper` を使う。

SAM3 を利用する場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"; .\scripts\start-windows.ps1
```

Platform 別の詳細は [docs/setup-gpu.md](docs/setup-gpu.md) を参照する。

## Workflow

上部の「次の工程」は、未生成または stale の最初の stage を 1 つ実行する。魚眼有効領域のような
手動確認ではそこで停止する。

```text
Source inspection
  -> Frame extraction
  -> Fisheye region confirmation       (Native fisheye)
  -> Pinhole reprojection              (Pinhole rig)
  -> SAM3 masks                        (optional)
  -> Feature extraction
  -> Feature matching
  -> Sparse reconstruction
  -> Gravity alignment
  -> LFStudio export
```

Feature matching だけを変更しても frame decode と feature extraction は再実行しない。Mapper や
BA だけを変更した場合は reconstruction 以降だけが stale になる。各 Inspector の「生成」「再生成」
「クリア」で stage 単位に操作できる。

旧 run が `Failed to parse options - SiftExtration.max_image_size` または
`SiftExtraction.max_image_size` で停止した場合は「特徴抽出」だけをクリアして再生成する。COLMAP
4.1 の正しい option は `FeatureExtraction.max_image_size` で、実装と regression test はこの
namespace を使用する。

## 推奨既定値

| 項目 | 既定 |
|---|---|
| Frame selection | Sharpness-first、1 fps、候補 5 枚 |
| SAM3 | 長辺 1024 px、dilation 8 px |
| Feature | SIFT、長辺 2048 px、最大 8192 features |
| Matcher | Brute-force、Sequential overlap 4 |
| Pair validation | 最大 16,384 matches、最小 15 inliers、guided matching off |
| Mapper | Global Mapper + view-graph calibration |
| Bundle Adjustment | CPU。Doctor が cuDSS capability を確認した場合だけ GPU を選択可能 |
| Alignment | Exposure-synchronized IMU auto、軌跡径を 1 model unit に正規化 |
| LFStudio | MRNF + GUT、`max_cap=1,000,000` |

標準と高品質 preset は、実写比較で最良だった SIFT + Brute-force を共通の基礎にする。高品質 preset
は長辺 3072 px、最大 16,384 features、最大 32,768 matches とし、時間と memory を多く使う。
ALIKED + LightGlue は極端に弱い texture の代替として手動選択できるが、暗所だから常に優位とは
限らない。

主な詳細 parameter の意味:

- `max_image_size`: feature 抽出時の長辺上限。0 は COLMAP 既定
- `max_num_features`: 1 image に保持する feature 上限。増やすと時間と memory も増える
- `peak_threshold`: SIFT の低 contrast feature を除く閾値。0 は COLMAP 既定
- `edge_threshold`: 線状 edge feature の保持を制御。0 は COLMAP 既定
- `affine_shape + DSP`: viewpoint / scale 耐性を上げるが CPU cost が大きい
- `max_num_matches`: 1 image pair で幾何検証する対応点上限
- `min_num_inliers`: 幾何検証済み pair を残す最小 inlier 数
- `guided_matching`: 初回 geometry を使う 2 回目の探索。難しい pair 向けだが低速
- `view_graph_calibration`: Global Mapper 前に相対回転と intrinsics を調整
- `ba_use_gpu`: 同じ BA 目的関数を CUDA/cuDSS で解く。SIFT GPU とは別 capability

## 実写 reconstruction benchmark

入力は 38.17 秒の Insta360 X5 dual 3840×3840 HEVC。104 rig frame / 208 image を同じ
camera rig と mask で比較した。

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | **27,202** | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s¹ | **9.7 s** | **17.9 s** | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s¹ | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |
| COLMAP ALIKED + Brute-force + Incremental | 243.0 s¹ | 9.7 s | 88.2 s | 208/208 | 25,634 | 1.112 px |

¹ 4096-feature ALIKED extraction を共有した比較値。

最終 SIFT 再実行は 208/208 image、27,244 points、mean / median / P95 reprojection =
0.913 / 0.846 / 1.732 px、139,839 observations。直前の ALIKED 結果は 10,733 points、
0.945 / 0.813 / 1.866 px、59,974 observations だった。

UI E2E の 37 rig frame / 74 image は 74/74 登録、5,761 points、mean / median / P95 =
0.872 / 0.781 / 1.782 px、gravity residual median / P90 = 1.599° / 3.537°。Project 作成から
export まで 3.4 分で、旧約 44.8 分から約 13.2 倍高速化した。主因は native matcher と、322 回の
random seek を stream 単位の sequential decode に置き換えたことにある。

Pinhole fallback は 12 時点 × 12 virtual camera = 144/144 image、1,667 points、mean
reprojection 1.085 px。合成 ERP 12 枚は 12/12 image、7,720 points、mean / median / P95 =
0.386 / 0.304 / 0.989 px。Camera-only model は quality gate で失敗にする。

## LFStudio training benchmark とノイズ除去の結論

同じ 208 image を LFStudio v0.5.3、MRNF、GUT、binary segment mask、3840²、30,000 iteration
で比較した。

| Training image / eval target | Feature pose | Gaussians | Eval PSNR | Eval SSIM |
|---|---:|---:|---:|---:|
| FastDVDnet / raw | ALIKED | 870,956 | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 879,324 | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 1,000,000 | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **1,000,000** | **24.920** | 0.8056 |

Saved render を同じ clean target に対して再計算した masked PSNR は、ALIKED pose で
FastDVDnet training 24.223 / raw training 24.489、SIFT pose で FastDVDnet training
**24.478** / raw training **25.028**。今回の夜間室内素材では、時間方向ノイズ除去により平面
noise は減った一方、multi-view reconstruction に必要な細部も失われ、原画より約 0.55 dB 悪化した。

この結果を受け、training image denoise stage、FastDVDnet / FFmpeg fallback、model download、
設定、UI、依存 package は削除した。Export は常に feature extraction と同じ原画を使用する。
Single-view の見た目が滑らかになることは、multi-view training の最終品質向上を意味しない。

7,000 iteration の既存 checkpoint は学習不足で、30,000 iteration の約 1/4 しか完了していなかった。
また 500k cap は残り約 17k iteration で topology を増やせないため、12GB GPU で検証した既定を
1M とする。Fresh 1M / 30k run は約 59–69 分。MRNF の grow/prune は時刻 seed を含むため、run
間に小さな揺らぎがある。

## Reconstruction mode

### Native fisheye

前後画像を `OPENCV_FISHEYE` camera として使用する。Front を rig reference、Back を 180°
rotation + metadata baseline として固定する。再投影による情報損失がなく、2 lens に視覚 overlap
がなくても同じ exposure の rig として 1 model に統合する。

### Pinhole rig

Dual fisheye / ERP を 6-view cubemap にする fallback。Known intrinsics / extrinsics を固定でき、
IGS+ や pinhole-only downstream に向く。

### Equirectangular

COLMAP 4.1 camera model ID 17 を使用する。LFStudio config は GUT 対応 MRNF / MCMC だけを生成
する。IGS+ は GUT と併用できず equirectangular を undistort できないため候補から除外する。

## Gravity alignment

COLMAP world orientation の gauge freedom を、INSV の accelerometer と各 exposure timestamp で
解く。

- `first_frame_timestamp`、`gyro_timestamp`、`is_raw_gyro` を metadata から読む
- 全 recording の device-frame acceleration を単純平均しない
- 24 個の右手系 signed-axis mapping から robust consensus を選ぶ
- Inlier、median / P90 angular residual、time offset を記録する
- `model_transformer` で rig、frame、image、point 全体を変換する
- GPS が無い scale は任意なので reference-camera trajectory diameter を 1 に正規化する

Dataset は `-Y up`、Web viewer は `diag(1,-1,-1)` で `+Y up` 表示する。UI の位置単位は meter
ではなく model unit。

## LFStudio で開く

Export Inspector に表示される次の 1 directory を dataset root として選ぶ。

```text
<workspace>/projects/<project-id>/export_dataset/
```

```text
export_dataset/
├── images/
├── masks/
├── sparse/0/
│   ├── rigs.bin
│   ├── cameras.bin
│   ├── frames.bin
│   ├── images.bin
│   └── points3D.bin
├── preview/
├── train_configs/
└── export_manifest.json
```

CLI の再現可能な dataset / config 指定:

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json --data-path <dataset>
```

LFStudio の training output directory は LFStudio 側で選択する。本アプリケーションはその path を
作成、表示、変更、移動しない。誤って `export_dataset/` 内に外部 output を作った場合、再生成や
クリアによる消失を防ぐため、管理外 item を検出して明示的に停止する。別 directory へ移動してから
export を操作する。

LFStudio v0.5.3 の GUI folder import は `train_configs/` を自動適用しない。Folder と camera pose
は読み込めるが、GUI training では MRNF、GUT、Segment mask を手動設定する。Binary mask は
white=valid / black=excluded。`mask_mode="segment"` は masked pixel を loss から外す。

### Camera の色

LFStudio の camera icon は per-camera photometric loss の相対 heatmap。Green は現在の最小、red
は最大、brown / orange は中間、white は未観測。Min / max は更新ごとに再正規化されるため、red
は pose failure や camera disable を意味しない。Selected / hovered camera も orange になる。

### Transform Inspector が 0 の場合

LFStudio は dataset camera pose を SceneNode transform ではなく `Camera.R/T` に保持し、center を
`C=-Rᵀt` で計算する。Generic Transform Inspector が local node の `0,0,0` を表示しても pose は
失われていない。本アプリケーションの Camera Inspector は実 center を表示し、export validation
は unique center 数と trajectory diameter を検査する。

## Stage statistics

各 stage は `manifest.extra` に利用可能な統計を保存し、対応する Inspector に全 scalar statistic を
表示する。

- Source: container、calibration、gravity sample、解像度、尺
- Frames: selected / candidates / rejection、FPS、時間範囲
- Reproject / masks: render 数、有効領域、coverage、detection、warning
- Features / matching: keypoint、descriptor、pair、inlier
- Reconstruction: registration、point、observation、track、reprojection、trajectory
- Alignment: consensus residual、inlier、time offset、normalization
- Export: camera model、mask、LFStudio validation、training profile、推奨 cap

LFStudio の loss / PSNR / SSIM は外部 training process の値であり、output directory も管理しないため
自動取得しない。Export Inspector には「外部管理のため取得不可」と明示する。

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

## Camera / source 拡張

Source 固有ロジックは `colmap/input_workspace.py` の builder に隔離する。新 camera は共通
`InputSpec` として images、optional masks、camera model / intrinsics、optional rig config、refine
policy を返す。Feature、Matcher、Mapper、Alignment、Export は source format を再解釈しない。
Fisheye circle は X5 固定値ではなく画像から推定し、UI で確認できる。

## Architecture

```text
React + TypeScript + Vite + Playwright
                  |
FastAPI + aiosqlite + WebSocket
                  |
multiprocessing spawn worker
                  |
FFmpeg / COLMAP / ONNX Runtime / optional SAM3
```

FastAPI process は heavy CUDA model を import しない。Stage は temporary directory に書き、成功後
だけ atomic replace する。Invalidation は線形順序ではなく artifact DAG を辿る。外部 application
の training result は stage ownership に含めない。

## Test

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
pnpm test:e2e
```

実 INSV の browser E2E:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 \
SPHERE_E2E_SOURCE=/absolute/path/to/video.insv \
pnpm test:e2e
```

## License

Project license は未確定。外部公開前に project 本体の license を決定する。

## References / legacy plugin

- [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): loader、GUT、mask、MRNF、loss heatmap
- [colmap/colmap](https://github.com/colmap/colmap): Global Mapper、ALIKED、rig、EQUIRECTANGULAR
- [AdrianEddy/telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata / IMU timestamp
- [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [BenjaminHenriksson/insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container 調査の参考
- [alexmgee/lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): 旧 plugin workflow の比較対象（GPL-3.0-or-later）
