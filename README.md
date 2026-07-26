# sphere-reconstruct

Insta360 dual-fisheye INSV、equirectangular 動画、ERP 画像列から COLMAP sparse model と
LichtFeld Studio 用 dataset を生成するローカル Web application。

画像抽出、mask、feature、matching、Mapper、gravity alignment、training image denoise、export
を独立 stage として扱う。各 stage は個別に生成・クリアでき、parameter や入力 fingerprint が
変わった consumer だけを無効化する。

## 機能

- 生の前後魚眼を `OPENCV_FISHEYE` 2-camera rig として解く Native fisheye mode
- Dual fisheye / ERP を cubemap へ変換する Pinhole rig fallback
- COLMAP 4.1 `EQUIRECTANGULAR` camera による ERP direct reconstruction
- Native SIFT / ALIKED N16ROT / ALIKED N32
- Brute-force / LightGlue と Sequential / Exhaustive / Vocab-tree の独立選択
- COLMAP Global Mapper / Incremental Mapper
- SAM3 dynamic mask と画像から推定する fisheye valid circle
- Exposure timestamp に同期した per-frame IMU gravity alignment
- FastDVDnet 5-frame temporal denoise と FFmpeg adaptive fallback
- LFStudio が直接開ける `export_dataset/` と安全な `training_outputs/`
- Stage cache、transitive invalidation、atomic publish、worker isolation
- 日本語・中国語・英語 UI、dockable layout、camera/point viewer
- Windows/Linux/macOS start script と dependency Doctor

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

起動後は `http://127.0.0.1:8787` を開く。全 script は frontend build、uv dependency sync、
Doctor を実行してから server を開始する。Windows で COLMAP が無ければ公式 4.1.1 package を
固定 SHA-256 で取得する。独立 GLOMAP package は使わず、COLMAP 内蔵 `global_mapper` を使う。
File browser の既定 root は Windows では検出した filesystem drive、Linux/macOS では user home。
`SPHERE_FILESYSTEM__ALLOWED_ROOTS` または config で明示すればこの既定を置き換えられる。

SAM3 を有効にする場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"; .\scripts\start-windows.ps1
```

## Workflow

上部の「次の工程」は未生成または stale の最初の stage を 1 つだけ実行する。Fisheye region
のような手動確認では停止する。

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
  -> Training image denoise            (optional, Native/ERP video)
  -> LFStudio export
```

Matching を変えても frame decode や feature extraction は再実行しない。Denoise は camera solve
後の training image だけを置き換えるため、camera pose、point cloud、gravity alignment は不変。
Pinhole rig の denoised training views は未対応なので、その mode の export は明示的に原画を使う。

旧 run が `Failed to parse options - SiftExtration.max_image_size` または
`SiftExtraction.max_image_size` で停止している場合は「特徴抽出」だけをクリアして再生成する。
COLMAP 4.1 の正しい option は `FeatureExtraction.max_image_size` で、現在の stage と regression
test はこの namespace を使用する。Frame 抽出からやり直す必要はない。

## 既定値

| 項目 | 既定 |
|---|---|
| Frame selection | Sharpness-first、1 fps、候補 5 枚 |
| SAM3 | 長辺 1024 px、dilation 8 px |
| Feature | SIFT、長辺 2048 px、最大 8192 |
| Matcher | Brute-force、Sequential overlap 4 |
| Mapper | Global Mapper + view-graph calibration |
| Bundle Adjustment | CPU（cuDSS capability がある場合だけ GPU を選択可能） |
| Alignment | Exposure-synchronized IMU auto |
| Denoise | **Off**。FastDVDnet `sigma=10` は比較・preview 用の optional stage |
| LFStudio | MRNF + GUT、`max_cap=1,000,000` |

ALIKED + LightGlue は暗所・弱 texture の High preset として残す。通常の X5 実写では SIFT +
Brute-force + Global Mapper が速度、registration、reprojection の Pareto front だった。

Bundle Adjustment（BA）は全 camera pose、intrinsics、3D point を同時に調整して reprojection
error を最小化する最終最適化である。CPU / GPU solver は主に所要時間と memory が異なり、
同じ目的関数を解く。Doctor が cuDSS 対応を確認できない環境では CPU が安全な既定となる。

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

最終 project を同じ 104 rig frame で再実行した SIFT 結果は 208/208 image、27,244 points、
mean/median/P95 reprojection = 0.913 / 0.846 / 1.732 px、139,839 observations。直前の ALIKED
結果は 10,733 points、0.945 / 0.813 / 1.866 px、59,974 observations だった。Denoise artifact
は feature branch から独立しているため、この再構成中も再生成せず保持された。

既定 UI E2E は 37 rig frame / 74 image を 74/74 登録し、5,761 points、mean/median/P95
reprojection = 0.872 / 0.781 / 1.782 px、gravity residual median/P90 = 1.599° /
3.537°。Project 作成から LFStudio export まで 3.4 分で、旧約 44.8 分から約 13.2 倍高速化
した。主因は native matcher と、322 回の random seek を置き換えた stream 単位の sequential
decode である。

Pinhole fallback は 12 時点 × 12 virtual camera = 144/144 image、1,667 points、mean
reprojection 1.085 px、gravity residual 1.910°。Camera-only model を成功扱いしない quality
gate を持つ。合成 ERP 12 枚は 12/12 image、7,720 points、mean/median/P95 = 0.386 /
0.304 / 0.989 px。実写 ERP sequence は未検証。

## LFStudio training benchmark

同じ 208 image を LFStudio v0.5.3、MRNF、GUT、binary segment mask、3840² で確認した。

| Training image / eval target | Feature pose | Iteration | Gaussians | Eval PSNR | Eval SSIM |
|---|---:|---:|---:|---:|---|
| Raw / raw（旧 config） | ALIKED | 7,000 | 95,165 | 22.197 | 0.7976 |
| Raw / raw（旧 500k resume） | ALIKED | 30,000 | **500,000** | — | — |
| FastDVDnet / raw | ALIKED | 30,000 | 870,956 | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 30,000 | 879,324 | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 30,000 | **1,000,000** | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **30,000** | **1,000,000** | **24.920** | 0.8056 |

既存 GUI checkpoint も 7,000 iteration / 88,632 Gaussians で止まっていた。ぼけの第一原因は
30,000 iteration の約 1/4 しか終わっていないこと。さらに完全 run は 500k cap に早期到達し、
残り約 17k iteration で topology を増やせなかった。この 2 原因を分離して確認したため、学習率
は変えず、公式 eval preset と同じ 1M だけを新しい下限にした。LFStudio runtime default の 5M
は eager VRAM allocation のため 12GB GPU の未検証既定にはしない。

Saved render を同一の denoised target に対して再計算した masked PSNR は、ALIKED で
FastDVDnet training 24.223 / raw training 24.489、SIFT で FastDVDnet training 24.478 / raw
training **25.028**。従って暗所素材でも training source の既定は raw とする。FastDVDnet は
平面 noise を減らすが、今回の multi-view optimization では細部損失を上回る改善が無かった。
Offline 値は saved 8-bit render による同条件比較で、LFStudio 内部の float metric とは約
0.1–0.2 dB 差がある。

MRNF の grow/prune は時刻 seed を含むため run 間に小さな揺らぎがある。各 fresh 1M / 30k run
は約 59–69 分。7k だけで止めず、30k checkpoint / PLY を最終成果物として使う。

## 暗所ノイズ除去

夜間室内の X5 image は sensor/ISP/HEVC の相関ノイズを含む。Single-frame generative denoiser は
view ごとに別の texture を生成し得るため既定にしない。FastDVDnet は source-rate の前後 2 枚
ずつ、計 5 frame から中心 frame の noise residual を予測し、明示的な optical-flow warp や
座標 resampling を行わない。

3 時点 × 2 lens の 3840² 実写比較:

| Sigma | Mean absolute change | RGB mean shift | Wall high-frequency | Table edge energy |
|---:|---:|---:|---:|---:|
| 5 | 1.148 / 255 | (-0.201, -0.214, -0.116) | 81% | 95% |
| **10** | 1.900 / 255 | (-0.001, -0.281, -0.377) | **51%** | **83%** |
| 15 | 2.667 / 255 | (0.024, -0.399, -0.480) | 29% | 74% |

Sigma 10 は平面ノイズを約半分にし、文字・輪郭を多く残した。15 は平面には強いが texture と
色の変化が増える。512px core + 64px halo と 768px core の出力差は平均 0.033 / 255、95
percentile 0 だった。Network の receptive-field radius は 74 px なので、製品既定は境界を完全に
覆う 80 px halo とし、512px core を使う。

Model は MIT license の clipped-noise checkpoint を immutable commit から取得し、
SHA-256 `8118974ac7defaa5037f73caf87e0cb53efcfa49ae77d55c05ab187f59e55949`
を検証してから `<workspace>/.models/fastdvdnet/` に publish する。Torch/CUDA が無い場合は
FFmpeg `atadenoise` を source-rate で適用する。どちらも抽出後の疎な 1fps image を時間平均
しない。

208 枚を software decode した初回 full run は 4,368.5 秒（21.0 秒/image）。同じ 5 枚の
3840² frame を source 中央まで decode する実測は software 24.4 秒、CUDA/NVDEC 2.8 秒だった。
Hardware frame を `yuv420p` に戻してから RGB 化すると両経路の 5 image SHA-256 が完全一致
したため、現在は `yuv420p` 入力で実動画 probe に成功した場合だけ CUDA decode を自動使用する。
未検証 pixel format は画質優先で software decode を維持する。

## Reconstruction mode

### Native fisheye

前後画像を `OPENCV_FISHEYE` camera として使用する。Front を rig reference、Back を 180°
rotation + metadata baseline として固定する。再投影による情報損失がなく、2 lens に視覚 overlap
がなくても同一 exposure の rig として 1 model に統合する。

### Pinhole rig

Dual fisheye / ERP を 6-view cubemap にする fallback。Known intrinsics/extrinsics を固定でき、
IGS+ や pinhole-only downstream に向く。Training image temporal denoise は未対応。

### Equirectangular

COLMAP 4.1 camera model ID 17 を使用する。LFStudio は GUT 対応 MRNF/MCMC だけを生成する。
IGS+ は GUT と併用できず equirectangular を undistort できないため候補から除外する。

## Gravity alignment

COLMAP world orientation の gauge freedom を、INSV の約 1kHz accelerometer と各 exposure
timestamp で解く。

- 全 recording の device-frame acceleration を平均しない
- `first_frame_timestamp`、`gyro_timestamp`、`is_raw_gyro` を metadata から読む
- 24 個の右手系 signed-axis mapping から robust consensus を選ぶ
- Inlier、median/P90 angular residual、time offset を記録する
- `model_transformer` で rig、frame、image、point 全体を変換する
- GPS が無い scale は任意なので reference-camera trajectory span を 1 model unit に正規化する

Dataset は `-Y up`、Web viewer は `diag(1,-1,-1)` で `+Y up` 表示する。UI の位置単位は
meter ではなく model unit。

## LFStudio で開く

次を dataset root として選ぶ。

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

Training output は dataset の外へ書く。

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --output-path <project>/training_outputs/<run>
```

LFStudio が dataset 内へ既定 `output/` を作っても、export 再生成前に unmanaged item を
`training_outputs/` へ移す。使用中で移動できない場合は削除せず error にする。

LFStudio v0.5.3 の GUI folder import は `train_configs/` を自動適用しない。Folder 自体と camera
pose は正しく読み込めるが、GUI training では MRNF、GUT、Segment mask を手動設定する必要が
ある。再現可能な既定経路は上記 CLI であり、`<run>` は実行ごとに固有名へ置き換える。

Binary mask は white=valid / black=excluded。`mask_mode="segment"` は masked pixel を loss
から外し、その領域の Gaussian opacity も抑制する。

### Camera の色

LFStudio の camera icon は per-camera photometric loss の相対 heatmap。Green は現在の最小、
red は最大、brown/orange は中間、white は未観測。Min/max は更新のたび再正規化されるため、
red は pose failure や camera disable を意味しない。Selected/hovered camera も orange に
override される。

### Transform Inspector が 0 の場合

LFStudio は dataset camera pose を SceneNode transform ではなく `Camera.R/T` に保持し、center
を `C=-Rᵀt` で計算する。Generic Transform Inspector が local node の `0,0,0` を表示しても
pose は失われていない。本アプリの Camera Inspector は実 center を表示し、export validation は
unique center 数と trajectory diameter を検査する。

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

[denoise]
model_path = ""
device = "auto"
hardware_decode = "auto"

[sam3]
repo_path = ""
checkpoint_path = ""
device = "cuda:0"
dtype = "bfloat16"
```

詳細は [docs/setup-gpu.md](docs/setup-gpu.md)。

## Camera/source 拡張

Source 固有ロジックは `colmap/input_workspace.py` の builder に隔離する。新 camera は共通
`InputSpec` として images、optional masks、camera model/intrinsics、optional rig config、
refine policy を返す。Feature、Matcher、Mapper、Alignment、Export は source format を再解釈
しない。Fisheye circle は X5 固定値ではなく画像から推定し、UI で確認できる。

## Architecture

```text
React + TypeScript + Vite + Playwright
                  |
FastAPI + aiosqlite + WebSocket
                  |
multiprocessing spawn worker
                  |
FFmpeg / COLMAP / Torch / ONNX Runtime / SAM3
```

FastAPI process は heavy CUDA model を import しない。Stage は temporary directory に書き、成功
後だけ atomic replace する。Windows native library が handle を短時間保持する場合だけ対象を
限定して retry する。再生成可能な artifact と LFStudio training result の ownership は分離する。
Invalidation は線形順序ではなく artifact DAG を辿るため、feature / matching / Mapper を変更しても
独立した denoise artifact は保持され、両 branch を消費する export だけが stale になる。

## Test

```bash
cd backend
uv sync --extra dev --extra imaging --extra denoise
uv run pytest -q
uv run ruff check
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

Project license は未確定。FastDVDnet 由来の network definition と checkpoint は MIT license で、
notice を package に同梱する。外部公開前に project 本体の license を決定する。

## References / legacy plugin

- [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): loader、GUT、mask、MRNF、loss heatmap
- [colmap/colmap](https://github.com/colmap/colmap): Global Mapper、ALIKED、rig、EQUIRECTANGULAR
- [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet): 5-frame residual denoiser（MIT）
- [AdrianEddy/telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata / IMU timestamp
- [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [alexmgee/lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): 旧 workflow の比較対象（GPL-3.0-or-later）
- [BenjaminHenriksson/insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container 調査の参考
