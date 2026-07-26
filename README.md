# sphere-reconstruct

Insta360 の dual-fisheye INSV、equirectangular 動画、equirectangular 画像列から、
COLMAP sparse reconstruction と LichtFeld Studio 用 3D Gaussian Splatting dataset を
生成するローカル Web アプリケーション。

処理は独立 stage に分割されている。特徴抽出、matching、Mapper、重力整列、export を
個別に再生成・クリアできるため、Matcher を変えるだけで画像 decode や特徴抽出から
やり直す必要はない。

## 主な機能

- 生の前後魚眼を `OPENCV_FISHEYE` 2-camera rig として直接 reconstruction
- cubemap pinhole rig fallback
- COLMAP 4.1 の `EQUIRECTANGULAR` camera model による ERP direct reconstruction
- COLMAP native SIFT / ALIKED N16ROT / ALIKED N32
- Brute-force / LightGlue と Sequential / Exhaustive / Vocab-tree の独立選択
- Incremental Mapper / COLMAP 内蔵 Global Mapper
- SAM3 dynamic-object mask と camera 非依存の fisheye valid-circle 推定
- exposure timestamp と同期した IMU gravity alignment
- LFStudio で `export_dataset/` をそのまま選べる export layout
- stage cache、transitive invalidation、atomic output replace、WebSocket progress
- 日本語・中国語・英語 UI、dockable IDE layout、point-cloud/camera viewer
- 起動時 dependency diagnostics と Windows COLMAP auto installer

## 推奨 workflow

上部の「次の工程」ボタンは、未生成の最初の工程を 1 つだけ実行する。Fisheye region の
ような手動確認が必要な箇所では自動的に停止する。

```text
Source inspection
  -> Frame extraction
  -> Fisheye region confirmation       (Native fisheye のみ)
  -> Pinhole reprojection              (Pinhole rig のみ)
  -> SAM3 masks                        (任意)
  -> Feature extraction
  -> Feature matching
  -> Sparse reconstruction
  -> Gravity alignment
  -> LFStudio export
```

各 stage の「クリア」は、その stage と真に依存する後続成果物を invalidate する。例えば
`match_features` のクリアは `extract_features` を残し、`reconstruct` 以降だけを無効化する。

## 既定値

実写 X5 dataset の wall-clock と reconstruction quality の Pareto 比較から、現在の既定は
次の通り。

| 項目 | 既定 |
|---|---|
| Frame extraction | Sharpness-first、1 fps、候補 5 枚 |
| SAM3 | 長辺 1024 px、mask dilation 8 px |
| Feature | `SIFT`、長辺 2048 px、最大 8192 features |
| Matcher | `SIFT_BRUTEFORCE` |
| Pairing | Sequential、overlap 4 |
| Mapper | Global Mapper + view-graph calibration |
| Alignment | exposure-synchronized IMU auto |
| Export config | MRNF / MCMC / IGS+（camera model に不適合な config は生成しない） |

ALIKED + LightGlue は暗所・弱テクスチャで track を救う High preset として残す。
Global Mapper は規模が小さい素材でも initial-pair search を避けられるため、動画入力の既定に
適している。

## 実写 benchmark

入力: Insta360 X5、38.17 秒、dual 3840×3840 HEVC。旧 Spatial 設定で 104 rig frames
（208 images）を選択し、同じ画像・mask・camera rig で比較した。

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | **27,202** | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s¹ | 9.7 s | 17.9 s | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s¹ | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |
| COLMAP ALIKED + Brute-force + Incremental | 243.0 s¹ | 9.7 s | 88.2 s | 208/208 | 25,634 | 1.112 px |

¹ ALIKED 4096-feature benchmark の共有 extraction。通常 workflow は約 1 fps のため
image 数はこの 104-frame benchmark より少ない。

最終既定 UI test は 37 rig frames / 74 images を全登録し、5,761 points、mean/median/P95
reprojection error = 0.872 / 0.781 / 1.782 px、gravity residual median/P90 =
1.599° / 3.537°だった。

Pinhole fallback は 12 時点 × 12 virtual cameras = 144/144 images を登録し、1,667 points、
mean reprojection 1.085 px、gravity residual 1.910°、全 pipeline 77.5s。6 時点の極端に疎な
入力は 0 points になったため、現在は quality gate が camera-only reconstruction の export を
拒否する。

Equirectangular 分支は平移視差を持つ合成 ERP 12 枚で 12/12 images、7,720 points、
mean/median/P95 reprojection = 0.386 / 0.304 / 0.989 px。Export は GUT 対応の MRNF/MCMC
だけを生成した。実写 ERP sequence は未入手のため、実素材の追加検証は別途必要。

旧処理が約 45 分かかった主因は次の 2 点だった。

1. Python LightGlue が 812 pairs に約 26 分 44 秒を使用。
2. Frame extraction が 322 回の個別 FFmpeg seek を行い、2 秒 GOP を繰り返し decode。

現在は COLMAP native matcher と stream ごとの single-pass sequential decode を使用する。
同じ 38.17 秒素材を新規 project 作成から LFStudio export まで Playwright で操作した既定 UI
workflow は **3.4 分**で完了し、旧約 44.8 分から約 13.2 倍高速化した。内訳は
frame extraction 90.8s、SAM3 52.6s、SIFT 12.7s、matching 16.0s、Global Mapper 5.7s、
alignment 1.3s、export 0.3s。
Point count だけでは default を選ばず、registration、reprojection、track length、trajectory、
最終 LFStudio validation quality を合わせて評価する。

## Reconstruction mode

### Native fisheye

INSV の前後画像をそのまま `OPENCV_FISHEYE` camera として使う。Front を rig reference、
Back を 180° rotation + `offset_v3` baseline として固定する。Pinhole reprojection による
情報損失がなく、2 lens に視覚 overlap がなくても同一 frame rig として 1 model に統合する。

### Pinhole rig

Dual fisheye または ERP を 6-view cubemap に変換する fallback。既知 intrinsics と rig
extrinsics を固定できる。LFStudio の IGS+ を使いたい場合や、downstream が fisheye camera
を扱えない場合に選ぶ。

### Equirectangular

COLMAP 4.1 の camera model ID 17 を使い、ERP を球面 camera のまま解く。LFStudio training
では GUT 対応の MRNF/MCMC を使う。IGS+ は GUT と併用できないため生成候補から除外する。

## Gravity alignment

COLMAP world orientation には gauge freedom があるため、再構成ごとに上下が変わる。INSV の
約 1 kHz accelerometer を各 exposure timestamp に同期し、各 front camera pose から world
up を求める。

- 全期間の device-frame acceleration は平均しない
- Insta360 metadata の `first_frame_timestamp` / `gyro_timestamp` / `is_raw_gyro` を使用
- encoded image と IMU の signed axis mapping は右手系 24 候補から robust 選択
- angular residual、inlier count、time offset を `alignment.json` に記録
- Sparse model 全体を `model_transformer` で変換し、rig/frame/points の整合を維持
- GPS のない SfM scale は任意なので、reference-camera trajectory span を 1 model unit に正規化

LFStudio/COLMAP dataset は `-Y up`、Web viewer は表示時に `diag(1,-1,-1)` を掛けた `+Y up`
として扱う。UI の span/position の単位は meter ではなく model unit。

## LFStudio への読み込み

Export 完了後、LFStudio では次の folder をそのまま選ぶ。

```text
<workspace>/projects/<project-id>/export_dataset/
```

```text
export_dataset/
├── images/
├── masks/                         # 生成時のみ
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

`export_dataset/dataset/` という追加階層は存在しない。`export_manifest.json` には image
欠落、camera trajectory collapse、camera model、mask 数を含む loadability check が入る。
Binary mask は white=valid / black=excluded で export され、推奨 config は
`mask_mode="segment"` を設定する。

LFStudio の generic Transform inspector が camera node position を `0,0,0` と表示しても、
COLMAP pose が消えたことを意味しない。LFStudio は実 pose を camera の `R/T` に保持し、
`images.bin` の camera center `C=-Rᵀt` を training と frustum 表示に使う。本アプリは
trajectory diameter と unique camera center 数を別途表示する。

## インストールと起動

### Windows

```powershell
.\scripts\start-windows.ps1
```

または:

```cmd
scripts\start-windows.cmd
```

COLMAP が未設定・未導入なら、公式 Windows CUDA package 4.1.1 を SHA-256 検証付きで
`.runtime/tools/` に導入する。独立 GLOMAP package は使わない。

### Linux

```bash
./scripts/start-linux.sh
```

### macOS

```bash
brew install colmap
./scripts/start-macos.sh
```

互換入口として `scripts/run.sh` と `scripts/run.ps1` も残している。全 platform で
`uv`、`pnpm 9.15`、FFmpeg/FFprobe、COLMAP 4.1+ を診断してから起動する。

SAM3 を有効にする場合:

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

```powershell
$env:SPHERE_WITH_SAM3="1"; .\scripts\start-windows.ps1
```

起動後は `http://127.0.0.1:8787` を開く。設定メニューにも同じ environment diagnostics が
表示される。

## 設定

既定設定は `runtime/config.toml`。別ファイルは `SPHERE_CONFIG` で指定する。

```toml
[workspace]
root = "./workspace"

[filesystem]
allowed_roots = ["D:/"]

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

環境変数は Pydantic の nested 形式を使える。例:

```text
SPHERE_BINARIES__COLMAP=D:/tools/colmap/bin/colmap.exe
SPHERE_WORKSPACE__ROOT=D:/sphere-workspace
```

## Camera 拡張

Camera/source 固有ロジックは `colmap/input_workspace.py` の builder に隔離されている。
新しい camera を追加するときは、共通 `InputSpec` として次を出力する builder を登録する。

- images と optional masks
- camera model / intrinsics
- optional rig config
- refine policy
- image count と dimensions

Feature、Matcher、Mapper、Alignment、Export stage は source camera を再解釈しない。Fisheye
valid circle は X5 固定値ではなく画像から初期推定し、UI で確認・保存する。

## 開発とテスト

```bash
cd backend
uv sync --extra dev --extra imaging
uv run pytest -q
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
pnpm test:e2e
```

実 INSV を UI から最後まで検証する場合:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 \
SPHERE_E2E_SOURCE=/absolute/path/to/video.insv \
pnpm test:e2e
```

E2E は project 作成、source 設定、各「次の工程」、manual fisheye region、alignment、export、
LFStudio folder contract まで browser 経由で確認する。Source file は削除しない。

## Architecture

```text
React + TypeScript + Vite + Playwright
                  |
FastAPI + aiosqlite + WebSocket
                  |
multiprocessing spawn worker
                  |
FFmpeg / COLMAP / ONNX Runtime / SAM3
```

FastAPI process は Torch/CUDA を import しない。重い native runtime は worker subprocess に
隔離され、停止時は process tree ごと終了する。Stage output は一時 directory に書き、成功後
にのみ atomic replace する。Windows で native library/Defender が終了直後に directory handle
を保持する場合に限り、`PermissionError` を最大 6 秒再試行する。

## ライセンス

ライセンスは未確定。外部公開前に決定する。

## 参考・旧プラグイン

この実装は独立実装であり、旧 plugin の GPL code はコピーしていない。調査・UX 比較の参考を
文書末尾にまとめる。

- [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): dataset loader、camera model、GUT、mask contract
- [colmap/colmap](https://github.com/colmap/colmap): COLMAP 4.1、Global Mapper、ALIKED、rig、EQUIRECTANGULAR
- [AdrianEddy/telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata、raw IMU、timestamp normalization
- [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow): IMU orientation semantics
- [alexmgee/lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): 旧 UI/workflow の比較対象（GPL-3.0-or-later）
- [BenjaminHenriksson/insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container 調査の参考
