# sphere-reconstruct

Insta360 X5 の INSV から, COLMAP ベースの 3DGS 向けデータセットを生成するローカル Web
アプリケーション. 既定は生の前後魚眼をそのまま解く **Native 魚眼 (前後強制 rig)**,
fallback として pinhole rig cubemap も選べる. Equirectangular (ERP) 動画/画像入力も
受け付ける.

旧 [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin) (GPL-3.0-or-later) は行動参考のみで, 実装コードは流用していない. 独立に再実装.

## パイプライン (V1 主線)

再構成は 2 モード. reconstruct ステージの `reconstruction_mode` で切替える.

**Native 魚眼 (既定, native_fisheye)** — 前後魚眼を全 FoV / 原画素のまま解く:

```
INSV
  -> ソース検査 (INSPECTED)
  -> 前後同期フレーム抽出 (EXTRACTED)
  -> SAM3 魚眼マスク: 円形有効領域 + 動体除外 (MASKED)
  -> COLMAP: 前後 2 レンズ OPENCV_FISHEYE + 強制物理 rig (RECONSTRUCTED)
  -> データセット書き出し (EXPORTED)
```

front を参照, back を「Y 軸 180deg + 物理 baseline」で強制固定するため, 前後半球に
視覚的重なりが無くても必ず 1 モデルへ合流する. pinhole 再投影が不要で特徴損失が無い.
魚眼の有効領域 (円) は UI で手動調整でき (レンズ端の反射/汚れを外周から除外), 動体
(人 / 自撮り棒 / 三脚 / 影) は SAM3 で検出し膨張させて除外する.

**Pinhole rig (fallback, pinhole_rig)** — 6-view cubemap pinhole に再投影して解く:

```
INSV -> 検査 -> 抽出 -> pinhole rig 再投影 -> SAM3 マスク -> COLMAP (12 仮想カメラ rig) -> 書き出し
```

各ステージは冪等. 入力ハッシュ・パラメータハッシュ・実装バージョンで再計算判定. 中間
結果は tmp に書いてから原子リプレース. native_fisheye では reproject_views を自動
スキップし, generate_masks は生魚眼用 (fisheye レイアウト) の SAM3 マスクを作る.

## アーキテクチャ

```
React Web UI (Vite + TypeScript)
        |
        v
FastAPI (127.0.0.1 のみ, uvicorn)
        |
        +-- SQLite (aiosqlite): project / job / stage state
        |
        +-- Worker サブプロセス (multiprocessing.spawn)
                  |
                  +-- CUDA / Torch / COLMAP / ffmpeg
```

FastAPI プロセスは CUDA を絶対に触らない. Worker が独立プロセスで, キャンセルと VRAM 解放を確実にする.

## ディレクトリ

```
sphere-reconstruct/
├── backend/
│   ├── pyproject.toml            # uv 管理, Python 3.12
│   ├── src/sphere_reconstruct/
│   │   ├── main.py               # FastAPI エントリ
│   │   ├── api/                  # projects / jobs / events(WS) / previews / settings
│   │   ├── domain/               # Project / Artifact / PipelineState
│   │   ├── pipeline/             # Engine / Stage 抽象 / Manifest
│   │   ├── stages/               # 各パイプラインステージ実装
│   │   ├── insta360/             # INSV footer / offset_v3 / MEI キャリブ / IMU
│   │   ├── imaging/              # sampling / projection / masks / thumbnails
│   │   ├── sam3/                 # 手動パス方式 (HF Token 不使用)
│   │   ├── colmap/               # CLI runner / rig / model / web preview
│   │   └── infrastructure/       # database / processes / filesystem
│   └── tests/
├── frontend/
│   ├── package.json              # pnpm, Vite, React 18, TypeScript
│   ├── vite.config.ts
│   └── src/
│       ├── pages/                # Projects / Source / Frames / Masks / Reconstruction / Settings
│       ├── features/             # 機能単位のロジック
│       ├── viewers/              # three.js + R3F ビューア
│       ├── components/           # 汎用 UI
│       ├── api/                  # OpenAPI 生成型 + fetch ラッパ
│       └── hooks/
├── runtime/
│   └── config.toml               # ワークスペース / 許可された FS root / SAM3 パス
└── docs/                         # spec / phase 計画
```

## 前提ソフトウェア

外部バイナリはユーザ側で用意. 起動時とヘルスチェックで存在を検証する.

- Python 3.12 (uv が管理)
- Node.js 18+ / pnpm
- ffmpeg (INSV デコードとフレーム抽出)
- COLMAP CLI (`colmap` バイナリ)
- SAM3 リポジトリ + チェックポイント (手動配置, HuggingFace 経由の自動 DL は行わない)
- NVIDIA GPU + CUDA (SAM3 / COLMAP GPU 用)

## 使い方 (単体アプリとして起動)

backend が frontend の build を静的配信するので, Electron 不要でブラウザで使える.

```bash
# Linux/macOS. CUDA 機は SPHERE_WITH_SAM3=1 を付ける.
SPHERE_WITH_SAM3=1 ./scripts/run.sh
# -> http://127.0.0.1:8787 をブラウザで開く
```

```powershell
# Windows
$env:SPHERE_WITH_SAM3="1"; .\scripts\run.ps1
```

`runtime/config.toml` で workspace / allowed_roots / 外部バイナリ / SAM3 パスを設定する.
CUDA マシンのセットアップは [docs/setup-gpu.md](docs/setup-gpu.md) を参照.

## 開発の始め方

Backend:

```bash
cd backend
uv sync --extra dev --extra imaging
uv run uvicorn sphere_reconstruct.main:app --reload --host 127.0.0.1 --port 8787
```

Frontend (Vite dev server, API は 8787 へ proxy):

```bash
cd frontend
pnpm install   # または npm install
pnpm dev       # http://127.0.0.1:5173
```

## フェーズ計画

| Phase | 内容 | 状態 |
|-------|------|------|
| 1 | FastAPI + React 骨組 / SQLite / Worker / WebSocket / Job ライフサイクル | 完了 |
| 2 | INSV footer / offset_v3 / IMU / MEI キャリブ | 完了 (実 X5 で検証) |
| 3 | 前後同期抽出 / **空間抽出 (2層多基準)** / pinhole rig | 完了 |
| 4 | SAM3 手動パス / 推論 / マスク (縮小->推論->拡大) | 完了 (4070Ti で検証) |
| 5 | COLMAP CLI / **rig 拘束** / SIFT / ALIKED / エクスポート | 完了 (実機で検証) |
| 6 | 3D Viewer (three.js + R3F) / Frames / Masks / 点群 + カメラ | 完了 |
| 7 | 起動スクリプト / 環境チェック / クラッシュ復旧 / 静的配信 | 完了 |
| 8 | **Native 魚眼再構成 (前後強制 rig) / ALIKED 全解像度 + OOM 自動 CPU** | 完了 (実機で検証) |

### 再構成モード (native_fisheye / pinhole_rig)

**Native 魚眼 (既定)**: 前後 2 レンズを OPENCV_FISHEYE の 2 センサー rig として解く.
front を参照, back を「Y 軸 180deg (quat [0,0,1,0]) + 物理 baseline」で**強制固定**する.
この相対姿勢は正常素材の合流結果から実測した (front->back 回転 179.87deg, 12 フレーム
std 0.16deg). front/back は視覚的に重ならないが, 同名フレームを 1 つの rig frame とみなす
ことで必ず 1 モデルへ合流する. baseline (offset_v3 のレンズ中心間距離 ~32mm) は metric
スケールのアンカーにもなる. rig 外参は精修せず実測値で固定する.

実機比較 (暗所低解像 2880^2, 前後各 10 フレーム):

| 構成 | モデル数 | 登録 | 点数 | 再投影 |
|------|---------|------|------|--------|
| Native 魚眼, rig 無し | 2 (分裂) | 85% | — | — |
| Native 魚眼, **強制 rig + SIFT** | 1 | **100% (20/20)** | 767 | 0.81px |
| Native 魚眼, **強制 rig + ALIKED** | 1 | **100% (20/20)** | **3521** | 1.19px |

強制 rig で前後分裂が解消し 100% 合流する. 暗所では ALIKED が点数で圧倒 (4.6x), 精度は
SIFT が上. Pinhole rig (fallback) は正常素材 6 フレームで 72/72 (100%), 0.81px.

**pinhole rig 拘束の効果 (参考)**: 6-view cubemap = 12 仮想カメラの既知相対姿勢を
COLMAP に与えることで, 登録率が 16/96 → **96/96 (100%)**, 平均再投影誤差 0.64px.
offset_v3 を一切精修しない剛性 rig でも 144/144 (100%), 0.94px の亜画素精度が出るため,
offset_v3 が幾何的に正確であることが裏付けられる.

**空間抽出 (2層多基準)**: 固定時間間隔ではなく,
- 快速層 (安価): 時間 + Laplacian ブレ + 過曝/欠曝 + IMU 回転差
- 精確層 (高価): SIFT 特徴数 + 光流中央値 + 前選択フレームからの運動量
の 2 段で候補を絞り, 「視覚/運動の変化量」で等間隔に高品質フレームを選ぶ.
`selection_mode = interval | sharpness | spatial` で切替.

**特徴 backend (SIFT vs ALIKED+LightGlue)**: reconstruct の `feature_backend` で選ぶ.
この COLMAP ビルドは SIFT のみ (deep features 非対応) なので, ALIKED は外部 onnxruntime で
抽出/マッチして COLMAP DB に書き込む (database_creator -> keypoints -> LightGlue ->
matches_importer). 魚眼は縮小すると角分解能が落ちるため ALIKED は**全解像度**で抽出し,
`extraction_device = auto` は空き VRAM から GPU/CPU を選び, 実行時 OOM も CPU へ
フォールバックする (8K の 3840^2 は GPU では OOM するため CPU 抽出になる). 通常素材では
SIFT の方が高精度, 弱テクスチャ/暗所では ALIKED が救済する. モデルは SAM3 同様 config
パス + `scripts/fetch_aliked_models.py` で取得 (git には入れない).

今後: 物理レンズ姿勢からの Derived Pinhole 生成 (訓練出力用, 2 度目の COLMAP を
回さない), COLMAP loop closure (faiss 形式 vocab tree), PB (`.insv.pb`) 完全パーサ.

## ライセンス

未確定. コミット前に決定する.

## 謝辞

- Insta360 X5 INSV 構造の逆解析については [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch) の PIPELINE.md および FINDINGS.md を参照 (実装は独立に行っている).
- パイプライン全体像は旧 [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin) から学んだが, 実装コードは流用していない.
