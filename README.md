# sphere-reconstruct

Insta360 X5 の INSV から, キャリブレーション済み pinhole rig ベースの COLMAP データセットを生成するローカル Web アプリケーション. Equirectangular (ERP) 動画/画像入力も受け付ける.

旧 [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin) (GPL-3.0-or-later) は行動参考のみで, 実装コードは流用していない. 独立に再実装.

## パイプライン (V1 主線)

```
INSV / ERP
  -> ソース検査 (INSPECTED)
  -> フレーム抽出 (EXTRACTED)
  -> pinhole rig 再投影 (REPROJECTED)
  -> SAM3 マスク生成 (MASKED)
  -> COLMAP Incremental Mapper (RECONSTRUCTED)
  -> データセット書き出し (EXPORTED)
```

各ステージは冪等. 入力ハッシュ・パラメータハッシュ・実装バージョンで再計算判定. 中間結果は tmp に書いてから原子リプレース.

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
| 3 | 前後同期抽出 / 空間抽出 (鮮鋭度) / pinhole rig | 完了 |
| 4 | SAM3 手動パス / 推論 / マスク (縮小->推論->拡大) | 完了 (4070Ti で検証) |
| 5 | COLMAP CLI / **rig 拘束** / SIFT / Sequential Matching / エクスポート | 完了 (実機で検証) |
| 6 | 3D Viewer (three.js + R3F) / Frames / Masks / 点群 + カメラ | 完了 |
| 7 | 起動スクリプト / 環境チェック / クラッシュ復旧 / 静的配信 | 完了 |

**rig 拘束の効果 (実機, X5 8 frames)**: 前後レンズ 6 視点 = 12 仮想カメラの既知
相対姿勢を COLMAP に与えることで, 登録率が 16/96 → **96/96 (100%)**, 点数 936 →
2795, 平均再投影誤差 0.64px. 低視差の手持ち 360 素材でも全フレームが登録される.

今後: COLMAP loop closure (COLMAP 4.x は faiss 形式 vocab tree が必要), PB
(`.insv.pb`) 完全パーサ (PB を伴う個体の INSV サンプル入手待ち; 現状は offset_v3
(ASCII) で完全に校正できている).

## ライセンス

未確定. コミット前に決定する.

## 謝辞

- Insta360 X5 INSV 構造の逆解析については [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch) の PIPELINE.md および FINDINGS.md を参照 (実装は独立に行っている).
- パイプライン全体像は旧 [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin) から学んだが, 実装コードは流用していない.
