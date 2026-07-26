# GPU / native dependency setup

Windows 11 + RTX 4070 Ti、Linux NVIDIA、macOS CPU/Metal 周辺で必要になる runtime をまとめる。
通常は platform 別 start script を使い、手作業の前に doctor の結果を確認する。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

## Windows

```powershell
.\scripts\start-windows.ps1
```

この script は次を順に行う。

1. pnpm frontend install/build
2. uv backend sync
3. COLMAP が無ければ公式 4.1.1 CUDA zip を取得
4. SHA-256 を検証して `.runtime/tools/` へ atomic install
5. FFmpeg、COLMAP capability、workspace、SAM3、CUDA runtime を診断
6. FastAPI を起動

自動 COLMAP install を禁止する場合:

```powershell
$env:SPHERE_SKIP_AUTO_INSTALL_COLMAP="1"
.\scripts\start-windows.ps1
```

CPU package を明示導入する場合:

```powershell
backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cpu
```

独立した `glomap.exe` は導入しない。GLOMAP は COLMAP 4.1 の `global_mapper` に統合済み。

## COLMAP の CUDA 表示について

`COLMAP ... with CUDA` は SIFT/ONNX/CUDA support を示すが、Ceres bundle adjustment の
CUDA/cuDSS support を保証しない。公式 Windows 4.1.1 CUDA package は test machine 上で
次の警告を出し、BA を CPU へ戻した。

```text
Requested to use GPU for bundle adjustment, but Ceres was compiled without CUDA support.
Requested to use GPU for bundle adjustment, but Ceres was compiled without cuDSS support.
```

Doctor は sibling/system cuDSS を確認し、`gpu_bundle_adjustment=false` を表示する。この場合は
UI の BA GPU を off にする。ALIKED/SIFT GPU は別機能なので引き続き利用できる。

## COLMAP native ALIKED

COLMAP 4.1.1 は次を内蔵する。

- `ALIKED_N16ROT`
- `ALIKED_N32`
- `ALIKED_BRUTEFORCE`
- `ALIKED_LIGHTGLUE`

Model option の既定値には URL、filename、SHA-256 が含まれ、未配置なら COLMAP が download・
verify する。既存 model を config で指定してもよい。

```toml
[aliked]
extractor_path = "D:/models/aliked-n16rot.onnx"
matcher_path = "D:/models/aliked-lightglue.onnx"
```

Windows の COLMAP ONNX CUDA provider は `cudnn64_9.dll` を必要とする。SAM3 用 Torch が
同じ backend venv にある場合、runner は `site-packages/torch/lib` を subprocess `PATH` に
追加する。Doctor の `cuda_runtime.cudnn` で確認できる。

cuDNN が無い環境では feature GPU を off にするか、対応 runtime を導入する。失敗を黙って
CPU に隠す処理は行わない。

## SAM3

SAM3 は任意。利用時だけ extra を同期する。

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

```bash
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

現状は repository と checkpoint を config で指定する。

```toml
[sam3]
repo_path = "D:/models/sam3-main"
checkpoint_path = "D:/models/sam3.pt"
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 1024
```

Runtime は worker process 内でのみ import する。`cuda:0` は SAM3 builder の制約に合わせて
内部で `cuda` へ正規化する。Inference 前の既定長辺は 1024 px。

## FFmpeg

INSV の 2 本の HEVC stream を decode できる build が必要。

```toml
[binaries]
ffmpeg = "D:/tools/ffmpeg/bin/ffmpeg.exe"
ffprobe = "D:/tools/ffmpeg/bin/ffprobe.exe"
```

Frame extraction は source frame index を select filter に渡し、stream ごとに 1 回だけ順次
decode する。以前の 1 frame 1 process / random seek 方式は使わない。

## Linux

Distribution package または source build の COLMAP 4.1+ と FFmpeg を用意する。

```bash
./scripts/start-linux.sh
```

ALIKED GPU を使う場合は COLMAP の ONNX Runtime CUDA provider と cuDNN が必要。Doctor が
capability と runtime path を表示する。

## macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

macOS では NVIDIA CUDA を前提にしない。SIFT/CPU ALIKED と CPU Mapper を利用する。

## Service deployment

Remote machine で service manager を使う場合も、起動 command の前に `SPHERE_CONFIG` を
設定する。

```text
SPHERE_CONFIG=D:/path/to/config.remote.toml
```

Backend code を更新した後は service を再起動する。Frontend dist は static file なので、
copy 後に browser reload すればよい。Running worker を止めるときは command line が
`sphere_reconstruct` / `uvicorn` に一致する PID だけを対象にし、machine 上の全 Python を
一括停止しない。

## 検証

```bash
cd backend
uv run pytest -q
uv run sphere-doctor
```

```bash
cd frontend
pnpm build
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 pnpm test:e2e
```
