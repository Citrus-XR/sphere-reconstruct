# sphere-reconstruct

Raw multi-fisheye、stitch 済み 360° video / image、通常 video、phone photo を同じ COLMAP
reconstruction へ統合し、LichtFeld Studio が直接 load できる dataset を作る local Web application。

Project は一つの primary source と任意個の supplemental source を持つ。Video だけを抽出し、still folder は
original image を capture として収集する。各 Step は独立して clear / regenerate でき、重い成果物は temporary
directory から成功時だけ atomic publish する。

## Quick start

### Windows

```powershell
.\scripts\start-windows.ps1
```

Command Prompt は `scripts\start-windows.cmd`。Script は frontend build、uv sync、GPU 診断、必要な pinned
COLMAP / jpegtran / FAISS tree の install、Doctor、server 起動を行う。

```powershell
$env:SPHERE_WITH_SAM3="1"
$env:SPHERE_WITH_DENSE="1"
.\scripts\start-windows.ps1
```

### Linux / macOS

```bash
./scripts/start-linux.sh
SPHERE_WITH_SAM3=1 SPHERE_WITH_DENSE=1 ./scripts/start-linux.sh
```

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

起動 URL は `http://127.0.0.1:8787`。Root path も UI を返し、API は `/api/...`。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は actual FFmpeg decoder、COLMAP 4.1、Global Mapper、Ceres CUDA / cuDSS、FAISS tree、jpegtran、
ONNX CUDA、SAM3 を別 capability として調べる。Platform / build 詳細は
[docs/setup-gpu.md](docs/setup-gpu.md)。独立 `glomap.exe` は不要で、COLMAP 4.1 の `global_mapper` を使う。

## Supported source

| Input | Media | Prepared camera | Role |
|---|---|---|---|
| Insta360 `.insv` | Video | 2× `THIN_PRISM_FISHEYE` + calibrated physical rig | Primary / supplemental |
| Stitch 済み 360° video | Video | `EQUIRECTANGULAR` | Primary / supplemental |
| Stitch 済み 360° image folder | Images | `EQUIRECTANGULAR` | Primary / supplemental |
| 通常 / phone video | Video | `SIMPLE_RADIAL` | Primary / supplemental |
| 通常 / phone image folder | Images | EXIF group ごとの `SIMPLE_RADIAL` | Primary / supplemental |

別メーカーの stitched ERP はすぐ追加できる。Raw camera は container / metadata 固有 adapter が必要だが、
downstream は vendor-neutral `CameraSystem` だけを読む。Vendor SDK は native backend や必須 dependency にしない。
新 format の実装契約は [docs/adding-360-camera-formats.md](docs/adding-360-camera-formats.md)。

## Pipeline

```text
Sources
  -> inspect_source
  -> extract_frames                     # video: PTS-based decode / still: collect
  -> fisheye valid-region editor
  -> prepare_images                     # canonical images / cameras / rigs
       ├─ generate_feature_masks -> extract_features -> match_features -> reconstruct
       └─ generate_training_masks ---------------------------------------------┐
  -> align_reconstruction               # rotation only                         │
  -> restore_metric_scale               # only independently observable scale   │
  -> position_ground                    # Y translation only                    │
  -> dense_initialization               # optional RoMaV2 native-ray seed       │
  -> export_dataset <-----------------------------------------------------------┘
```

| Step | Output | Invalidates |
|---|---|---|
| Inspect | source identity、camera system、IMU / shutter metadata | Source branch 全体 |
| Extract | selected captures、real PTS、pairing statistics | Prepare 以降 |
| Prepare | canonical image catalog、camera groups、rig config | Masks / SfM 以降 |
| Feature masks | SfM keep mask | Feature / match / SfM / export |
| Training masks | final training keep mask | Export |
| Feature | COLMAP DB、descriptor、input workspace | Match 以降 |
| Match | verified two-view graph | SfM 以降 |
| Reconstruct | registered camera / sparse points | Alignment 以降 |
| Align | gravity rotation | Scale 以降 |
| Restore scale | metric model、または unscaled の明示 | Ground 以降 |
| Ground | local-ground origin | Dense / export |
| Dense seed | sparse を保持した optional added points | Export |
| Export | LFStudio dataset root、preview、configs、statistics | なし |

Progress の正本は SQLite event stream。Refresh / WebSocket reconnect 後も `/stages` snapshot から復元する。
数値 progress は Step 内で単調、総量不明 phase は indeterminate、artifact publish 前は最大 99%、成功後だけ 100%。
SAM3 mask は PNG の atomic write 後すぐ preview できる。

## General defaults

| Category | Default | Reason |
|---|---|---|
| Reconstruction | Native fisheye | Derived pinhole を作らず source pixel を保持 |
| Feature | SIFT | 実測で ALIKED より速く reprojection も低い |
| Matcher | Brute-force | SIFT の安定経路 |
| Pairing | Auto | Single=Sequential、small mixed=Exhaustive、large mixed=Vocab-tree |
| Loop closure | On | 周回 trajectory の drift を抑える |
| Transitive | 1 pass | Track を伸ばし pair 爆発を避ける |
| View graph calibration | Auto | Unknown intrinsics だけに適用 |
| Mapper | Global + gated Incremental fallback | Clean DB と continuity gate を保つ |
| BA GPU | Capability dependent | Ceres CUDA + cuDSS の両方が必要 |
| Alignment | IMU rotation | Scene scale を変更しない |
| Scale | Observable constraint only | Fixed baseline の自己一致を meter evidence にしない |
| Ground | Trajectory-local mode | Water / roof / point-density bias を避ける |
| Dense seed | Off | Camera-view 指標は上がるが free-view 改善が安定しない |
| LFStudio | MRNF UI defaults + GUT + mask | Aggressive eval LR と PPISP を default にしない |

### Quality presets

| Preset | Feature max edge | Features / image | Matches / pair | BA local / global |
|---|---:|---:|---:|---:|
| Draft | 1536 | 4096 | 8192 | 15 / 50 |
| Standard | 2048 | 8192 | 16384 | 25 / 100 |
| High | 3072 | 16384 | 32768 | 40 / 200 |

Standard が default。High は large outdoor / weak texture の final run 用。

## Frame extraction

```toml
[frame_extraction]
hwaccel = "cuda"
require_hwaccel = true
score_workers = 0
```

Hardware method list だけでなく actual source の一 frameを decode probe する。Required hardware が失敗した時は
software fallback を隠さない。Raw dual-fisheye は二 stream を一 process / 一 demux pass で decodeする。

Extractor は container packet の PTS / duration を presentation 順へ並べ、required sensor の frame count / PTS
sequence を検証する。Current dual stream tolerance は 0.5 ms。Selected capture は `frame_index / fps` ではなく実 PTS を持つ。
Spatial targets も PTS から選ぶので、VFR、non-zero start、gap に対応する。

Sharpness、exposure、feature count は全 sensor の worst value、motion は maximum。Lens 0 だけ良い capture を
採用しない。`score_workers=0` は logical CPU 全数、各 OpenCV instance は internal thread 1。

RTX 4070 Ti、3840² HEVC、35.96 s sample:

| Decoder | Speed | CUDA / software difference |
|---|---:|---:|
| CUDA | **8.98× realtime** | MAE 0、maximum 0 |
| Software HEVC | 0.85× realtime | Reference |

## Camera calibration、rig、rolling shutter

Adapter は manufacturer metadata を次へ正規化する。

```text
camera_system.json
  reference_sensor_id
  sensors[]
    id / image_key
    calibration image crop transform
    sensor-local projection
    cam_from_rig quaternion + translation
    shutter type / readout / scan direction / timestamp reference
```

Core の projection / rig / Prepare は `offset_v3`、X5、合成 calibration canvas を知らない。Current adapter は
`offset_v3` の二つの MEI sensor、full 6DoF relative extrinsic、local principal point を出力する。理想 180°
rotation や baseline 軸だけへ丸めない。

X5 の protobuf field 27 は calibration reference 5376×5376 から 5312×5312 への centered window crop を示す。
Adapter は各辺 32 px を引いてから 5312→decoded size を scale する。旧実装は 5376→3840 と直結して focal を
1.204819% 過小評価していた。Crop 修正後の 3840 px focal は lens0 3092.747、lens1 3106.128。Principal point
差は約 0.06 px しかないため center check では発見できず、外参で seam が改善しても円や直線を曲げていた。

MEI は COLMAP / LFStudio の model semantics を同時に評価した `THIN_PRISM_FISHEYE` へ近似する。

| Sensor | COLMAP RMS | LFStudio RMS | Combined maximum |
|---|---:|---:|---:|
| lens0 | 0.0396 px | 0.0421 px | 0.247 px |
| lens1 | 0.1029 px | 0.1112 px | 0.590 px |

旧 radial-only fit の lens1 maximum 4.464 px より小さい。Physical circle は calibrated forward ray
`theta < 89.55°` と交差し、principal point offset を無視した scalar clamp を使わない。Valid-region editor は
任意 frame / sensor を preview に使える。基準円は画像中心 `(0.5, 0.5)` に固定して半径だけを変更するため、
sensor / projection の歪みを center drift で隠さない。Lens ごとに独立した円 brush の operation list を持ち、
`add` は keep area を加え、`subtract` は不要物を除く。最終 mask は calibrated ray domain と custom region の積で、
feature、training、crop、export が同じ形を使う。

X5 metadata の readout は 21.244001 ms。Current default は readout 中の `|omega|` が 0.8° 以下になる frame を
優先する risk filter であり、dewarp ではない。Short clip の median は 0.423°、Parktest は median 0.411°、
P95 1.022°、maximum 8.482°。典型でも約 7–18 px、極端時は 100 px 超に相当する。

True correction には scan direction、frame timestamp reference、sensor crop、IMU-to-rig rotation、gyro bias、
video↔IMU offset / drift が必要。Unknown のまま top-to-bottom remap を default-enable しない。Correction は将来
raw sensor ごとの独立 Step とし、SAM / feature の前で capture-center native grid へ一回 resample する。
Opaque stitched ERP は stitch time-map が無ければ単純な row model を適用できない。

## Two SAM3 mask Steps

Default long edge は両 Step 2048 px、dilation 8 px。

```text
Feature:  person,camera operator,person's shadow,animal,sky,tree,vehicle,airplane,water
Training: person,camera operator,person's shadow
```

Feature mask は動く tree / cloud / water / person 等を SfM から広く除外し、Training mask は final detail を残す。

| Feature | Training | Feature / match | Export |
|---|---|---|---|
| On | On | Feature mask | Training mask |
| On | Off | Feature mask | Feature mask |
| Off | On | Physical valid region | Training mask |
| Off | Off | Physical valid region | Physical valid-region mask |

SAM3 は circle 外を含む full image で object context を認識し、結果だけを geometric valid region と交差する。
Video propagation は multi-prompt で遅く independent detection より miss が残ったため採用しない。

## Feature、matching、sparse reconstruction

| UI | Current parameter | Meaning |
|---|---|---|
| Feature image limit | `max_image_size` | `FeatureExtraction.max_image_size` |
| Features / image | `max_num_features` | Keypoint 上限 |
| Affine shape + DSP | `affine_shape + DSP` | View robustness、CPU cost 大 |
| SIFT thresholds | `peak_threshold` / `edge_threshold` | Response / edge filter |
| Feature matcher | `matcher_type` | Brute-force / LightGlue |
| Image pairing | `pairing` | Sequential / Exhaustive / Vocab-tree / Auto |
| Loop closure | `loop_closure` | Revisit edge |
| Transitive | `transitive_matching` | Track extension |
| Matches / pair | `max_num_matches` | Geometry verification 前の上限 |
| Two-view inliers | `min_num_inliers` | Verified pair の minimum |
| Guided matching | `guided_matching` | Geometry-guided second pass |
| Mapper | `mapper` | Global / Incremental |
| View graph calibration | `view_graph_calibration` | Unknown camera calibration |
| BA GPU | `ba_use_gpu` | Ceres CUDA + cuDSS |

旧 typo `SiftExtration.max_image_size` / `SiftExtraction.max_image_size` は COLMAP 4.1 で無効。

Single video Sequential は offset 1 / 2 / 4 / 8、loop closure は official 256K FAISS tree。COLMAP 4.1.1 の
folder-major rig bug は PR #4591 を backport する。Generalized rig verification は small sequential / loop graph
へ一回だけ適用し、one-pass transitive 後の巨大 graph へ繰り返さない。

Parktest clean graph は raw 253,061 pairs、verified 243,869 pairs、43,484,839 inliers。旧 scheduling は
6 h 16 min、現在は数時間の final RANSAC pass を避ける。FAISS indexing は CPU / sequential outer loop なので
GPU utilization が低いのは正常。

Global Mapper は最大三 deterministic seed を試し、registration、point count、trajectory continuity gate を満たす
model だけを採用する。全 seed が失敗した時は original DB から Incremental fallback。Continuous primary は
capture ごとの sensor center を平均し、default `maximum step / P95 <= 10`。

| Parktest final | Value |
|---|---:|
| Registered | 3388 / 3388 |
| Points3D | 791,750 |
| Mean reprojection | 0.941 px |
| Median / P95 step | 0.210 / 0.238 m |
| Maximum / P95 | 2.51× |
| Global Mapper | 69 min 4 s |

Fixed-rig intrinsics BA + weak-frame filter は reprojection を 0.941 px から 0.829 px へ改善し、training visual も
改善した。ただし thin pole の residual ghost は残る。

## Gravity、scale、ground、dense seed

Gravity alignment は IMU / camera trajectory の time offset と軸 permutation を評価し、model 全体へ rotation だけを
適用する。Parktest は 1,694 frame 中 1,501 inliers、offset 15 ms。

IMU acceleration の二重積分を metric scale に使わない。Fixed rig の reconstructed spacing が設定 baseline と
一致することは自己一致であり独立 evidence ではない。Shared stereo parallax、VIO、control point 等が無ければ
unscaled と明示する。Ground は trajectory 周囲の local mode から Y translation だけを求める。

RoMaV2 dense initialization は COLMAP camera model の pixel を native ray へ戻し、certainty、feature mask、
parallax、ray gap、native reprojection、voxel dedupe、hard cap を通った点だけを sparse modelへ追加する。Default
off。Plugin の pinhole 書換えは使わない。

## LFStudio export

Export Inspector が表示する一つの directory が dataset root。その directory 自体を LFStudio で選ぶ。
Application は LFStudio output path を制御しない。

```text
export_dataset/
├── images/sources/<source-id>/...
├── masks/sources/<source-id>/...
├── sparse/0/{rigs,cameras,frames,images,points3D}.bin
├── preview/{reconstruction.json,points.bin}
├── train_configs/{train_config.mrnf.json,train_config.mcmc.json,recommendations.json}
└── export_manifest.json
```

Fisheye JPEG は valid region を含む MCU boundary で lossless crop し、principal point、2D observation、mask を
同じ offset で更新する。Export は registered image だけを含み、camera center `C=-R^Tt`、rig/frame、mask count
を loader smoke で検証する。LFStudio は `rigs.bin` で pose を修正しないため、`images.bin` の pose が正本。

Stock LFStudio v0.5.3 / current master の `THIN_PRISM_FISHEYE` inverse は、各 iteration で元の distorted UV から
解く代わりに前回 UV から non-radial delta を繰り返し減算する。Lens1 では 2048 training scale でも maximum
約 11.4 px の forward/inverse 不一致となり、円を非対称に変形する。Default export workaround は SfM の精密な
THIN_PRISM model を保持し、学習用 `cameras.bin` だけを internally consistent な `OPENCV_FISHEYE` approximation
へ変換する。画像は再 resample しない。Crop-correct 3840 scale の approximation maximum は lens0 1.58 px、
lens1 4.52 px で、stock inverse より小さい。修正版 LFStudio を使う場合は Export Inspector で無効化できる。
Source build 用の修正は [scripts/patches/lichtfeld-thin-prism-inverse.patch](scripts/patches/lichtfeld-thin-prism-inverse.patch)。
`undistort=true` にも prism packing bug があるため workaround にしない。

Recommended training:

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --max-width 2048 --headless --train
```

- MRNF UI defaults、GUT、segment mask
- `undistort=false`
- PPISP / novel-view controller off
- Stock LFStudio THIN_PRISM workaround on
- max width 2048、general cap 2M、30,000 iterations

PPISP は exposure / vignetting / response の appearance model で denoiser ではない。旧 eval preset は means LR
6.4 倍、scaling LR 約 2.86 倍で Parktest の sky splat を ground へ崩したため使わない。Camera icon の brown / red
は relative photometric loss heatmap で camera disable ではない。

Sparse points は surface ではなく multi-view feature sample。地面が連続した点の床に見えないこと自体は正常。
Gaussian / sparse distribution statistics は warning として表示し、真の遠景を自動削除しない。

## Measured comparisons

### Feature / matcher: 38.17 s raw X5、104 captures / 208 images

| Feature / Matcher / Mapper | Feature | Match | Mapper | Registered | Points | Mean reproj. |
|---|---:|---:|---:|---:|---:|---:|
| Python ALIKED + LightGlue + Incremental | 89.7 s | 1604.1 s | 428.9 s | 208/208 | 52,505 | 1.242 px |
| **COLMAP SIFT + Brute-force + Global** | **28.9 s** | 33.1 s | 25.8 s | 208/208 | 27,202 | **0.917 px** |
| COLMAP ALIKED + Brute-force + Global | 243.0 s | **9.7 s** | **17.9 s** | 208/208 | 19,060 | 1.041 px |
| COLMAP ALIKED + LightGlue + Global | 243.0 s | 193.0 s | 24.7 s | 208/208 | 24,665 | 1.273 px |

### Mixed camera: 35.96 s X5 + iPhone 16 Pro stills

| Result | Primary only | + phone photos |
|---|---:|---:|
| Registered primary | 70 / 70 | 70 / 70 |
| Registered phone | — | 10 / 10 |
| Verified cross-source pairs | — | 168 |
| Points3D | 10,544 | 11,268 |
| Mean reprojection | 1.003 px | 1.010 px |

### RoMaV2 seed: same raw X5

| Initialization | Seed | 7k GS | Runtime | PSNR | SSIM | Free-view |
|---|---:|---:|---:|---:|---:|---|
| **Sparse** | 22,356 | 212,256 | **23:01** | 21.176 | 0.7477 | わずかに鮮明 |
| Dense + 10k | 32,356 | 313,236 | 25:49 | 21.623 | 0.7510 | 中間 |
| Dense + 50,671 | 73,027 | 683,732 | 25:38 | **22.104** | **0.7572** | 安定した改善なし |

Camera-view metric は上がるが 1M cap へ達し、free-view benefit が安定しないため default Sparse。

### Night denoise

FastDVDnet は raw target に対する SIFT masked PSNR を 25.028 から 24.478 へ約 0.55 dB悪化させた。
Detail hallucination / temporal inconsistency risk もあるため denoise Step、model、UI、dependency は削除済み。

### Raw calibration / official ERP reference

同じ 38.17 s recording の raw dual-fisheye と、official app が前後 3 s を trim して出力した 7680×3840 ERP を
別 project で比較した。Official ERP sparse reconstruction は 80/80 registered、23,176 points、mean reprojection
1.0859 px、trajectory continuous。

| Input / training | Images | Runtime | PSNR | SSIM | Observation |
|---|---:|---:|---:|---:|---|
| Legacy raw sparse | 190 | 23:01 | 21.176 | 0.7477 | Baseline |
| Calibrated raw sparse | 190 | 33:49 | 20.492 | 0.7430 | Seam は改善、円形変形と遠方 ghost 残存 |
| Calibrated raw + low-RS frame filter | 190 | 33:32 | — | — | 目視改善は小さい |
| **Official stitched ERP reference** | 80 | **30:21** | **21.770** | **0.8611** | 円形は良好、公式 seam に大きい局所 offset |

PSNR / SSIM は各 dataset 自身の training cameras に対する値で、camera / target が異なる行を直接 ranking しない。
Official ERP が高い SSIM を得た事実は trainer が単独で破綻していない control になるが、raw lens calibration と
official optical-flow / rolling-shutter stitch の効果を分離しない。Official output:

```text
lfstudio-runs/roma-short-official-erp-reference-1m-2048/
roma-short-official-erp-reference.ply
```

全 PLY は 1,000,000 records finite。同じ P95 scene-radius normalization の maximum-scale P99 は legacy raw
0.0506、calibrated raw 0.0720、low-RS raw 0.0677、official ERP 0.0711。Scale tail だけでは visual quality を
判定できない。

Overlap RoMa feature から約 0.06° rig correction を推定した実験は cross-sensor track と SfM residual を悪化させた。
Parallax / dynamics に引かれたため採用せず metadata extrinsic を保持する。Official ERP 自体も optical-flow warp を
含むので、raw lens との単純 global rotation difference を subpixel external calibration に使わない。

上表の raw training は window crop 修正前かつ stock LFStudio THIN_PRISM inverse の結果で、現在の quality
baseline にはしない。Seam 改善は full rig extrinsics の効果、残った shape deformation は内参 crop と consumer
inverse の二つの独立 bug で説明できる。Low-motion single-lens interior pilot では crop 修正により lens0 の
SO(3) RANSAC inlier が 50.9%→88.3%、angular median が 0.342°→0.108°へ改善した。

## Frontend

React 19 + TypeScript + Vite、Scene View は React Three Fiber / Three.js。Primary toolbar は
[liquid-glass-react](https://github.com/rdev/liquid-glass-react) の stable mode を限定的に使い、通常 panel / button は
CSS fallback。Icon は Fluent collection。

Scene grid は finite double-sided plane、depth-test on / depth-write off。Wheel は通常 zoom、right-look 中だけ
movement speed を 0.001×–16×で変更して中央へ倍率を表示する。Middle drag は screen-plane pan。Near / Far は
scene scale と speed に追従する。Theme background は applied theme と同じ update で反映する。

Source / Photo hierarchy は default collapsed。Inspector statistics も default collapsed。Photo と Dataset Camera は
source、capture、projection、mask channel を共通形式で表示する。

## Architecture

```text
Source adapter registry
  -> inspected topology + camera_system.json
  -> PTS-qualified captures
  -> vendor-neutral image catalog / camera groups / rig
  -> masks / SfM / similarity transforms / export
```

API process は orchestration、native / ML は worker subprocess。Project mutation は active job 中 409、cancel は
その worker process tree だけを終了する。Source adapter ID は closed Enum ではなく registry key。現在の
remaining extension debt は multi-resource DB と sensor-array capture manifest で、guide に migration order を記載する。

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

Actual UI path:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 \
SPHERE_E2E_SOURCE=/path/to/short.insv \
pnpm --dir frontend test:e2e
```

Gaussian PLY statistics:

```bash
backend/.venv/bin/python scripts/analyze_lfstudio_ply.py result.ply
```

Repository は CI workflow を持たない。

## License

Project 全体は [GNU General Public License v3.0 or later](LICENSE)、SPDX `GPL-3.0-or-later`。
Downloaded binary / model は各 component の license に従う。

## References and legacy plugin

- [COLMAP](https://github.com/colmap/colmap): camera、rig、Global Mapper、ALIKED、ERP
- [COLMAP PR #4591](https://github.com/colmap/colmap/pull/4591): sequential rig pairing fix
- [GLOMAP paper](https://arxiv.org/abs/2407.20219): global positioning
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio): loader、GUT、MRNF、mask、PPISP
- [PPISP](https://github.com/nv-tlabs/ppisp): appearance compensation
- [RoMaV2](https://github.com/Parskatt/RoMaV2): optional dense correspondence
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): multi-vendor motion metadata
- [Gyroflow](https://github.com/gyroflow/gyroflow): time / orientation / rolling shutter reference
- [Insta360 Desktop Media SDK](https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp): official stitch reference only
- [liquid-glass-react](https://github.com/rdev/liquid-glass-react): MIT glass decoration
- [Icônes Fluent](https://icones.js.org/collection/fluent) / [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons): UI icons
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): original legacy plugin reference

Frontend notices は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
