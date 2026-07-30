# sphere-reconstruct

Insta360 INSV、stitch 済み 360° video / image、通常 video、phone photo を同じ COLMAP
reconstruction へ統合し、LichtFeld Studio が直接 load できる dataset を作る local Web application。

Project は 1 個の primary source と任意個の supplemental source を持つ。Media container、projection、
physical camera、rig、mask purpose を別 data として保持するため、魚眼、ERP、通常 camera を同じ scene に
混在できる。処理は独立 Step に分かれ、重い成果物だけを個別に clear / regenerate できる。

## Quick start

### Windows

```powershell
.\scripts\start-windows.ps1
```

Command Prompt では `scripts\start-windows.cmd`。NVIDIA GPU / driver / compute capability を診断し、
必要な COLMAP runtime、jpegtran、FAISS vocabulary tree を固定 SHA-256 で自動導入する。

SAM3 も導入する場合:

```powershell
$env:SPHERE_WITH_SAM3="1"
.\scripts\start-windows.ps1
```

### Linux

COLMAP 4.1+、FFmpeg、FFprobe を system package で用意する。

```bash
./scripts/start-linux.sh
SPHERE_WITH_SAM3=1 ./scripts/start-linux.sh
```

### macOS

```bash
brew install colmap ffmpeg
./scripts/start-macos.sh
```

起動後は `http://127.0.0.1:8787` を開く。Start script は frontend build、uv sync、dependency install、
Doctor、server の順に実行し、native command が失敗した時点で停止する。

```text
GET /api/system/doctor
cd backend && uv run sphere-doctor
```

Doctor は workspace、filesystem roots、actual FFmpeg decoder、COLMAP 4.1、Ceres CUDA / cuDSS、FAISS tree、
jpegtran、SAM3 を検査する。Platform / GPU の詳細は [docs/setup-gpu.md](docs/setup-gpu.md)。

独立 `glomap.exe` は不要。COLMAP 4.1 に統合された `global_mapper` を使い、Windows start script が
COLMAP runtime 自体を取得する。

## Supported sources

| Input | Media | COLMAP camera | Role |
|---|---|---|---|
| Insta360 `.insv` | Video | 2× `OPENCV_FISHEYE` + physical rig | Primary / supplemental |
| Stitch 済み 360° video | Video | `EQUIRECTANGULAR` | Primary / supplemental |
| Stitch 済み 360° image folder | Images | `EQUIRECTANGULAR` | Primary / supplemental |
| 通常 / phone video | Video | `SIMPLE_RADIAL` | Primary / supplemental |
| 通常 / phone image folder | Images | EXIF group ごとの `SIMPLE_RADIAL` | Primary / supplemental |

別メーカーの 360 camera は stitched ERP としてすぐ追加できる。Raw dual-fisheye は camera 固有の
calibration adapter が必要で、native adapter は現在 Insta360 INSV に対応する。

Still folder は frame extraction を通らない。混合 project では video source だけを decode / select し、
still は original file を capture として catalog する。Source group は Scene Hierarchy で個別に折り畳める。
Phone photo に time sync は不要だが、primary source と共通 texture、parallax、cross-source match が必要。

## Pipeline Steps

```text
Sources
  -> inspect_source
  -> extract_frames                 # video だけ。still folder は収集
  -> fisheye valid-region editor    # 任意 frame / front / back
  -> prepare_images                 # canonical images、camera groups、rigs
       ├─ generate_feature_masks -> extract_features -> match_features -> reconstruct
       └─ generate_training_masks ---------------------------------------------┐
  -> align_reconstruction           # rotation only                            │
  -> restore_metric_scale           # observable stereo baseline -> meters     │
  -> position_ground                # trajectory-local ground -> Y=0           │
  -> export_dataset <---------------- resolved training / feature mask ---------┘
```

| Step | Output | 下流 invalidation |
|---|---|---|
| Source inspection | container、calibration、IMU、source metadata | Source branch 全体 |
| Frame extraction | selected captures | Prepare 以降 |
| Image preparation | canonical images、camera catalog、rig | Masks / feature / SfM 以降 |
| Feature masks | SfM 用 mask artifact | Feature / match / SfM / export |
| Training masks | final training 用 mask artifact | Export だけ |
| Feature extraction | database、descriptors、input workspace | Match 以降 |
| Matching | verified two-view graph | SfM 以降 |
| Sparse reconstruction | registered cameras / points | Alignment 以降 |
| Gravity alignment | rotation 済み sparse model | Scale 以降 |
| Restore real size | metric sparse model、または unscaled の明示 | Ground / export |
| Position from predicted ground | origin 補正済み model | Export |
| Export | LFStudio dataset root、preview、configs | なし |

Artifact は temporary directory へ書き、成功時だけ atomic replace する。Inspector statistics は default で
折り畳む。Object array を含む全統計を展開でき、Photo / Dataset Camera Inspector は共通 field と mask
channel を同じ形式で表示する。

## Progress contract

Progress の正本は SQLite event stream。Page refresh、WebSocket reconnect、別 browser で開始した job でも
`/stages` snapshot から percentage と activity を復元する。

- 数値 progress は Step 全体で単調
- 総量不明 phase は 0% と偽装せず indeterminate 表示
- Artifact publish 前は最大 99%、manifest / project state 更新後だけ 100%
- Late event は job ID で分離
- FFmpeg / COLMAP の image、block、batch、iteration counter を解析
- FAISS indexing、transitive iteration、SAM3 prompt / image を個別表示
- SAM3 PNG は atomic write 後すぐ preview index へ公開

## Default workflow

General default は X5 indoor / outdoor と mixed phone-photo test の実測を基準にする。

| Category | Default | Reason |
|---|---|---|
| Reconstruction mode | Native fisheye | Derived pinhole を作らず source pixel を維持 |
| Feature | SIFT | ALIKED より速く、再投影誤差も良かった |
| Matcher | Brute-force | SIFT の安定経路 |
| Pairing | Auto | Single=Sequential、small mixed=Exhaustive、large mixed=Vocab-tree |
| Loop closure | On | 周回 video の drift を抑える |
| Transitive | 1 iteration | 3-view track を作り、候補 pair の指数的膨張を避ける |
| View-graph calibration | Auto | Unknown intrinsics のみ。固定 physical calibration は skip |
| Mapper | Global + gated fallback | Global を試し、失敗時は original DB から Incremental |
| BA GPU | Capability dependent | Ceres CUDA + cuDSS の両方がある時だけ On |
| Alignment | IMU rotation | Scale を変更しない |
| Scale | Observable physical baseline only | 共有 stereo 視差なしでは meter と推測しない |
| Ground | Trajectory-local median | 水面 / roof / point density bias を抑える |
| LFStudio | MRNF UI default + GUT + masks | Distorted camera を扱い、未検証の appearance 補正を既定にしない |

### Quality presets

| Preset | Feature max edge | Features / image | Matches / pair | BA local / global |
|---|---:|---:|---:|---:|
| Draft | 1536 | 4096 | 8192 | 15 / 50 |
| Standard | 2048 | 8192 | 16384 | 25 / 100 |
| High | 3072 | 16384 | 32768 | 40 / 200 |

Standard が general default。High は large outdoor / weak texture の final run 用で、時間と memory が増える。

## Frame extraction and speed

```toml
[frame_extraction]
hwaccel = "cuda"
require_hwaccel = true
score_workers = 0
```

`auto` は FFmpeg の method list だけを信用せず actual source の 1 frame で decoder を probe する。
`require_hwaccel=true` は failure を明示 error にし、software fallback を隠さない。INSV は 2 video stream を
1 process / 1 demux pass で decode し、採用 candidate JPEG を再 encode しない。

RTX 4070 Ti、3840×3840 HEVC、35.96 s sample:

| Decoder | Speed | CUDA / software pixel difference |
|---|---:|---:|
| CUDA | **8.98× realtime** | MAE 0、maximum 0 |
| Software HEVC | 0.85× realtime | reference |

旧 parktest extraction は software decode と lens ごとの重複 I/O により 1694 captures で 3509.906 s。
これは修正前 baseline で、現在の expected path ではない。

## Camera calibration and valid region

INSV native は heuristic `f=W/3.5、k=0` を使わない。`offset_v3` の front / back MEI calibration を別々の
8-parameter `OPENCV_FISHEYE` へ least-squares fit し、source ごとの 2-camera physical rig にする。

COLMAP perspective fisheye は 180° 以下の forward hemisphere 専用。Physical circle が 180° を越える
場合、単一の circle radius は使わない。Calibrated `OPENCV_FISHEYE` の principal point、fx / fy、k1–k4
から各 pixel の ray angle を判定し、`theta < 89.55°` と user-selected physical circle の積集合を feature /
training mask にする。Physical circle と principal point がずれても背面 ray は残らない。Parktest fit:

| Lens | RMS | Maximum | Physical radius / width | Ray limit |
|---|---:|---:|---:|---:|
| Front | 0.525 px | 1.564 px | 0.448096 | 89.55° |
| Back | 1.541 px | 4.464 px | 0.468613 | 89.55° |

旧 scalar-circle clamp は principal-point offset を表現できず、Parktest export の white training pixels に
front 1.61%、back 0.96% の hemisphere 外領域を残した。最初の LFStudio run は 5,200 iter で停止し、mask
を calibrated-ray intersection へ移行してから最初から再実行した。

Valid-region editor は抽出済みの任意 frame と lens を切り替えられる。SAM3 は circle 外を含む full image で
object context を認識し、推論後に valid region と合成する。円外に胴体、円内に腕だけ見える人物でも context
を失わない。Region schema v2 は SVG と同じ image-edge coordinates。旧 schema は自動 offset せず UI に
review / resave を要求し、user 調整値を勝手に移動しない。

## Two independent SAM3 mask Steps

両 Step の default long-edge inference limit は 2048 px、dilation は 8 px。

```text
Feature:  person,camera operator,person's shadow,animal,sky,tree,vehicle,airplane,water
Training: person,camera operator,person's shadow
```

Feature prompt は特徴照合から除外する対象。風で動く tree、sky / cloud、water、animal、vehicle を広く除外。
Training prompt は final detail を残す。Scene に存在しない追加 object prompt は誤検出を増やすため
含めない。

| Feature | Training | Feature / match | Export |
|---|---|---|---|
| On | On | Feature mask | Training mask |
| On | Off | Feature mask | Feature mask |
| Off | On | Physical valid region | Training mask |
| Off | Off | Physical valid region | Physical valid-region mask |

Export は 1 mask channel だけを書く。SAM3 channel が無い場合も camera-model physical valid region を書き、
fisheye padding を黒い training target と誤認させない。White=keep、black=ignore。

### Video propagation decision

Parktest front 64 frames / person prompt:

| Method | Time | Missed frames |
|---|---:|---:|
| Independent images、shared embedding | 約 55 s / 13 prompts 相当 | 8 |
| Forward propagation | 21.20 s / 1 prompt | 14 |
| Bidirectional 16-frame | 20.44 s / 1 prompt | 7 |
| Bidirectional 8-frame | 23.06 s / 1 prompt | 4 |

Propagation は prompt ごとに再実行するため multi-prompt では遅く、8-frame でも独立 detection より漏れる。
速度と recall の両方で default にできないため採用しない。

## Feature extraction and matching

UI parameter:

| Display | Current name | Meaning |
|---|---|---|
| Feature image limit | `max_image_size` | 長辺上限。正しい COLMAP 4.1 flag は `FeatureExtraction.max_image_size` |
| Features / image | `max_num_features` | Keypoint 上限 |
| Affine shape + DSP | `affine_shape + DSP` | Viewpoint robustness、CPU cost 大 |
| SIFT thresholds | `peak_threshold` / `edge_threshold` | Response / edge-like keypoint filter |
| Feature matcher | `matcher_type` | Brute-force / LightGlue |
| Image pairing | `pairing` | Sequential / Exhaustive / Vocab-tree / Auto |
| Loop closure | `loop_closure` | 離れた revisit edge を追加 |
| Transitive | `transitive_matching` | 1 pass で 3-view 以上の track を追加 |
| Matches / pair | `max_num_matches` | Geometry verification 前の上限 |
| Two-view inliers | `min_num_inliers` | Verified pair の最小 inlier |
| Guided matching | `guided_matching` | Geometry-guided second pass、default Off |
| Mapper | `mapper` | Global / Incremental |
| View graph calibration | `view_graph_calibration` | Unknown intrinsics / rotation graph の calibration |
| BA GPU | `ba_use_gpu` | Ceres CUDA + cuDSS |

旧 typo `SiftExtration.max_image_size` と `SiftExtraction.max_image_size` は COLMAP 4.1 では無効。

Single-source Sequential は quadratic overlap 4、つまり 1 / 2 / 4 / 8 frame offset。Loop closure は official
256K FAISS tree を使う。旧 FLANN tree は file が存在しても crash するため、installer と Doctor は header を
検証する。256K indexing は CPU / sequential outer loop、SIFT pair matching は GPU。Parktest clean run では
indexing が約 56 分、GPU pair matching が約 4 分だった。

COLMAP 4.1.1 の folder-major rig pairing bug は sensor directory 境界を temporal neighbor と誤認する。
Pinned runtime は upstream [PR #4591](https://github.com/colmap/colmap/pull/4591) の same-camera guard を
backport する。Matching は clean database から作り、sequential / loop graph に generalized rig verification
を 1 回適用してから transitive extension を行う。拡張 edge の individual two-view geometry は維持し、巨大な
final graph 全体へ generalized RANSAC を繰り返さない。

Parktest clean run は raw 253,061 pairs、verified 243,869 pairs、43,484,839 inliers を得た。旧 scheduling は
transitive 後の巨大 graph 全体へ rig verification を適用したため総計 6 h 16 min を要した。Sequential / loop
graph だけを generalized verification する現在の scheduling は、その数時間の final pass を避ける。

Transitive pass は 1 回で十分。Parktest で 1 回目 265 batches に対し、2 回目は 2492 batches（約 249 万
pairs）へ膨張した。閉環後の 2 / 3 回目は quasi-exhaustive になり、速度と outlier risk の両方で不利。

## Sparse reconstruction and continuity gate

Global Mapper を最大 3 deterministic seed で試す。Primary registration、point count、trajectory continuity を
満たす最初の model を採用する。全 Global seed が失敗した場合、view-graph calibration 前の original DB から
Incremental Mapper を実行し、同じ gate を再適用する。

Continuous video は capture ごとに sensor camera center を平均し、相隣 step の P95 と maximum を比較する。
Default `max / P95 <= 10×`。評価に十分な連続 capture が無ければ silent pass しない。旧 parktest model は
3388/3388 images を登録しながら discontinuity を持ったため reject された。Clean database、修正版 pairing、
feature mask、1-pass transitive から再計算した final model は seed 0 で gate を通過した。

| Metric | Rejected old model | Accepted final model |
|---|---:|---:|
| Registered images | 3388 / 3388 | **3388 / 3388** |
| Points3D | — | **791,750** |
| Mean reprojection | — | **0.941 px** |
| Median / P95 step | 0.792 / 1.112 | **0.210 / 0.238 m** |
| Maximum step | 62.92 | **0.597 m** |
| Maximum / P95 | 56.6× | **2.51×** |
| Outlier jumps | 2 | **0** |
| Global Mapper time | — | 69 min 4 s |

Registration 数だけを成功判定に使わない。Global positioning は verified pair translation を直接拘束せず、
3-view 以上の bearing track で位置を解くため、pair が多くても piecewise jump は起こり得る。

Bundle Adjustment は pose、intrinsics、3D points を同じ reprojection objective で最適化する。Feature GPU と
BA GPU は別 capability。Official Windows CUDA archive の Ceres は CPU-only。Pinned CUDA 12.8 runtime は
native sm89、dense CUDA、sparse cuDSS 0.8.0.10 を Doctor と実 mapper log の両方で検証する。Native
ALIKED 用には cuFFT、NVRTC、cuDNN 9.20 と ONNX CUDA provider も app-local に含める。

## Gravity, real scale, and ground origin

### Gravity alignment

Primary INSV の exposure timestamp と accelerometer から gravity consensus を推定し、model 全体へ rotation
だけを適用する。Dataset は `-Y up`。旧 trajectory-diameter normalization は physical scale を壊すため廃止。
Parktest は 1,694 frame 中 1,501 inliers、time offset 15 ms で整列し、camera vertical span は 5.725 から
2.029 m になった。

### Restore real size

IMU orientation だけでは SfM scale は決まらない。Acceleration 二重積分は bias が距離へ発散するため、
scale 推定には使わない。Known physical sensor baseline は、同じ capture の複数 sensor が十分な 3D point を
共有し、baseline が reprojection objective から観測可能な場合だけ meter scale の根拠にする。

Rig 外参を固定すると reconstructed sensor distance は設定 baseline と常に一致するため、その一致だけで metric
と判定するのは循環論法。Parktest の背面 dual-fisheye は共有 stereo 視差が不足し、intrinsics BA 後も camera
trajectory scale が 1.72 倍変化した。従ってこの source は meter と表示せず、VIO / control point / overlapping
stereo など独立の尺度拘束が得られるまで unscaled とする。

### Position from predicted ground

Primary trajectory 周囲の local points を集め、camera sample ごとの ground mode を等 weight で集約する。
遠方の water、roof、point density が origin を支配しない。Terrain は変形せず Y translation だけを適用する。
Parktest は 246,780 support points、91.3% trajectory coverage から Y を −0.579 m 平行移動した。予測 camera
height median 0.532 m は、抽出 frame で確認できる低い手持ち姿勢と一致した。

## LFStudio export

Export Inspector が表示する 1 個の directory が dataset root。その directory 自体を LichtFeld Studio で
選ぶ。Application は LFStudio output path を表示・変更・削除しない。

```text
export_dataset/
├── images/sources/<source-id>/...
├── masks/sources/<source-id>/...       # resolved channel がある時だけ
├── sparse/0/{rigs,cameras,frames,images,points3D}.bin
├── preview/{reconstruction.json,points.bin}
├── train_configs/
│   ├── train_config.mrnf.json
│   ├── train_config.mcmc.json
│   └── recommendations.json
└── export_manifest.json
```

Native fisheye JPEG は valid circle を含む MCU boundary で `jpegtran` lossless crop し、camera principal
point、2D observation、mask を同じ offset で更新する。Export は registered image だけを含み、camera center
`C=-Rᵀt` の non-zero trajectory、rig/frame、image/mask count を LFStudio loader smoke で検証する。

GUT training では `undistort=false` のまま original `OPENCV_FISHEYE` model と distortion coefficient を
rasterizer へ渡す。LFStudio import 時の `Undistort: ... -> ...` log は hypothetical crop metadata の事前計算で、
training RGB / mask をその小さい解像度へ変換した記録ではない。Actual image loader の `max-width` が学習 tensor
解像度を示す。

疎点群は surface reconstruction ではなく、複数 view で再現できた feature の 3D sample だけを表す。従って地面が
連続した点の床にならないこと自体は正常。Export statistics は point radius の median / P95 / P99 / maximum と
P99-to-median ratio を表示し、ratio が 5 を超える場合は `wide_sparse_point_distribution` を出す。遠方の山や建物も
含み得るため自動削除せず、学習 hyperparameter の異常と区別して診断する。

## Recommended LFStudio training

```text
LichtFeld-Studio --config <dataset>/train_configs/train_config.mrnf.json \
  --data-path <dataset> --max-width 2048 --headless --train
```

- Strategy: MRNF
- Renderer: GUT
- Mask: resolved segment mask
- PPISP + novel-view controller: Off
- Maximum width: 2048
- General Gaussian cap: 2M
- Iterations: 30,000

生成 config は LFStudio v0.5.3 の `mrnf_defaults()` を基準にする。`eval/mrnf_optimization_params.json` は
benchmark 用で、UI default より means LR 6.4 倍、scaling LR 約 2.86 倍、SH LR 2.5 倍、opacity LR
約 2.08 倍高い。parktest では scene scale 22.895 により実効 means LR が 0.00293 となり、完走しても
geometry が壊れたため general preset として使用しない。

parktest の実測:

| Config | Runtime result | Geometry result |
|---|---|---|
| Eval preset、2.4M / 2304、PPISP controller | 約 7k で OOM | 未評価 |
| Eval preset、2.4M / 2304、PPISP controller off | 約 10k で OOM | 未評価 |
| Eval preset、2M / 2048、PPISP on / controller off | 30,000 完走 | **不合格**。28,600 splat が 10 m 超、3,238 splat が 50 m 超 |
| LFStudio UI default + GUT + segment mask、2M / 2048、PPISP off | 30,000 完走、61分18秒 | 数値改善、visual acceptance 待ち |

不合格 PLY の巨大 splat は青 / 白が中心で、空が地面へ現れる直接原因だった。PPISP を完全に無効化しても
eval preset で再現するため、PPISP 単独の問題ではない。PPISP は exposure、vignetting、white balance、
camera response を学習する appearance model で denoiser ではなく、現時点では opt-in experiment とする。

UI default の完走 PLY は median maximum scale が 0.0921 m から 0.0298 m、P99 が 14.08 m から 6.58 m、
10 m 超が 28,600 から 12,434、10 m 超かつ opacity 0.5 超が 8,858 から 1,091 へ減った。改善は明確だが
50 m 超が 1,404 残り maximum 2,345 m の低 opacity splat もあるため、数値だけで品質合格にせず LFStudio で
sky / ground separation を visual acceptance する。

Camera icon の brown / red は relative photometric loss heatmap で、camera disable を意味しない。Generic
SceneNode Transform が 0 でも actual pose は COLMAP `R/t` にある。

## Measured comparisons

### Features: 38.17 s X5、104 rig captures / 208 images

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
| Registered phone | — | **10 / 10** |
| Verified cross-source pairs | — | **168** |
| Cameras | 2 | 3 |
| Points3D | 10,544 | **11,268** |
| Mean reprojection | 1.003 px | 1.010 px |

### Night-scene denoise

FastDVDnet preprocessing と raw training を 30k まで比較した。

| Training / target | Pose | PSNR | SSIM |
|---|---:|---:|---:|
| FastDVDnet / raw | ALIKED | 24.154 | 0.8043 |
| Raw / raw | ALIKED | 24.413 | 0.8046 |
| FastDVDnet / FastDVDnet | SIFT | 24.640 | **0.8470** |
| **Raw / raw** | **SIFT** | **24.920** | 0.8056 |

同じ clean target の SIFT masked PSNR は denoised training 24.478、raw training 25.028。Denoise は約
0.55 dB 悪化し、detail hallucination / temporal inconsistency risk も増えるため、denoise Step、model、UI、
dependency は削除した。Night scene は raw image + SIFT pose を基準にし、appearance 補正は別の opt-in
比較として扱う。

### Parktest pose refinement

Fixed-rig BA で per-sensor focal length、principal point、fisheye distortion と各 frame pose を再最適化し、
弱い 46 / 3,388 images を training dataset から除いた。COLMAP mean reprojection error は 0.941 px から
0.829 px へ低下し、front camera の 2 px 超 observation は 17.5% から 9.1%、back は 12.5% から 7.7% へ
減少した。同じ MRNF + GUT + mask、2M / 2048、PPISP off の 30k training は visual comparison で detail と
全体画質が改善し、遠景の大 Gaussian も問題を生じなかった。ただし細い pole 等は理想精度に未達で、次の改善は
Gaussian cap ではなく calibration prior、weak-frame policy、または visual-inertial pose constraint を対象にする。

## Frontend UI

Frontend は React 19 + TypeScript + Vite、Scene View は React Three Fiber / Three.js。Toolbar と primary
action island は [liquid-glass-react](https://github.com/rdev/liquid-glass-react) の stable `standard` mode を
progressive decoration として使い、Firefox / Safari の部分対応時も CSS glass fallback で操作性を保つ。
Icon は [Icônes Fluent collection](https://icones.js.org/collection/fluent) で選び、Microsoft official
`@fluentui/react-icons` の atomic SVG path だけを import する。

Scene grid は finite double-sided plane、depth-test on / depth-write off。Ground positioning 済み model は
viewer Y=0 に置き、opaque alpha-tested point depth の後ろへ描画するため、view rotation 時に grid が消える、
点群を透過して前面へ張り付く、巨大 infinite plane の precision flicker を避ける。

Scene View の wheel は fly / middle-pan の移動速度を `0.125x` から `16x` の倍数段階で変更し、変更値を view
中央へ短時間表示する。Wheel 自体は camera position を前後移動しないため、inspection 中に視点を崩さない。

## Architecture and extension

```text
project_source table
  ├─ one primary source
  └─ N supplemental sources
         ↓
source-scoped inspect / extract / prepare
         ├─ feature mask -> feature workspace -> matching -> SfM
         └─ training mask ------------------------------------┐
                                                              ↓
rotation -> metric scale -> ground origin -> registered-only export
```

新しい raw camera は adapter、calibration、prepare implementation を追加する。Global pipeline に camera 名の
条件分岐を増やさず、camera catalog、per-sensor intrinsics、rig contract を実装する。Fisheye region は X5
constant ではなく source calibration / user region から導出する。

API process は orchestration、native / ML は worker subprocess。Project mutation は active job 中 409、cancel
はその worker process tree だけを終了する。

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

Actual UI path test:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:8787 \
SPHERE_E2E_SOURCE=/path/to/short.insv \
pnpm --dir frontend test:e2e
```

## License

Project 全体は [GNU General Public License v3.0 or later](LICENSE)、SPDX identifier
`GPL-3.0-or-later`。Downloaded COLMAP、Ceres、NVIDIA runtime、model は各 component の license に従う。

## References and legacy plugin

- [COLMAP](https://github.com/colmap/colmap): camera、rig、Global Mapper、ALIKED、ERP
- [COLMAP PR #4591](https://github.com/colmap/colmap/pull/4591): sequential rig pairing fix
- [COLMAP PR #4590](https://github.com/colmap/colmap/pull/4590): panorama benchmark / consistency gate
- [GLOMAP paper](https://arxiv.org/abs/2407.20219): global positioning formulation
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio): loader、GUT、MRNF、mask、PPISP
- [PPISP](https://github.com/nv-tlabs/ppisp): appearance compensation / controller
- [liquid-glass-react](https://github.com/rdev/liquid-glass-react): MIT-licensed progressive glass decoration
- [Icônes Fluent collection](https://icones.js.org/collection/fluent) / [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons): UI icon source
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser): Insta360 metadata
- [Gyroflow](https://github.com/gyroflow/gyroflow): IMU semantics
- [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch): INSV container research
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): original legacy plugin workflow reference

Frontend dependency attribution and retained MIT notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
