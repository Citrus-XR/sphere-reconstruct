# Sphere Reconstruct

Raw multi-fisheye、stitch 済み 360° media、通常 video、phone photo を共通の capture / camera / rig model へ正規化し、COLMAP sparse reconstruction と LichtFeld Studio training dataset を生成する desktop-oriented pipeline である。

License は **GPL-3.0-or-later**。Original source は変更せず、各 Step の成果物は個別に clear / regenerate できる。

## Current recommendation

- Frame extraction: **Spatial optical flow**。Quality gate の後、camera motion に基づく可変間隔で frame を選ぶ
- Raw dual-fisheye: versioned omnidirectional calibration を読み、image preparation 内部で exact `OMNI -> OPENCV_FISHEYE` normalization
- Feature / matcher: SIFT + brute-force
- Resolution limits: source projection / native dimensions から自動算出し、mixed source は項目ごとの最大要求を採用
- Calibrated dual-fisheye mapper: **Incremental Mapper**
- Video source: COLMAP の video global-BA schedule を使い、最終 BA を保ったまま中間 global BA の過密実行を避ける
- Incremental output: Mapper 内の逐次 color sampling を無効化し、終了後に全 observation から parallel final RGB extraction
- LFStudio: MRNF + GUT + segment mask、PPISP off、controller off、`undistort=false`
- Training width cap: 2048。SfM / rectification resolution を 2048 に固定する意味ではない

COLMAP 4.1.1 の `OPENCV_FISHEYE` は各 sensor の forward hemisphere、すなわち単眼 180° 以下を前提とする。Native dual-fisheye は各 sensor を光軸から 89.55° までに制限し、physical circle / brush validity と交差させる。前後 sensor は独立した fisheye と固定 rig のまま保持し、ERP を生成しない。元の単眼 180° 超の周辺 overlap は SfM に渡さない。Native dual-fisheye で Global Mapper を選ぶと UI warning を表示するが、選択自体は変更しない。

## Quick start

### Windows

```powershell
scripts\start-windows.ps1
```

または:

```bat
scripts\start-windows.cmd
```

Windows launcher は startup log を `runtime/logs/launcher-*.log` に保存し、失敗した工程と exit code を表示する。CMD window は結果表示後に key 入力を待つ。正常起動後は browser を開き、window を閉じても server は background で継続する。二回目の起動は既存の healthy server を使う。導入済み環境の download / build を省く場合は `-SkipSetup`、browser を開かない場合は `-NoBrowser` を付ける。自動実行では `SPHERE_LAUNCHER_NO_PAUSE=1` を設定する。

### Linux

```bash
./scripts/start-linux.sh
```

### macOS

```bash
./scripts/start-macos.sh
```

Launcher は uv / pnpm environment、FFmpeg、COLMAP、jpegtran、optional model を診断し、不足項目を明示する。既定 server は `127.0.0.1:8787`。詳細は [Runtime / GPU setup](docs/setup-gpu.md) を参照する。

### AI Agent による初回 setup

AI Agent に setup を依頼する前に、必要な optional feature と runtime configuration を環境変数で明示する。これらは launcher / application の設定であり、AI Agent 自体を起動する変数ではない。未指定の optional feature は導入対象にしない。以下の `<config-file>` と `<media-directory>` は利用環境の設定ファイルと素材 directory に置き換える。

```bash
# SAM3 mask generation と RoMaV2 dense initialization が必要な場合だけ有効にする。
export SPHERE_WITH_SAM3=1
export SPHERE_WITH_DENSE=1

# 既定の runtime/config.toml 以外を使う場合だけ指定する。
export SPHERE_CONFIG="<config-file>"

# Source browser が参照してよい media root。JSON array で指定する。
export SPHERE_FILESYSTEM__ALLOWED_ROOTS='["<media-directory>"]'
```

PowerShell では同じ値を `$env:SPHERE_WITH_SAM3`、`$env:SPHERE_WITH_DENSE`、`$env:SPHERE_CONFIG`、`$env:SPHERE_FILESYSTEM__ALLOWED_ROOTS` に設定する。

Agent には次の手順を依頼する。まず対象 OS、shell、GPU の有無、上記 optional feature を確認する。次に対象 platform の launcher を実行し、Doctor の結果から不足する `uv`、`pnpm`、FFmpeg / FFprobe、COLMAP、`jpegtran`、model / checkpoint を特定する。依存同期は `uv sync --locked --inexact` を使い、導入済み optional dependency を保持する。診断は `.venv` の Python から直接実行し、再同期しない。Windows launcher は未設定の COLMAP、`jpegtran`、vocabulary tree に project-local installer を使う。Linux / macOS の native binary は system package manager で導入し、`runtime/config.toml` または `SPHERE_CONFIG` の `[binaries]` に path を設定する。不足項目を導入した後は launcher を再実行し、Doctor が必要な capability を確認できた時だけ server を起動する。

OS の package manager、GPU driver、CUDA toolkit、media root のように自動判定できない項目だけは、Agent が不足内容と実行する install command を確認してから導入する。Python と frontend の依存関係は launcher がそれぞれ `backend/.venv` と `frontend` に導入するため、global Python package / npm package として導入しない。

Windows launcher は共通 Python service manager で background 起動する。Linux / macOS の launcher は foreground 起動するため、初回 setup 後に terminal / SSH session から独立して backend を維持する場合は同じ manager を使う。

```bash
# Linux / macOS
backend/.venv/bin/python scripts/server_service.py start
backend/.venv/bin/python scripts/server_service.py status
backend/.venv/bin/python scripts/server_service.py stop
```

```powershell
# Windows
backend\.venv\Scripts\python.exe scripts\server_service.py start
backend\.venv\Scripts\python.exe scripts\server_service.py status
backend\.venv\Scripts\python.exe scripts\server_service.py stop
```

## Source model

| Input | Media | Projection | Role |
|---|---|---|---|
| Insta360 `.insv` | Video | calibrated dual-fisheye | Primary / supplemental |
| Generic stitched 360 | Video / images | equirectangular | Primary / supplemental |
| Generic camera | Video / images | perspective | Primary / supplemental |
| Phone folder | Images | EXIF-assisted perspective | Supplemental detail |

複数 source では video だけ frame extraction を行い、image folder は original still を capture として登録する。Source group と Photos hierarchy は個別に fold できる。

Adapter は manufacturer metadata を vendor-neutral な `camera_system.json` へ変換する。

```text
camera_system.json
  coordinate_system
  reference_sensor_id
  sensors[]
    id / image_key
    calibration_image_transform
    projection intrinsics
    cam_from_rig rotation + translation
    shutter type / readout / scan direction / timestamp reference
```

Device-specific crop、distortion、baseline、rolling-shutter readout は adapter に閉じ込め、他 camera の default にしない。Insta360 adapter は機種名で parameter を固定せず、素材内で確認できた V3 / V6 calibration を選択する。未知 version は明示 error とする。新 format の contract は [Adding 360 camera formats](docs/adding-360-camera-formats.md) にまとめている。

## Pipeline

```text
inspect_source
  -> extract_frames
  -> source_region
  -> prepare_images
  -> generate_feature_masks
  -> generate_training_masks
  -> extract_features
  -> match_features
  -> reconstruct
  -> align_reconstruction
  -> restore_metric_scale
  -> scene_alignment
  -> cleanup_sparse
  -> dense_initialization
  -> export_dataset
```

| Step | Output | Notes |
|---|---|---|
| Inspect | metadata、adapter、camera system、IMU | original media は immutable |
| Frames | selected capture、PTS、quality、RS risk | multi-video sensor を同時選択 |
| Region | source・sensor ごとの有効領域 | 魚眼は既定 r=0.5 の中心固定円、普通カメラ/ERP は全画像。各 source に add/subtract brush |
| Prepare | canonical catalog、camera groups、rig | internal fisheye normalization を含む |
| Feature mask | SfM から除く dynamic region | 独立 regenerate |
| Training mask | final training から除く region | 独立 regenerate |
| Features | COLMAP DB、known cameras / rigs | SIFT / ALIKED |
| Matching | verified graph | sequential / exhaustive / vocab tree |
| Reconstruct | sparse model、base Scene View preview | Global / Incremental、final point RGB completion |
| Gravity | world rotation | IMU と trajectory の time alignment |
| Metric | similarity scale | observable physical evidence が必要 |
| Scene alignment | yaw rotation + Y translation | orthogonal wall confidence と local ground confidence を記録 |
| Sparse cleanup | full-track stability filtering | 全観測の不確実性と capture 留保予測で不安定点を除去 |
| Dense seed | optional RoMaV2 points | default off |
| Export | LFStudio-loadable folder | output root は一つだけ表示 |

Top bar の連続生成は未生成 Step を既定値で順番に実行し、failure / cancel / stop で停止する。Internal fisheye normalization は独立 Step として hierarchy、進捗、統計、clear / regenerate を表示する。Feature Mask、Training Mask、Features を直接実行した場合も backend execution plan が先に normalization を生成または cache reuse する。

## Progress contract

実行中 Step は次の二つを独立して保持する。

- `progress_event`: 最新の数値 progress と count（例: `1487/1488`）
- `activity_event`: 最新の phase / native tool activity（例: Global refinement）

Activity-only 行は percentage や count を上書きしない。Steps hierarchy は percentage と短い detail を同時に表示し、Inspector は count と current activity を別行で表示する。黄色の Step が UI 設定と生成済み manifest の差で stale になった場合は、`min_sharpness: 0 → 135` のように未適用 parameter を行内へ表示する。数値 progress がまだ無い間は quarter-arc を SVG viewBox 中心で回転し、座標移動を animation として使わない。Page reload、WebSocket reconnect、別 browser から開始した job でも `/stages` snapshot から両方を復元する。Backend request が失敗した時は cached `running` snapshot を実行中として扱わず、「Backend disconnected」と明示して elapsed timer / spinner / Stop を停止する。大量の per-image / per-pair update は `kind=progress` として扱い、Console へ spam しない。

Backend は単一 API process で稼働し、Job の開始、成果物クリア、project 削除、source 変更、有効領域保存を project ごとの排他制御で直列化する。DB に `queued` / `running` Job がある場合、競合する request は HTTP 409 を返す。Cancel は Worker の停止を確認してから `cancelled` にする。他 project の操作、読み取り、UI 設定保存はこの排他制御の対象外であり、filesystem 処理中も DB transaction を保持しない。

Worker の起動失敗は Job を `failed` にし、元の診断を event に記録する。Stage の成功は出力公開、downstream invalidation、manifest の原子置換、project state 更新が完了した時点で確定する。公開途中の例外も Job / Stage を `failed` にし、`running` の表示だけを残さない。途中まで書かれた manifest は cache として公開しない。

成果物クリア dialog は「全 Step」「Source inspection / Frames を保持」「custom checklist」を提供する。Custom checklist の明示的な選択は project UI state に保存し、project switch / browser reload 後も project ごとに復元する。Upstream Step を選ぶと consumer closure を UI でも自動選択し、backend は fisheye normalization を含む同じ closure を一つの quarantine transaction で処理する。全対象を同一 filesystem 内の `.pipeline/trash/<transaction>` へ rename してから manifest / state を切り替える。`export_dataset` は一つの Stage output として扱い、LFStudio が作成した `output/` や `.licht` を含む内容も同時に削除する。

Windows file lock などで rename が失敗した場合、移動済みの成果物を rollback し、復元完了時は HTTP 423 を返す。Rollback 自体が失敗した場合は backup を削除せず、元の場所と backup の対応を `recovery_required.json` に記録し、error log に保管場所を残す。Rename 後の物理削除だけが失敗した場合は bounded retry を行い、残件を `pending_cleanup.json` と API `pending_cleanup` に公開して次回操作で再試行する。自動削除の対象はこの pending marker がある確定済みの残件だけであり、未確定の quarantine と復旧待ち backup は除外する。Clear success 時は in-flight preview query を cancel し、削除対象の query cache を null にするため Scene View に旧 point cloud を残さない。

Incremental Mapper の denominator は image 数ではなく COLMAP の rig frame 数である。同一 capture の二つの fisheye sensor は 2 images / 1 frame と数える。COLMAP の `num_reg_frames` は次の image を登録する直前の値なので、次の registration / Global refinement が成功を示した時点で 1 frame を commit する。Registration、visible point、Global refinement pass を structured detail として表示する。Global refinement 内部の track merge / retriangulation は CPU task で、GPU BA utilization と同じ意味ではない。

Sparse reconstruction publish 時に `reconstruct/preview` を同時生成するため、次の alignment Step を待たず Scene View へ camera と point を表示する。Gravity alignment、scale、scene alignment、cleanup、dense が生成された場合は preview API が最も下流の完成済み preview を選ぶ。

## Exact fisheye normalization

Raw sensor image は camera metadata だけを書き換えず、RGB、validity、camera model を同時に変換する。

```text
target OPENCV pixel
  -> target fisheye ray
  -> source omnidirectional pixel
  -> one backward resample
```

- Same-resolution、Lanczos4、PNG
- Sensor ごとに別 intrinsics / distortion を使用
- INSV MP4 stream order は offset calibration lens order と逆なので、抽出時に stream1→lens0 / stream0→lens1 へ正規化
- `cam_from_rig`、capture timestamp、camera center は変更しない
- SAM、SIFT、matching、SfM、export は同じ rectified catalog を読む
- Physical circle と custom add/subtract region は同じ map で nearest resample

V6 radtan-pro は `a=p1+p3·r²`、`b=p2+p4·r²` とし、水平 decentering を `(r²+2x²)·a + 2xy·b`、垂直を `2xy·a + (r²+2y²)·b` として適用する。Sensor order とこの非対称項は二眼間で相殺されないため、どちらも adapter contract の一部である。

96 capture の比較:

| Geometry | Crop-correct THIN | Exact OPENCV |
|---|---:|---:|
| Registered | 192 / 192 | 192 / 192 |
| Points3D | 28,060 | 29,974 |
| Observations | 169,246 | 181,549 |
| Mean reprojection | 1.105 px | 1.080 px |
| P95 reprojection | 1.987 px | 1.947 px |

Two-pass diagnostic は lens0 / lens1 で 42.02 / 40.47 dB、MAE 0.445 / 0.390。Production は一回だけ resample する。

Physical lens が 180° を超えても、stock COLMAP / LFStudio は `z <= 0` ray を拒否するため consumer contract を 89.55° half-angle に制限する。Experimental full-fisheye COLMAP patch は same-capture pair を回復したが、Global Mapper の later track / positioning stage が stereo constraint を保持しなかったため production には採用していない。

## Source valid region and masks

Source selector で各素材を選び、source・sensor ごとに独立した有効領域を保存する。魚眼の base circle は center `(0.5, 0.5)`、既定・最大 radius `0.5`。保存済みの超過半径は起動時に `0.5` へ補正し、brush は保持する。普通カメラ・ERP は円形制限のない全画像から開始する。各 sensor の operation は:

- `add`: keep region を追加
- `subtract`: blue flare、hand、camera body 等を局所除外

除外範囲は半透明黒・明るいピンクの斜線・最終 mask の輪郭で表示し、暗部でも見分けられる。表示色は実際の出力 mask を変えない。Frame extraction 前でも魚眼の radius と Save / Discard を表示する。Preview、frame switch、brush は frame extraction 後に有効になる。Discard は server 保存値へ戻し、Undo は直前の一筆全体を戻す。Step は有効な source の保存状態を集計し、未保存は「未設定」、一部保存は設定済み数、全 source の保存・座標確認が済めば「完了」とする。未保存 draft は完了数に含めず、無効 source は集計対象外とする。既定領域のまま使う source も Save で設定済みになる。

SAM3 は二つの独立 Step を持つ。

```text
Feature default:  person,camera operator,person's shadow,animal,sky,vehicle,water
Training default: person,camera operator,person's shadow
Inference long edge は Auto。Perspective は 2048、dual-fisheye は最大 3072、ERP は最大 4096 を基準にし、mixed source は最大要求を使う。
```

片方だけ enabled の場合はその mask を利用可能な feature / export 側へ使う。両方 enabled なら broad Feature mask と detail-preserving Training mask を分離する。生成済み image は全体完了前でも Camera / Frame Inspector で preview できる。

Per-image coverage は「semantic 除外率」であり、sky / water も含む。Prompt 別の coverage と sky 以外の union coverage を manifest に保存し、Feature mask の coverage warning は sky 以外の領域で判定する。Training mask は全除外率で判定する。Prompt 間の領域は重なるため、個別 coverage の和は全除外率にならない。この統計処理は出力 mask を変更しない。Threshold exceed を一画像一行の Console warning にせず、Inspector と aggregate statistics で確認する。

## Feature and matching

| Setting | Default | Meaning |
|---|---|---|
| Feature | SIFT | measured stable baseline |
| `max_image_size` | Auto | Perspective 2048、dual-fisheye native edge、ERP horizontal edge。Mixed は最大要求 |
| `max_num_features` | Auto | Perspective / pinhole view 8192、dual-fisheye 16384、ERP 32768。Mixed は最大要求 |
| Matcher | Brute-force | broad camera / texture support |
| Pairing | Auto | single source sequential、small mixed exhaustive |
| Rig visibility guard | On for calibrated rig | 同一 capture で consumer-valid angular cap が完全に非交差の sensor edge だけを除外。隣接 capture と重なる視錐は保持 |
| Cross-source temporal guard | Off | 同期が保証された video source pair にだけ明示的に使う。独立 recording と静止画 source の visual match は時間で除外しない |
| Loop closure | On for sequential | verified long-range edge を時刻だけで除外しない。vocab tree required |
| Transitive | one pass | 3-view track を増やす |
| `two_view min_num_inliers` | 15 | weak valid edge を残す |
| Guided matching | Off | optional geometry-guided pass |

Large continuous source では verified pair 数と correspondence 数を確認する。Rig verification は frame 間の対応をまとめて検証した後に image pair へ分配するため、分配後の一部 pair が内点閾値未満になることがある。Matching 統計は全 verified pair と、指定内点閾値以上 / 未満の pair を分ける。Mapper は独自の `min_num_matches`（既定 15）を各 pair に適用するため、この二つの閾値を変更した場合は区別する。Loop closure と transitive expansion の組合せは graph を大きくし、Incremental Mapper の CPU-only track merge / retriangulation を支配する場合がある。

Reconstruction 統計は二観測点の数・割合、および solver 診断を含む。`Linear solver failure` は Ceres が棄却した試行 step であり、BA 全体の終了失敗とは区別する。`Bundle adjustment failed:` は別件数として記録する。CLI が成功終了しても途中の診断を消さない。二観測点は冗長性が低いが、実際の三角角・残差を満たした点まで一律に飛点と判定しない。

Sparse reconstruction の未指定値は厳格 preset（filter / merge / complete 1 px、filter / triangulation angle 5°、create / continue 0.75°）を使う。保存済みの明示値は保持し、Inspector から COLMAP 既定値・一般・厳格を選択できる。

[Sparse cleanup](docs/strict-sparse-cleanup.md) は全 track の位置不確実性と capture 単位の留保予測で既存点を評価する。既定は最低 3 capture、pixel noise 仮定 1 px、相対誤差 2%、原 point / 留保予測の再投影 P95 2 px・最大 4 px。同一 capture の二眼は一組にまとめ、通常写真・phone video の perspective camera と native fisheye は自身の camera model で処理する。保持点と camera の座標は変えず、道路に平面を仮定しない。TestO2 では 110,919 点から 55,034 点を保持し、厳格 split 版の大きな欠損を抑えた。Mixed 実素材の画質改善と最終 Gaussian の完全な浮遊点除去は未実証。Cleanup は独立 Step / preview / statistics を持ち、clear すると scene-aligned model と比較できる。

Frame extraction の default は Spatial optical flow。Candidate quality gate の後、前回採用 frame からの optical-flow motion を使って間隔を決める。Frame hierarchy の右端は Laplacian variance による sharpness score。Spatial sharpness threshold は 0–2000、抽出済み primary frames の lower 20% から一度 auto-fill し、その後の手動変更を上書きしない。Float field は editing 中の `.5` を保持し、blur / Enter で `0.5` へ canonicalize する。

FFmpeg の frame-index 選択は、等間隔部分を圧縮し、残りの条件を平衡な加算式にまとめてから文字数で分割する。通常動画と dual-lens は同じ生成処理を使う。文字数だけの制限では、不規則な候補列が式木の深さ上限を超え、十分な空きメモリがあっても `Cannot allocate memory` で停止するためである。選択する frame index と順序は維持する。

Spatial optical flow は quality gate が長い時間空洞を作らないよう、最大 bridge 間隔（default 4 秒）も持つ。露出不良または最小 feature 数未満の frame は bridge に使わない。sharpness / rolling-shutter だけで除外された候補は、通常選択では除外を維持し、空洞を埋める時だけ再評価する。

- 視差 + 鮮鋭度: target motion の帯域で十分な視差を持つ候補から最も鮮鋭な frame を選ぶ。default。
- 鮮鋭度優先: bridge 窓内で最も高品質な frame を選ぶ。
- 視差優先: target motion に最も近い frame を選び、品質は tie-break に使う。

Optional RoMaV2 dense initialization は native camera ray、certainty、mask、parallax、ray gap、reprojection、voxel dedupe を通した point だけを追加する。魚眼 view を含む場合だけ depth seed は光軸から 85° 以内に制限し、有効候補の certainty 下位 8% を除外する。この depth guard は魚眼の物理有効領域や SfM mask とは別である。UI quality は `Turbo / Fast / Base / High`、runtime setting は `turbo / fast / base / precise` へ明示変換する。High は RoMaV2 Precise の 800 / 1280 px bidirectional path。Dense pair progress は detail event で表示し、Console へ一対一の行を残さない。

## Mapper decision

### Incremental Mapper

Default mapper かつ calibrated dual-fisheye の current recommendation。Generalized frame registration と full correspondence graph の retriangulation により、sensor をまたぐ temporal tracks を保持する。

### Global Mapper

Perspective、ERP、または cross-sensor observability が十分な graph では高速な選択肢。Native dual-fisheye では warning を表示する。

192 camera dataset の比較:

| Mapper | Points | Observations | Cross-sensor points | Path diameter | Mean reproj. |
|---|---:|---:|---:|---:|---:|
| Global fixed rig | 29,974 | 181,549 | 179 | 135.890 | 1.080 px |
| **Incremental fixed rig** | **51,960** | **288,040** | **5,760** | **2.403** | 1.235 px |

二つの trajectory shape は Sim3 後 RMS 0.058% とほぼ同じ。Global の問題は cross-sensor track loss と physical baseline に対する gauge scale であり、registration 192 / 192 だけでは quality を保証しない。

`ba_use_gpu` は Ceres の linear solve に CUDA/cuDSS を許可する。Small local BA は transfer overhead を避けるため CPU を選ぶ。Global refinement の track completion、merge、retriangulation、residual construction も CPU であり、Step 全体が GPU task になるわけではない。

Incremental Mapper の final retriangulation は最後の per-image color sampling より後に point を追加できる。そこで production は `Mapper.extract_colors=0` とし、geometry 完了後に COLMAP `color_extractor` を全 thread で一度だけ実行する。Rewritten `images.bin` は採用せず、元の camera / image ordering を保持したまま colored `points3D.bin` だけを置換する。`point_color_completion` は exact-black point の before / after を Reconstruct statistics に記録する。

## Gravity, scale, and scene coordinates

Gravity alignment は IMU と reconstructed trajectory の time offset / axis permutation を評価し、model 全体へ rotation だけを適用する。

Metric scale は同一 capture の複数 sensor が共有 3D point を持つ場合だけ physical baseline を evidence として使う。Fixed rig に baseline value があるだけでは scale observable ではない。IMU acceleration の単純二重積分は drift-unbounded なので meter scale に使わない。

Scene coordinate alignment は metric scale を前提にしない。重力整列済み model から最大 100,000 点を deterministic uniform sampling し、primary camera path の下方にある dominant horizontal plane のうち camera に最も近い面を ground として Y=0 へ移す。Wall orientation はさらに最大 20,000 点に制限し、ground band を除外して vertical plane を batched line-RANSAC で抽出する。十分な support と 90° consistency を持つ隣接 wall pair が得られた場合だけ Y 軸 yaw を最寄りの Manhattan axis へ回転し、屋外や非 Manhattan scene では rotation を適用しない。全点数に依存するのは model read / final transform だけで、幾何推定量は上限固定である。

## LFStudio export

Export Inspector が表示する directory 自体を LFStudio で開く。Application は `export_dataset` 全体を Stage output として扱うため、clear / regenerate は LFStudio output directory も含めて削除する。

Export model source は生成済み Step から `dense_initialization`、`cleanup_sparse`、`scene_alignment` の順で決める。Dense toggle を off にしても passthrough Step を生成する必要はなく、選択した source は `model_source` として export manifest / Inspector statistics に記録する。

```text
export_dataset/
├── images/sources/<source-id>/...
├── masks/sources/<source-id>/...
├── sparse/0/{rigs,cameras,frames,images,points3D}.bin
├── preview/{reconstruction.json,points.bin}
├── train_configs/{train_config.mrnf.json,train_config.mcmc.json,recommendations.json}
└── export_manifest.json
```

Rectified PNG は validity bounds で pixel crop する。Legacy JPEG だけ jpegtran MCU crop を使う。Camera principal point、2D observations、mask は同じ offset で更新する。LFStudio は `images.bin` の world-to-camera pose を正本として読む。

Image / mask は materialize と完全 decode 検証を同じ logical-CPU worker pool で一度だけ行う。Crop / jpegtran が成功した output はその時点の検証済み size を引き継ぎ、最終 dataset validation は path、camera reference、記録済み size、camera model を検査する。生成直後の全 PNG を serial に再 decode しない。

Recommended CLI example:

```bash
lichtfeld-studio train --data <export_dataset> --config <export_dataset>/train_configs/train_config.mrnf.json
```

Training output、loss、PSNR は LFStudio が生成する。Application は利用可能な external metrics だけ Export Inspector に表示するが、export Stage を clear / regenerate した時は dataset root 内の全内容を削除する。

## Frontend and Scene View

Frontend は React 19 + TypeScript + Vite、Scene View は React Three Fiber / Three.js。Primary toolbar だけ `liquid-glass-react` の stable mode を限定利用し、通常 panel / button は theme-aware CSS を使う。Icon は Fluent collection。

Desktop の既定配置は左 Hierarchy、中央 Scene View / Console / Inspector の tab、右 Steps。狭い画面では全 panel を一組の tab にまとめ、desktop と別の layout を保存する。Step / Photo の選択で Inspector を開き、scene 内 camera の選択では Inspector の tab を点灯する。成果更新と新規 log も対応する tab を短く点灯し、background update で active tab を切り替えない。

Scene grid は finite double-sided plane、depth-test on / depth-write off。成果が無い場合も斜め上から見た grid と座標軸を描画する。Point cloud の取得中や更新時に Canvas を作り直さず、同じ project の点群更新・clear で現在の視点を変えない。初回 data load だけ自動 framing し、以後は F で明示的に framing する。Point は antialiased circular sprite を使用し、RGB byte を linear vertex color へ変換して tone mapping を適用せず描画する。Geometry / material は交換時に dispose する。

Wheel は通常 zoom、right-look 中だけ movement speed を 0.001×–16×で変更し、中央へ倍率を表示する。Middle drag は screen-plane pan。Near / Far clip は scene scale と movement speed に追従する。Theme background は applied theme と同じ update で反映する。非表示の Scene View は keyboard input と連続 rendering を停止する。

Camera は常時 9 px の camera gizmo icon として一 draw call で overlay 描画し、scene extent によって巨大化させない。Pick は world-space ray threshold を使わず 8 px の screen-space radius で決定し、空白の左 click は selection を解除する。重なる icon は camera から最も近い visible camera を選ぶ。選択した camera だけ projection view を表示し、Perspective は aspect-aware rectangular frustum、fisheye は circular boundary、ERP は spherical guide を使う。Inspector の native fisheye image / mask / overlay も circular clip、pinhole / ERP は元の rectangular layout を保つ。

Source / Photos hierarchy と Inspector statistics は default collapsed。Photo と Dataset Camera は source、capture、projection、mask channel を共通形式で表示する。Checkbox は light / dark theme の両方で unchecked border を背景から分離し、checked state は accent fill と白い checkmark を表示する。Header は primary source の path だけを表示し、補助 source の件数 badge を付けない。

永続設定の正本は workspace の SQLite DB。`GET/PATCH /api/preferences` に theme、language、最後の project、desktop / compact layout、viewer 表示、console filter を保存する。Language 未指定は browser language、theme `auto` は OS preference に従うが、browser storage は読み書きしない。`PATCH /api/projects/{id}/ui-state` は stage parameters、reconstruction mode、clear checklist、Inspector selection、camera pose を project ごとに部分更新する。保存 request は直列化し、失敗時は未送信 patch を保持して再試行ボタンと診断を表示する。Project switch / create / delete の前に保存を完了させ、background / pagehide では keepalive request を送る。未送信変更を突然の browser process 終了から復旧する保証はない。既存の browser-only layout / theme は移行しないが、server にある工程 parameter は保持する。

「Generate all pending steps」は一つの server pipeline job を開始する。Parameter と optional mask skip を開始時に固定し、browser を閉じても server worker が継続する。完了済み Step の cache 判定と downstream invalidation は engine が担当する。

Sharp Frames の比較と抽帧改善候補は [抽帧品質レビュー](docs/frame-quality-review.md) を参照。

## Development and validation

```bash
uv run --project backend pytest -q backend/tests
cd frontend
pnpm exec tsc --noEmit
pnpm exec playwright test
pnpm build
```

Backend test は projection、rig、mask、progress snapshot、Mapper frame counter、gravity、scale、bounded Manhattan / ground alignment、export を検証する。UI smoke test は source、Steps、localized controls、live mask preview、Scene View、path copy、theme、progress detail を user-visible route から通す。

CI workflow は repository に置かない。Validation は local / controlled machine で実行する。

## References and third-party work

- [COLMAP](https://github.com/colmap/colmap)
- [GLOMAP](https://github.com/colmap/glomap)
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)
- [RoMaV2](https://github.com/Parskatt/RoMaV2)
- [SAM3](https://github.com/facebookresearch/sam3)
- [PPISP](https://github.com/nv-tlabs/ppisp)
- [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser)
- [Gyroflow](https://github.com/gyroflow/gyroflow)
- [liquid-glass-react](https://github.com/rdev/liquid-glass-react)
- [Icônes Fluent](https://icones.js.org/collection/fluent) / [Fluent UI System Icons](https://github.com/microsoft/fluentui-system-icons)
- [Lichtfeld Densification Plugin](https://github.com/shadygm/Lichtfeld-Densification-Plugin)
- [lichtfeld-360-plugin](https://github.com/alexmgee/lichtfeld-360-plugin): original legacy plugin reference

Third-party license details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
