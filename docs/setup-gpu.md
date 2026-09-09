# Runtime / GPU setup

Development checkout と packaged runtime の起動、dependency diagnosis、CUDA、COLMAP、optional model、UI progress contract を説明する。Machine-specific media path や experiment output path は記録しない。

## Launchers

### Windows

```powershell
scripts\start-windows.ps1
```

```bat
scripts\start-windows.cmd
```

### Linux

```bash
./scripts/start-linux.sh
```

### macOS

```bash
./scripts/start-macos.sh
```

Launcher は backend / frontend environment を作成し、runtime config を読み、server と Vite frontend を起動する。Python package は uv、Node package は pnpm で管理する。Global Python / npm install は使用しない。

### Background service

初回 launcher 実行後は `scripts/server_service.py` が Windows / Linux / macOS 共通の background lifecycle を提供する。Windows Task Scheduler、systemd、launchd を application contract にしない。Windows では detached + breakaway process、POSIX では independent session を作り、共通 PID record、stdout / stderr log、health check を使う。

```text
server_service.py start [--port 8787]
server_service.py status
server_service.py restart
server_service.py stop
```

Python executable は `backend/.venv` の platform 固有 path を使う。PID record は `runtime/backend.pid.json`、log は `runtime/logs/backend.stdout.log` と `backend.stderr.log`。Backend restart 時は前回の `running / queued` Job を `interrupted by restart` として終了し、未確定 Stage は再実行可能な状態へ戻す。

同じ workspace に対する API service は一つだけ起動する。Project ごとの排他制御はこの API process が所有し、Worker の稼働状態は SQLite の Job record で共有する。Frontend build の静的配信は解決後の path を `frontend/dist` 内に限定し、上位 directory や外部を指す symlink を配信しない。未定義の `/api` route は HTML に置き換えず HTTP 404 を返す。

## Runtime config

`runtime/config.toml` または `SPHERE_CONFIG` で指定した TOML を読む。Environment variable は `SPHERE_` prefix と `__` delimiter を使う。

```toml
[server]
host = "127.0.0.1"
port = 8787

[workspace]
root = "./workspace"

[filesystem]
allowed_roots = []

[binaries]
ffmpeg = ""
ffprobe = ""
colmap = ""
jpegtran = ""
vocab_tree = ""

[frame_extraction]
hwaccel = "auto"
require_hwaccel = false
score_workers = 0
```

Empty binary value は `PATH` resolution を使う。Filesystem browser は `allowed_roots` 内だけを表示する。

## Doctor

Settings の Doctor は file existence だけでなく capability を検査する。

- FFmpeg decoder / encoder と selected hardware decode
- FFprobe
- COLMAP version / command set
- Global Mapper
- Final point `color_extractor`
- Ceres CUDA / cuDSS dense / sparse BA
- ONNX CUDA provider
- FAISS vocabulary tree
- jpegtran（legacy JPEG crop のみ）
- SAM3 package / checkpoint
- RoMaV2 package / checkpoint

Feature が利用できない場合は disabled reason を UI に表示する。Software fallback を CUDA success と表示しない。

## FFmpeg hardware decode

`frame_extraction.hwaccel = "auto"` は representative frame を実際に decode し、動作した backend を採用する。

| Platform | Typical backend |
|---|---|
| NVIDIA Windows / Linux | CUDA / NVDEC |
| Intel Windows / Linux | QSV / VAAPI |
| AMD Linux | VAAPI |
| macOS | VideoToolbox |

`require_hwaccel = true` は hardware decode が使えない時に error とする。`score_workers = 0` は logical CPU 全数。Raw dual-fisheye は二 stream を一 demux pass で decode し、Packet PTS で sensor pairing を検証する。

## COLMAP installation

Generic installer:

```bash
uv run scripts/install_colmap.py
```

Windows CUDA / cuDSS build:

```powershell
scripts\build_colmap_cuda_ba.ps1 -OutputDirectory <output>
```

Build は次を pin / bundle する。

- COLMAP 4.1.1
- Ceres CUDA build
- cuDSS runtime
- cuDNN / ONNX CUDA runtime
- target CUDA architectures
- COLMAP PR #4591 semantic backport

Generated capability metadata を Doctor が読み、`ba_use_gpu` の availability を決める。System CUDA install が存在するだけでは GPU BA capability と判定しない。

## BA and CPU/GPU boundaries

`ba_use_gpu` は Ceres linear solver に CUDA/cuDSS を許可し、geometry model や loss は変更しない。

- 50 images 未満の small local BA は transfer overhead を避けて CPU を選ぶ
- Large BA は CUDA dense / cuDSS sparse solver を選択できる
- Residual / Jacobian preparation の一部は CPU
- Track completion、merge、retriangulation は CPU
- Global Positioning GPU と BA GPU は別 capability

Incremental Mapper の `Retriangulation and Global bundle adjustment` は複数の CPU / GPU phase をまとめた upstream label である。Label 中に BA があっても phase 全体が GPU になる意味ではない。UI は registered rig frame count と current phase を別 detail として表示する。

`Linear solver failure. Failed to compute a finite step.` は無効な trial step の棄却である。Ceres は trust radius を縮小して再試行するため、warning 件数と BA 終了失敗数は一致しない。実際に解を利用できない場合の `Bundle adjustment failed:` を別途確認する。Reconstruction Inspector は両方の件数、log file、local/global context を記録する。GPU を許可しても小規模 local BA は CPU のままなので、GPU 切替だけをこの warning の解決策にしない。詳細診断には stock CLI の `--log_level 3` を使う。

根拠: [Ceres step 検証](https://github.com/ceres-solver/ceres-solver/blob/bac1127f9ef672405bd0d2d9c84e809ae89bd239/internal/ceres/levenberg_marquardt_strategy.cc#L106-L129)、[COLMAP BA 終了判定](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/estimators/bundle_adjustment_ceres.cc#L556-L563)。

Video source では COLMAP の `ModifyForVideoData()` と同じ `ba_global_frames_ratio=1.4` / `ba_global_points_ratio=1.4` を使う。final global BA は省略せず、成長ごとの中間 global BA だけを減らす。これは low-texture 画像を削る設定ではなく、同じ観測グラフに対する mapper の実行 schedule である。

[Native dual-fisheye 安定性の実測](fisheye-stability.md)では、同じ観測 database に対して厳格 preset、local BA 12 近傍、一般 preset を比較している。全画像登録と warning 数だけでは点群品質や metric scale の正しさを判定できない。

### Incremental Mapper の三角測量設定

Inspector の `Reconstruct > Incremental Mapper > COLMAP Advanced` で、三角測量の品質ゲートをプリセットまたは個別値で指定できる。新規 UI 設定と API の省略 parameter は厳格 preset を使用し、保存済みの明示値は保持する。これらは `--Mapper.*` オプションであり、Global Mapper には適用されない。値の単位は再投影誤差が pixel、角度が degree である。空欄または `0` を個別入力した項目は、その項目の COLMAP 既定値へ委譲する。

| 設定 | COLMAP 4.1.1 既定値 | 一般 | 厳格（アプリ既定） |
|---|---:|---:|---:|
| `filter_max_reproj_error` | 4.0 px | 1.5 px | 1.0 px |
| `filter_min_tri_angle` | 1.5° | 3.0° | 5.0° |
| `tri_create_max_angle_error` | 2.0° | 1.0° | 0.75° |
| `tri_continue_max_angle_error` | 2.0° | 1.0° | 0.75° |
| `tri_merge_max_reproj_error` | 4.0 px | 1.5 px | 1.0 px |
| `tri_complete_max_reproj_error` | 4.0 px | 1.5 px | 1.0 px |
| `tri_min_angle` | 1.5° | 3.0° | 5.0° |

COLMAP 既定値は登録数と遠景の coverage を優先する。一般は低視差・高残差の点を減らしながら接続を保ち、厳格は不安定な点をさらに除外する代わりに登録数と遠距離点が減る可能性がある。プリセットを選択すると 7 項目が同時に設定され、個別値を編集すると `カスタム` に切り替わる。Global Mapper を選択している間はこの Incremental 専用設定を表示しない。

## GLOMAP and fisheye

COLMAP 4.1.1 の `OPENCV_FISHEYE` は forward hemisphere camera である。

- `ImgFromCam` は `z <= 0` を reject
- Default `CamRayFromImg` は `z > 0`
- Effective field of view は 180° 以下

Native dual-fisheye + Global Mapper selection には warning を表示する。Warning は mapper を自動変更しない。

Current recommendation:

- Default / calibrated dual-fisheye: Incremental Mapper
- Perspective / ERP / well-connected graph: Global Mapper を選択可能

192 camera test では Global fixed rig が 179 cross-sensor points、Incremental fixed rig が 5,760 を保持した。Experimental more-than-180° extension は same-capture verification を回復したが、Global Mapper の later stage が stereo constraint を保持しなかったため production dependency にしない。

Matching は calibrated rig の同一 capture edge を consumer-valid angular cap で検査する。`cam_from_rig` の optical axis separation が二つの half-angle の和を超える場合だけ、その二視図 edge は物理的に共有 ray を持てないため除外する。これは adapter が報告する外参と validity から導かれ、機種固有の閾値ではない。時間の異なる capture と angular cap が重なる sensor pair は保持する。

Rig 検証の `min_num_inliers` は集約した frame pair に適用され、`CALIBRATED_RIG` の各 image pair はそれ未満になることがある。Mapper の `min_num_matches` を下回る pair は database cache が既に除外する。その pair を再度削除しても BA の観測は改善しない。Matching 統計は総 pair 数と内点閾値以上の pair 数を併記する。根拠: [rig inlier の分配](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/estimators/two_view_geometry.cc#L514-L557)、[cache の閾値](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/scene/database_cache.cc#L39-L45)。

## Fisheye runtime contract

Raw adapter は source-local intrinsics、calibration crop、`cam_from_rig`、shutter、physical/custom validity を出力する。Image preparation の内部 dependency が次を実行する。

```text
target OPENCV pixel -> target ray -> source omnidirectional pixel
```

RGB は same-resolution Lanczos4 で一回 resample、validity は nearest。Mask / SfM / export は rectified catalog を読む。Camera metadata だけを交換して source pixels を残す方法は禁止する。Device-specific constants は adapter に置く。
INSV extraction は MP4 stream1→calibration lens0、stream0→calibration lens1 の canonical sensor mapping を適用する。V6 radtan-pro の非対称 decentering / prism 項を含む source ray から backward remap を構築する。

Fisheye normalization は独立 Stage として UI hierarchy、進捗、統計、clear / regenerate を公開する。Feature Mask、Training Mask、Feature extraction の単独 Job も同じ Job 内で normalization を先に生成または cache reuse してから target Stage を実行する。

## Optional models

### SAM3

```toml
[sam3]
repo_path = ""
checkpoint_path = ""
device = "cuda:0"
dtype = "bfloat16"
max_inference_size = 2048
confidence_threshold = 0.5
```

`sam3.max_inference_size` は UI 以外から parameter を省略した場合の backend fallback。UI の Auto resolution は source ごとの projection と native dimensions を使い、Perspective 2048、dual-fisheye 最大 3072、ERP 最大 4096 を解決する。Feature extraction は Perspective 2048、native dual-fisheye の source edge、ERP horizontal edge を使い、0 を手動指定した場合は COLMAP の無制限既定へ委譲する。Feature count は Perspective / pinhole view 8192、dual-fisheye 16384、ERP 32768。Pinhole view は dual-fisheye edge の 1/2、ERP width の 1/4 を 256 px 単位へ切り上げる。複数 source は各 parameter の最大要求を採用する。

Feature / Training は別 Step、prompt、manifest を持つ。Missing checkpoint は Doctor が表示する。Per-image coverage は manifest / Inspector に保存し、threshold warning を一画像一行 Console へ出さない。

### RoMaV2

```bash
uv run scripts/install_romav2.py
```

Dense initialization は optional、default off。Native rays、certainty、mask、parallax、ray gap、reprojection、voxel dedupe を通した point だけを追加する。魚眼 view を含む場合だけ、depth seed は光軸から 85° 以内に制限し、有効候補の certainty 下位 8% を除外する。これは魚眼の物理有効領域や SfM mask を変更しない。

Scene coordinate alignment は metric scale の成否から独立して実行する。Ground は最大 100,000 sample、Manhattan yaw は最大 20,000 sample / 2,048 candidate line へ固定し、point count が増えても推定計算量を増やさない。Gravity-aligned model の primary camera path 下方から最も近い dominant horizontal plane を Y=0 へ移し、十分な vertical / horizontal support、scene 内交点、10° 以下の orthogonality residual を持つ wall pair がある場合だけ yaw を揃える。その後に [full-track sparse cleanup](strict-sparse-cleanup.md) を実行する。最低 3 capture、全 track の条件付き不確実性半径と再三角化位置差 2%、原 point と capture 留保予測の reprojection P95 2 px / 最大 4 px を使う。Pixel noise の仮定は 1 px。同一 capture の二眼を一緒に留保し、異なる source は非同期でも各 camera model で扱う。距離だけの削除や道路への平面拘束はしない。Cleanup output があれば Dense / Export が読み、clear すれば Scene-aligned preview へ戻る。

UI `High` は package API に存在しない文字列を直接渡さず、RoMaV2 `precise` setting（800 / 1280 px、bidirectional）へ変換する。Supported runtime setting は `turbo / fast / base / precise`。Unknown setting は model load 前に validation error とする。

### Vocabulary tree

```bash
uv run scripts/install_vocab_tree.py
```

Sequential loop closure と large mixed-source vocab pairing に使用する。Missing tree で loop closure を要求した場合は error とする。

### jpegtran

```bash
uv run scripts/install_jpegtran.py
```

Legacy JPEG の lossless MCU crop にだけ使用する。Rectified PNG は pixel crop のため jpegtran を必要としない。

Export の crop、decode validation、hard-link materialization は logical CPU 全数の worker pool を使う。Crop / jpegtran の完了時に image size と decode success を記録し、dataset validation では同じ大容量 PNG を再 decode しない。未変換 hard-link は link 前に worker 内で完全 decode するため、corrupt image / mask の検出を省略しない。

## Progress and recovery

Worker は SQLite event table を single progress source とする。

- `progress_event`: 最新の numeric progress / counter
- `activity_event`: 最新の phase / tool activity
- `kind=progress`: Inspector / hierarchy 用、Console 非表示
- `kind=log`: lifecycle、warning、failure、Console 表示

`GET /api/projects/{id}/stages` は activity と numeric snapshot を別々に返す。WebSocket reconnect 後も count と phase を復元できる。Activity-only event が numeric progress を上書きしてはならない。Frontend freshness は各 Stage の normalized manifest parameters と現在の UI parameters を比較し、差がある黄色行には old / new value を明示する。HTTP polling が失敗した場合、Frontend は最後の `running` snapshot を stale として表示し、elapsed timer、spinner、Stop control を止める。Backend startup は中断 Job / Stage に `finished_at` と `interrupted by restart` を記録し、その Job の一時 Stage directory と frame-selection scratch を削除する。

Job の開始と成果物を変更する API は、同じ project の ownership check と変更処理を一つの排他区間に置く。`queued` / `running` Job が存在する場合、run / rerun、Stage clear、project 削除、source 追加・削除・primary 変更、有効領域保存は HTTP 409 を返す。Cancel 中も Worker が停止するまで Job の稼働状態を保持し、完了後に `cancelled` を確定する。Scratch cleanup も同じ排他制御を使うため、次の Job の一時出力を削除しない。別 project の操作、読み取り、UI 設定保存は待たせない。

SQLite は独立 SQL 用の autocommit 接続と、複数文の transaction 用の専用接続を分離する。明示 transaction は直列化して `BEGIN IMMEDIATE` から commit / rollback まで保持し、別 request の `commit()` / `rollback()` が途中の変更を確定・破棄しない。Filesystem の処理や Worker 停止を待つ間は DB transaction を保持しない。

Worker を起動できない場合は Job を `failed` にして原因を event に残し、`queued` のまま放置しない。Stage の公開処理も実行 lifecycle の例外境界に含める。Downstream invalidation が成功してから manifest を一時 file 経由で原子置換し、project state 更新後に成功を確定する。公開失敗は Job / Stage の両方を `failed` にし、不完全な manifest を cache として再利用しない。

COLMAP Incremental の counter は `num_reg_frames` を読む。この値は次の image registration を試す直前の registered count なので、次の registration または Global refinement が success を証明した時点で pending frame を commit する。Failure 行では commit しない。Dual-fisheye の同一 capture は 2 images だが 1 rig frame であるため、`InputSpec.frame_count` を denominator にする。Long-running native command は structured count / phase を表示し、raw line は secondary activity として保持する。

Incremental Mapper は final retriangulation 後に全 point を再着色しないため、Mapper 内の incremental color sampling を無効化し、終了後に `color_extractor --num_threads -1` を実行する。Colored output からは `points3D.bin` だけを採用し、LFStudio validation split に影響する `images.bin` ordering は元のまま保持する。Geometry summary が変化した場合は publish せず error にする。

Current Mapper は upstream model write が完了時に行われる。中断 recovery を必要とする大規模 run では COLMAP snapshot / resume support を別途有効にする必要がある。

Stage clear / downstream invalidation は quarantine rename transaction を使う。Custom clear の明示 selection は project UI state に保存し、consumer closure は表示時に再計算するため implied selection を保存値へ混在させない。Backend は requested Stage の consumer closure を先に確定し、全対象を `.pipeline/trash/<transaction>` へ移す。`export_dataset` は外部 tool の `output/` や `.licht` を含めて一つの Stage output として削除する。

Rename failure は移動済みの成果物を rollback し、復元に成功した場合は HTTP 423 を返す。Rollback も失敗した場合は残る backup を保持し、`recovery_required.json` に元の path、backup path、復元エラーを記録する。Error log の保管場所と対応表を使って復旧する。物理削除だけが失敗した場合は bounded retry 後に `pending_cleanup.json` を作り、API の `pending_cleanup` に表示する。次回の自動 retry は pending marker のある確定済み残件だけを対象とし、未確定の quarantine と復旧待ち backup を削除しない。Frontend は clear 前の in-flight artifact query を cancel し、preview / frames / masks / export cache を削除結果に応じて null 化する。

## UI defaults

- Generate all は success 後に次 Step を開始し、failure / cancel / stop で停止
- Internal fisheye normalization は独立 hierarchy Step として詳細進捗を表示する
- Fisheye region は extraction 前でも radius / Save / Discard を表示
- Photos hierarchy は default collapsed
- Spatial sharpness は 0–2000、primary score から一度 auto-fill
- Feature size limit の下に primary width / height を表示
- Float field は `.5` を受け、commit 後 `0.5` に正規化
- Running Step は percentage、count、activity、elapsed time を同時表示
- Dense / SAM / matching の高頻度 update は Console へ spam しない
- Checkbox は unchecked / checked / disabled を light / dark theme の両方で識別可能にする
- Camera は fixed-pixel gizmo と 8 px screen-space pick、blank-click deselection、selected-only projection guide を使い、fisheye guide / Inspector preview は circular にする
- Browser refresh は最後に開いた project を復元し、保存 ID が存在しない場合だけ先頭 project へ fallback する

## LFStudio

Export directory 自体を dataset root として渡す。Application は `export_dataset` を所有し、clear / regenerate では LFStudio output を含む全内容を削除する。Recommended profile は MRNF + GUT + segment masks、PPISP / controller off、`undistort=false`、width cap 2048。

Dense initialization は optional。Export は dense artifact が存在する時だけそれを使用し、Dense off / cleared の場合は scene-aligned sparse model と preview へ明示的に fallback する。LFStudio 自身の CUDA context / VRAM は application worker と別 process であり、system stats は process owner を混同しない。GPU 統計用の `nvidia-smi` は 3 秒で timeout し、timeout・通信失敗・request cancel 時には subprocess を終了して回収する。監視 request のたびに停止した probe を残さない。

## Validation

```bash
uv run --project backend pytest -q backend/tests
cd frontend
pnpm exec tsc --noEmit
pnpm exec playwright test
pnpm build
```

Runtime archive test は executable presence、command availability、capability metadata、small reconstruction route を確認する。UI smoke test は reload / reconnect 後の numeric progress と activity detail を user-visible route から検証する。
