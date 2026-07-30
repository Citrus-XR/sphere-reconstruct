# 360 camera format adapter guide

この文書は、新しい 360 camera、raw multi-fisheye container、metadata sidecar、または
stitch 済み panorama を追加する AI / developer の実装契約である。特定メーカーの SDK を
pipeline backend に据えるための手順ではない。Vendor SDK や公式 app の出力は ground truth、
互換性確認、任意の derived-product generator としてだけ扱い、物理 sensor の native path と混同しない。

## 設計原則

Pipeline core が知るものは、source adapter ID、sensor topology、正規化済み camera system、capture、
motion track だけである。Container box、metadata key、メーカー座標、stream ordinal、sidecar file 名は
adapter 内で正規化する。

- `lens0` / `lens1` は container 順の opaque sensor ID であり、front / back を意味しない。
- 回転名は常に `target_from_source`、rig 外参は `cam_from_rig` とする。
- 長さは meter、gyro は rad/s、時刻は integer nanosecond または明記した second とする。
- Principal point は sensor-local image 座標へ変換してから core へ渡す。
- 不明な値を推測 default で埋めない。`unknown` として correction を無効化する。
- 固定 rig、stitch 済み ERP、virtual pinhole、rolling-shutter correction は別の概念である。

現在の code は `domain/camera_system.py` を vendor-neutral calibration boundary とし、
`insta360/camera_system.py` だけが `offset_v3` の合成画布と軸規約を知る。Source adapter ID は
closed Enum ではなく `domain/source.py` の registry で検証する。

## 三つの artifact boundary

### Source bundle

将来の multi-file adapter は一つの `path` に依存せず、media、calibration、motion、time-map を
resource として保持する。目標 schema は次の形にする。

```json
{
  "schema_version": 1,
  "source_id": "uuid",
  "adapter": {"id": "vendor.camera_format", "version": "1"},
  "resources": [
    {"id": "media-0", "purpose": "media", "path": "capture-a.bin", "ordinal": 0},
    {"id": "media-1", "purpose": "media", "path": "capture-b.bin", "ordinal": 1},
    {"id": "calibration", "purpose": "calibration", "path": "capture.json", "ordinal": 2}
  ],
  "adapter_options": {}
}
```

現行 DB は source ごとに一つの `path` を持つ。単一 container はこのまま追加できるが、dual-file、
明示 sidecar、外部 time-map を必要とする format を追加する前に `project_source_resource` へ正規化する。
Runtime compatibility branch を増やさず、DB migration で既存 `path` を `media-0` resource へ移す。

### Inspected source と camera system

Inspection はメーカー依存 metadata をそのまま下流へ渡さない。次を確定する。

- adapter ID / version と probe evidence
- sensor 数、stable sensor ID、resource / stream mapping
- decoded size、codec、pixel rotation / mirror
- 全 frame PTS / duration と sensor 間 skew
- calibration provenance と camera-system artifact path
- motion track と shutter capability
- opaque stitched / stitched with time-map / raw sensors の区別

現行 `camera_system.json` は次の schema を使う。

```json
{
  "version": 1,
  "coordinate_system": {
    "handedness": "right",
    "x_axis": "right",
    "y_axis": "down",
    "z_axis": "forward",
    "length_unit": "meter",
    "pixel_origin": "top_left_pixel_center_0"
  },
  "calibration_source": "offset_v3",
  "reference_sensor_id": "lens0",
  "sensors": [
    {
      "id": "lens0",
      "image_key": "lens0",
      "calibration_image_transform": {
        "reference_width": 5376,
        "reference_height": 5376,
        "crop_x": 32.0,
        "crop_y": 32.0,
        "crop_width": 5312,
        "crop_height": 5312
      },
      "projection": {
        "model": "mei",
        "width": 5312,
        "height": 5312,
        "xi": 2.0,
        "fx": 4278.3,
        "fy": 4277.33,
        "cx": 2662.63,
        "cy": 2649.84,
        "k1": 0.18366432,
        "k2": 2.07332635,
        "k3": -3.27984834,
        "p1": -0.00005305,
        "p2": 0.00065176
      },
      "cam_from_rig": {
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation_xyz": [0.0, 0.0, 0.0]
      },
      "shutter": {
        "type": "rolling",
        "readout_time_ms": 21.244001,
        "scan_direction": "unknown",
        "timestamp_reference": "unknown"
      }
    }
  ]
}
```

`projection.width/height/cx/cy` は各 sensor の local reference image に対する値である。例えば
二つの sensor が横並びの 10752×5376 calibration canvas を使っていても、sensor 1 の `cx` から
5376 を引く処理は adapter 内で終える。Core の `imaging/projection.py` に canvas offset を入れない。

新しい native model が MEI でない場合、`CalibratedSensor` の projection を discriminated union とし、
camera-model registry に exact pixel↔ray 実装を追加する。未知 model を無理に MEI へ詰め替えない。

### Capture manifest

任意 sensor 数へ進む時の canonical capture は、special key ではなく sensor frame array を使う。

```json
{
  "capture_index": 12,
  "source_capture_index": 12,
  "timestamp_ns": 500000000,
  "frames": [
    {
      "sensor_id": "lens0",
      "path": "extract_frames/sources/id/lens0/frame_000012.jpg",
      "source_frame": 12,
      "pts_ns": 500000000,
      "duration_ns": 41666667
    }
  ]
}
```

現行 manifest は `lens0` / `lens1` / `image` key を残している。三 sensor 以上、dual-file、sensor ごとの
time offset を追加する前に、Prepare、preview、valid-region editor をこの array schema へ一度に移行する。
Runtime で旧 key と新 key の両方を読む分岐は作らず、artifact migration と stage invalidation を行う。

## Adapter 実装手順

### 1. Probe と source identity

拡張子だけで判定しない。Magic、container brand、metadata signature、stream topology を読み、confidence と
evidence を返す。別 format と曖昧なら自動選択せず UI に候補を出す。Wrong extension の正しい file を検出し、
同じ拡張子の無関係な MP4 を拒否する test を作る。

Adapter は sensor order を stream ordinal だけで front / back と命名しない。Selfie mode、mirror flag、
camera rotation、旧機種の dual-file naming を evidence から確定する。確定できなければ `lens0` のような
opaque ID を保つ。

### 2. PTS と sensor pairing

`frame_index / fps` は VFR、non-zero start、gap、B-frame、drop frame で誤る。現行 extractor は
container packet の PTS / duration を読み、B-frame を PTS で presentation 順へ戻して全 sensor の sequence と
frame count を検証する。新 adapter も次を満たす。

- PTS は各 stream で strictly increasing
- 全 required sensor frame が一 capture に揃うか、atomic failure
- 許容 skew を adapter contract に明記
- Selected capture の時刻は実 PTS
- Source hash、adapter version、resource hash が stage invalidation に入る

Frame quality は全 required sensor で評価する。現行 dual-fisheye は sharpness、exposure、feature count の
minimum と optical-flow motion の maximum を使う。一方だけ blurred / clipped の frame を選ばない。

### 3. Intrinsics と pixel transform

Metadata の reference canvas から decoded sensor image までの crop、scale、90° rotation、mirror を式として
記録し、principal point と distortion domain に同じ変換を適用する。非正方、off-center、rotated fixture を
必ず用意する。

Projection test は center、四象限、valid boundary で `pixel -> ray -> pixel` を検証する。Forward hemisphere
だけを consumer が扱える場合、physical circle をそのまま通さず ray angle と交差させる。

COLMAP と LFStudio の同名 model が同じ式とは限らない。現在の `THIN_PRISM_FISHEYE` は COLMAP と LFStudio
v0.5.3 で tangential / prism の適用位置が異なるため、両 consumer residual を同時に最小化し、個別 RMS / max
を記録する。Consumer 変換 artifact は native calibration provenance と分離する。

Forward 式だけでなく `ray -> pixel -> ray` を consumer 実装そのもので検証する。Stock LFStudio の
THIN_PRISM inverse は fixed-point iteration が前回 UV から delta を繰り返し減算するため forward と一致せず、
non-radial 項が大きい sensor で円を非対称に変形する。修正版 build を保証できない export は、native / COLMAP
camera を変更せず、training camera だけを internally consistent な OPENCV_FISHEYE approximation へ変換する。
原因箇所は [LFStudio v0.5.3 Cameras.cuh](https://github.com/MrNeRF/LichtFeld-Studio/blob/d8c50c6a3e2273cb74130a6e9023de8d068af52d/src/training/rasterization/gsplat/Cameras.cuh#L1147-L1160)。
正しい fixed point は毎回 `uv = uv_distorted - delta(uv)` とし、元の observed UV を保持する。Patched build の
roundtrip test が通るまで `undistort=true` へ逃げない。この経路にも prism packing bug がある。Repository の
source patch は [`scripts/patches/lichtfeld-thin-prism-inverse.patch`](../scripts/patches/lichtfeld-thin-prism-inverse.patch)。

### 4. Rig extrinsics

各 transform の向き、quaternion order、translation の意味を source code permalink と real fixture で確認する。
`cam_from_rig` の camera center は `C = -R^T t` である。Reference sensor は identity に正規化し、他 sensor は
full 6DoF relative transform を使う。理想 180° や baseline 軸だけへ丸めない。

Native fisheye と derived pinhole は同じ camera system を読む。Pinhole 側で別の Euler convention を再実装しない。
現在の code は native / pinhole とも canonical quaternion から rotation と camera center を導出する。

### 5. Shutter と motion

Readout magnitude だけでは rolling-shutter correction を実行できない。Sensor ごとに最低でも次が必要である。

- global / rolling / unknown
- readout duration
- top-to-bottom / bottom-to-top / left-to-right / right-to-left
- frame timestamp が exposure start / center / end のどれか
- encoded crop と physical sensor scan coordinate の関係
- `R_rig_from_imu`、gyro bias、video↔IMU clock offset / drift

これらが欠ける場合、現在と同様に high-motion frame risk filtering だけを使い、status を `corrected` にしない。
Generic raw-sensor correction を実装する場合は独立 Step を extraction 後、SAM / feature 前に置き、各 sensor を
同一 capture-center pose の同一 native grid へ一回だけ resample する。Frame 間 stabilization は行わない。

Opaque stitched ERP に単純な row-time model を適用してはいけない。ERP の longitude ごとに元 sensor と scan row
が異なり、時刻は通常 `t(u,v)` の二次元 field になる。Stitch mesh / time-map が無い場合は diagnostics と frame
rejection だけを提供する。

Production reference:

- [telemetry-parser Insta360 parser](https://github.com/AdrianEddy/telemetry-parser/blob/77a3b810a0e0f64688a90546c5aaf24c9dba00bd/src/insta360/record.rs#L94-L178)
- [telemetry-parser orientation mapping](https://github.com/AdrianEddy/telemetry-parser/blob/77a3b810a0e0f64688a90546c5aaf24c9dba00bd/src/insta360/mod.rs#L149-L176)
- [Gyroflow rolling-shutter row timestamps](https://github.com/gyroflow/gyroflow/blob/b5e8828f82c150676e48a7c2e3db39c97392f606/src/core/stabilization/frame_transform.rs#L220-L257)
- [Gyroflow visual RS synchronization](https://github.com/gyroflow/gyroflow/blob/b5e8828f82c150676e48a7c2e3db39c97392f606/src/core/synchronization/find_offset/rs_sync.rs#L74-L180)
- [Gyroflow iterative source-row ST map](https://github.com/gyroflow/gyroflow/blob/b5e8828f82c150676e48a7c2e3db39c97392f606/src/core/stmap.rs#L89-L103)

### 6. Valid region、mask、export

Valid region は source / sensor ごとに保持し、camera model の ray domain と user-adjusted physical region の積を
使う。SAM は object context のため full decoded image を入力できるが、出力 mask は geometric validity と交差する。
比較用の基準円は image center に固定し、中心 offset を intrinsics の代用にしない。任意形状は resolution-independent
な ordered circle-stamp operation (`add` / `subtract`) として sensor ごとに保存し、mask renderer、bounding box、
training crop が同じ順序で適用する。

Geometry correction、crop、resize を追加した場合、RGB、feature mask、training mask、principal point、2D
observation を同じ transform で更新する。Mask は nearest、image は一回の高品質 resample を使う。

Export は registered image だけを含め、LFStudio loader roundtrip で camera center と representative pixel ray を
確認する。`rigs.bin` / `frames.bin` を trainer が読むと仮定せず、`images.bin` に焼き込まれた pose を検証する。

## Error pattern の切り分け

| Pattern | 主な原因 |
|---|---|
| 静止 / low-gyro frame でも radius / azimuth に固定した residual | Intrinsics / projection model |
| Sensor 境界で一定の spherical offset、gyro と無関係 | Rig extrinsics |
| Signed scan coordinate × angular velocity に比例し、回転方向で符号反転 | Rolling shutter |
| Frame 全体が angular velocity と相関して回転 | IMU clock offset |
| Seam で不連続または局所 mesh pattern | Stitching |
| Rotational correction 後も近距離だけ残る | Translational rolling shutter / depth dependence |

Held-out frame で fixed lens basis と rolling-shutter basis のどちらが residual を説明するか比較する。Calibration に
使った frame と評価 frame を混ぜない。

## 必須 test matrix

- Probe: magic、wrong extension、missing / duplicate resource、ambiguous format
- Timing: non-zero start、VFR、B-frame、drop / duplicate、unequal sensor count、allowed skew
- Decode: hardware / software pixel tolerance、atomic multi-sensor capture
- Projection: grid roundtrip、boundary、non-square crop / scale / rotation / mirror
- Rig: known baseline / direction、quaternion norm / rotation determinant、real overlap epipolar residual
- Shutter: all four scan directions、timestamp reference、offset / drift、identity / no-motion
- Selection: sensor 1 だけ blurred / clipped の fixture を reject
- Pipeline: inspect → extract → prepare → both masks → feature → match → reconstruct → export
- UI: 全 sensor preview、valid-region edit、source group、localized error / statistics
- Trainer: LFStudio load、camera center / ray roundtrip、短い training smoke
- Safety: cancel は temporary artifact だけを消し、original resource を保持

Real golden data は少なくとも current short X5、Parktest、generic official ERP、perspective control、旧 dual-file、
selfie / reversed mapping、synthetic VFR rig を含める。

## 現在までの実測 evidence

- 同一個体の short clip と Parktest は calibration ID `197632` と完全に同じ `offset_v3` を持つ。
- 両 file の window crop は 5376²→5312²。旧 5376→3840 direct scale は focal を 1.204819% 過小評価した。
  Low-motion single-lens pilot で crop 修正は lens0 angular median を 0.342°→0.108°へ改善した。
- Full relative quaternion は理想 180° ではなく約
  `[-0.0017243, -0.0019477, 0.9999962, -0.0008950]`、baseline は約 32.273 mm。
- MEI → joint THIN_PRISM fit は lens0 で COLMAP / LFStudio RMS 0.0396 / 0.0421 px、lens1 で
  0.1029 / 0.1112 px。ただし stock LFStudio inverse bug は lens1 で training scale maximum 約 11.4 px となる。
- Readout は 21.244001 ms。Short selected frame の frame 内回転は median 約 0.423°、Parktest は
  median 約 0.411°、P95 約 1.022°、maximum 8.482°。Risk filtering は correction ではない。
- Overlap feature だけで rig 回転を上書きした実験は parallax / dynamic object に引かれ、cross-sensor track と
  SfM residual を悪化させた。Metadata extrinsics を保持する。
- Official app stitch は optical-flow / rolling-shutter warp を含むため、raw lens と ERP の単純な global rotation
  difference を subpixel external calibration として使わない。

## Definition of done

新 adapter は sample が読み込めるだけでは完了しない。Adapter 固有 code が probe / metadata / extraction mapping
へ隔離され、core にメーカー名分岐が増えていないこと、全 PTS / sensor / projection / rig / shutter test が通ること、
real UI path から export と LFStudio loader smoke が通ること、held-out geometry report が既存 adapter を退行させない
ことを満たして初めて default-enabled とする。

[Insta360 Desktop Media SDK](https://github.com/Insta360Develop/Desktop-MediaSDK-Cpp) のような proprietary
SDK は、公式 stitch との A/B や optional derived ERP generator に使う余地はあるが、この application の
native backend、必須 dependency、calibration source of truth にはしない。
