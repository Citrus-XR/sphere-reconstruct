# Adding 360 camera formats

この文書は新しい 360 camera、raw multi-fisheye container、metadata sidecar、stitched panorama を追加する AI / developer 向け contract である。実装は素材から検証可能な metadata と projection contract だけに依存する。

## Principles

1. Vendor parsing と geometry core を分離する
2. Sensor-local pixel coordinates へ正規化する
3. Rotation 名は `target_from_source`
4. Rig 外参は `cam_from_rig`
5. Unit は meter、quaternion は wxyz
6. Original source は immutable
7. Unknown value を heuristic default で隠さない
8. RGB、mask、camera、observation は同じ transform chain を使う

Device-specific数値は adapter / parsed metadata にだけ置く。Current camera の crop、focal、baseline、shutter readout を別 device の default にしない。複数 device で同じことが specification / test により確認された場合だけ generic core へ昇格する。

## Adapter output

Adapter は source inspection と `camera_system.json` を生成する。

```json
{
  "calibration_source": "vendor_metadata_version",
  "coordinate_system": {
    "handedness": "right",
    "x_axis": "right",
    "y_axis": "down",
    "z_axis": "forward",
    "length_unit": "meter",
    "pixel_origin": "top_left_pixel_center_0"
  },
  "reference_sensor_id": "sensor0",
  "sensors": [
    {
      "id": "sensor0",
      "image_key": "sensor0",
      "calibration_image_transform": {
        "reference_width": 0,
        "reference_height": 0,
        "crop_x": 0,
        "crop_y": 0,
        "crop_width": 0,
        "crop_height": 0
      },
      "projection": {
        "model": "omni",
        "width": 0,
        "height": 0,
        "xi": 0,
        "fx": 0,
        "fy": 0,
        "cx": 0,
        "cy": 0,
        "distortion_model": "radtan_pro",
        "distortion_parameters": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      },
      "cam_from_rig": {
        "rotation_wxyz": [1, 0, 0, 0],
        "translation_xyz": [0, 0, 0]
      },
      "shutter": {
        "type": "unknown",
        "readout_time_ms": null,
        "scan_direction": "unknown",
        "timestamp_reference": "unknown"
      }
    }
  ]
}
```

Unsupported intrinsics を別 model と偽装しない。New projection は `domain/camera_system.py` と numerical projection module に明示追加する。

Insta360 は UI / persistence 上で単一 adapter とし、camera model 名から calibration を推測しない。素材内の versioned calibration について、V3 は 5-parameter radtan、V6 は 13-parameter radtan-pro として解釈する。同一素材に複数 version がある場合は最高の対応 version を選び、未知 version しかない場合は inspection を失敗させる。

Container stream ordinal と calibration sensor ordinal を同一視しない。INSV は MP4 stream1 を calibration lens0、stream0 を calibration lens1 として sensor-local artifact へ正規化する。新 adapter も decoded stream から canonical sensor ID への写像を明示し、単なる列挙順を camera identity にしてはならない。

V6 radtan-pro の coefficient order は `k1..k5,p1..p4,s1,s3,s2,s4`。`a=p1+p3·r²`、`b=p2+p4·r²` とした時、水平項は `a·(r²+2x²)+2bxy`、垂直項は `2axy+b·(r²+2y²)` である。Forward と analytic Jacobian は同じ contract を実装する。

## Inspection

Inspection は少なくとも次を記録する。

- adapter id / version
- media kind
- sensor count
- stream codec / dimensions / FPS / time base
- packet PTS range
- calibration block version / checksum
- lens model / image transform
- physical rig extrinsics
- exposure / IMU / shutter metadata availability
- warnings / unsupported reason

Opaque vendor blob は parsed field と raw reference を区別する。Calibration validity が不足する raw format は native reconstruction を開始しない。

## Frame extraction

Multi-sensor video は同じ demux decision から capture を作る。

```json
{
  "index": 42,
  "source_frame": 420,
  "timestamp_sec": 17.5,
  "sensor0": "...",
  "sensor1": "...",
  "score": {
    "rolling_shutter_motion_deg": 0.2
  }
}
```

Requirements:

- Packet PTS を presentation order で扱う
- Sensor skew を測り、許容値を超えた capture を error / reject
- Frame selection は全 required sensor の worst quality を使う
- Video source だけ decode、image source は original still を直接登録
- Cancellation / progress は decode、score、selection、write の各 phase を報告

Exposure timestamp がある場合、encoded PTS と exposure clock を混同しない。

## Calibration image transform

Metadata calibration canvas と decoded sensor image が異なる場合、transform を必ず明示する。

```text
reference sensor canvas
  -> crop / window
  -> decoded sensor-local image
  -> optional rectification target
```

Principal point と focal を同じ順序で transform する。Half-pixel convention を混在させない。Crop は rig extrinsics を変更しない。

Validation:

- reference corner / center mapping
- round-trip pixel error
- scaled resolution
- odd dimensions
- asymmetric principal point
- sensor order swap failure

## Projection

Projection module は vectorized forward / inverse を提供する。

```text
project(ray, intrinsics) -> pixel + validity
unproject(pixel, intrinsics) -> unit ray + validity
```

Tests:

- optical axis
- azimuth signs
- 0° / edge angles
- distortion monotonicity
- forward/inverse round trip
- source crop scaling
- invalid / non-finite domain

Camera model approximation の RMS だけを採用条件にしない。Maximum error、azimuth distribution、consumer implementation difference も測る。

Scene View の selected camera guide は model family へ対応させる。Perspective は rectangular frustum、`*FISHEYE` は circular view boundary、`EQUIRECTANGULAR` は spherical guide とする。Native fisheye Inspector preview は circular clip、reprojected pinhole / ERP は rectangular layout を保つ。Camera pick は device-independent な screen pixel distance で判定し、world scale を threshold に使わない。新 camera model を追加する場合は classification、blank-click deselection、pick boundary の UI test も更新する。

## Internal consumer normalization

Raw non-consumer projection は `rectify_fisheye` artifact を独立した user-visible Step として生成する。下流 Stage から直接実行した場合も dependency plan が先に生成するため、adapter が normalization を bypass することはできない。

```text
target consumer pixel
  -> target camera ray
  -> source calibrated pixel
  -> backward image sample
```

Contract:

- Same-resolution one-pass resample
- RGB: Lanczos4
- Validity / hand-painted region: nearest
- Camera group と rig config を同時更新
- Pose、timestamp、sensor center は維持
- SAM、features、matching、SfM、dense seed、export は rectified catalog を読む
- Matching は同一 capture の sensor pair について、`cam_from_rig` の optical axis 間隔が両 consumer-valid half-angle の和を超える時だけ edge を除外する
- Adapter option で bypass しない

Camera metadata だけを consumer model に交換し、source pixels を残す方法は禁止。Image と camera ray が一致しないためである。

## More-than-180° fisheye

COLMAP 4.1.1 と LFStudio v0.5.3 の `OPENCV_FISHEYE` は forward hemisphere implementation である。

- Projection は `z <= 0` を reject
- Default unprojection は `z > 0`
- Consumer domain は half-angle 90° 未満

Physical lens が 180° を超える場合でも、stock consumer へ back-facing ray を渡さない。UI は native dual-fisheye + Global Mapper selection に warning を表示する。

Current experiment では physical overlap を使える COLMAP extension が 86 / 96 same-capture pair と 2,862 inliers を作ったが、GLOMAP が later stages で stereo track を保持しなかった。したがって extension は production dependency ではない。

Future adapter は次を別々に記録する。

- physical sensor coverage
- consumer-valid forward coverage
- removed overlap / blind region
- training-valid region

Outer overlap を SfM-only auxiliary constraint に使う場合、final LFStudio export から auxiliary camera / back-facing pixel を除外し、stock loader compatibility を smoke test する。

## Rig extrinsics

`cam_from_rig` は:

```text
p_cam = R_cam_from_rig * p_rig + t_cam_from_rig
C_rig = -R^T t
```

Reference sensor は identity に正規化する。他 sensor は full 6DoF を保持し、ideal 180° rotation や単一 baseline axis へ丸めない。

Tests:

- Unit quaternion
- Camera-center baseline
- Sensor order
- Same capture frame membership
- Non-overlapping same-capture sensor edge is rejected; overlapping caps and different captures are retained
- Relative rotation / translation reproduction after mapper / export
- `images.bin` pose consistency

Fixed rig が model に書かれていることは metric scale evidence ではない。同じ world point を同 capture の複数 sensor が観測する必要がある。

## Rolling shutter and synchronization

Shutter fields:

- global / rolling / unknown
- readout time
- scan direction
- exposure start / center / end reference
- sensor exposure offset
- IMU-to-rig rotation
- gyro bias
- video / IMU offset and drift

Unknown scan direction で row dewarp を default-enable しない。Risk filter と correction を区別する。Correction を実装する場合、source raw grid で行い、後段 rectification と不要な多重 resample を避ける。

Validation は static、constant rotation、reversed scan、zero readout、clock drift を含む。

## Valid region and object masks

ソース有効領域は source ID と sensor ID ごとに独立する。魚眼 source は `lens0` / `lens1` の中心固定円を持ち、既定・最大半径は `r=0.5`。UI の slider・drag と API validation は同じ上限を使う。保存済みの超過半径は起動時に `0.5` へ補正し、それ以外の設定・brush は保持する。Perspective / ERP source は `main` の全画像領域から開始し、円形制限を持たない。いずれも add/subtract brush を保存できる。

保存先は `<project>/source_regions.json`。`views` の各値は `kind: circle | full` と `operations` を持ち、円形の場合だけ `cx/cy/r` を含む。Brush の `x/y` は EXIF 表示向きを適用した画像の幅/高さに対する比、半径 `r` は画像幅に対する比である。`stroke_id` は pointer down から up までの一筆を識別し、Undo は一筆全体を戻す。旧 `fisheye_regions.json` は起動時に保存済み半径・brush を維持して新形式へ移行する。

UI の source selector は主素材・補助素材の全 source を列挙し、選択した source の frame だけをプレビューする。切替中の未保存 draft は source ごとに保持する。Step の完了判定は有効 source の `saved && !needs_review` を集計し、未設定・設定済み数・全件完了を表示する。Save の応答を editor と Step の共通 query cache へ反映し、保存直後に更新する。無効 source と未保存 draft は完了判定に含めない。除外範囲は半透明黒に明るいピンクの斜線と輪郭を重ね、暗部でも識別できる。輪郭は最終的な除外 mask から作り、重複 brush の内部境界や add で復元した部分には残さない。この表示は保存・出力 mask の値に影響しない。Frame がない状態でも魚眼の半径を保存できる。領域変更の保存は `prepare_images` 以降を無効化する。

Native image には解析的な valid region を適用し、魚眼正規化や pinhole 再投影では RGB と同じ remap で bitmap validity を生成する。Source validity は SAM3 の有効/無効とは独立し、COLMAP feature mask、dense initialization、training export mask に適用する。

Screen-space artifact の例:

- lens flare / blue ring
- camera body
- hand / operator attached to camera
- stitching border

これらは world geometry ではないため feature extraction から除外する。Local artifact に対して uniform radius を過剰に縮めない。Object SAM mask は physical validity の後に合成する。

Per-image coverage、threshold exceed、detections は manifest に保存し、Inspector が live partial manifest から読む。Coverage exceed を一画像一行の Console warning にしない。Progress は `kind=progress` の image / prompt ticks で駆動し、diagnostic record と log verbosity を結合しない。

## Mapper selection

Mapper selection は user-visible で、warning は自動変更を行わない。

### Incremental

Calibrated multi-camera frame を generalized pose として順次登録し、full correspondence graph から complete / merge / retriangulate する。Current dual-fisheye recommendation。

### Global / GLOMAP

Global rotation / position を高速に解く。Perspective / ERP / strongly connected rig graph に有効。Native >180° dual-fisheye は unsupported overlap と sensor separation risk を UI で警告する。

Acceptance は registration count だけでなく:

- per-sensor observations
- cross-sensor 3D points
- same-capture shared points
- trajectory continuity
- baseline / path ratio
- per-sensor reprojection residual
- free-view thin edge / far landmark visual

を記録する。

## Export contract

Export は loader-ready folder を生成する。

```text
images/
masks/
sparse/0/
preview/
train_configs/
export_manifest.json
```

Requirements:

- Registered images only
- Image / mask dimensions match camera
- Crop offset を principal point / observation へ適用
- `images.bin` pose が canonical
- Camera model が LFStudio supported set に含まれる
- Back-facing / auxiliary SfM camera を final training set に残さない
- Rectified PNG は pixel crop、実 JPEG だけ jpegtran MCU crop
- Per-source registration と warnings を manifest に保存

## New adapter checklist

### Parse

- [ ] adapter registry id
- [ ] stream / image enumeration
- [ ] true timestamps
- [ ] calibration version and checksum
- [ ] sensor-local crop
- [ ] projection intrinsics
- [ ] full rig extrinsics
- [ ] shutter metadata

### Geometry

- [ ] forward / inverse tests
- [ ] crop / scale tests
- [ ] rig center / rotation tests
- [ ] physical vs consumer FOV report
- [ ] same-capture sensor validation

### Pipeline

- [ ] inspect
- [ ] video-only extraction
- [ ] physical region
- [ ] internal mandatory normalization artifact
- [ ] both SAM Steps
- [ ] feature / matching
- [ ] both mapper choices and warning behavior
- [ ] Full-track cleanup が camera model、capture grouping、相対不確実性を検証し、保持点と camera 座標を変えない
- [ ] gravity / metric / scene-coordinate alignment
- [ ] export loader smoke

### Safety

- [ ] original media survives clear / cancel
- [ ] corrupt input raises
- [ ] no device-local absolute path
- [ ] no vendor constant in generic core
- [ ] no silent heuristic calibration fallback

## Evidence record

When a new format is accepted, document:

- camera / firmware / metadata version
- calibration transform
- projection round-trip metrics
- sensor overlap and consumer FOV
- timing / shutter evidence
- feature / matching / mapper metrics
- LFStudio loader result
- known limitations
- source reference URLs with commit permalinks

Local experiment paths、private media names、temporary repository paths は documentation に書かない。
