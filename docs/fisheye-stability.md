# Native dual-fisheye の sparse reconstruction 安定性

TestO2 では、局所 BA の近傍数を増やしても点群品質の改善は確認できなかった。三角測量閾値を緩和すると観測と点数は増えたが、残差も増えたため、既存の厳格設定と成果物を維持する。2026-09-08 の Mapper 比較を以下に示す。2026-09-09 の [固定 pose 再三角化](fixed-pose-triangulation.md) は留保画像での予測 coverage を増やしたが、利用者の目視比較では大きな改善がなく、品質改善策として採用しない。一つの実写 dataset の結果であり、他の機種に適用する既定値を決める根拠にはしない。

## 比較条件と再実行

COLMAP 4.1.1、Incremental Mapper、native `OPENCV_FISHEYE` 2 sensors、276 captures / 552 images。既存の rectified images と同じ matched database を各 case 専用 directory にコピーした。再抽出・再マッチング・ERP 合成は行わず、intrinsics と sensor-from-rig pose を固定した。COLMAP source は変更していない。

共通設定は `random_seed=0`、CPU BA、local/global iteration 上限 40/200、video 用 global BA growth ratio 1.4。

- `strict_repeat`: 既存の厳格 preset、local BA 近傍数は stock default の 6。
- `strict_local12`: 厳格 preset を維持し、`--Mapper.ba_local_num_images 12` だけを追加。
- `medium`: [一般 preset](setup-gpu.md#incremental-mapper-の三角測量設定)、local BA 近傍数は 6。

Backend environment から次の script を使う。出力先には未使用 directory を指定する。

```text
python scripts/benchmark_sparse_stability.py <project-directory> <new-experiment-directory> --colmap <colmap-executable> --detach
python scripts/audit_fisheye_support.py <project-directory>
```

Benchmark は `mapper.log`、`sparse/0`、`result.json` を case ごとに保存し、終了した case を `results.json` に集計する。Solver 診断は trial step の棄却と BA 終了失敗を区別する。Point color extraction を省略するため、そのまま training dataset として扱わない。今回の保存先は workspace の sibling にある `experiments/testo2-stability-20260908`。

## 全三組の結果

| 指標 | 厳格・再実行 | 厳格・12 近傍 | 一般 |
|---|---:|---:|---:|
| Registered images | 552 / 552 | 552 / 552 | 552 / 552 |
| Points | 110,919 | 110,582 | 212,889 |
| Observations | 932,832 | 934,925 | 1,861,664 |
| Mean point reprojection error | 0.423510 px | 0.423735 px | 0.591289 px |
| P95 point reprojection error | 0.661389 px | 0.659185 px | 0.954840 px |
| Mean track length | 8.4100 | 8.4546 | 8.7448 |
| Two-observation points | 26.64% | 26.09% | 24.86% |
| Rejected linear-solver steps | 5 | 10 | 20 |
| Failed BA terminations | 0 | 0 | 0 |
| Runtime | 485 s | 555 s | 722 s |

元の production run は step 棄却 13 回、BA 終了失敗 0 回だった。厳格な再実行と point count、残差統計、track 統計が一致し、capture center の similarity alignment 後の最大差は約 `1.92e-12` model units。Warning 数だけでは出力品質を評価できないことを示す。

各 model から約 5,000 点を等間隔抽出し、track 内最大三角測量角を計算した。P05 / median / P95 は厳格で 5.53° / 10.85° / 34.48°、12 近傍で 5.55° / 10.78° / 33.98°、一般で 3.41° / 7.71° / 31.13°。一般 preset は低視差側の点を保持するが、その追加点が正しい遠景か誤対応かはこの集計だけでは判別できない。

## 軌跡と尺度

同一 capture の二 sensor center の平均を取り、厳格な再実行に similarity alignment した。誤差の正規化には reference capture-center bounding-box diagonal を使う。

| 指標 | 厳格・12 近傍 | 一般 |
|---|---:|---:|
| Candidate → reference scale | 0.946717 | 1.760325 |
| P95 center difference / reference span | 0.0227% | 0.0491% |
| Maximum center difference / reference span | 0.4250% | 0.0870% |

全 case で sensor 間距離は約 0.032081195、optical axis の内積は約 -0.999953764 のまま、276 captures すべてに二 sensor が存在する。Rig が分解された結果ではない。一方、trajectory diameter は 54.8382 / 57.9509 / 31.1380 model units と異なる。Baseline が同じなので、この差を単なる単位換算と扱わない。Similarity alignment はこの尺度差を取り除くため、小さい誤差を metric accuracy の証明にしない。

Stock global BA は最初の frame pose と別 frame の translation 一成分を固定し、local BA は三つの point 座標を固定する。Calibrated rig の場合もこの inherited anchor がある。Normalization は rig baseline 自体も scale するため、今回の差を normalization だけで説明することはできない。ただし、anchor が差の主因であるかは未確定である。

根拠: [global gauge](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/estimators/bundle_adjustment_ceres.cc#L388-L410)、[local gauge](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/sfm/incremental_mapper.cc#L959-L965)、[reconstruction transform と rig baseline](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/scene/reconstruction.cc#L785-L803)。

同一 capture に stereo overlap がなくても、姿勢変化や異なる時刻の cross-sensor track が尺度を拘束する場合がある。Application の `rig_baseline_not_observable` は十分な直接 stereo evidence が得られないという保守的な判定であり、運動からの metric recovery が数学的に不可能という証明ではない。

この比較では ground truth、既知距離、独立した depth 測定を使用していない。低い残差、長い track、全画像登録、軌跡の再現性は有用な診断だが、繰り返し模様に対する誤対応や空中の点を否定しない。厳格 preset が幾何的に最適と確定したわけではなく、candidate への置換を正当化する改善が得られなかったため現状を維持する。

## 有効視野と matching 統計

抽出 database の全 keypoint を camera ray に戻して検査した。

| Sensor | Keypoints | 光軸から 90° 以上 | 最大 off-axis angle |
|---|---:|---:|---:|
| lens0 | 4,743,897 | 0 | 89.549875° |
| lens1 | 5,335,452 | 0 | 89.561696° |

有効 half-angle は 89.55°。lens1 の 4 点だけが subpixel localization によりこの境界をわずかに越えるが、90° には達していない。Raw source の 180° を超える外周は現行 SfM の入力に残っていない。さらに crop を狭める根拠は得られなかった。

Optical axis separation は 179.449030°、二つの有効 half-angle の和は 179.1°。既存の visibility filter は幾何的に重ならない同一 capture の 37 pair を除外済みだった。Raw pair 6,040、verified pair 5,744 のうち、inlier 15 未満は 372 pair。その 369 pair が cross-sensor `CALIBRATED_RIG` だったが、Mapper log 自体が `ignored 372` を記録していた。これらを database から再削除しても、今回の BA input は改善しない。

Rig の inlier threshold は集約した frame pair に適用され、その後 image pair に分配される。個別 pair が threshold 未満であることだけを破損と判定しない。`tri_ignore_two_view_tracks` は孤立した二観測の correspondence component を対象とし、最終 point の track length が 2 になる全ケースを禁止する設定ではない。

根拠: [rig inlier 分配](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/estimators/two_view_geometry.cc#L514-L557)、[Mapper cache threshold](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/scene/database_cache.cc#L279-L302)、[two-view predicate](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/scene/correspondence_graph.cc#L353-L362)。

## 採用した application 修正

Reconstruct は棄却された solver step と実際の BA 終了失敗を別々に表示する。Matching は threshold 以上／未満の pair 数を併記し、点群統計は二観測点の割合を明示する。Feature SAM3 の coverage は sky を含む semantic exclusion として表示し、prompt 別と sky 以外の面積を新しい実行で保存する。Feature coverage warning の判定では sky を除き、実際の mask pixel は変更しない。

既存成果物と manifest は書き換えていない。過去に保存していない prompt 別面積を推定値で補完せず、新しい詳細統計は対応 Stage の次回実行で生成する。新しい機種固有 crop 値、cross-sensor edge の一律削除、local BA 近傍数の既定変更は導入していない。

## 弱テクスチャの追加診断

Production SIFT model の各 observation 周辺で 9×9 Sobel gradient の平均を計算し、点の最大観測角、track 長、最近接 camera 距離と比較した。これは texture の proxy であり、surface material の分類ではない。

- 弱い texture の下位 20% は 22,184 点で、texture proxy の中央値 41.37、track 長中央値 3。
- 強い texture の上位 20% は同数で、texture proxy の中央値 227.90、track 長中央値 6。
- 弱い texture かつ最大観測角 8° 未満は 6,673 点。角度中央値 6.40°、track 長中央値 2、点 error P95 0.728 px。
- 弱い texture の下位 20% に限定して `track <3` かつ角度 `<8°` を組み合わせると 3,527 点、全体では 11,786 点が候補になる。
- 全体の角度中央値は 10.85°。5° の深度条件数はおよそ `1/sin(5°)=11.5` 倍で、reprojection error が小さくても depth は不安定になり得る。

このため、`max_reprojection_error` だけで空中点を除外するのは不十分である。全点に `min_track_length=3` を適用するだけでも 29,548 点を失い、近景の細部を壊す可能性がある。実用的な候補は、遠景または弱 texture の点に対してだけ、`track >= 3`、十分な triangulation angle、異なる時刻への分散、複数 view の reprojection consistency を組み合わせる方法である。今はその判定を production cleanup に自動追加していない。

### ALIKED + LightGlue の対照

SIFT の代替として、同じ native dual-fisheye workspace で ALIKED N16ROT + LightGlue を 2048 px / 4096 keypoints、sequential overlap 4、同じ strict Mapper 設定で実行した。LightGlue が guided matching をサポートしないため guided は無効、ALIKED 経路には vocab-tree loop closure がない。

| 指標 | SIFT production | ALIKED isolated run |
|---|---:|---:|
| Registered images | 552 | 552 |
| Points | 110,919 | 39,020 |
| Mean reprojection error | 0.423510 px | 0.412708 px |
| P95 reprojection error | 0.661389 px | 0.678751 px |
| Mean track length | 8.4100 | 6.5448 |
| Median track length | 4 | 3 |
| Two-observation ratio | 26.64% | 34.61% |
| Trajectory diameter | 54.8382 | 40.4602 |

ALIKED は今回の設定では弱 texture の multi-view 拘束を増やさず、点数と track 支持が減った。抽出解像度と pairing contract は production SIFT と完全には一致しないため、この設定を採用しない根拠であり、モデル系列全体への結論ではない。追加実験は SIFT と native fisheye を維持する。

## 改善の検証順序

1. Camera-center の移動量と視差を区別して調べ、近接 capture の overlap を保つ。現行 spatial flow は縮小画像の Farneback flow であり、回転補償した並進視差ではない。同じ場所で回転した frame の増加は、深度の観測拘束を増やすとは限らない。
2. 既存の intrinsics / rig / poses を固定した条件付き比較を行い、点の変更と軌跡の変更を分離する。元の calibration や軌跡の絶対精度が検証済みという意味ではない。
3. Capture 全体を留保し、残りの観測だけで三角化して留保画像への投影を検証する。さらに track を時間的に前後へ分け、独立に得た XYZ の差を測る。道路や橋面の勾配・曲率を保持するため、全体を単一平面に拘束しない。
4. 時間分割で安定した低視差点の追加は検証済みだが、目視で大きな改善がなかったため採用しない。[Sparse cleanup 比較](strict-sparse-cleanup.md) では元の点も除去対象とする。Split policy は浮遊点を減らしたが大きな欠損を生じた。同条件の LFStudio training と目視比較を経て、full-track の条件付き不確実性と capture 単位の留保予測を正式 cleanup に採用した。Coverage 増加や削除数だけを修復の判定にせず、既存点の誤対応・深度不確実性と pose / calibration の系統誤差を区別する。単眼 depth に基づく geometry の置換は行わない。

COLMAP の公式 tutorial は、白壁や無地の机などの弱 texture には追跡可能な拘束が少なく、視点の並進が必要であると説明している。FAQ にある dense MVS の解像度や PatchMatch window の調整は、現在の sparse SfM へそのまま適用できない。根拠: [COLMAP tutorial](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/doc/tutorial.rst#L125-L147)、[FAQ feature extraction](https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/doc/faq.rst#L43-L53)、[弱 texture と運動](https://github.com/colmap/colmap/issues/830#issuecomment-602847248)。
