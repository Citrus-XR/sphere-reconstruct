# 画像だけを使う混合素材の姿勢検証

360 動画とスマートフォン画像の混合再構成では、異常 camera の削除だけで姿勢の精度を保証できない。部屋の内側に残った誤登録、反復する壁面への登録、orientation の誤りも検証対象とする。LiDAR 点群、depth map、ARKit pose を求解条件に使わず、画像特徴と calibrated 360 camera から得た幾何を使う。

## RoomTest の初期状態

2026-09-24 の調査開始時点では、全 2,670 画像を joint Global Mapper で登録していた。主素材は 997 captures / 1,994 images、スマートフォンは 676 images。公開済み export は外れ値除去後の 571 phone images を含むが、そのうち 13 images は surviving sparse observation がゼロ、36 images は 10 observations 未満だった。点の除去後も旧 pose が残るため、export の整合性と姿勢の正しさは別に確認する必要がある。観測ゼロだけで誤 pose と断定するものでもない。

既存 matching は SIFT と vocabulary-tree retrieval を使っていた。連続する phone images の 675 pairs 中、116 pairs は raw matching 自体が未実施だった。前後 4 selected frames の 2,694 pairs 中、raw pair があるのは 1,456 pairs、15 inliers 以上の verified pair は 1,413 pairs。壁面、似た家具、反射面を含む画像では、retrieval で選ばれた画像への対応だけに依存する問題がある。

## Matching の変更

MatchFeatures 2.12 は vocabulary-tree retrieval の後に、同一 source / sensor の時間近傍で未実施の pair を補う。`overlap` は selected capture の順位で測り、異なる sensor や source を一つの sequence に混ぜない。既存 raw matches と verified geometry は保持し、失敗済み pair を同じ条件で繰り返さない。

動画は extraction の capture 順序を利用する。通常の写真集合は時間順と仮定せず、撮影順が確認できる source だけ `ordered_image_source_ids` で明示する。このリストは MatchFeatures API parameter であり、現在の stage settings UI には専用の選択欄を設けていない。

```json
{
  "pairing": "auto",
  "overlap": 4,
  "ordered_image_source_ids": ["PHOTO_SOURCE_ID"]
}
```

CLI は official COLMAP 4.2 の `matches_importer --match_type pairs` を使う。画像名の pair list から特徴 matching と幾何検証を実行し、既存の feature extraction を再利用する。Temporal interpolation で camera pose を作る処理ではない。

## 再登録と評価

Reconstruct 2.12 の primary-first path で 360-only model を作成し、primary pose と既知の calibration を固定して補助 camera を Incremental Mapper で登録する。推測した phone focal length は trusted prior にしない。Matching を追加する実験は同じ primary model を使い、全体の再構成やフレーム抽出を繰り返さない。

最小手順の比較用に、`scripts/prepare_fixed_primary_reference.py` は既存の primary pose を固定し、phone observation を除いた primary observation だけで点座標を再三角化する。3 captures 以上で観測された track を対象とし、reprojection error と triangulation angle で検証する。この参照は camera pose と track identity を joint model から継承するため、独立した 360-only reconstruction とは区別する。`--reference-report` でこの制約を姿勢検証の report に引き継ぐ。

`scripts/experiment_sequence_completion.py` は隔離 database で不足 pair を補完する。`--matches-only` で matching だけを先に完了でき、後の実行で `--database` にその database を指定すると再利用できる。`--ordered-source` は順序が確認できる source、`--calibration-model` は画像から求めた calibration の参照を指定する。現在の experiment driver は SIFT brute-force を対象とし、production matcher 自体は既存の matcher type を引き継ぐ。

`scripts/audit_mixed_pose_geometry.py` は primary-only tracks と cross-source correspondences で旧 / 新 phone pose を比較する。

- 座標系の整列には primary camera だけを使い、20 capture block の交互分割で fitting / heldout を分ける。
- PnP は一部の correspondence を fitting から除外し、別の correspondence への投影で検証する。
- `--probe-focal` は同じ分割で per-image focal estimation も比較する。
- 移動量の比較は両 model に存在する同じ隣接 pair を使い、欠落数を別に報告する。

Mapper が既に使った image matches を audit が含むため、これを SfM 全体から独立した ground truth と扱わない。反復物体の誤対応や planar calibration の曖昧性は、低い reprojection error だけでは解消しない。

## LFS の検証

LFS の camera loss heatmap は camera ごとの photometric loss EMA を相対的に着色する。[対象 build の実装](https://github.com/MrNeRF/LichtFeld-Studio/blob/395c7f31edc3790fbe013f2bd9bed9f9af1e45e7/src/training/trainer.cpp#L2634) は観測済み camera の最小 / 最大値で正規化する。赤い camera frame は pose error の直接検出結果ではなく、全 camera が緑になることを採用条件にしない。

`scripts/prepare_pose_training_comparison.py` は共通の phone cohort、同じ primary camera、同じ seed points、同じ RGB / masks を持つ二つの dataset を生成する。Phone pose とその estimated calibration が比較対象となる。再登録できなかった画像を報告し、除外による loss 改善を pose 修正に数えない。

`scripts/benchmark_lfs_training.py --max-width 2048` で解像度を明示し、同じ iteration、strategy、capacity、mask mode と評価分割を使う。Current export の診断 training と、この共通条件での比較は区別する。Windows の `--detach` は WMI で起動し、SSH session の終了から supervisor を分離する。

## RoomTest の実測結果

再登録と二条件の 7,000 iterations training は 2026-09-24 に完了し、2026-09-27 に report と render を確認した。実験名は `fixed-primary-repair`。不足していた 1,238 pairs を補完し、既存の 360 pose を固定した条件付き参照を用いた。LiDAR、depth、ARKit pose は使用していない。独立した 360-only Global Mapper の二つの seed は主軌跡に大きな跳躍または発散を生じたため、今回の採用結果には使っていない。したがって、別 dataset での主素材再構成まで検証済みとはしない。

- 360 は 1,994 images を維持し、camera center の最大変化は約 `1.05e-14` model units。
- Phone は 676 images 中 665 を再登録した。旧 export の 571 images との共通集合は 568、新たに含まれる画像は 97、旧 export から登録できなかった画像は 3。登録件数の増加を全画像の精度保証とはしない。
- 同一の 660 adjacent pairs の移動量 p95 は `151.073 → 0.454`、最大値は `1,335,723.666 → 5.270` model units。メートル尺度は確定していない。
- 主軌跡への距離は p95 `0.402`、最大 `0.943` model units。主軌跡から離れる outlier 判定はゼロだが、隣接移動量は `1400→1401` と `1423→1424` が p95 の 10 倍を超える。他にも大きな移動や回転が残り、部屋の内側にあることだけで正確とは判定しない。
- Primary-only correspondence の支持判定は共通画像で `334 → 357`。Inlier 合計は `206,890 → 197,774` と減っており、全指標が一様に改善した結果ではない。

再登録できなかった phone frame は `1377, 1378, 1402, 1403, 1437, 1595–1600`。このうち `1402, 1403, 1437` は旧 export に存在した。削除による改善を避けるため、以下の A/B は両方に存在する同じ 568 phone images と 1,994 primary images、同じ 354,625 primary-only seed points を使用した。

| Official heldout metric / 7,000 iterations | 旧 pose | 新 pose |
| --- | ---: | ---: |
| PSNR (dB) | 21.547941 | 21.674349 |
| SSIM | 0.850071 | 0.849228 |

全体 PSNR は小幅に改善し、SSIM はわずかに低下した。以下は事前選択した同一視点の JPEG render に対する診断 PSNR で、official heldout metric とは異なる。Training views を含み、単一 seed の結果である。

| Phone frame | 旧 pose (dB) | 新 pose (dB) | Render の確認 |
| --- | ---: | ---: | --- |
| 001144 | 14.66 | 25.00 | 入口付近からベッド側の壁へ修正 |
| 001147 | 12.85 | 24.04 | 机側の照明からベッド側の照明へ修正 |
| 001156 | 10.88 | 21.66 | 同じくベッド側の照明へ修正 |
| 001230 | 9.11 | 19.67 | 窓外から室内のテレビと窓へ修正 |
| 001496 | 11.32 | 15.33 | 浴室の向きは改善、鏡・ガラス部分は不十分 |

正常視点 `000997, 001120` と primary `000350` の診断 PSNR はそれぞれ `0.52, 0.20, 0.31 dB` 低下した。特定の誤登録が明確に改善した一方、全視点の完全修復や training 品質の一様な向上は確認できていない。

根拠は実験内の `registration/report.json`、`audit/report.json`、`final-trajectory.json`、`training-comparison/report.json` と `training-comparison/contact.jpg`。Color extraction 後の sparse model は 1,352,651 points、純黒点は 4。旧 experiment report の黒点数は色抽出前の値だったため、完成モデルを再読込して確認した。現在の driver は色抽出後にこの統計を更新する。A/B は primary-only seed points を使っているので、完成モデルの全点・全 665 phone images を用いた training の成績とは区別する。

2026-09-27 に改善後の model を RoomTest へ反映した。旧 reconstruction と下流成果物・manifest をバックアップし、完成した model と matching database を取り込んで、alignment、scale、scene alignment、export だけを再実行した。追加の camera 削除は行っていない。Export は 2,659 images / 1,352,651 points と全画像分の training masks を含み、`training_ready` 検証と公開元 model のファイルハッシュ検証を通過した。未解決の trajectory warning は import receipt と幾何 report に残し、この export を全 camera の修正完了とは扱わない。
