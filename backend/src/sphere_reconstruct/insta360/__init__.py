"""insta360 パッケージ.

INSV コンテナ (MP4) と, その末尾に貼り付けられた Insta360 独自フッタを扱う.
また外部 `.insv.pb` の読み取り, IMU (0x0300) 記録の解釈, offset_v3 / MEI 拡張
畜れみモデルへのマッピングを提供する.

参考:
- INSV/PB の内部構造とレコードタグの一部は
  https://github.com/BenjaminHenriksson/insv-stitch の PIPELINE.md / FINDINGS.md
  の逆解析結果を参考にしている. コードは独立に書いている.
- X5 サンプル 1 台での観察を汎化しないこと. IMU-to-camera 行列, ストリーム順
  (前 / 後), xi の範囲は個体差 / ファーム差の可能性がある.
"""
