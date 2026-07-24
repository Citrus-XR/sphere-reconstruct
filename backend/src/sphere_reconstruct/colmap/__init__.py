"""colmap パッケージ.

COLMAP CLI を子プロセスとして駆動する. pycolmap を API プロセスで直接使わない
方針 (spec §7):
- ログをリアルタイムに捕捉できる
- 正確にキャンセルできる (プロセス kill)
- クラッシュ隔離
- COLMAP バイナリのパスを手動指定 / ビルド切替が容易

このパッケージは Worker プロセス側で使う. model.py の binary reader だけは
純粋なので API からも利用可能.
"""
