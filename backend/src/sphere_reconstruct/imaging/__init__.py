"""imaging パッケージ.

ffmpeg / ffprobe / OpenCV による画像処理を Worker 側で行う.
FastAPI プロセスからは import しない (OpenCV や numpy を引きずらないため).
"""
