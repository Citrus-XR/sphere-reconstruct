#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "$0")/.." && pwd)"
for dependency in colmap ffmpeg ffprobe; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "$dependency がありません. 先に 'brew install colmap ffmpeg' を実行してください." >&2
        exit 1
    fi
done
exec "$repository_dir/scripts/start-linux.sh"
