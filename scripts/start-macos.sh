#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "$0")/.." && pwd)"
if ! command -v colmap >/dev/null 2>&1; then
    echo "COLMAP がありません. 先に 'brew install colmap' を実行してください." >&2
    exit 1
fi
exec "$repository_dir/scripts/start-linux.sh"
