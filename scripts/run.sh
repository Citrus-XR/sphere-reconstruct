#!/usr/bin/env bash
set -euo pipefail
repository_dir="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$(uname -s)" == "Darwin" ]]; then
    exec "$repository_dir/start-macos.sh"
fi
exec "$repository_dir/start-linux.sh"
