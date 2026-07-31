#!/usr/bin/env bash

set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $0 MODEL [MODEL ...] [--dry-run] [--force]

Models:
  must3r
  vggt
  da3-streaming
  vggt-long
  mast3r-slam
  vggt-omega
  all

External checkout variables:
  DA3_STREAMING_PATH
  VGGT_LONG_PATH
  MAST3R_SLAM_PATH

All files are stored below the repository's ignored weights/ directory.
EOF
}

models=()
forwarded=()
while (($#)); do
    case "$1" in
        --dry-run|--force)
            forwarded+=("$1")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            models+=("$1")
            ;;
    esac
    shift
done

((${#models[@]})) || {
    usage >&2
    exit 2
}

if [[ " ${models[*]} " == *" all "* ]]; then
    models=(must3r vggt da3-streaming vggt-long mast3r-slam vggt-omega)
fi

for model in "${models[@]}"; do
    case "${model}" in
        must3r)
            script="download_must3r.sh"
            ;;
        vggt)
            script="download_vggt.sh"
            ;;
        da3-streaming|da3_streaming)
            script="download_da3_streaming.sh"
            ;;
        vggt-long|vggt_long)
            script="download_vggt_long.sh"
            ;;
        mast3r-slam|mast3r_slam)
            script="download_mast3r_slam.sh"
            ;;
        vggt-omega|vggt_omega)
            script="download_vggt_omega.sh"
            ;;
        *)
            echo "Unknown model: ${model}" >&2
            usage >&2
            exit 2
            ;;
    esac
    "${SETUP_DIR}/${script}" "${forwarded[@]}"
done
