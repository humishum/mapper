#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--force]

Set MAST3R_SLAM_PATH to the pinned MASt3R-SLAM checkout. The script creates
an upstream checkpoints symlink to Mapper's ignored weights directory.
EOF
}

parse_common_args "$@" || {
    usage
    exit 0
}

repo_dir="$(resolve_external_repo "${MAST3R_SLAM_PATH:-}" "main.py" "MAST3R_SLAM_PATH")"
destination="${WEIGHTS_ROOT}/mast3r_slam"
base_url="https://download.europe.naverlabs.com/ComputerVision/MASt3R"

download_file \
    "${base_url}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    "${destination}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
download_file \
    "${base_url}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" \
    "${destination}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
download_file \
    "${base_url}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" \
    "${destination}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
link_external_directory "${destination}" "${repo_dir}/checkpoints"

log "MASt3R-SLAM checkpoints ready under ${destination}"
