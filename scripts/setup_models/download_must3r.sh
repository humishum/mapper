#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    echo "Usage: $0 [--dry-run] [--force]"
}

parse_common_args "$@" || {
    usage
    exit 0
}

destination="${WEIGHTS_ROOT}/must3r"
base_url="https://download.europe.naverlabs.com/ComputerVision/MUSt3R"

# The post-CVPR 512 checkpoint is the current upstream recommendation.
download_file "${base_url}/MUSt3R_512.pth" \
    "${destination}/MUSt3R_512.pth" \
    "8854f948a8674fb1740258c1872f80dc"
download_file "${base_url}/MUSt3R_512_retrieval_trainingfree.pth" \
    "${destination}/MUSt3R_512_retrieval_trainingfree.pth" \
    "f7c133906bcfd4fe6ee157a9ffa85a23"
download_file "${base_url}/MUSt3R_512_retrieval_codebook.pkl" \
    "${destination}/MUSt3R_512_retrieval_codebook.pkl" \
    "1125d80b9de940de2655d19b3ff78bb5"

log "MUSt3R weights ready under ${destination}"
