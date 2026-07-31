#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--force]

VGGT-Omega is gated. Request access at:
https://huggingface.co/facebook/VGGT-Omega
Then authenticate with HF_TOKEN or 'hf auth login'.
EOF
}

parse_common_args "$@" || {
    usage
    exit 0
}

destination="${WEIGHTS_ROOT}/vggt_omega"
hf_download \
    "facebook/VGGT-Omega" \
    "${destination}" \
    "vggt_omega_1b_512.pt"

log "VGGT-Omega weights ready under ${destination}"
