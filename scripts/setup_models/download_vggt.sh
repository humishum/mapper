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

destination="${WEIGHTS_ROOT}/vggt/VGGT-1B"
hf_download \
    "facebook/VGGT-1B" \
    "${destination}" \
    "model.pt" \
    "860abec7937da0a4c03c41d3c269c366e82abdf9"

log "VGGT weights ready under ${destination}"
