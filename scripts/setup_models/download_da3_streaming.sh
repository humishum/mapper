#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--force]

Set DA3_STREAMING_PATH to either the Depth-Anything-3 checkout or its
da3_streaming directory. The script creates an upstream weights symlink;
all checkpoint bytes remain in Mapper's ignored weights directory.
EOF
}

parse_common_args "$@" || {
    usage
    exit 0
}

destination="${WEIGHTS_ROOT}/da3_streaming"
shared_salad="${WEIGHTS_ROOT}/shared/dino_salad.ckpt"
streaming_dir="$(resolve_da3_streaming_dir)"

hf_download \
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1" \
    "${destination}"
download_file \
    "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
    "${shared_salad}"
link_file "${shared_salad}" "${destination}/dino_salad.ckpt"
link_external_directory "${destination}" "${streaming_dir}/weights"

log "DA3-Streaming weights ready under ${destination}"
