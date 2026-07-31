#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--force]

Set VGGT_LONG_PATH to the pinned VGGT-Long checkout. This installs only the
VGGT backend's required weights, not the optional Pi3 and MapAnything models.
EOF
}

parse_common_args "$@" || {
    usage
    exit 0
}

repo_dir="$(resolve_external_repo "${VGGT_LONG_PATH:-}" "vggt_long.py" "VGGT_LONG_PATH")"
destination="${WEIGHTS_ROOT}/vggt_long"
shared_salad="${WEIGHTS_ROOT}/shared/dino_salad.ckpt"
vggt_checkpoint="${WEIGHTS_ROOT}/vggt/VGGT-1B/model.pt"

vggt_args=()
[[ "${DRY_RUN}" != "1" ]] || vggt_args+=(--dry-run)
[[ "${FORCE_DOWNLOAD}" != "1" ]] || vggt_args+=(--force)
"${SETUP_DIR}/download_vggt.sh" "${vggt_args[@]}"
download_file \
    "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt" \
    "${shared_salad}"
download_file \
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" \
    "${destination}/dinov2_vitb14_pretrain.pth"
download_file \
    "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz" \
    "${destination}/ORBvoc.txt.tar.gz"
if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: extract ORBvoc.txt from ${destination}/ORBvoc.txt.tar.gz"
elif [[ ! -f "${destination}/ORBvoc.txt" || "${FORCE_DOWNLOAD}" == "1" ]]; then
    tar -xzf "${destination}/ORBvoc.txt.tar.gz" -C "${destination}"
fi
link_file "${shared_salad}" "${destination}/dino_salad.ckpt"
link_file "${vggt_checkpoint}" "${destination}/model.pt"
link_external_directory "${destination}" "${repo_dir}/weights"

log "VGGT-Long weights ready under ${destination}"
