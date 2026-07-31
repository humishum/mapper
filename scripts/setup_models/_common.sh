#!/usr/bin/env bash

set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SETUP_DIR}/../.." && pwd)"
WEIGHTS_ROOT="${REPO_ROOT}/weights"
DRY_RUN="${DRY_RUN:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

log() {
    printf '[model-setup] %s\n' "$*"
}

die() {
    printf '[model-setup] error: %s\n' "$*" >&2
    exit 1
}

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[model-setup] dry-run:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

ensure_dir() {
    run mkdir -p "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

parse_common_args() {
    while (($#)); do
        case "$1" in
            --dry-run)
                DRY_RUN=1
                ;;
            --force)
                FORCE_DOWNLOAD=1
                ;;
            -h|--help)
                return 2
                ;;
            *)
                die "unknown argument: $1"
                ;;
        esac
        shift
    done
}

download_file() {
    local url="$1"
    local destination="$2"
    local expected_md5="${3:-}"
    local partial="${destination}.partial"

    ensure_dir "$(dirname -- "${destination}")"
    if [[ -f "${destination}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
        if [[ -z "${expected_md5}" ]] || printf '%s  %s\n' "${expected_md5}" "${destination}" | md5sum --check --status; then
            log "already present: ${destination#"$REPO_ROOT/"}"
            return 0
        fi
        die "checksum mismatch for existing file: ${destination}"
    fi

    require_command curl
    run curl --fail --location --retry 3 --continue-at - --output "${partial}" "${url}"
    if [[ "${DRY_RUN}" != "1" && -n "${expected_md5}" ]]; then
        printf '%s  %s\n' "${expected_md5}" "${partial}" | md5sum --check --status \
            || die "checksum mismatch after downloading ${url}"
    fi
    run mv -f "${partial}" "${destination}"
}

link_file() {
    local target="$1"
    local link_path="$2"

    ensure_dir "$(dirname -- "${link_path}")"
    if [[ -L "${link_path}" ]]; then
        [[ "$(readlink -f -- "${link_path}")" == "$(readlink -f -- "${target}")" ]] \
            || die "refusing to replace unrelated symlink: ${link_path}"
        return 0
    fi
    [[ ! -e "${link_path}" ]] || die "refusing to replace existing path: ${link_path}"
    run ln -s "${target}" "${link_path}"
}

link_external_directory() {
    local target="$1"
    local link_path="$2"

    ensure_dir "${target}"
    if [[ -L "${link_path}" ]]; then
        [[ "$(readlink -f -- "${link_path}")" == "$(readlink -f -- "${target}")" ]] \
            || die "refusing to replace unrelated symlink: ${link_path}"
        return 0
    fi
    if [[ -e "${link_path}" ]]; then
        die "${link_path} already exists. Move its files into ${target}, remove the empty directory, and rerun."
    fi
    run ln -s "${target}" "${link_path}"
}

resolve_da3_streaming_dir() {
    local candidate="${DA3_STREAMING_PATH:-${REPO_ROOT}/../Depth-Anything-3}"
    if [[ -f "${candidate}/da3_streaming.py" ]]; then
        printf '%s\n' "${candidate}"
    elif [[ -f "${candidate}/da3_streaming/da3_streaming.py" ]]; then
        printf '%s\n' "${candidate}/da3_streaming"
    else
        die "DA3-Streaming checkout not found; set DA3_STREAMING_PATH"
    fi
}

resolve_external_repo() {
    local configured="$1"
    local marker="$2"
    local variable_name="$3"
    [[ -n "${configured}" ]] || die "${variable_name} must point to the pinned upstream checkout"
    [[ -f "${configured}/${marker}" ]] || die "${configured} does not contain ${marker}"
    printf '%s\n' "${configured}"
}

hf_download() {
    local repo_id="$1"
    local destination="$2"
    local filename="${3:-}"
    local revision="${4:-}"
    local args=(
        "${SETUP_DIR}/_hf_download.py"
        --repo-id "${repo_id}"
        --local-dir "${destination}"
    )
    [[ -z "${filename}" ]] || args+=(--filename "${filename}")
    [[ -z "${revision}" ]] || args+=(--revision "${revision}")
    [[ "${FORCE_DOWNLOAD}" != "1" ]] || args+=(--force)

    require_command uv
    run uv run python "${args[@]}"
}
