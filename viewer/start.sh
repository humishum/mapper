#!/usr/bin/env bash
set -euo pipefail

VIEWER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${VIEWER_DIR}/.." && pwd)"
FRONTEND_DIR="${VIEWER_DIR}/frontend"
MODE="dev"

usage() {
    cat <<'EOF'
Usage: ./viewer/start.sh [--production|--dev] [--no-basemap]

Starts the Mapper catalog API and Giro3D viewer together. Press Ctrl+C to
stop both services.

Options:
  --production   Build and serve the optimized frontend (recommended for demos)
  --dev          Run the Vite development server (default)
  --no-basemap   Disable external basemap tile requests
  -h, --help     Show this help

Environment overrides:
  MAPPER_CATALOG_PATH  SQLite catalog path
  MAPPER_HOST          Backend bind address (default: 127.0.0.1)
  PORT                 Backend port (default: 8000)
  VITE_HOST            Frontend bind address (default: 127.0.0.1)
  VITE_PORT            Frontend port (default: 5173)
EOF
}

while (($#)); do
    case "$1" in
        --production)
            MODE="production"
            ;;
        --dev)
            MODE="dev"
            ;;
        --no-basemap)
            export VITE_BASEMAP_ENABLED=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

export MAPPER_CATALOG_PATH="${MAPPER_CATALOG_PATH:-/home/ape/mapper_output/phase1_fresh/catalog.sqlite3}"
export MAPPER_HOST="${MAPPER_HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export VITE_HOST="${VITE_HOST:-127.0.0.1}"
export VITE_PORT="${VITE_PORT:-5173}"
export VITE_API_PROXY_TARGET="${VITE_API_PROXY_TARGET:-http://127.0.0.1:${PORT}}"

if [[ ! -r "${MAPPER_CATALOG_PATH}" ]]; then
    echo "Catalog is not readable: ${MAPPER_CATALOG_PATH}" >&2
    exit 1
fi

PYTHON=""
SYSTEM_PYTHON="$(command -v python || true)"
for candidate in \
    "${VIEWER_DIR}/venv/bin/python" \
    "${REPO_DIR}/.venv/bin/python" \
    "${SYSTEM_PYTHON}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]] && "${candidate}" -c \
        'import fastapi, pyarrow, pyproj, uvicorn' >/dev/null 2>&1; then
        PYTHON="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON}" ]]; then
    echo "Viewer backend dependencies are missing." >&2
    echo "Install them with: python -m venv viewer/venv && viewer/venv/bin/pip install -r viewer/requirements.txt" >&2
    exit 1
fi

if ! command -v npm >/dev/null 2>&1 || [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    echo "Viewer frontend dependencies are missing." >&2
    echo "Install them with: npm --prefix viewer/frontend ci" >&2
    exit 1
fi

cleanup() {
    trap - INT TERM EXIT
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill "${FRONTEND_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

cd "${REPO_DIR}"
if [[ "${MODE}" == "production" ]]; then
    export MAPPER_RELOAD="${MAPPER_RELOAD:-false}"
else
    export MAPPER_RELOAD="${MAPPER_RELOAD:-true}"
fi
"${PYTHON}" -m viewer.backend.server &
BACKEND_PID=$!

cd "${FRONTEND_DIR}"
if [[ "${MODE}" == "production" ]]; then
    npm run build
    npm run preview -- --host "${VITE_HOST}" --port "${VITE_PORT}" --strictPort &
else
    npm run dev -- --host "${VITE_HOST}" --port "${VITE_PORT}" --strictPort &
fi
FRONTEND_PID=$!

echo "Mapper viewer started"
echo "Mode:     ${MODE}"
echo "Frontend: http://${VITE_HOST}:${VITE_PORT}"
echo "Backend:  http://${MAPPER_HOST}:${PORT}"
echo "Catalog:  ${MAPPER_CATALOG_PATH}"
echo "Press Ctrl+C to stop both processes."

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
