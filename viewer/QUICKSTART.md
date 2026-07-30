# Viewer quick start

## Prerequisites

- Python 3.12
- Node.js 18 or newer
- npm
- a populated Phase 1 SQLite catalog

GPU access is not needed to run API, schema, or frontend unit tests. A working
GPU is required for meaningful Giro3D visual/picking validation and the GTX
1070 performance gates; software rendering is not an acceptance substitute.

## Install

```bash
cd viewer
python -m venv venv
venv/bin/pip install -r requirements.txt
cd frontend
npm ci
```

## Start

From the repository root, the live five-artifact catalog can be viewed with
one command:

```bash
./viewer/start.sh
```

Open `http://127.0.0.1:5173`. API health is available at
`http://127.0.0.1:8000/health`.

For an optimized demo build, or a fully offline session:

```bash
./viewer/start.sh --production
./viewer/start.sh --production --no-basemap
```

The launcher defaults `MAPPER_CATALOG_PATH` to
`/home/ape/mapper_output/phase1_fresh/catalog.sqlite3`. Override that
environment variable to view another catalog. Press Ctrl+C once to stop both
the API and frontend.

To use a different basemap:

```bash
VITE_BASEMAP_URL='https://example.test/{z}/{x}/{y}.png' \
VITE_BASEMAP_ATTRIBUTION='Example tiles' \
./viewer/start.sh
```

To run without external tiles:

```bash
./viewer/start.sh --no-basemap
```

## Expected workflow

1. Pan or zoom the overview. Catalog footprints and aligned markers update
   after a short debounce.
2. Choose an aligned marker or a separately listed unaligned local scene.
3. Enter detail to stream COPC progressively, toggle trajectory and point
   styles, and inspect source provenance.
4. Return to overview; its prior camera is restored and detail requests and GPU
   resources are disposed.
5. Select another site without reloading the page.

Unaligned packages never receive an inferred geographic placement. Their detail
view includes the stored rejection reason.

## Checks

From the repository root:

```bash
pytest
cd viewer/frontend
npm test
npm run build
```

See `frontend/README.md` for the frontend contract and
`../docs/phase2_viewer_integration_plan.md` for acceptance requirements.
