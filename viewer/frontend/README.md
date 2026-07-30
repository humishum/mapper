# Mapper Phase 2 viewer

The viewer is one React 18 application with two disposable Giro3D renderer
modes:

- **Overview** uses an EPSG:4978 (ECEF) Giro3D `Globe` with a tiled
  OpenLayers-backed basemap, catalog footprints, and lightweight markers.
  Overview never creates a `COPCSource`, so merely seeing a site cannot request
  detailed geometry.
- **Detail** creates a separate origin-centred Giro3D instance for one site.
  It streams one or two COPCs, camera trajectories, and point provenance.
  Returning to overview aborts application requests and disposes the complete
  instance, sources, workers, cached node attributes, geometries, and controls.

This renderer switch keeps close-range coordinates precise and makes inactive
site teardown explicit. The application cross-fades between modes and preserves
the previous overview camera. The accepted tradeoff is that detail has no
basemap underneath it and the transition is not a literal globe-to-point
flight.

## Quick start

Start the canonical Phase 1 API from the repository root:

```bash
uv run uvicorn viewer.backend.api:create_app --factory --host 127.0.0.1 --port 8000
```

Then start Vite:

```bash
cd viewer/frontend
npm ci
npm run dev
```

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`. A different API
origin can be supplied at build time with `VITE_API_ROOT`.

Basemap configuration:

```bash
VITE_BASEMAP_ENABLED=true
VITE_BASEMAP_URL=https://tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_BASEMAP_ATTRIBUTION="© OpenStreetMap contributors"
```

`VITE_BASEMAP_ENABLED=false` produces a plain shaded globe and makes no tile
request. The URL must be compatible with OSM `{z}/{x}/{y}` placeholders.

## Architecture

Renderer-independent types live in `src/types/contracts.ts`. They cover scene
mode, catalog records, layers, float64 camera and transform state, loading,
picking, disposal, and performance. React coordinates those contracts; Giro3D
is isolated under `src/renderers`.

The data flow is:

1. The overview camera produces visible WGS84 bounds.
2. `useCatalog` waits 150 ms, aborts the superseded query, splits an
   antimeridian crossing into two valid backend bounding boxes, and merges the
   catalog records by opaque artifact ID.
3. Unaligned artifacts are queried separately and shown as local-only scenes.
4. Selection lazily fetches artifact detail and `manifest.json`. Both retain
   the canonical `/api/v1` paths.
5. The manifest and its selected metrics, sources, and tabular sidecars are
   validated in-browser against vendored canonical Phase 1 JSON Schemas before
   they are used.
6. For aligned artifacts, all geometry and trajectories use the authoritative
   row-major float64 transform:

   ```text
   M_render = T(-active_origin_ecef) × M_artifact_to_ecef
   ```

   Unaligned artifacts use their native local frame, never an inferred global
   position. Their alignment rejection reason remains visible.
7. A second same-site COPC is accepted only when its ECEF anchor is within
   10 km. More distant artifacts force a return to overview.

The versioned URL (`v=1`) stores mode, up to two active artifacts, selected
source, overview/detail cameras, and per-layer visibility, opacity, point size,
and color mode. Unsupported attribute modes remain visible but disabled.

## Detail resource policy

- Global visible-point budget: 2,000,000, divided across visible COPCs.
- Giro3D subdivision threshold: `1`.
- Source and display decimation stride: `2`.
- CPU and GPU geometry limits: 256 MiB each. Giro3D's global COPC cache is
  configured to a 256 MiB byte capacity before any renderer is created.
  Runtime memory is sampled from the Giro3D instance and exposed in the
  renderer-neutral performance state. A pressure guard raises SSE and clears
  the scene if either measured pool exceeds its limit.
- Unused node cleanup delay: `0` when leaving visibility.
- Picked-node attributes: 32 MiB / 24-entry LRU bound.

Point picking first uses Giro3D to resolve artifact, node, point index, and
local coordinate. Only that node's `PointSourceId`/`SourceIndex`,
`Confidence`, and `ContributorCount` buffers are requested with
`COPCSource.getNodeData`. The source ID is then resolved through the generic
catalog sources endpoint; `window`, `submap`, `keyframe_group`, `batch`, and
`capture` kinds use one UI.

`poses.parquet` is read by hyparquet from the immutable asset endpoint. It uses
HTTP `HEAD`/byte ranges, column projection, row-range support, and an abort
signal; point parsing stays in the browser. COPC range reads use the same
abort-aware fetch path, so superseded detail scenes cannot keep issuing range
requests.

## API types and schemas

The backend OpenAPI document is checked in at `schemas/openapi.v1.json`.
Regenerate the frontend wire types reproducibly after an API contract change:

```bash
npm run generate:api
```

This runs `openapi-typescript` against the checked-in document and writes
`src/api/openapi.ts`. The manifest, sources, and tabular schemas are vendored
under `src/schemas` so browser validation is deterministic and works offline.
Refresh those copies whenever the canonical schemas change.

## Verification

```bash
npm run generate:api
npm test
npm run lint
npm run build
```

Unit/component coverage includes row-major ECEF transforms, antimeridian
queries, URL recovery/round-trip, global point allocation, bounded attribute
cache eviction, schema/API failures, generic sources, absence of COPC requests
during overview queries, and unsupported color controls.

Meaningful Giro3D picking, WebGL disposal, trajectory agreement, and cold/warm
performance validation require the repository browser benchmark harness and a
real GPU. Software-rendered CI is intentionally not treated as evidence for the
GTX 1070 performance gates. The harness exercises seven scenarios, including a
multi-site transition that waits for geometry and trajectories and performs
point picking plus source inspection after the transition.
