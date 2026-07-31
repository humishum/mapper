# Phase 2 — Viewer Integration and Existing-Data Onboarding

Status: **complete — accepted 2026-07-26**

Parent plan: [`visualization_platform_plan.md`](visualization_platform_plan.md)

## Current data status

- COPC migration is intentionally partial:
  - Phase 0: 146.9M-point `mp7` plus voxel variants.
  - Phase 1 legacy acceptance: `kings_canyon_2` and `tahoe_ridge_2`.
  - Phase 1 fresh packages: one MUSt3R and one VGGT run.
  - Remaining legacy inventory: 30 `aligned_pointcloud.ply` files,
    approximately 14.9 GB.
- Current storage:
  - `/home/ape/mapper_output/phase0_spike/artifacts/`: Phase 0 standalone
    benchmark COPCs.
  - `/home/ape/mapper_output/phase1_acceptance/`: Phase 1 standalone converted
    COPCs.
  - `/home/ape/mapper_output/phase1_fresh/`: Phase 1 complete packages.
  - `/home/ape/mapper_output/phase1_fresh/catalog.sqlite3`: current populated
    package catalog.
- At Phase 2 kickoff, the catalog contained only the two fresh packages. Both
  were unaligned because GPS validation failed, so they could be shown as
  local scenes but not geographic footprints.
- Standalone COPCs are not automatically discoverable by the Phase 2 viewer.
  They must first be wrapped in valid packages and registered.
- Phase 2 will onboard three existing fixtures without rerunning
  reconstruction:
  - `mp7` as the large performance package.
  - MUSt3R `kings_canyon_2`.
  - VGGT `tahoe_ridge_2`.
- Telemetry-only georeferencing will be attempted for Kings Canyon and Tahoe
  Ridge using saved poses and original video telemetry. Failed validation
  leaves the package explicitly unaligned.
- Phase 2 completion state:
  - the three fixtures are contained below
    `/home/ape/mapper_output/phase2_packages/`;
  - `/home/ape/mapper_output/phase1_fresh/catalog.sqlite3` contains the two
    fresh packages plus the three onboarded fixtures; and
  - the pre-registration two-artifact catalog is preserved at
    `/home/ape/mapper_output/phase1_fresh/catalog.sqlite3.pre-phase2-20260726`.

## Architecture and interfaces

- [x] Migrate the frontend shell to TypeScript/TSX while retaining React 18
  and Vite 5.
- [x] Pin Giro3D 2.0.3, Three.js 0.180.0, OpenLayers 10.8.0, and hyparquet
  1.26.2.
- [x] Keep Phase 1 API paths unchanged; formalize their response models and
  generate frontend types from OpenAPI.
- [x] Validate manifests and sidecars against the existing package schemas in
  the browser.
- [x] Define renderer-neutral contracts for scene mode, catalog artifacts,
  layers, camera state, loading, picking, disposal, and performance.
- [x] Use one React application with two internal renderer modes:
  - ECEF globe overview with basemap and catalog footprints.
  - Origin-centered local detail scene with COPC, trajectory, and inspection.
- [x] Cross-fade between modes without page navigation and restore the previous
  overview camera.
- [x] Record why this was chosen in ADR-005:
  - Predictable close-range precision.
  - Clean network/GPU teardown when leaving a site.
  - No custom rebasing of Giro3D's globe internals.
  - Accepted cost: no basemap underneath close-detail geometry and no
    literally continuous globe-to-point flight.
  - A persistent camera-relative ECEF scene remains a future option if Giro3D
    gains a proven relative-to-eye or high/low-coordinate path.
- [x] Preserve authoritative float64 ECEF state. Place aligned detail artifacts
  with:

  ```text
  M_render = T(-active_origin_ecef) × M_artifact_to_ecef
  ```

- [x] Allow two same-site detailed COPCs under one shared two-million-point
  budget. Artifacts more than 10 km from the active origin require an overview
  transition.
- [x] Open unaligned artifacts in local mode with no geographic marker and a
  visible rejection reason.
- [x] Store versioned URL state for mode, active artifacts, selected source,
  camera/target, and layer display settings.

## Implementation checklist

### Existing-data onboarding

- [x] Add a `package-existing-copc` publisher command that accepts an existing
  COPC, legacy reconstruction directory, optional source video, output package
  root, and catalog path.
- [x] Validate hierarchy compatibility, dimensions, checksums, source
  distribution, and bounds before adopting an existing COPC.
- [x] Hardlink compatible COPCs into the package when possible, copying when
  necessary; packages must never reference files outside their root.
- [x] Republish incompatible COPCs through the pinned Phase 1 writer without
  rerunning reconstruction.
- [x] When missing, derive `ContributorCount=1` and record that migration
  default explicitly; never invent confidence values.
- [x] Convert saved legacy `poses.npz` and window metadata into canonical
  `poses.parquet` and `sources.json`.
- [x] If an original video is found, extract telemetry and run the Phase 1
  GPS/pose alignment pipeline.
- [x] Register successful alignment with an ECEF transform and footprint.
  Register failed or missing telemetry as unaligned with a specific rejection
  reason.
- [x] Package and register:
  - Canonical 146.9M-point `mp7`.
  - `must3r-kings-canyon-2-sources.copc.laz`.
  - `vggt-tahoe-ridge-2-sources.copc.laz`.
- [x] Do not convert the remaining 30 legacy PLYs during Phase 2.

### Overview and catalog

- [x] Render a Giro3D ECEF globe with catalog footprints and lightweight
  markers.
- [x] Use configurable OSM-compatible tiles through `VITE_BASEMAP_URL`,
  `VITE_BASEMAP_ATTRIBUTION`, and `VITE_BASEMAP_ENABLED`.
- [x] Query visible catalog bounds after a 150 ms debounce and split
  antimeridian-crossing queries.
- [x] List unaligned packages separately as local scenes.
- [x] Fetch manifests and artifact details only after selection.
- [x] Never load COPC nodes while an artifact is merely visible in overview.

### Detail viewer

- [x] Create the local renderer around the selected artifact's ECEF anchor.
- [x] Apply the same relative transform to COPC geometry and trajectories.
- [x] Start from Phase 0 defaults:
  - Two-million-point global budget.
  - SSE/subdivision threshold `1`.
  - Decimation stride `2`.
  - CPU and GPU geometry pools each capped at 256 MB.
- [x] Provide visibility, opacity, point size, loading state, and
  RGB/elevation/source/confidence coloring.
- [x] Disable attribute modes not present in the package.
- [x] Load pose columns from `poses.parquet` through HTTP ranges using
  hyparquet.
- [x] Dispose removed entities, workers, cached attributes, and geometry when
  leaving detail.
- [x] Keep source kinds generic so future SLAM submaps and keyframe groups work
  without viewer redesign.

### Picking and inspection

- [x] Use Giro3D picking for artifact, node, point index, and local coordinate.
- [x] Fetch only the picked node's provenance attributes through
  `COPCSource.getNodeData`.
- [x] Resolve `PointSourceId` through `sources.json`.
- [x] Show capture/run/source, frame range, coordinates, confidence,
  contributor count, alignment status, and RMSE.
- [x] Use a bounded, abortable node-attribute cache.
- [x] Keep point parsing entirely in the browser.

### Legacy viewer removal

- [x] Remove deck.gl, MapLibre, axios, hex decoding, `/api/locations`, and the
  100k-point cap.
- [x] Remove the legacy PLY/hex backend and obsolete dependencies and
  configuration.
- [x] Delete the Dash folder visualizer only after source coloring passes
  acceptance.
- [x] Retain `scripts/ply_viewer/` as a documented local debugging tool.
- [x] Update viewer startup scripts to launch the Phase 1 FastAPI API and Vite
  frontend.

## Validation and completion gates

- [x] Use the fresh 200k VGGT package for fast tests, fresh 11.2M MUSt3R
  package for unaligned tests, and canonical `mp7` for large benchmarks.
- [x] Build two deterministic synthetic aligned packages hundreds of
  kilometres apart for repeatable multi-site tests.
- [x] Treat real Kings Canyon and Tahoe georeferencing as field validation;
  failed GPS does not block Phase 2 if synthetic multi-site tests pass.
- [x] Test transforms, URL state, API and schema failures, antimeridian
  queries, budget allocation, and unsupported attributes.
- [x] Test overview, detail, and local transitions; catalog errors; warnings;
  controls; source inspection; and trajectory agreement.
- [x] Verify overview makes zero detailed COPC requests.
- [x] Verify site A to overview to site B works without a page reload.
- [x] Verify inactive-site requests are aborted and cannot repopulate disposed
  state.
- [x] Verify unaligned artifacts never receive inferred global placement.
- [x] Add Playwright/CDP cold/warm, orbit, dive, comparison, site-transition,
  and throttled-cancellation benchmarks.
- [x] Pass:
  - First coarse `mp7` geometry under 1 second warm and 2.5 seconds cold.
  - Moving-view p95 under 20 ms.
  - No repeated steady-state frames or long tasks over 50 ms.
  - At most two million visible points.
  - CPU and GPU geometry pools each at or below 256 MB.
  - Zero detailed requests for an inactive distant site.
  - Stable geometry, trajectory, and picking after transitions.

## Documentation consistency

- [x] Mark Phase 2 in progress at kickoff and complete only after the gates
  above pass.
- [x] Update D9 from “one scene” to “one application and viewport with
  overview/detail renderer modes.”
- [x] Correct aligned ENU versus unaligned native-local wording.
- [x] Correct provenance terminology to `PointSourceId`, `ContributorCount`,
  and optional `Confidence`.
- [x] Clarify metric-only voxel consolidation.
- [x] Replace historical PDAL writer language with the pinned Rust writer and
  hierarchy-flattening rule.
- [x] Correct the documented publisher path to `src/publisher`.
- [x] Update viewer README, quick start, and architecture diagrams.
- [x] Mark Phase 1 ADRs accepted and finalize ADR-005 from Phase 2 evidence.
- [x] Keep terrain, offline basemaps, temporal comparison, hosting/auth, bulk
  legacy migration, and SLAM reconstruction outside Phase 2.

## GPU requirement

GPU access is not required for COPC packaging, catalog registration, API work,
TypeScript migration, or most automated tests. A functioning GPU is required
for meaningful Giro3D visual validation, picking validation, geometry-memory
measurements, and the Phase 2 performance gates. Software rendering is not an
acceptable substitute for the GTX 1070 acceptance measurements.
