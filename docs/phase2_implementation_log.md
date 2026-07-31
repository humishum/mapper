# Phase 2 implementation log

Status: **complete**
Started: 2026-07-26
Completed: 2026-07-26
Plan: [`phase2_viewer_integration_plan.md`](phase2_viewer_integration_plan.md)

This is the evidence log for completed Phase 2 implementation, real-package
onboarding, persistent catalog registration, and GTX 1070 acceptance.

## Work tracks

| Track | Scope | Evidence state |
| --- | --- | --- |
| Existing COPC onboarding | adoption/republication, legacy pose/source conversion, optional telemetry alignment, package registration | Implemented; all three real fixtures passed the production CLI into disposable `/tmp` package/catalogs |
| Catalog API | typed `/api/v1` responses, OpenAPI, antimeridian queries, range data plane, legacy backend removal | Implemented and verified |
| Viewer frontend | TypeScript shell, overview/detail renderers, lifecycle, URL state, layers, picking/inspection | Implemented; unit, lint, and production-build checks pass |
| Existing fixture registration | `mp7`, Kings Canyon MUSt3R, Tahoe Ridge VGGT | Persistently packaged, registered, rehashed, and served through the catalog API |
| Synthetic multi-site fixture | two distant aligned packages plus a same-origin comparison package | Implemented and verified (~670 km between sites) |
| Browser benchmark harness | cold/warm, orbit, dive, compare, transition, cancellation | Seven-scenario functional and GTX 1070 acceptance runs passed |
| GPU acceptance | visual placement, picking, memory, frame time on GTX 1070 | Passed; standard suite 7/7 and canonical mp7 1/1 |

## Decisions recorded during implementation

- D9 is one React application and viewport with separately owned ECEF overview
  and origin-centered detail renderer modes.
- Aligned artifacts use metre ENU and an authoritative float64
  artifact-local-to-ECEF transform. Unaligned artifacts retain native-local
  coordinates and never receive a geographic marker.
- Detail placement uses
  `T(-active_origin_ecef) × M_artifact_to_ecef` for both geometry and
  trajectories.
- Provenance names are LAS `PointSourceId`, required
  `ContributorCount`, and optional meaningful `Confidence`.
- Voxel consolidation is permitted only when source units are known to be
  metric.
- Production COPC publication uses bounded LAZ staging, pinned Rust
  `copc-converter` v0.11.0, validation, and terminal hierarchy flattening.

These decisions are captured in
[`adrs/002-coordinate-frames-and-ecef-placement.md`](adrs/002-coordinate-frames-and-ecef-placement.md),
[`adrs/003-copc-and-artifact-contract.md`](adrs/003-copc-and-artifact-contract.md),
[`adrs/004-direct-copc-serving.md`](adrs/004-direct-copc-serving.md),
[`adrs/005-two-mode-giro3d-viewer.md`](adrs/005-two-mode-giro3d-viewer.md),
and
[`adrs/007-provenance-consolidation.md`](adrs/007-provenance-consolidation.md).

## Fixture inventory confirmed

The following inputs exist and were read during implementation:

| Role | Input |
| --- | --- |
| Large benchmark | `/home/ape/mapper_output/phase0_spike/artifacts/mp7-full.copc.laz` (955,111,955 bytes) |
| MUSt3R existing COPC | `/home/ape/mapper_output/phase1_acceptance/must3r-kings-canyon-2-sources.copc.laz` (150,401,162 bytes) |
| VGGT existing COPC | `/home/ape/mapper_output/phase1_acceptance/vggt-tahoe-ridge-2-sources.copc.laz` (125,608,650 bytes) |
| Kings Canyon legacy run | `/home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/outputs/kings_canyon_2` |
| Tahoe legacy run | `/home/ape/mapper_output/vggt_trail_test_20260111_013711/outputs/tahoe_ridge_2` |
| Kings Canyon video | `/home/ape/Documents/MapperGoProVids/kings_canyon_2.MP4` |
| Tahoe video | `/home/ape/Documents/MapperGoProVids/tahoe_ridge_2.MP4` |

The pre-Phase-2 catalog contains the two fresh packages only: an 11,207,020
point unaligned MUSt3R artifact and a 200,228 point unaligned VGGT artifact.

A disposable `/tmp/mapper-phase2-multisite-browser-v2` build also proved the
deterministic fixture generator end to end. It created and registered two
aligned 200,228-point site packages approximately 670 km apart plus a distinct
same-origin comparison package. All three include canonical pose and source
sidecars. Their manifests declare the COPC's actual RGB, `PointSourceId`,
`ContributorCount`, and `Confidence` capabilities; 15 source records sum
exactly to 200,228 points. This is test evidence only and does not modify the
user catalog.

### Real Kings Canyon onboarding audit

The production CLI was exercised end to end with the real MUSt3R fixture,
legacy window tree, and original GoPro video. All writes stayed below the
disposable directory `/tmp/mapper-phase2-kings.GZYV90`; source data and the
populated user catalog were opened read-only.

```bash
.venv/bin/python -m src.publisher.cli package-existing-copc \
  /home/ape/mapper_output/phase1_acceptance/must3r-kings-canyon-2-sources.copc.laz \
  /home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/outputs/kings_canyon_2 \
  /tmp/mapper-phase2-kings.GZYV90/package \
  --catalog /tmp/mapper-phase2-kings.GZYV90/catalog.sqlite3 \
  --source-video /home/ape/Documents/MapperGoProVids/kings_canyon_2.MP4 \
  --memory-limit 512M --threads 4 \
  --temp-dir /tmp/mapper-phase2-kings.GZYV90
```

The command completed in 6.47 seconds. GoPro extraction retained 14,533 IMU
samples, while no GPS samples passed the Phase 1 coordinate/fix/precision
gates. The result was therefore correctly registered as unaligned with the
specific reason `gps_telemetry_unavailable`, unknown native-local units, and no
origin, ECEF transform, CRS, or footprint. It still published all five
available representations: points, 719 canonical poses, IMU telemetry, 18
window source records, and migration metrics.

Independent validation of the finished package and disposable catalog proved:

- 25,667,473 header, hierarchy, source-distribution, manifest, and catalog
  points agree;
- the COPC has 1,241 root-reachable nodes and one hierarchy page;
- all 18 source counts sum exactly to 25,667,473;
- source and packaged geometry are both 150,401,162 bytes with SHA-256
  `e7f86000eae1620c7d05094f63f9ea8cdd8238fd086b6669165955e7a9d3a067`;
- hardlinking was unavailable in the managed execution environment, so the
  intended copy fallback was used; the result is not a symlink and every
  artifact resolves inside the package root;
- canonical pose columns are float64 with an int64 `frame_index`; canonical
  IMU columns are float64; and
- the catalog contains one matching unaligned artifact, five matching
  representation records, and no footprint row.

No implementation defect was exposed by this real-data run.

### Real Tahoe Ridge onboarding audit

The same production CLI was then exercised with the real VGGT Tahoe Ridge
fixture, all 340 legacy windows, and the original GoPro video. All writes
stayed below `/tmp/mapper-phase2-tahoe.tZqubV`; external source data and the
user catalog remained read-only.

```bash
.venv/bin/python -m src.publisher.cli package-existing-copc \
  /home/ape/mapper_output/phase1_acceptance/vggt-tahoe-ridge-2-sources.copc.laz \
  /home/ape/mapper_output/vggt_trail_test_20260111_013711/outputs/tahoe_ridge_2 \
  /tmp/mapper-phase2-tahoe.tZqubV/package \
  --catalog /tmp/mapper-phase2-tahoe.tZqubV/catalog.sqlite3 \
  --source-video /home/ape/Documents/MapperGoProVids/tahoe_ridge_2.MP4 \
  --memory-limit 512M --threads 4 \
  --temp-dir /tmp/mapper-phase2-tahoe.tZqubV
```

The command completed in 6.13 seconds. Telemetry extraction retained 4,931
quality GPS samples and 54,828 IMU samples, so the Phase 1 GPS/pose aligner ran
against all 2,716 recovered poses. The metric-scale-locked fit was explicitly
rejected as `alignment_rmse_exceeds_threshold`: estimated RMSE was 103.987 m
against the 8.0 m gate, median residual was 93.637 m, and 2,715
correspondences produced a 0.9845 robust inlier fraction. The approximately
482.22 m GPS track and 0.2562 clock-peak quality were retained as independent
diagnostics. The package correctly remained metre native-local with no origin,
ECEF transform, CRS, or footprint; it did not publish a plausible but invalid
global placement.

Independent validation of the package and disposable catalog proved:

- 17,000,000 header, hierarchy, source-distribution, manifest, and catalog
  points agree;
- the COPC has 969 root-reachable nodes, one hierarchy page, and 340 exact
  source IDs;
- all 340 `window` records sum exactly to 17,000,000 points;
- source and packaged geometry are both 125,608,650 bytes with SHA-256
  `f93dc9afd7833706f2609c1d064d1369b7e15d6b334073bea85cd4c6d67fe62d`;
- the managed environment again used the copy fallback; the geometry is not a
  symlink and every artifact resolves inside the package root;
- the six representations are points, 2,716 canonical poses, GPS telemetry,
  IMU telemetry, sources, and metrics; and
- the catalog contains one matching unaligned metre artifact, six
  representation rows, 340 source rows, and no footprint.

No implementation defect was exposed by the Tahoe Ridge real-data run.

### Real mp7 pinned-republish onboarding audit

The production CLI was finally exercised with the canonical 146.9 million
point mp7 benchmark and its 43-window legacy run. This fixture deliberately
lacks `ContributorCount`, so it covers the pinned converter path rather than
the direct-adoption path. All writes stayed below
`/tmp/mapper-phase2-mp7.NNlC2W`; the input COPC, legacy tree, and user catalog
remained read-only. No source video exists for this legacy run.

```bash
.venv/bin/python -m src.publisher.cli package-existing-copc \
  /home/ape/mapper_output/phase0_spike/artifacts/mp7-full.copc.laz \
  /home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/outputs/mp7 \
  /tmp/mapper-phase2-mp7.NNlC2W/package \
  --catalog /tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3 \
  --memory-limit 1G --threads 4 \
  --temp-dir /tmp/mapper-phase2-mp7.NNlC2W
```

The bounded republish completed in 353.73 seconds; the largest observed
disposable directory footprint was 5.9 GB. It produced canonical point format
7 with 0.001 coordinate scale, zero offset, and one hierarchy page. With no
video or telemetry, the package correctly remained unaligned with
`source_video_missing`, unknown local units, and no origin, ECEF transform,
CRS, or footprint.

Independent full-file scans and package/catalog validation proved:

- input and output both contain exactly 146,911,634 points; the republished
  COPC has 6,448 root-reachable nodes through depth 11 and no invalid or
  unreachable nodes;
- every one of the 146,911,634 output points has the explicit uint16 migration
  default `ContributorCount=1` (minimum 1, maximum 1, zero exceptions), while
  `Confidence` is absent rather than invented;
- all 43 output `PointSourceId` counts exactly equal both the input COPC
  distribution and `sources.json`, whose counts sum to 146,911,634;
- input SHA-256
  `be3a17c0be2fb3b1f51e1e2273de5dd4009862813ce633a389a360d33ddbf23f`
  was identical during onboarding and the post-run independent read, while
  the expected republished output is 957,383,311 bytes with SHA-256
  `8b6c40eb64c13309c1c2e6eedbd7a749db8f93002f9294215e5febeacd4391ce`;
- input bounds were preserved to the declared 0.001 canonical quantization;
  manifest validation independently rechecked every declared size and
  checksum;
- canonical poses contain 1,694 rows covering every frame index from 0
  through 1,693, and the four representations are points, poses, sources, and
  metrics;
- the manifest explicitly records `missing_contributor_count`,
  `ContributorCount=1 (legacy migration default)`, `confidence_invented:
  false`, and pinned converter version `0.11.0`;
- there are no symlinks or paths resolving outside the package root; and
- the disposable catalog contains one matching unaligned artifact, four
  representation rows, 43 source rows, and the exact point count.

No implementation defect was exposed by the mp7 real-data run.

## Completion evidence checklist

Before changing this log, the Phase 2 plan, or ADR-005 to complete/accepted,
record direct evidence for:

- all Python and frontend unit/contract tests;
- deterministic OpenAPI/type generation and production frontend build;
- valid registration and containment of the three adopted real fixtures;
- synthetic multi-site overview → A → overview → B without reload;
- zero overview and inactive-site detailed COPC requests;
- abort/disposal race resistance;
- transform, trajectory, URL, schema-failure, attribute-capability, and picking
  checks;
- cold/warm, orbit, dive, comparison, transition, and throttled cancellation
  browser reports; and
- GTX 1070 frame-time, first-geometry, visible-point, CPU/GPU pool, long-task,
  and post-transition stability gates.

## Automated evidence recorded

On 2026-07-26:

- `.venv/bin/pytest -q`: **66 passed, 1 skipped**.
- Ruff over the new publisher, backend, fixture, and focused test surfaces:
  **passed**.
- `npm run generate:api`: regenerated the checked-in TypeScript wire types.
- `npm test`: **19 passed across 8 frontend test files**.
- `npm run lint`: **passed**.
- `npm run build`: **passed**. Vite reports a non-blocking 2.26 MB bundle-size
  warning; this is not an acceptance-performance result.
- `npm run benchmark:validate`: **4 harness-contract checks passed**.
- `npm run benchmark:list`: **7 required Playwright scenarios discovered**.
- The complete Playwright suite against
  `/tmp/mapper-phase2-multisite-browser-v2` passed **7/7 in 30.3 seconds**
  using the explicit `functional` evidence mode. All seven reports validate
  against `benchmarks/viewer/report.schema.json`.
- Functional browser evidence proved zero overview and inactive-site detailed
  requests, one-navigation site transitions, trajectory readiness, same-origin
  comparison, Source-control selection, resolved provenance inspection,
  renderer disposal, throttled aborts, and absence of stale site-A commits.
  [`../benchmarks/viewer/FUNCTIONAL_EVIDENCE.md`](../benchmarks/viewer/FUNCTIONAL_EVIDENCE.md)
  records the exact commands and observations.
- A separate focused functional run opened the real fresh 200,228-point VGGT
  and 11,207,020-point MUSt3R packages through a disposable catalog copy. It
  passed **1/1 in 4.7 seconds**, proved zero overview asset requests, local-only
  listing, explicit `gps_unavailable` warnings, null global placement, one
  document navigation, and the shared point budget.
  [`../benchmarks/viewer/UNALIGNED_FUNCTIONAL_EVIDENCE.md`](../benchmarks/viewer/UNALIGNED_FUNCTIONAL_EVIDENCE.md)
  records the exact read-only setup and observations.
- A dedicated canonical `mp7` functional run passed **1/1 in 6.4 seconds**
  against the disposable republished package. It proved cache-disabled and
  cache-enabled ranged COPC streaming, Source mode, cold/warm trajectory
  readiness, overview isolation, detail disposal, and the two-million-point
  budget across the 146,911,634-point artifact.
  [`../benchmarks/viewer/MP7_FUNCTIONAL_EVIDENCE.md`](../benchmarks/viewer/MP7_FUNCTIONAL_EVIDENCE.md)
  records the exact command and clearly labels the timings as SwiftShader
  diagnostics rather than GTX acceptance.
- `bash -n viewer/start.sh` and `git diff --check`: **passed**.
- `python -m viewer.backend.api.export_openapi` regenerated
  `schemas/openapi.v1.json` byte-for-byte.
- The three real COPCs passed streaming inspection:
  header/hierarchy counts agree, all nodes are root-reachable, bounds are
  finite and consistent, checksums were computed, and every
  `PointSourceId` distribution sums to the point count.
- Each real COPC distribution exactly matches its legacy window metadata:
  `mp7` 146,911,634 points/43 windows, Kings Canyon 25,667,473 points/18
  windows, and Tahoe Ridge 17,000,000 points/340 windows.
- Canonical pose recovery produced 1,694 `mp7`, 719 Kings Canyon, and 2,716
  Tahoe de-duplicated frame records.
- The real Kings Canyon CLI/package/catalog audit described above passed
  independently, including video telemetry extraction, copy fallback,
  manifest/package validation, COPC reinspection, and catalog readback.
- The real Tahoe Ridge audit also passed, exercised valid GPS plus the complete
  Phase 1 aligner, and proved that a 103.987 m rejected fit remains explicitly
  unaligned while preserving GPS/IMU sidecars and metric native-local units.
- The real mp7 audit exercised the pinned republisher under a 1 GB writer
  limit and proved explicit `ContributorCount=1`, absence of invented
  `Confidence`, exact point/source preservation, canonical poses, containment,
  checksums, and catalog registration across all 146,911,634 points.
- The final headed hardware suite used Chrome 119 and
  `ANGLE (NVIDIA Corporation, NVIDIA GeForce GTX 1070/PCIe/SSE2, OpenGL
  4.5.0)`. The deterministic standard suite passed **7/7 in 40.6 seconds**:
  576.4 ms cold, 79.5 ms warm, 16.68 ms orbit/dive p95, zero steady-state
  long tasks over 50 ms, bounded geometry, zero inactive-site requests, stable
  site transitions, and two verified aborts in the throttled supersession
  scenario. All seven JSON reports validate against the closed report schema.
- The final canonical `mp7` GTX 1070 run passed **1/1 in 16.6 seconds**:
  643.8 ms cold, 184.1 ms warm, 16.68 ms orbit/dive p95, zero steady-state
  long tasks over 50 ms, 309,925 visible points, and 4,738,688 CPU/GPU geometry
  bytes. It also proved ranged streaming, post-transition trajectory
  stability, source coloring, renderer disposal, and a resolved provenance
  pick.
- The `mp7` screenshot was inspected at original resolution. It shows
  high-contrast source regions, the trajectory, and inspection fields for
  node/index, local coordinate, Source 0, capture, run, frames,
  ContributorCount=1, missing confidence, and unaligned status. This satisfied
  the visual gate; `scripts/folder_visualizer.py` and the direct
  Dash/Plotly/Dash-Bootstrap dependencies were removed.
- A headed GTX 1070 run over the real fresh 200,228-point VGGT and
  11,207,020-point MUSt3R packages passed **1/1 in 12.5 seconds** in explicit
  `acceptance` mode. It rejects software WebGL and retained local-only
  placement, `gps_unavailable` warnings, null global placement, overview
  request isolation, and one-navigation transitions.
- [`../benchmarks/viewer/HARDWARE_ACCEPTANCE.md`](../benchmarks/viewer/HARDWARE_ACCEPTANCE.md)
  records the environment, commands, DPMS validity control, exact metrics,
  and acceptance conclusion. ADR-005 is now Accepted.

`mp7` correctly reports `missing_contributor_count` and therefore takes the
pinned republish path that writes explicit `ContributorCount=1`. Kings Canyon
and Tahoe are directly adoptable unless accepted telemetry alignment requires
their coordinates to be republished into ENU.

## Persistent promotion and catalog registration

After explicit approval on 2026-07-26, the three fully validated disposable
packages were copied without rerunning reconstruction or publication into:

| Fixture | Persistent package root | Points | Representations | Sources |
| --- | --- | ---: | ---: | ---: |
| `mp7` | `/home/ape/mapper_output/phase2_packages/mp7` | 146,911,634 | 4 | 43 |
| Kings Canyon MUSt3R | `/home/ape/mapper_output/phase2_packages/must3r-kings-canyon-2` | 25,667,473 | 5 | 18 |
| Tahoe Ridge VGGT | `/home/ape/mapper_output/phase2_packages/vggt-tahoe-ridge-2` | 17,000,000 | 6 | 340 |

The existing catalog was preserved at
`/home/ape/mapper_output/phase1_fresh/catalog.sqlite3.pre-phase2-20260726`
before registration. The backup independently reports the original two
artifacts. The live
`/home/ape/mapper_output/phase1_fresh/catalog.sqlite3` now reports five
artifacts and binds the three new IDs to the persistent roots above.

Independent `PackageValidator` passes rechecked every persistent file,
sidecar contract, size, checksum, and containment rule. No persistent package
contains a symlink, and no temporary `.incoming` directory remains. Direct
full-file SHA-256 reads matched the manifests:

- `mp7`:
  `8b6c40eb64c13309c1c2e6eedbd7a749db8f93002f9294215e5febeacd4391ce`;
- Kings Canyon:
  `e7f86000eae1620c7d05094f63f9ea8cdd8238fd086b6669165955e7a9d3a067`;
- Tahoe Ridge:
  `f93dc9afd7833706f2609c1d064d1369b7e15d6b334073bea85cd4c6d67fe62d`.

Finally, the production FastAPI application was started against the persistent
catalog. It returned all five artifacts, resolved the persistent `mp7`
representation, and served `bytes=0-1023` as `206 Partial Content` with a
1,024-byte body, `Accept-Ranges: bytes`, content range
`bytes 0-1023/957383311`, and the expected geometry SHA-256 ETag.

A final strict headed browser run then exercised the promoted `mp7` directly
through that live catalog. It passed **1/1 in 16.3 seconds** on the GTX 1070:
641.9 ms cold, 483.9 ms warm, 16.68 ms orbit/dive p95, zero steady-state long
tasks over 50 ms, 309,925 visible points, 4,607,129 CPU/GPU geometry bytes,
and zero overview asset requests. The original-resolution screenshot shows
the five-scene catalog, high-contrast Source coloring, trajectory, and
resolved Source 0 provenance inspection.

## Remaining acceptance work

None. All Phase 2 implementation, real-data onboarding, persistent
registration, documentation, and GTX 1070 acceptance gates are closed.
