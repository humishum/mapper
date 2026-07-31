# Phase 1 review — correctness and canonical publisher

Status: **complete**

Plan: [`visualization_platform_plan.md`](visualization_platform_plan.md)
§ Phase 1

## Delivery checklist

- [x] Explicit `AlignmentResult`; durable alignment contains float64
  model-to-artifact-local and artifact-local-to-ECEF transforms, quality
  status, scale, inliers, errors, anchor, clock estimate, and rejection reason.
- [x] Timestamp-based GPS/pose interpolation, quality filtering, robust ECEF
  anchor, weighted robust Umeyama/Kabsch, metric-scale locking, and explicit
  alignment rejection.
- [x] GPS, IMU, and camera pose plain-Parquet contracts and physical schema
  validation.
- [x] MUSt3R reads points, RGB, and confidence directly from its scene object;
  threshold-family PLY round trips are removed. VGGT confidence remains
  attached to its in-memory point cloud.
- [x] Window alignment can return independently transformed source units.
  Durable provenance stores each unit's exact transform. The package contract
  also accepts SLAM submaps, keyframe groups, and batches, so fixed VRAM
  windows are not a permanent architecture requirement.
- [x] Bounded-memory LAZ staging, cross-source voxel consolidation, and
  canonical COPC publication with `PointSourceId`, optional `Confidence`, and
  `ContributorCount`.
- [x] Strict reconstruction-package v1 manifest, sources, metrics, and tabular
  schemas; atomic manifest-last writer; containment, checksum, COPC, sidecar,
  transform, and physical Parquet validation.
- [x] SQLite catalog with stable opaque identities, WGS84 R-tree footprints,
  validation-first idempotent registration, and exclusion of unaligned
  artifacts from spatial queries.
- [x] Range-capable immutable asset API with `GET`, `HEAD`, single ranges,
  `206`/`416`, ETags, cancellation-aware streaming, and safe path resolution.
- [x] One-shot bounded-memory legacy PLY migration CLI.
- [x] Viewer and legacy utility data roots are configurable rather than
  machine-coded.
- [x] End-to-end package assembly crosses the real converter boundary, validates
  the package, and can register it in the catalog. Full suite: 48 passed,
  1 skipped.

## COPC writer decision

Adopt pinned Rust
[`copc-converter` v0.11.0](https://github.com/360-geo/copc-converter)
through Mapper's publisher wrapper.

The unmodified converter stayed below a 1 GiB limit on the 146,911,634-point
fixture, but its three-level hierarchy paging caused Giro3D 2.0.3 to issue
2,081 eager range requests and spend 47.43 seconds initializing. Mapper's
wrapper now flattens only the terminal hierarchy EVLR while the output is still
disposable, then independently revalidates the complete structure before an
atomic rename. Point chunks are not rewritten.

With that mitigation:

- 146,911,634 points published in 248.25 seconds with 715.8 MiB peak RSS;
- all 6,455 nodes and all 43 source counts remained exact and reachable;
- Giro3D initialization fell to 4 requests and 27–89 ms;
- a 2M-point view still decoded correctly.

The hierarchy mitigation is mandatory for this pinned writer/viewer
combination and is covered by read-back validation.

## Limited migration acceptance

We reused existing reconstructions instead of rerunning every video:

- MUSt3R `kings_canyon_2`: 25,667,473 points, 18 source windows, unaligned
  non-metric geometry (therefore no centimetre voxel assumption).
- VGGT `tahoe_ridge_2`: 17,000,000 points, 340 source windows, metric geometry.

Both conversions preserve exact source distributions and produce a
root-reachable, single-page COPC hierarchy:

| Model | Acceptance artifact | SHA-256 |
| --- | --- | --- |
| MUSt3R | `/home/ape/mapper_output/phase1_acceptance/must3r-kings-canyon-2-sources.copc.laz` | `e7f86000eae1620c7d05094f63f9ea8cdd8238fd086b6669165955e7a9d3a067` |
| VGGT | `/home/ape/mapper_output/phase1_acceptance/vggt-tahoe-ridge-2-sources.copc.laz` | `f93dc9afd7833706f2609c1d064d1369b7e15d6b334073bea85cd4c6d67fe62d` |

The adjacent `*.copcstats.json` reports record the independent hierarchy
checks. Package orchestration is separately exercised through the real pinned
converter and validates manifest, sidecars, checksums, physical Parquet
schemas, and COPC structure before returning success.

## Fresh model-to-package acceptance

We also ran one short, 230-frame Kings Canyon capture through each live model
adapter. These are complete model → source alignment → consolidation → COPC →
package validation → catalog registration runs, not legacy-file conversions.

| Model | Run | Sources | Staged points | COPC points | COPC bytes | COPC SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MUSt3R | `run_e67eb10916cf43a2bd0577eb73ac02ff` | 6 | 11,207,020 | 11,207,020 | 91,573,194 | `001c9c3e8dab9279c7735627683455c0bec58e1aa5e2a95c9fd21959f81d99e2` |
| VGGT | `run_f584269ca82a4a8586e0398b4c61274f` | 15 | 750,000 | 200,228 | 1,435,703 | `343147928f65359770a30d961a1c8ba8d817bab1fba94c0b5e28c5920e8a73ce` |

MUSt3R correctly remained non-metric, so centimetre voxel consolidation was
skipped. VGGT marked its local output as metric and applied the configured 2 cm
confidence-aware consolidation; `PointSourceId` follows the highest-confidence
contributor and `ContributorCount` preserves overlap multiplicity. Both
packages contain five registered representations, exact source rows, validated
Parquet, an independently validated COPC, and no old aligned PLY or metadata
sidecar. Their disposable scratch directories are empty.

The capture's GoPro GPS samples failed the coordinate/fix/precision gates.
Consequently both manifests explicitly say `unaligned` with
`gps_unavailable`, and neither advertises an origin, ECEF transform, CRS, or
WGS84 footprint. This exercises the negative global-coordinate contract.

VGGT's stock fp32-weight/518 px profile exhausted the GTX 1070's 8 GiB VRAM
even with six-frame windows; reducing the window alone did not materially
change the roughly 7.55 GiB footprint. The adapter now supports
`mixed_float16`: its memory-heavy autocast-compatible aggregator is loaded as
fp16, its prediction heads remain float32 as required by VGGT, and aggregator
tokens are restored to float32 at that boundary. The checked-in
`vggt_test.yaml` is deliberately a fast 168 px smoke profile with 20-frame
windows. A 518 px quality run still belongs on a larger GPU or in a separate
quality/performance profile.

## Coordinate storage clarification

Each point artifact stays in a small, near-origin `artifact_local` frame.
For a successful GPS alignment this is metre ENU; geometry is not rewritten
into large ECEF coordinates. The manifest stores the full float64 row-major
4×4 `artifact_local -> ECEF` transform and a longitude/latitude/ellipsoidal
height origin. The catalog derives its WGS84 footprint from that transform.

For rejected alignment, the package stores the local geometry plus the
rejection reason, but no global origin, CRS, ECEF transform, or WGS84
footprint. A legacy initial-GPS marker is not treated as authoritative
placement.

## Deliberate boundaries

- Raw GoPro gravity is not applied until a calibrated sensor-to-camera/model
  extrinsic exists. Assuming the sensor vector is already in model space can
  corrupt orientation.
- Fixed reconstruction windows remain a GPU execution strategy, not a package
  concept. A future continuous SLAM pipeline with loop closure should emit
  submaps/keyframe groups (or one continuous source) through the same
  publisher.
- A 2 cm voxel is meaningful only after geometry is metric. Unaligned
  non-metric runs skip metre-based consolidation rather than silently damaging
  geometry.

## Verification

- `pytest -q -rs`: **48 passed, 1 skipped**. The skip is the opt-in real-video
  telemetry test requiring `TELEMETRY_TEST_VIDEO`; the two fresh runs above
  exercised real GoPro telemetry.
- Targeted Ruff checks for the Phase 1 Python surface pass.
- `git diff --check` passes.
