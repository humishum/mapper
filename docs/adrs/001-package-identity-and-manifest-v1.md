# ADR-001: Reconstruction package identity and manifest v1

- Status: Accepted
- Date: 2026-07-25

## Context

Reconstruction models emit incompatible native files and coordinate conventions. The
viewer needs one stable contract that supports point, mesh, splat, and pose-only runs without
inferring identity or placement from directory names.

## Decision

Each package has stable opaque `capture_id`, `run_id`, and `artifact_id` values and a required
`manifest.json` conforming to `schemas/manifest.v1.json`. Concrete files and trees are typed
representations with their own opaque ID, relative path, byte size, SHA-256 checksum, media
type, and format. No geometry representation is universally required.

The canonical geometry frame is `artifact_local`, near the origin. An authoritative float64
row-major `artifact_local -> ECEF` affine transform provides global placement. The WGS84
origin is always ordered longitude, latitude, ellipsoidal height. Correcting placement changes
the manifest rather than rewriting the geometry.

Every alignment attempt is recorded. `unaligned` packages may remain metric or non-metric but
must not claim a WGS84 footprint, origin, projection, CRS, or ECEF transform. Other alignment
states require metre units, an ECEF transform, origin, and footprint.

`sources.json` generalizes provenance units to windows, submaps, keyframe groups, batches, or
captures. `metrics.json` records reconstruction, alignment, validation, and publication
measurements. Pose and telemetry Parquet files declare their columns in the manifest and follow
`schemas/tabular-contracts.v1.json`.

The manifest is written atomically and last. Registration independently revalidates the
contract, containment, sizes, sidecars, and checksums.

## Consequences

- Native model outputs stay optional under `raw/`; consumers never depend on them.
- Manifests are descriptive and reproducible, including dirty Git state and lossy operations.
- Global placement remains precise while point buffers remain small and local.
- Updating bytes requires updating the corresponding checksum and re-registering the package.
- Antimeridian-spanning artifacts need explicit split footprints; the convenience bounds
  projector rejects them.
