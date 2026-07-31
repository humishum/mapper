# Phase 2 deterministic multi-site fixture

This builder adopts one existing, structurally valid COPC into three complete
package-v1 directories and registers them in one SQLite catalog. `site_a` and
`site_b` have distinct ENU-to-ECEF placements in San Francisco and Las Vegas;
their anchors are roughly 670 km apart. `site_a_comparison` has a distinct
package/artifact identity at the same authoritative ECEF anchor as `site_a`.
This supports both distant overview → detail transitions and deterministic
two-COPC comparison under one shared local origin.

Run from the repository root:

```bash
python -m benchmarks.viewer.fixtures \
  /path/to/points.copc.laz \
  /path/to/phase2-multisite \
  --summary /path/to/phase2-multisite/fixture.json
```

The output contains `site_a/`, `site_b/`, `site_a_comparison/`, and
`catalog.sqlite3`. Geometry is hardlinked when the source and output share a
filesystem and otherwise copied; packages never contain symlinks or paths
outside their roots. The input is opened read-only and is never modified. Use
a new output directory for each build; existing site package directories are
rejected to avoid silently overwriting benchmark evidence.

Given identical COPC bytes and dependency versions, manifests and source
sidecars are byte-for-byte reproducible. The summary includes the package
roots, artifact IDs, WGS84 anchors, and float64 ECEF transforms needed by
browser automation.

The source COPC must contain canonical `PointSourceId` and
`ContributorCount` dimensions. The builder reads the real LAS/COPC header,
advertises RGB and optional `Confidence` only when present, and scans
`PointSourceId` values to create one deterministic `sources.json` record per
observed source ID with its exact point count. This scan is intentional: source
coloring and inspection benchmarks must not rely on invented provenance.
