# ADR-006: SQLite catalog layout

- Status: Accepted
- Date: 2026-07-25

## Context

The local viewer needs fast discovery by footprint, time, and quality status, but geometry
must remain immutable range-readable files. PostgreSQL/PostGIS, object storage, and an
application-managed point database would add operational cost without a current local need.

## Decision

Use SQLite as an index over validated filesystem packages. Normal tables contain capture,
run, logical artifact, representation, source, and layer-default records. An SQLite R-tree
indexes WGS84 artifact bounds by the artifact table's integer surrogate key; public identity
remains the opaque string ID.

Package registration is validation-first and one `BEGIN IMMEDIATE` transaction. Re-registering
the same package path and IDs replaces its indexed records idempotently. Reusing a run or
artifact ID from another package path is a conflict. Validation failure or any SQL error leaves
no partial registration.

Only globally placed artifacts enter the R-tree. Consequently, any bbox query necessarily
excludes unaligned artifacts. Non-spatial discovery still returns unaligned packages for
local-scene viewing.

Representations store relative paths and checksums, not file bytes. The asset service resolves
a representation ID through its package root, verifies path containment and size, and streams
the file with single-range `GET`/`HEAD`, immutable cache headers, ETags, and cancellation
points. It never imports or invokes point readers or reprojection code.

## Consequences

- Catalog files are disposable indexes; JSON manifests remain the source of truth.
- SQLite WAL supports concurrent local readers while registration is serialized.
- R-tree bounds are a coarse filter; exact polygon intersection can be added above it if a
  concrete query requires it.
- Moving a package requires explicit re-registration rather than silently repointing an ID.
