# ADR-002: Coordinate frames and float64 ECEF placement

- Status: Accepted
- Date: 2026-07-25

## Context

Reconstruction outputs are small relative to ECEF coordinates. Sending absolute
ECEF values through float32 render buffers destroys close-range precision,
while pretending unaligned native coordinates are geographic creates
convincing but incorrect maps.

## Decision

Aligned artifacts use a metre ENU `artifact_local` frame and carry an
authoritative float64 row-major transform to ECEF. Unaligned artifacts keep
their explicitly declared native-local frame, have no geographic footprint or
ECEF transform, and open only in local detail mode with their alignment
rejection reason visible.

The detail renderer chooses an active ECEF origin and applies:

```text
M_render = T(-active_origin_ecef) × M_artifact_to_ecef
```

The identical relative transform is applied to geometry, poses, and other
detail artifacts. URL and catalog placement state retain float64 values.

## Consequences

- Re-alignment changes metadata instead of rewriting COPC bytes.
- Close-range buffers remain near the origin.
- Two aligned artifacts can share a detail scene only when their anchors are
  within 10 km.
- Unaligned artifacts can never acquire an inferred marker or footprint.
