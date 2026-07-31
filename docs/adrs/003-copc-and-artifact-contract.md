# ADR-003: COPC and the typed artifact contract

- Status: Accepted
- Date: 2026-07-25

## Context

The viewer must consume outputs from different reconstruction models without
depending on model-native folders or requiring every run to emit the same
geometry kind.

## Decision

Point geometry is published as COPC/LAZ in a typed representation within a
schema-validated package. No geometry slot is universally required. Provenance
uses LAS `PointSourceId`, `ContributorCount` records consolidation
multiplicity, and `Confidence` is present only when the producing model
supplied meaningful confidence.

Metric geometry may be voxel-consolidated under a recorded policy. Non-metric
geometry is never assigned a centimetre voxel size.

## Consequences

- Point, pose-only, mesh, and future splat runs share one package contract.
- Consumers discover capabilities from manifest dimensions instead of
  guessing.
- Missing confidence disables confidence coloring; it is never synthesized.
