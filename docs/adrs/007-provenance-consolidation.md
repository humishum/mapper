# ADR-007: Provenance through consolidation

- Status: Accepted
- Date: 2026-07-25

## Context

Windowed reconstruction observes surfaces repeatedly. Consolidation must
reduce duplicates without erasing which capture/run/source produced the
surviving point.

## Decision

LAS `PointSourceId` identifies a generic source record in `sources.json`;
source kinds are not limited to today's windows. `ContributorCount` is
required and records how many input points contributed to a consolidated
point. When meaningful `Confidence` exists, the highest-confidence contributor
supplies the surviving attributes and `PointSourceId`. Without confidence,
selection is deterministic and no confidence is invented.

Metric-only voxel policy and all point-count changes are recorded in the
manifest and metrics.

## Consequences

- The viewer can color and inspect windows, submaps, keyframe groups, or later
  source kinds without redesign.
- Fine-grained multi-observation lineage, if needed, belongs in a sidecar
  rather than UUIDs on every point.
