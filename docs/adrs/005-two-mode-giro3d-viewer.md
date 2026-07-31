# ADR-005: Two-mode Giro3D viewer

- Status: Accepted
- Date: 2026-07-26

## Context

The product needs an ECEF world overview and stable centimetre-scale
inspection of local reconstructions. Keeping both scales in one persistent
absolute-coordinate scene would require custom rebasing of Giro3D globe
internals without a proven relative-to-eye or high/low path.

## Decision

Use one React application and viewport with two independently owned Giro3D
renderer modes:

1. An ECEF globe overview renders the configurable basemap, catalog footprints,
   and lightweight markers. Merely visible artifacts never initialize a COPC
   source.
2. An origin-centered detail renderer owns selected COPC geometry,
   trajectories, picking, and inspection. Aligned artifacts use the relative
   transform from ADR-002; unaligned artifacts remain native-local.

Transitions cross-fade without page navigation. Returning to overview restores
its previous camera. Leaving detail aborts requests and disposes entities,
workers, cached attributes, and geometry.

## Rationale

- Predictable close-range precision.
- Clean network and GPU teardown when leaving a site.
- No custom rebasing of Giro3D globe internals.

The accepted cost is that close detail has no basemap underneath and the
transition is not a literally continuous globe-to-point flight. A persistent
camera-relative ECEF scene remains a future option if Giro3D gains a proven
relative-to-eye or high/low-coordinate path.

## Acceptance

Accepted on 2026-07-26 after Phase 2 proved:

- deterministic multi-site overview → A → overview → B transitions with one
  document navigation;
- zero overview and inactive-distant-site detailed requests;
- abort/disposal resistance with no stale state commit;
- stable geometry, trajectories, Source coloring, and resolved provenance
  picking after renderer transitions; and
- GTX 1070 cold/warm, moving-view, long-task, point-budget, and CPU/GPU memory
  gates using hardware WebGL.

The canonical 146,911,634-point `mp7` run measured 643.8 ms cold, 184.1 ms
warm, 16.68 ms p95 for both orbit and dive, zero steady-state long tasks over
50 ms, 309,925 visible points, and 4,738,688 CPU/GPU geometry bytes. The
standard seven-scenario suite passed 7/7. See
[`../../benchmarks/viewer/HARDWARE_ACCEPTANCE.md`](../../benchmarks/viewer/HARDWARE_ACCEPTANCE.md).
Software rendering was not used as acceptance evidence.
