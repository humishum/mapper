# ADR-004: Direct COPC serving and LOD policy

- Status: Accepted
- Date: 2026-07-25

## Context

The legacy viewer parsed PLY on the backend, encoded point chunks into JSON,
and capped loads at 100,000 points. That path prevents progressive hierarchy
traversal and wastes CPU and network bandwidth.

## Decision

Serve immutable package artifacts directly with HTTP byte ranges and
validators. Giro3D's COPC source performs hierarchy traversal, selection, LAZ
decoding, and point parsing in the browser. The backend never parses or
reprojects points for a view request.

Detail starts with a two-million-visible-point global budget,
SSE/subdivision threshold 1, decode stride 2, and separate 256 MB CPU and GPU
geometry pools. Inactive requests must be aborted and disposed state must not
be repopulated.

## Consequences

- Overview discovery performs zero COPC node requests.
- Browser caches and range requests make warm starts cheap.
- The range server remains format-agnostic and future artifact kinds can
  reuse it.
