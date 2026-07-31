# COPC publisher

Mapper publishes canonical LAS/LAZ shards with
[`copc_converter`](https://github.com/360-geo/copc-converter), pinned to
**v0.11.0**. The binary is Rust-based and uses an out-of-core count/distribute/build
pipeline. The release URL and SHA-256 are recorded in
`tools/copc_converter.lock.json`; the binary itself is not committed.

Install the verified Linux x86-64 release:

```bash
python scripts/install_copc_converter.py
```

The runtime resolves an explicitly configured executable, then
`MAPPER_COPC_CONVERTER`, then the repo-local install, then `PATH`.

## Python API

Stage each source unit (a reconstruction window now, a SLAM submap later) as a
bounded-memory LAZ shard:

```python
from src.publisher import write_laz_shard

write_laz_shard(
    "staging/source-000.laz",
    points=point_cloud.points,
    colors=point_cloud.colors,  # may be None
    confidence=point_cloud.confidence,
    source_index=0,
)
```

`source_index` and `contributor_count` can be scalars or per-point arrays. RGB,
when present, is expanded exactly from 8-bit to LAS 16-bit (`value * 257`).
Colored shards are LAS 1.4 point format 7; geometry-only shards use format 6.
Both use scale `0.001`, offset `0`. `Confidence` float32 is included when
supplied (or explicitly required by `LasStagingConfig`), while
`ContributorCount` uint16 is included by default.

Publish one file, a directory, or an explicit shard list:

```python
from src.publisher import CopcPublisher, CopcPublisherConfig

result = CopcPublisher(
    CopcPublisherConfig(memory_limit="4G", threads=4)
).publish(
    ["staging/source-000.laz", "staging/source-001.laz"],
    "package/points.copc.laz",
)
```

Cross-source 2 cm consolidation is a separate disk-backed step:

```python
from src.publisher import consolidate_laz_shards

consolidated = consolidate_laz_shards(
    "staging",
    "staging-voxel2cm",
)
```

All contributors for a voxel are routed to the same bounded scratch bucket.
With confidence, the highest-confidence record retains XYZ/RGB/source index;
without confidence, the lowest source index wins deterministically. Incoming
contributor counts are always summed. A skewed bucket that exceeds the
configured hard point cap fails before it can violate the memory contract.

The destination appears atomically only after the converter succeeds and the
publisher verifies:

- LAS 1.4 point format 6 or 7, canonical scale/offset and dimension schema;
- header count equals both the input count and hierarchy count;
- every hierarchy node is valid and reachable from the root;
- the complete `PointSourceId` distribution matches the input shards.

Temporary distribute/build data and partial output are removed after success or
failure. Use `validate_source_distribution=False` only for an explicitly
documented performance tradeoff; structural and schema checks always run.

The wrapper also flattens v0.11.0's deeply paged hierarchy before validation.
The converter's three-level page boundaries produced 2,078 pages for `mp7`;
Giro3D 2.0.3 eagerly loads the complete tree, turning those pages into 2,081
initial range requests and a 47.43 s localhost initialization. Flattening
changes only the hierarchy EVLR (not point chunks), restoring the one-request
hierarchy behavior measured in Phase 0. Set `flatten_hierarchy=False` only for
a client that demonstrably loads hierarchy pages lazily.

## Legacy migration

The one-shot command memory-maps a binary PLY, restores window provenance from
the legacy metadata, stages bounded LAZ shards, optionally consolidates, writes
and validates the COPC, then removes all staging:

```bash
python -m src.publisher.cli migrate-ply \
  /path/to/aligned_pointcloud.ply \
  /path/to/points.copc.laz \
  --source-window-dir /path/to/windows \
  --memory-limit 4G \
  --temp-dir /scratch
```

Add `--voxel-size 0.02` only when source units are known to be metres. Legacy
GPS marker metadata is deliberately not interpreted here as authoritative
placement; the package/manifest layer must register these conversions as
`unaligned` unless a separate alignment result proves otherwise.

## Package an existing COPC

Use `package-existing-copc` when a validated COPC already exists and
reconstruction must not be rerun:

```bash
python -m src.publisher.cli package-existing-copc \
  /path/to/existing-sources.copc.laz \
  /path/to/legacy/reconstruction \
  /path/to/packages/capture-run \
  --catalog /path/to/catalog.sqlite3 \
  --source-video /path/to/original.MP4 \
  --memory-limit 1G \
  --threads 4 \
  --temp-dir /scratch
```

`--source-video` is optional. When omitted, the command also checks for the
legacy `metadata.json` `video_name` directly inside the reconstruction
directory. If neither exists, the package is registered as unaligned with
rejection reason `source_video_missing`. When a video is found, the command
extracts GPS/IMU telemetry and runs the Phase 1 GPS/pose alignment.
Missing telemetry, missing poses, extraction failure, or an alignment quality
failure each remains an explicit unaligned reason; no global placement is
inferred.

Before adoption, the command streams and validates the COPC hierarchy,
dimensions, finite bounds, SHA-256, and complete `PointSourceId` distribution.
It also requires the distribution to agree exactly with the legacy
`windows/window_NNN/metadata.json` point counts when those files exist.
Compatible input is hardlinked into `geometry/points.copc.laz`, falling back to
a copy across filesystems. The manifest always points inside the package.

Incompatible scale, offset, point format, paged hierarchy, or a successful
alignment causes streaming republication through pinned
`copc_converter` v0.11.0. A missing `ContributorCount` is staged explicitly as
`1` and recorded in the point artifact and migration metrics as a legacy
default. `Confidence` is preserved only when present and is never synthesized.

Saved per-window `poses.npz` archives are de-duplicated by original frame index
and written to `cameras/poses.parquet`; window metadata becomes `sources.json`.
Successful GPS alignment republishes geometry and poses into artifact-local
ENU and records a float64 ENU-to-ECEF transform plus WGS84 footprint. The
command validates the finished package and registers it in the requested
catalog before returning success.

## Evaluation evidence

On 2026-07-25, v0.11.0 was evaluated against the Phase 0 fixtures:

- Two LAZ shards, 240,000 total points, 256 MB configured memory: 0.52 s wall,
  103.3 MiB peak RSS. All points, RGB, two source IDs, float32 `Confidence`,
  and uint16 `ContributorCount` round-tripped through `laspy`.
- The 146,911,634-point `mp7-full.copc.laz` fixture, re-published with a 1 GB
  configured limit, four threads, LZ4 scratch and packed node storage: 248.25 s
  wall and 715.8 MiB peak RSS. Final flattened output was 957,342,373 bytes.
- Independent byte-level validation found 6,455 nodes and 146,911,634
  header/hierarchy points, zero invalid keys and zero unreachable nodes. The
  compatibility postprocess reduced 2,078 hierarchy pages to one; Giro3D
  initialization fell from 47.43 s/2,081 requests to 27–89 ms/four requests.
  A streaming read confirmed the exact counts of all 43 `PointSourceId` values.

The large input was already COPC-compressed, so this timing includes decoding
and re-encoding it and is not a direct speed comparison with Phase 0's PLY
writer. The result establishes the important property: peak RSS stayed below
the configured 1 GB budget instead of Phase 0's 12.9 GB.

Two real one-shot legacy migrations also passed with a 512 MB converter limit:

- MUSt3R `kings_canyon_2`: 25,667,473 points / 18 sources → 150,401,162
  bytes, 1,241 nodes, one hierarchy page, 62.85 s end-to-end, 568.8 MiB
  process peak RSS.
- VGGT `tahoe_ridge_2`: 17,000,000 points / 340 sources → 125,608,650
  bytes, 969 nodes, one hierarchy page, 54.50 s end-to-end, 514.8 MiB
  process peak RSS.

Both retained exact per-window source counts and passed header/hierarchy/root
reachability validation. They remain local-frame, unaligned artifacts.
