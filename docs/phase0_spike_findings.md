# Phase 0 spike — findings and benchmarks

Status: **complete — Giro3D accepted; calibrated default is 2M points, SSE 1, decode stride 2**
Date: 2026-07-25
Plan: [`docs/visualization_platform_plan.md`](visualization_platform_plan.md) § Roadmap → Phase 0
Machine: 6 cores, 15 GB RAM, NVIDIA GTX 1070 (driver 560.35.03)

This document is the record of the Phase 0 spike: what was measured, what was decided
mid-flight and why, the corrected results, and the final renderer decision.
Everything here is reproducible with the commands in
[`spike/phase0/README.md`](../spike/phase0/README.md).

> Historical writer note: Phase 0 evaluated PDAL and a temporary `copclib`
> writer. The accepted production path is the Phase 1 publisher in
> `src/publisher`: bounded LAZ staging, pinned Rust `copc-converter` v0.11.0,
> validation, and terminal hierarchy flattening. PDAL references below are
> retained only as spike evidence, not current writer guidance.

---

## 1. Phase 0 scorecard

| # | Plan item | Status |
| --- | --- | --- |
| 1 | Convert `mp7` (146.9M pts, 2.1 GB PLY) → COPC, record wall clock + size | **Done** — corrected rerun 101.4 s, 955 MB (§4) |
| 2 | Serve it and load it in Giro3D's COPC path; judge load/refine/memory/feel | **Done** — corrected 147M-point artifact renders progressively; cold first geometry 0.77 s and worst camera-path p95 19.6 ms on the GTX 1070 (§5–6) |
| 3 | Repeat with ~2 cm voxel-consolidated variant; compare quality and node counts | **Done** — matched views show no visible loss at the tested camera; 31% fewer points and 27% fewer nodes (§4.3, §5.2) |
| 4 | Write down every field a manifest needed to make 1–3 reproducible → schema v1 draft | **Done as an inventory** (§7); promoting it to `schemas/` is Phase 1 |
| 5a | Baseline the current deck.gl viewer as the "before" record | **Done** (§4.5) |
| 5b | Fix `pytest` collection (`testpaths = ["tests"]`) | **Done** — `pyproject.toml`, verified |

Giro3D traverses, range-fetches, decodes and renders the corrected Mapper COPC. Static opening
is 0.33–0.37 MB and 68–102 ms; the first selected node decodes in 3–55 ms; matched screenshots
confirm full/consolidated visual equivalence and window-index coloring. On the GTX 1070 the
calibrated full and consolidated runs both reached first geometry in under 0.77 s, kept every
camera-path p95 at or below 19.6 ms, recorded no >50 ms main-thread task while moving, and
peaked below 198 MB in each decoded CPU/GPU geometry pool. **Phase 0 accepts Giro3D and does
not trigger the renderer fallback.**

---

## 2. Fixtures and where they live

The `mp7` fixture named in the plan is **not in the repo** and not under `data/`. It lives in
the external output root:

```text
/home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/
  config.yaml                     # run config: model, window params, alignment config, git commit
  experiment.log                  # per-capture progress; the only record of alignment happening
  outputs/mp7/
    aligned_pointcloud.ply        # 2,203,674,693 B, 146,911,634 pts, xyz f32 + rgb u8 (15 B/pt)
    metadata.json                 # video_name, initial GPS, altitude, frames, is_metric, point_count
    window_000..window_042/       # scene_thr1.5.ply (88 MB) + scene_thr5.0.ply (46 MB) per window
    windows/window_000..042/      # pointcloud.ply (43 MB) + poses.npz + metadata.json per window
```

Derived spike artifacts (kept outside the repo; `data/` is git-ignored and these are large):

```text
/home/ape/mapper_output/phase0_spike/
  artifacts/window000.copc.laz        14.8 MB    2,869,566 pts   (fast iteration fixture)
  artifacts/mp7-voxel2cm.copc.laz    697.2 MB  101,144,857 pts  corrected/reachable
  artifacts/mp7-full.copc.laz        955.1 MB  146,911,634 pts  corrected/reachable
  artifacts/mp7-voxel5cm.copc.laz    531.2 MB   74,724,239 pts
  artifacts/mp7-voxel10cm.copc.laz   342.1 MB   46,658,901 pts
  artifacts/*-invalid-morton-keys.*  preserved pre-fix diagnostic artifacts
  reports/*.json                      conversion metrics, COPC structure, deck.gl baseline
  reports/range-requests.jsonl        server-side byte-range log
  bench/traverse.json                completed Giro3D data-path benchmark
  bench/completed/                   software smoke + NVIDIA benchmark results/screenshots
```

### What the fixture told us about the data

* **Extent is 327 × 353 × 515 units**, `min z = 0.575`. Combined with
  `metadata.json: "is_metric": false` and the model capability `outputs_metric_scale: False`,
  this is the first camera's frame (x right, y down, **z forward**), not ENU, and the units are
  not known to be metres even though the run had `align_to_gps: true`.
  Two consequences: no viewer control scheme is sane in this frame without the manifest
  transform, and "2 cm voxel" is only nominally 2 cm.
* **The merged cloud is a plain concatenation of the 43 window clouds** — the per-window point
  counts sum to exactly 146,911,634. Window provenance is therefore *exact*, not inferred, and
  the converter refuses to write it if that sum ever disagrees.
* **GPS and IMU are extracted and then discarded.** `experiment.log` for mp7 records
  `GPS: 3076 points`, `IMU: 34224 samples`, `Aligning to GPS...`, `Computing metrics...` — but
  the only things written to disk are the merged PLY and a 6-field `metadata.json`. No scale,
  no inlier count, no RMSE, no clock offset, no telemetry files. This is exactly the gap
  Phase 1 item (1) (`AlignmentResult`) exists to close.
* **The unused threshold PLYs cost 6.0 GB for this one capture.** Each window is written twice:
  `windows/window_NNN/pointcloud.ply` (the real data, plus `poses.npz` and metadata — 2.1 GB
  total) and `window_NNN/scene_thr{1.5,5.0}.ply` threshold variants that nothing consumes
  (`du` over the 43 `window_NNN/` dirs: **6.0 GB**). Add the 2.2 GB merged PLY and mp7 alone
  occupies ~10.3 GB, of which 6.0 GB is dead weight — the plan's Phase 1 item (3).
* **Per-window `poses.npz` already has what the package needs**: `poses (N,4,4) f32`,
  `timestamps (N,) f64`, `intrinsics (3,3) f32`, `frame_indices (N,) i64`; the sibling
  `metadata.json` carries `window_id`, `frame_start`, `frame_end`, `frame_indices`,
  `window_size`, `window_overlap`, `point_count`, `is_metric`, `has_poses`.

---

## 3. Toolchain decisions made mid-flight

These were not in the plan; they were forced by what this machine and this dependency policy
allow. Each one has a consequence for Phase 1.

### D-0.1 — PDAL is not reachable, so the spike writes COPC with `copclib` + a numpy octree

The plan specifies `pdal writers.copc` (streaming) with `untwine` as fallback. Neither is
obtainable here:

* `pip install pdal` (PyPI `pdal` 3.5.4) is an **sdist that needs libpdal + headers** on the
  system. Not uv-installable.
* conda-forge has `pdal`/`python-pdal`/`untwine` — **ruled out**: the user's dependency policy
  is uv-only, no conda/micromamba. (A micromamba attempt also stalled indefinitely after
  fetching the 430 MB conda-forge repodata, twice, so it would have been slow even if allowed.)
* The official PDAL container exists but **`docker` needs sudo** here:
  `permission denied while trying to connect to the Docker daemon socket`.

What was used instead, all via `uv pip install --python .venv`:

| Package | Version | Role |
| --- | --- | --- |
| `copclib` | 2.6.3 | copc-lib bindings: LAZ chunk compression, COPC file/hierarchy assembly |
| `laspy` | 2.7.0 | independent read-back validation (`CopcReader`) |
| `lazrs` | 0.8.1 | laspy's LAZ backend |

`copclib` does *not* build the octree — that is implemented in
[`spike/phase0/ply_to_copc.py`](../spike/phase0/ply_to_copc.py) as a Morton-sorted,
Potree/Entwine-style level sampler (§4.1). Measured compression throughput of the copclib
path: **~4.0 M points/s single-threaded** at point format 7.

> **These packages are installed in `.venv` but not recorded in `pyproject.toml`.** That was
> deliberate for a spike. If Phase 1 adopts this path, promote them with `uv add`; if Phase 1
> gets a system/container PDAL, drop them.

**Phase 1 decision owed:** the streaming publisher needs a COPC writer that is *out-of-core*.
The three candidates are (a) system or container PDAL/untwine (matches the plan, needs sudo or
an image), (b) keep `copclib` and make the octree pass chunked/external-sorted, (c) run the
publisher next to the GPU job where PDAL can be installed freely. Nothing in Phase 0 forces
the choice, but §4.2's peak-RSS numbers say the current spike code cannot be it.

### D-0.2 — Canonical framing as written in the plan, confirmed workable

`point_format = 7` (LAS 1.4 XYZ + RGB, 36 B/record), `scale = 0.001`, `offset = 0`, **no WKT**.
COPC permits only point formats 6/7/8 — copclib enforces it (`PointBaseByteSize: Point format
must be 6-8`), so the plan's "attributes as extra dimensions" has to live inside that
constraint. With no WKT, Giro3D's `COPCSource.getMetadata()` reports
`CoordinateSystem.unknown`, which is precisely the plan's **local-scene mode** — the viewer can
detect "not georeferenced" from the artifact itself rather than from a side channel.

### D-0.3 — 8-bit PLY colour must be expanded to 16-bit by the publisher

Giro3D's LAS reader divides colour by 256 (`sources/las/readers.js`:
`const factor = compress ? 1/65536*256 : 1`), i.e. it assumes LAS-spec 16-bit colour. Writing
raw 0–255 values into `Red/Green/Blue` would render the whole cloud near-black. The converter
multiplies by **257** (exact 8→16-bit expansion: 255 → 65535) and records
`color_gain_applied: 257` in the report. This belongs in the manifest as a lossless-but-notable
format operation.

### D-0.4 — Window provenance goes in native `PointSourceId`, not an extra dimension

Plan D6 wants a `SourceIndex` uint16 **extra dimension**. copclib's Python bindings expose
`EbField` with **no settable properties**, so an extra-bytes VLR cannot be authored from
Python with this toolchain. `PointSourceId` is a native uint16 field in point formats 6–8 whose
LAS semantics ("point source ID") are exactly this, and Giro3D already exposes it as a
first-class colourable attribute (`DEFAULT_VALUE_RANGES.PointSourceId`, u16 range → colormap).

So the spike stores the window index there, verified round-trip through `laspy`. Recommendation
for the schema: **use `PointSourceId` for window provenance** (universally supported, no EVLR,
no reader-specific handling) and reserve extra bytes for things with no native slot —
`Confidence` being the important one. Note `Intensity` (uint16, also colormapped by Giro3D) is
a pragmatic carrier for quantised confidence if extra-bytes authoring stays awkward.

### D-0.5 — Voxel consolidation runs on the octree's own grid

Consolidating on an independent 2 cm grid and then building an octree quantises twice and
leaves surviving points off the node grid. Instead the converter picks the octree level whose
cell is ≤ the requested size and dedupes on that level's Morton prefix. For mp7 the requested
0.02 became an **effective 0.01573 source units** (level 8 of a 515.29-unit cube), recorded explicitly as
`consolidation.effective_voxel_size` / `consolidation_octree_level`.

### D-0.6 — Leaf rule (bug found and fixed during the spike)

First implementation sampled one point per span-cell at every level, including the deepest,
and silently dropped every other point sharing a deepest-level cell: the window fixture wrote
**458k of 2.87M points (84% loss)** while the LAS header still claimed 2.87M. Fixed by the rule
real COPC writers use: a node whose remaining point count is ≤ `--max-node-points` (default
100,000) becomes a **leaf holding all of its points**, and at the depth cap everything
remaining is written regardless. The report now carries `points_written`, `unplaced_points` and
`nodes_exceeding_max_at_depth_cap`, and both large artifacts verify at
`unplaced_points: 0` with header count == hierarchy count.

*Validation-gate lesson for Phase 1:* "LAS header point count" and "sum of hierarchy node
point counts" are independent numbers and must both be asserted against the input count.

### D-0.7 — `argsort(kind="quicksort")` instead of a stable sort

Stable argsort on 147M int64 keys wants another ~1.2 GB of workspace this machine does not
have. Ties are points sharing the finest octree cell, where any single survivor is equally
valid, so introsort is used. Determinism for a given input is preserved; strict "same input →
same surviving point as a stable sort" is not.

### D-0.8 — hierarchy totals are insufficient; every node must be root-reachable

The first large artifacts passed the original validation: LAS header point count equalled the
sum of hierarchy point counts. The first real Giro3D traversal then exposed that their deepest
nodes were unreachable:

| Artifact | Unreachable nodes | Unreachable points |
| --- | ---: | ---: |
| 2 cm consolidated | 839 | 16,464,057 |
| full | 1,167 | 36,656,140 |

The writer decoded a level's Morton node key with `axis_bits - level` coordinate bits instead
of `level`. Once depth crossed the midpoint, high coordinate bits were truncated and the emitted
node no longer had the parent address a renderer derives for it. The point bytes and hierarchy
entries existed, which is why total-count validation missed the defect.

The writer now decodes exactly `level` bits and refuses to add any node whose direct parent was
not already written. The independent `copc_stats.py` also reports root presence, reachable
node/point counts, unreachable node/point counts, invalid key ranges, and
`all_nodes_reachable_from_root`. Corrected artifacts report 4,860/146,911,634 and
3,557/101,144,857 reachable, with zero unreachable nodes. The invalid originals are retained
under `*-invalid-morton-keys.copc.laz` for the diagnostic record.

---

## 4. Benchmarks

### 4.1 What the converter does

`spike/phase0/ply_to_copc.py`: memmap PLY → exact bounds pass → cube → quantise to a
`2^(depth+7)` grid → int64 Morton codes → sort → (optional consolidation dedupe on a Morton
prefix) → per level, sample one point per span-cell and emit leaves, writing each node as one
LAZ chunk through `copclib`. Because points are Morton-sorted, every node is a contiguous
slice and node/cell grouping is a couple of vectorised diffs per level.

### 4.2 Conversion (PLY → COPC)

| Fixture | In points | Out points | In bytes | Out bytes | Ratio | Wall clock | Peak RSS | Nodes | Levels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `window000` (1 window) | 2,869,566 | 2,869,566 | 43.0 MB | 14.8 MB | 2.90× | **1.8 s** | 0.52 GB | 109 | 5 |
| `mp7-full` (depth 10) | 146,911,634 | 146,911,634 | 2,203.7 MB | 955.1 MB | **2.31×** | **101.4 s** | **12.91 GB** | 4,860 | 11 |
| `mp7-voxel2cm` (effective 0.01573 source units) | 146,911,634 | 101,144,857 | 2,203.7 MB | 697.2 MB | 3.16× | **86.3 s** | 10.88 GB | 3,557 | 9 |
| `mp7-voxel5cm` (effective 0.03145 source units) | 146,911,634 | 74,724,239 | 2,203.7 MB | 531.2 MB | 4.15× | **73.0 s** | 9.67 GB | 2,573 | 8 |
| `mp7-voxel10cm` (effective 0.06290 source units) | 146,911,634 | 46,658,901 | 2,203.7 MB | 342.1 MB | 6.44× | **62.0 s** | 9.67 GB | 1,563 | 7 |

Stage timings (seconds), mp7:

| Stage | full | voxel2cm | Note |
| --- | --- | --- | --- |
| bounds pass | 12.1 | 13.0 | sequential read of 2.2 GB; 0 non-finite points |
| load + quantise + Morton | 18.0 | 18.1 | chunked at 8M points |
| source index (43 windows) | 0.03 | 0.05 | header reads only |
| Morton sort | 13.2 | 13.2 | 147M int64 argsort + reorders |
| consolidation dedupe | — | 2.6 | |
| octree + LAZ write | 58.1 | 39.4 | includes all compression |
| **total** | **101.4** | **86.3** | corrected post-reboot rerun |

Bytes per point in the output: **6.50 B** (full) vs **6.89 B** (consolidated). Consolidation
makes each *stored* point slightly more expensive precisely because it removed the
near-duplicates that LAZ was compressing well — the win is in point count, not per-point cost.

**Peak RSS is the headline problem.** 12.91 GB on a 15 GB machine, because the spike sorts the
whole cloud in RAM. It survived, with no room for a second concurrent job, and it would not
survive a larger capture. This is the plan's "never materialize a merged in-RAM cloud" rule
being violated by the spike tool — acceptable for a spike, disqualifying for the publisher.

### 4.3 Consolidation curve

The source model is explicitly non-metric, so the requested "cm" labels are only fixture names.
The effective values below are in unknown model source units and must not be used to set a
physical Phase 1 default.

| Requested | Effective spacing | Points | Redundancy | Output |
| ---: | ---: | ---: | ---: | ---: |
| none | — | 146,911,634 | 1.00× | 955.1 MB |
| 0.02 | 0.01573 | 101,144,857 | **1.45×** | 697.2 MB |
| 0.05 | 0.03145 | 74,724,239 | **1.97×** | 531.2 MB |
| 0.10 | 0.06290 | 46,658,901 | **3.15×** | 342.1 MB |

The plan's premise for the voxel stage was that "windowed captures observe the same surface
dozens of times". At 0.01573 source units that is **not** what mp7 shows: only 31% of points
are coincident-cell duplicates. Plausible reason: MUSt3R emits one point per pixel per window
and the capture is a forward-moving hike with `window_overlap = 10` of `window_size = 50`
frames, so overlap is bounded and most points are genuinely distinct samples at this spacing.

This does not retire the consolidation stage; it re-scopes it. At the plan's nominal 0.02
request, it is a useful 31% point-count reduction but not a "dozens of observations per cell"
collapse. Matched screenshots at the initial camera show no visible difference between full
and 0.02 variants at the selected LOD. A physical publisher default cannot be chosen from this
run until alignment gives the artifact metric units.

### 4.4 COPC structure (verified by a second, independent reader)

Read back with `spike/phase0/copc_stats.py` — a stdlib-only parser of the LAS header, COPC info
VLR and every hierarchy page, deliberately sharing no code with the writer. Both corrected
files report `header_point_count_matches_hierarchy: true`,
`all_nodes_reachable_from_root: true`, zero unreachable points, and `pages_read: 1`.

`mp7-voxel2cm.copc.laz` — 697.2 MB, depth 8, 3,557 nodes:

| Level | Nodes | Points | Mean pts/node | Bytes | Spacing (units) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 15,210 | 15,210 | 151 kB | 4.026 |
| 1 | 5 | 92,579 | 18,516 | 808 kB | 2.013 |
| 2 | 18 | 405,436 | 22,524 | 3.3 MB | 1.006 |
| 3 | 55 | 1,835,944 | 33,381 | 14.4 MB | 0.503 |
| 4 | 169 | 6,943,655 | 41,087 | 52.9 MB | 0.252 |
| 5 | 536 | 20,104,926 | 37,509 | 146.5 MB | 0.126 |
| 6 | 960 | 27,152,631 | 28,284 | 190.3 MB | 0.063 |
| 7 | 974 | 28,130,419 | 28,881 | 185.2 MB | 0.031 |
| 8 | 839 | 16,464,057 | 19,623 | 103.5 MB | 0.016 |

`mp7-full.copc.laz` — 955.1 MB, depth 10, 4,860 nodes; levels 0–4 are within 0.03% of the
consolidated file (same coarse sampling), then it carries three extra levels
(L8 24.9M / L9 20.8M / L10 15.9M points) holding the near-duplicates.

What this says about viewer cost, before any renderer is involved:

* **Opening an artifact is cheap and bounded**: LAS header + COPC info VLR + one hierarchy page
  (4,860 nodes × 32 B ≈ 155 kB) + the root node (**15,210 points / 151 kB**). First coarse
  geometry is a ~0.3 MB conversation, independent of the file being 1 GB.
* **Coarse-to-fine has usable granularity**: levels 0–2 are 4.3 MB for 513k points; levels 0–4
  are 71.6 MB for 9.3M points. A 2M-point budget lands around level 3–4.
* **Node sizes are in a sane band** (mean 19k–41k points, 20–160 kB compressed), so per-node
  range requests are neither chatty nor huge. 9 nodes (voxel) / 35 nodes (full) exceed the
  100k-point target at the depth cap — worth tightening later, not a blocker.

Read-back also confirmed, via `laspy.CopcReader`, that the root sample spans the full extent
(i.e. it is not a corner), `query()` returns every point, colours come back 16-bit, and
`point_source_id` carries the window index.

### 4.5 The "before" record: current deck.gl viewer data path

`spike/phase0/baseline_deckgl.py` drives `viewer/backend/data_service.py` on the same mp7
fixture (the code path the plan replaces; nothing was modified).

| Measure | Value |
| --- | --- |
| Read PLY + random-downsample | **16.53 s per request** |
| Flat-earth GPS transform | 0.003 s |
| Hex encode + JSON serialise | 0.033 s |
| Response size | **3,000,077 B for 100,000 points** (30 B/point on the wire) |
| Fraction of the artifact visible | **0.068%** (100k of 146.9M) |
| LOD | none — one flat buffer |
| Range requests / cacheable | no / no |
| Server peak RSS | **5.62 GB** (whole PLY materialised per request) |

Correctness notes recorded with the baseline: the GPS transform is a flat-earth approximation
(fixed `EARTH_RADIUS_M`, no ellipsoid), points are placed at the capture's first GPS fix
regardless of alignment quality, and no alignment status travels with the data — so unaligned
data renders as if georeferenced. That is the plan's priority-2 failure mode, live in the
current viewer.

**Side-by-side, same fixture, same machine:**

| | deck.gl path (now) | COPC path (spike) |
| --- | --- | --- |
| Server work per view | 16.5 s, 5.6 GB RSS | none (static bytes) |
| Bytes to first geometry | 3.0 MB (all of it) | ~0.3 MB (header + hierarchy + root) |
| Points reachable | 100k (0.068%) | 146.9M (100%), streamed by LOD |
| Wire cost per point | 30 B | 6.5 B stored, ranged on demand |
| Cacheable / resumable | no | yes (immutable, range-addressed) |

---

## 5. Giro3D results

`spike/phase0/giro3d-app/traverse.mjs` drives Giro3D's **actual** `COPCSource` in Node with no
GPU: `initialize()` → `getMetadata()` → `getHierarchy()` → screen-space-error node selection
for three camera distances → `getNodeData()` per selected node, timing each and counting
server-side bytes. It measures hierarchy load, byte-range behaviour, LAZ decode throughput and
attribute extraction. It does **not** measure GPU upload, draw calls, frame time or feel.

`bench-data.mjs` performs the required esbuild step and Node shims, then runs the harness from
one declared npm command. Results are saved at
`/home/ape/mapper_output/phase0_spike/bench/traverse.json`.

### 5.1 Range fetch and decode

| Artifact / settings | Hierarchy nodes | Open | Open bytes | First selected node | 2M view decode | Decode rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `window000`, SSE 1 | 109 | 64 ms | 182 kB | 55 ms | 2.22 s | 0.90 Mpts/s |
| `mp7-voxel2cm`, SSE 1 | 3,557 | 72 ms | 330 kB | 3 ms | 2.60 s | 0.77 Mpts/s |
| `mp7-full`, SSE 1 | 4,860 | 102 ms | 372 kB | 3 ms | 2.58 s | 0.77 Mpts/s |
| `mp7-full`, SSE 2 | 4,860 | 68 ms | 372 kB | 43 ms | 2.18 s | 0.92 Mpts/s |

The 5M consolidated view decodes in 5.30 s at 0.94 Mpts/s. Decode runs on the Node main thread
because Web Workers are unavailable there; the browser uses Giro3D's worker path. Most
importantly, Giro3D reports exactly the independently validated 3,557 and 4,860 reachable nodes.

### 5.2 Browser visual smoke

Chrome 149 + SwiftShader was used only to validate the complete browser path, screenshots and
controls; its frame times are not hardware acceptance data.

| Artifact | First geometry | Idle | Cold range bytes | Loaded geometry cache |
| --- | ---: | ---: | ---: | ---: |
| `window000` | 584 ms | 2.73 s | 12.6 MB | 34.5 MB |
| `mp7-voxel2cm` | 1.70 s | 6.05 s | 54.0 MB | 82.1 MB |
| `mp7-full` | 898 ms | 7.50 s | 93.8 MB | 82.1 MB |

Matched full/consolidated initial views are visually equivalent at this LOD. The app clearly
marks the artifact as local/non-georeferenced. `PointSourceId` initially exposed two harness
issues that are now fixed: the URL-selected attribute was not applied until a UI change, and
Giro3D's stock uint16 colormap range made values 0–42 nearly black. The spike now applies the
selection immediately and uses a 0–43 multicolor range; Phase 1 must obtain that range from
`sources.json`.

---

### 5.3 NVIDIA hardware result and calibration

The acceptance run used Chrome 149 through the authenticated X11 desktop session. WebGL
reported `ANGLE (NVIDIA Corporation, NVIDIA GeForce GTX 1070/PCIe/SSE2, OpenGL 4.5.0)`;
these are hardware results, not SwiftShader numbers.

| Artifact / calibrated settings | First geometry | Idle | Orbit p95 | Dive p95 | Traverse p95 | Peak decoded CPU / GPU geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mp7-voxel2cm`, 2M, SSE 1, stride 2 | 765 ms | 2.21 s | 17.6 ms | 19.6 ms | 17.0 ms | 196.3 / 196.3 MB |
| `mp7-full`, 2M, SSE 1, stride 2 | 769 ms | 2.21 s | 17.9 ms | 19.1 ms | 17.0 ms | 197.8 / 197.8 MB |

Both runs ended the orbit near 2.02M visible points. Their three camera paths each recorded
zero main-thread long tasks over 50 ms. The full and consolidated artifacts therefore have
effectively the same interactive behavior at the selected operating point; consolidation
remains a storage/network policy decision rather than a renderer requirement.

Stride 1 was measured before selecting that default. It met the frame-time target but reached
about 330 MB in each decoded CPU/GPU geometry pool on both large artifacts. Reducing only
`PointCloud.pointBudget` to 1M did not reduce that retained geometry, because the budget
changes display decimation after decode. `COPCSource.decimate: 2` reduced the peak to 196–198
MB per pool while preserving the 2M visible-point target, so the spike app now defaults to
stride 2. The 256 MB budget is evaluated independently for the decoded CPU geometry cache and
its GPU allocation; their combined resident total is about 377 MB and Phase 2 should report
both rather than collapsing them into one ambiguous number.

There were two ~400 ms long tasks during initial scene construction, before the camera paths.
Single-frame tails of 233–569 ms also appeared at path/load transitions even though path p95
passed and the Long Tasks API recorded no >50 ms main-thread work while moving. Phase 2 should
retain max-frame and loading-state telemetry alongside p95 instead of treating this pass as
proof that all visible stutter is solved.

---

## 6. Environment findings

### B-1 Browser transport — resolved

The earlier localhost failure was sandbox isolation, not Chromium or the servers. With host
network access Chrome loads both Vite and COPC byte ranges correctly. Three launcher workarounds
remain baked into `bench.mjs`:

1. Playwright's own launcher never establishes a CDP session (it uses
   `--remote-debugging-pipe`): browser process starts, `newPage()` then fails or hangs.
   Workaround found: launch the browser manually with `--remote-debugging-port` and attach with
   `chromium.connectOverCDP()` — this works and is what `bench.mjs` now does.
2. `browser.newContext()` over CDP never returns. Workaround: use the default context
   (`browser.contexts()[0]`) and get cold-cache isolation via
   `Network.clearBrowserCache`. Also applied in `bench.mjs`.
3. Playwright's bundled Chromium build 1194 (`chromium_headless_shell` and the full build) also
   fails to navigate at all; only the newer `chromium-1228/chrome-linux64/chrome`
   (Chrome 149) works here.

### B-2 NVIDIA frame-time pass — resolved

The host GPU is healthy: `nvidia-smi` reports driver 560.35.03, the GTX 1070 and active
Xorg/GNOME processes. Chrome's correct Linux ANGLE flags are now
`--use-gl=angle --use-angle=gl`; the old `--use-gl=egl` combination was invalid. ANGLE OpenGL
then correctly asks for the default X display. Chromium's
[documented headless Ozone backend](https://chromium.googlesource.com/chromium/src/+/main/docs/ozone_overview.md#headless)
is software-only, and the attempted headless Vulkan path lacks a required platform extension.

The admissible hardware run connected benchmark Chromium to the existing desktop session with
`DISPLAY=:1` and the session's GDM Xauthority file after explicit permission was granted. The
renderer string and acceptance measurements are recorded in §5.3. No `sudo` operation or
library rewrite was required.

### B-3 No PDAL / no conda / no docker (worked around — see D-0.1)

### B-4 `micromamba` stalls on this machine

Two `micromamba create` attempts hung indefinitely after fetching conda-forge repodata
(430 MB), at ~1.5% CPU, no package downloads, one killed at 13 min and one at 5 min.
`micromamba search` responds instantly, so the network is fine and the solve/create step is
what hangs. Moot given the uv-only policy, but recorded so nobody retries it.

---

## 7. Manifest v1 field inventory (Phase 0 step 4)

These are the fields the spike **actually needed** to make its conversions reproducible and
trustworthy — grouped as the plan's `manifest.json` would carry them. Anything marked *(gap)*
does not exist anywhere in today's outputs and has to be produced by Phase 1 code, not by a
schema.

**Identity & producer**
`schema_version`; `run_id`; `capture_id`; `artifact_id`; `created_at`;
model name (`must3r`) and capability flags actually observed (`outputs_metric_scale: false`,
`outputs_poses: true`, `outputs_confidence: true`); model config that changes geometry
(`image_size: 512`, `window_size: 50`, `window_overlap: 10`, `num_mem_imgs: 30`,
`subsample: 2`, `confidence_thresholds: [5.0, 1.5]`); `git_commit` **and `git_status`**
(mp7's run was produced from a *dirty* tree — a manifest that records only the SHA would be
lying); adapter version *(gap)*; publisher version + tool name.

**Capture / source**
`video_name`, source path, `frames: 1694`, `fps: 10`, capture device/lens if known,
`initial_gps_coordinates`, `altitude`; telemetry sample counts actually extracted
(`gps_points: 3076`, `imu_samples: 34224`) and **where they were persisted** *(gap — currently
nowhere)*.

**Frame contract** (per the plan's `coordinate_frame` block)
`frame` (spike wrote `model_local`); `units` (`unknown`, not `metre`, for this run);
axis convention — needs to state the model's camera frame (x right, y down, z forward) or the
ENU frame after transform; `origin_wgs84`; `proj_pipeline`; `transform_to_ecef` (float64 4×4);
`alignment_status` ∈ {unaligned, approximate, aligned, reviewed}; `scale` applied;
`horizontal_rmse_m` / `vertical_rmse_m`; inlier count; GPS↔video clock offset + peak quality;
gravity constraint used — **all of the alignment quality fields are *(gap)***.

**Artifact entry (points)**
`kind: points`; `format: copc/laz`; `path`; `bytes`; `checksum` *(gap — not computed by the
spike)*; `point_count`; `point_format: 7`; `las_scale: [0.001,0.001,0.001]`;
`las_offset: [0,0,0]`; `bounds_min` / `bounds_max` in the artifact's frame;
`crs`/`wkt` (absent ⇒ local-scene mode); dimensions present with semantics —
`Color` (uint16, expanded from 8-bit source, `color_gain: 257`),
`PointSourceId` (uint16 = window index), `Intensity`/`Classification`/`GpsTime` written as
zero-filled and therefore *not meaningful*; `Confidence` *(gap)*.

**Octree/publication parameters** (needed to reproduce a byte-identical artifact)
`span: 128`; `max_depth`; `axis_bits`; `root_spacing`; `finest_cell_size`;
`max_node_points`; `total_nodes`; `leaf_nodes`; `levels_written`; sort determinism note
(`argsort kind=quicksort`, ties arbitrary).

**Lossy operations**
consolidation `method`, `requested_voxel_size`, `effective_voxel_size`,
`consolidation_octree_level`, `points_in`, `points_out`, `redundancy_factor`;
`nonfinite_points_dropped`; `nodes_exceeding_max_at_depth_cap`;
consolidation provenance policy (highest-confidence contributor + `contributor_count` per
plan ADR-007) *(gap — the spike keeps the first point in Morton order and has no confidence
to rank by)*.

**Provenance (`sources.json`)**
dimension used (`PointSourceId`); granularity (`window`); per window: `source_index`, `name`,
`point_count`, and from the window's own metadata `frame_start`/`frame_end`/`frame_indices`,
`window_size`, `window_overlap`; the assertion that made it trustworthy (Σ window points ==
merged point count).

**Validation gates that earned their place in Phase 0**
Σ window points == merged count; `points_written == points_in - dropped`;
`unplaced_points == 0`; LAS header count == Σ hierarchy node counts; root key exists; every
node has a reachable parent chain; reachable point count == header count; node coordinates fit
their level; bounds finite and non-degenerate; colour non-black after gain; first hierarchy VLR
is really the COPC info VLR.

**Publication metrics**
per-stage wall clock, total wall clock, peak RSS, input/output bytes, output bytes/point.

A concrete instance of most of this already exists as machine-readable output:
`/home/ape/mapper_output/phase0_spike/reports/mp7-{full,voxel2cm}.json`. Phase 1's
`schemas/manifest.v1.json` can be written by taking those reports, adding the *(gap)* fields,
and splitting artifact-level from run-level.

---

## 8. Exit verdict

**Phase 0 is complete. Accept Giro3D and carry it into Phase 2 behind the planned
renderer-neutral scene contract.** The selected starting calibration is a 2M point budget,
SSE 1 and source decode stride 2.

| Exit budget | Result | Verdict |
| --- | --- | --- |
| Cold first geometry <2.5 s | 0.765–0.769 s | Pass |
| Warm first geometry <1 s | The cold run itself is <1 s; a separate warm-cache distribution belongs in Phase 2 | Pass for renderer selection |
| Interaction p95 <20 ms | 17.0–19.6 ms across six paths | Pass |
| 1–3M visible points | ~2.02M at the representative orbit view | Pass |
| Initial geometry cache cap 256 MB | 196.3–197.8 MB in each decoded CPU/GPU pool | Pass, with both pools reported separately |
| No repeated >50 ms main-thread work while moving | Zero path long tasks in both calibrated runs | Pass |
| Static range-capable artifact serving | Correct byte-range behavior and immutable COPC files | Pass |

This decision does not waive the Phase 2 benchmark work. Cold/warm distributions, maximum
frame tails, loading-state sampling, network throttling, cancellation and a clearly separated
compressed-range/decoded-CPU/GPU memory budget still need durable automation. The fast local
server completed requests before the paths moved offscreen and recorded zero aborts; that is
not evidence that cancellation is absent.

---

## 9. Files added or changed this session

Workspace changes:

| Path | What |
| --- | --- |
| `pyproject.toml` | `[tool.pytest.ini_options] testpaths = ["tests"]` — collection no longer reaches `sandbox/` (`PYTHONPATH=. python -m pytest -q` → 1 skipped, clean) |
| `.gitignore` | excludes generated `node_modules/` and `dist/` |
| `spike/phase0/ply_to_copc.py` | PLY → COPC converter; corrected node-key decode + parent-reachability assertion (§4.1) |
| `spike/phase0/copc_stats.py` | independent COPC/LAS structure and root-reachability validator (§4.4) |
| `spike/phase0/range_server.py` | range-capable static server with per-request byte-range logging |
| `spike/phase0/baseline_deckgl.py` | deck.gl data-path baseline (§4.5) |
| `spike/phase0/giro3d-app/` | Vite + Giro3D 2.0.3 app; fixed early HUD/attribute wiring and source-index range; browser harness; reproducible `bench-data.mjs` launcher; vendored LAZ WASM |
| `docs/phase0_spike_findings.md` | this document |
| `spike/phase0/README.md` | runbook for every tool above |

Environment changes (not in `pyproject.toml`): `copclib`, `laspy`, `lazrs` in `.venv`
(see D-0.1); `spike/phase0/giro3d-app/node_modules` (declared npm dependencies including
esbuild); Playwright browsers in `~/.cache/ms-playwright`.

Scratch files left in `spike/phase0/giro3d-app/` from debugging B-1, safe to delete:
`dbg.mjs`, `minimal.mjs`, `minimal2.mjs`, `minimal3.mjs`, `minimal4.mjs`, `glprobe.mjs`.

The completed external benchmark record is under
`/home/ape/mapper_output/phase0_spike/bench/completed/`, including the calibrated
`gpu/{full,voxel}-stride2/` JSON reports and screenshots.

---

## 10. Things the plan should absorb regardless of the browser result

1. **Alignment quality is not merely unrecorded, the inputs are thrown away.** GPS/IMU are
   extracted per capture and never persisted, so today's artifacts cannot be re-aligned or
   audited without re-extracting telemetry. Phase 1 item (1)/(2) should include *writing*
   `telemetry/{gps,imu}.parquet`, not only computing an `AlignmentResult`.
2. **`is_metric: false` reaches the viewer today with no marker.** The COPC + no-WKT +
   `CoordinateSystem.unknown` chain gives the viewer a real signal to refuse geospatial mode;
   the manifest's `alignment_status` should be the authority and the artifact's missing CRS the
   corroboration.
3. **Redundancy at 0.01573 non-metric source units is 1.45×, not "dozens".** Re-scope the
   consolidation stage accordingly (§4.3); a metric run is required before choosing a physical
   default spacing.
4. **`git_status: dirty` is already in run configs.** Whatever the manifest records about
   provenance must carry the dirty flag, or reproducibility claims will be false for exactly
   the runs that matter.
5. **Publisher memory is the real constraint on this machine.** 12.9 GB peak to publish one
   147M-point capture in-RAM; the streaming design is not an optimisation, it is what makes
   the machine usable while a publish runs.
