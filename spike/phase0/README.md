# Phase 0 spike tooling

Throwaway scaffolding for the Phase 0 spike in
[`docs/visualization_platform_plan.md`](../../docs/visualization_platform_plan.md).
Findings, benchmarks and remaining work live in
[`docs/phase0_spike_findings.md`](../../docs/phase0_spike_findings.md) — read that first.

**This is not product code.** It answers questions; it is meant to be deleted once Phase 1's
publisher exists. In particular `ply_to_copc.py` sorts the whole cloud in RAM (12.9 GB peak for
mp7), which is exactly what the plan forbids for the real streaming publisher.

Python deps used here are installed in the repo `.venv` but deliberately **not** in
`pyproject.toml`:

```bash
uv pip install --python .venv copclib laspy lazrs
```

Node deps are local to `giro3d-app/` (`npm install` there).

---

## 1. `ply_to_copc.py` — PLY → COPC with metrics

Converts a reconstruction PLY to COPC/LAZ in the model's local frame: LAS scale 0.001 /
offset 0, point format 7, 8-bit colour expanded to 16-bit (×257), window provenance in
`PointSourceId`, and a JSON metrics report next to the output.

```bash
MP7=/home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/outputs/mp7
OUT=/home/ape/mapper_output/phase0_spike

# full fidelity (every input point kept)
.venv/bin/python spike/phase0/ply_to_copc.py \
  --input  $MP7/aligned_pointcloud.ply \
  --output $OUT/artifacts/mp7-full.copc.laz \
  --max-depth 10 \
  --source-index-dir $MP7/windows \
  --assume-non-metric \
  --report $OUT/reports/mp7-full.json

# nominal 0.02 consolidated (effective 0.01573 source units; this fixture is non-metric)
.venv/bin/python spike/phase0/ply_to_copc.py \
  --input  $MP7/aligned_pointcloud.ply \
  --output $OUT/artifacts/mp7-voxel2cm.copc.laz \
  --voxel 0.02 \
  --source-index-dir $MP7/windows \
  --assume-non-metric \
  --report $OUT/reports/mp7-voxel2cm.json

# small fixture for fast iteration (one window, 2.87M points, ~2 s)
.venv/bin/python spike/phase0/ply_to_copc.py \
  --input  $MP7/windows/window_000/pointcloud.ply \
  --output $OUT/artifacts/window000.copc.laz \
  --report $OUT/reports/window000.json
```

Useful flags: `--limit N` (first N points only), `--max-node-points` (leaf threshold, default
100k), `--max-depth`, `--voxel`, `--chunk`, `--color-bits {8,16}`, `--wkt` (omit for a local
frame — omitting it is what puts the viewer in local-scene mode).

The redundancy curve was measured with `--voxel 0.05` and `--voxel 0.10`. Because this
fixture is non-metric and consolidation uses the octree grid, their effective source-unit
spacings are 0.03145 and 0.06290 respectively; see the findings rather than interpreting the
requested values as physical centimetres.

## 2. `copc_stats.py` — verify the result with a second reader

Stdlib only, shares no code with the writer: LAS header, COPC info VLR, every hierarchy page,
per-level node/point/byte counts.

```bash
.venv/bin/python spike/phase0/copc_stats.py \
  /home/ape/mapper_output/phase0_spike/artifacts/mp7-full.copc.laz \
  --json /home/ape/mapper_output/phase0_spike/reports/mp7-full.copcstats.json
```

Check both `header_point_count_matches_hierarchy: true` and
`all_nodes_reachable_from_root: true`. Matching counts alone can still hide orphaned nodes that
a renderer cannot traverse. `root_node_points/bytes` is the first-geometry payload.

A third opinion, if wanted — `laspy` reads COPC natively:

```python
from laspy import CopcReader
with CopcReader.open(path) as r:
    coarse = r.query(level=0)          # root-level sample only
    print(len(coarse), r.copc_info.spacing, coarse.point_source_id[:5])
```

## 3. `range_server.py` — range-capable local server

`python -m http.server` ignores `Range:` and would send the whole 1 GB file, so it cannot be
used to evaluate COPC streaming. This one answers `206 Partial Content`, sets CORS headers
copc.js needs, and logs every served range to JSONL (that log is how "bytes ranged" and
request counts are measured server-side, including client aborts).

```bash
.venv/bin/python spike/phase0/range_server.py \
  --root /home/ape/mapper_output/phase0_spike/artifacts \
  --port 8123 \
  --log  /home/ape/mapper_output/phase0_spike/reports/range-requests.jsonl

# verify
curl -s -r 0-99 -o /dev/null -D - http://127.0.0.1:8123/mp7-full.copc.laz | head -6
```

## 4. `baseline_deckgl.py` — the "before" record

Times the existing `viewer/backend/data_service.py` path (PLY read → random downsample →
flat-earth GPS transform → hex-in-JSON) on the same fixture. Nothing is modified; that path
gets no further engineering per the plan.

```bash
.venv/bin/python spike/phase0/baseline_deckgl.py \
  --ply /home/ape/mapper_output/must3r_011026/kings_canyon_must3r_20260110_202624/outputs/mp7/aligned_pointcloud.ply \
  --report /home/ape/mapper_output/phase0_spike/reports/baseline-deckgl-mp7.json
```

## 5. `giro3d-app/` — Giro3D scene, HUD and harnesses

```bash
cd spike/phase0/giro3d-app
npm install          # giro3d 2.0.3, three 0.180, ol, proj4, vite, playwright
npx vite --host 127.0.0.1 --port 5180
```

Open with an artifact URL (needs the range server from §3 running):

```text
http://127.0.0.1:5180/?url=http://127.0.0.1:8123/mp7-voxel2cm.copc.laz&budget=2000000&sse=1
```

Query params: `url`, `budget` (point budget), `sse` (subdivision threshold), `stride`
(`COPCSource` decode decimation, default 2), `size` (point size, 0 = auto), `attr`
(`Color` | `Z` | `PointSourceId` | `Intensity`), `sourceMax` (highest `PointSourceId`, 42 for
mp7), `up` (`-y` default for the MUSt3R camera frame, or `z`), and `autopath` (run a camera
path on load). Use `stride=1` only when full point decode is required; it exceeded the Phase 0
geometry-cache target on the two large artifacts.

The page exposes `window.__spike` for harnesses: `marks` (entityReady, metadata,
firstGeometry, idle), `snapshot()` (displayed points, decimation, draw calls, cpu/gpu geometry
memory, last/p95 frame time), `runPath(name, ms)` for the `orbit` / `dive` / `traverse` paths,
`waitForIdle()`, and `report()` for everything at once. The HUD shows the same numbers live and
displays the local-scene-mode warning when the artifact declares no CRS.

### 5a. `bench.mjs` — browser benchmark driver

```bash
cd spike/phase0/giro3d-app
npm run bench:browser -- \
  --out /home/ape/mapper_output/phase0_spike/bench \
  --gl swiftshader

# Run from this machine's active desktop session for real NVIDIA numbers.
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
npm run bench:browser -- \
  --out /home/ape/mapper_output/phase0_spike/bench \
  --gl egl \
  --only mp7-full-budget2m-stride2
```

Per scenario: fresh cache, load, three camera paths, screenshots, app marks, renderer counters,
long tasks, JS heap, and the server-side range log. Results land in
`bench/results-<gl>.json`.

Environment caveats baked into the script: it launches the browser itself
with `--remote-debugging-port` and attaches via `connectOverCDP` (Playwright's pipe transport
does not connect here), uses the default browser context (`newContext()` hangs over CDP), and
defaults to `--chrome /home/ape/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`
(Playwright's bundled 1194 builds fail to navigate at all). The sandbox hides the host display
and GPU devices; run the `egl` pass from the active desktop session (or explicitly grant the
benchmark access to that session). Frame times from `--gl swiftshader` are software
rasterisation and must not be read as the interaction budget. Chromium's headless Ozone backend
is software-only, so a real GPU pass requires the X11/Wayland session. On this machine the
accepted GTX 1070 calibration is `budget=2000000&sse=1&stride=2`; exact results are in the
findings and `bench/completed/gpu/`.

### 5b. `traverse.mjs` — GPU-free data-path harness (runs here)

Drives the real `COPCSource`: initialize → metadata → hierarchy → screen-space-error node
selection at three camera distances → `getNodeData()` per node, timing decodes and counting
server bytes. Measures IO + decode, not rendering.

Giro3D cannot be imported by Node directly (extensionless imports), so `bench-data.mjs` bundles
the harness with declared `esbuild` and supplies the required Node shims:

```bash
cd spike/phase0/giro3d-app
npm run bench:data -- \
  --artifacts http://127.0.0.1:8123 \
  --out /home/ape/mapper_output/phase0_spike/bench/traverse.json
```

The completed run and measurements are recorded in the findings.

---

## Scratch to delete

`giro3d-app/dbg.mjs`, `minimal.mjs`, `minimal2.mjs`, `minimal3.mjs`, `minimal4.mjs`,
`glprobe.mjs` — one-off probes from diagnosing the browser blocker (B-1). Kept only so the
diagnosis can be re-checked; nothing depends on them.
