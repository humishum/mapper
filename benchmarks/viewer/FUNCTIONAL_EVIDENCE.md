# Phase 2 functional browser evidence

## 2026-07-26 — deterministic multisite fixture

This run is **functional evidence only**. It is not GTX 1070 performance,
geometry-memory, visual source-color, or hardware-picking acceptance.

- Browser: Google Chrome `119.0.6045.199`, headless.
- WebGL renderer: ANGLE Vulkan, `SwiftShader Device (Subzero)`.
- Evidence mode recorded in every report: `functional`.
- Fixture catalog:
  `/tmp/mapper-phase2-multisite-browser-v2/catalog.sqlite3`.
- Fixture geometry: 200,228 points per package with 15 exact
  `PointSourceId` records. Manifests declare `PointSourceId`,
  `ContributorCount`, and `Confidence`; each package includes
  `cameras/poses.parquet`.
- Raw reports: `/tmp/mapper-phase2-browser-reports-final/*.json`.

Services were started with:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-multisite-browser-v2/catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.api:create_app --factory \
  --host 127.0.0.1 --port 8000

VITE_BENCHMARK_ENABLED=true \
VITE_API_PROXY_TARGET=http://127.0.0.1:8000 \
  npm --prefix viewer/frontend run dev -- --host 127.0.0.1
```

The coherent seven-scenario run was:

```bash
cd benchmarks/viewer
MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=functional \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_BENCH_REPORT_DIR=/tmp/mapper-phase2-browser-reports-final \
  npm test
```

Result: **7 passed in 30.3 seconds**.

Functional assertions demonstrated:

- overview issued zero detailed asset requests;
- cold and warm detail opened through ranged COPC requests;
- orbit and dive interaction completed under the global point budget;
- two same-origin COPCs were active together under one two-million-point
  budget;
- site A → overview → site B used one document navigation;
- both trajectories became ready after their transitions;
- Source color mode was enabled, selected, and retained before inspection;
- point inspection resolved a non-empty source ID and capture record;
- inactive distant-site request count remained zero;
- renderer generations were disposed during mode/site transitions;
- throttled site-A supersession left only site B active, produced aborts, and
  did not commit stale site-A state.

Selected report observations are included only for diagnostics:

| Scenario | Visible points | CPU/GPU bytes | Aborted | Inactive-site |
| --- | ---: | ---: | ---: | ---: |
| cold first geometry | 5,805 | 104,526 / 104,526 | 1 | 0 |
| warm first geometry | 5,805 | 104,526 / 104,526 | 3 | 0 |
| comparison | 11,610 | 209,052 / 209,052 | 0 | 0 |
| site transition | 5,805 | 116,136 / 116,136 | 18 | 0 |
| throttled cancellation | 5,805 | 104,526 / 104,526 | 7 | 0 |

The recorded cold/warm timings were approximately 890 ms and 237 ms, but they
are SwiftShader observations and must not be compared with or cited as the
hardware acceptance thresholds. Frame-time, long-task, memory, and picking
quality gates remain pending on the GTX 1070 target.
