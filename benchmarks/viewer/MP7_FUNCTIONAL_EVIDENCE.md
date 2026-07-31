# Canonical mp7 functional browser evidence

## 2026-07-26 — republished 146.9M-point package

This is **SwiftShader functional evidence only**. It is not GTX 1070 timing,
memory, visual source-color, trajectory-agreement, or hardware-picking
acceptance.

- Catalog: `/tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3`.
- Artifact: `art_4e05de949d7540108a56497d53017eb1`.
- Declared points: 146,911,634.
- Browser: Chrome `119.0.6045.199`.
- Renderer: ANGLE Vulkan, `SwiftShader Device (Subzero)`.

Start the services:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.api:create_app --factory \
  --host 127.0.0.1 --port 8000

VITE_BENCHMARK_ENABLED=true \
VITE_API_PROXY_TARGET=http://127.0.0.1:8000 \
  npm --prefix viewer/frontend run dev -- --host 127.0.0.1
```

Run the separate focused suite:

```bash
cd benchmarks/viewer
MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=functional \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_MP7_REPORT=/tmp/mapper-phase2-mp7-functional-report.json \
  npm run test:mp7
```

Result: **1 passed in 6.4 seconds**.

The run proved:

- overview issued zero `/api/v1/assets/` requests;
- the cache-disabled cold open issued 25 ranged COPC requests, including 206
  responses;
- the cache-enabled warm reopen issued 25 ranged COPC requests, including 206
  responses;
- Source color mode was enabled, explicitly selected, and retained;
- the trajectory emitted ready events for both cold and warm detail instances;
- the cold readiness sample displayed 41,857 points and the warm sample
  displayed 239,090 points, both under the 2,000,000 global budget;
- returning to overview disposed detail renderer generation 4 and cleared the
  active artifact list.

The raw focused report is
`/tmp/mapper-phase2-mp7-functional-report.json`. Its approximately 951 ms cold
and 592 ms warm first-geometry measurements are SwiftShader diagnostics only.
They must not be compared with or cited as satisfying the strict GTX 1070
thresholds, which remain in the standard seven-scenario acceptance suite.
