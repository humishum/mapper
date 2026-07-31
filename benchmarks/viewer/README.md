# Phase 2 browser benchmark harness

This Playwright/CDP harness measures the production viewer against the Phase 2
resource, isolation, cancellation, and transition gates. It records one JSON
report per scenario in `results/`; generated reports are intentionally ignored
by Git.

## Data and services

Build the deterministic distant-site fixture:

```bash
python -m benchmarks.viewer.fixtures \
  /path/to/valid-local.copc.laz \
  /tmp/mapper-phase2-sites \
  --summary /tmp/mapper-phase2-sites/fixture.json
```

Start the catalog API using the generated catalog:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-sites/catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.server:app --host 127.0.0.1 --port 8000
```

The managed Playwright configuration builds an optimized production artifact
with benchmark instrumentation enabled, then serves it through Vite preview.
Both the development and preview servers proxy `/api` and `/health` to the
configured API. The managed build disables the external basemap by default so
tile work cannot contaminate COPC measurements; explicitly set
`VITE_BASEMAP_ENABLED=true` only when basemap traffic is part of the intended
run. To use an already-running frontend, set
`MAPPER_BENCH_EXTERNAL_FRONTEND=1`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAPPER_BENCH_BASE_URL` | `http://127.0.0.1:5173` | Frontend URL |
| `MAPPER_BENCH_API_TARGET` | `http://127.0.0.1:8000` | Vite API proxy target |
| `MAPPER_BENCH_SITE_A` | `artifact_phase2_site_a` | First distant fixture artifact |
| `MAPPER_BENCH_SITE_B` | `artifact_phase2_site_b` | Second distant fixture artifact |
| `MAPPER_BENCH_COMPARE_ARTIFACT_ID` | `artifact_phase2_site_a_comparison` | Second COPC sharing site A's origin |
| `MAPPER_BENCH_REPORT_DIR` | `benchmarks/viewer/results` | JSON report directory |
| `MAPPER_BENCH_EXTERNAL_FRONTEND` | unset | Set to `1` to skip managed Vite startup |
| `MAPPER_BENCH_BROWSER_EXECUTABLE` | Playwright Chromium | Optional explicit Chrome/Chromium executable path |
| `MAPPER_BENCH_EVIDENCE_MODE` | `acceptance` | Set to `functional` for software-rendered integration evidence without GPU timing/memory gates |
| `VITE_BASEMAP_ENABLED` | `false` for managed benchmarks | Explicitly enable external basemap traffic |

The distant synthetic pair intentionally cannot be loaded simultaneously in
detail mode because the viewer enforces its 10 km shared-origin limit. The
fixture's third package supplies the same-site comparison artifact, so all
seven scenarios run with defaults. Override its ID only when benchmarking a
different same-site package.

## Commands

From `viewer/frontend`:

```bash
npm run benchmark:validate
npm run benchmark:list
npx playwright install chromium
npm run benchmark
```

For a preinstalled system browser, skip the Playwright browser download:

```bash
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome npm run benchmark
```

Acceptance evidence must use an optimized production build. The managed path
does this automatically and refuses to reuse an existing process on port 5173,
preventing an accidental Vite development/React StrictMode run. If
`MAPPER_BENCH_EXTERNAL_FRONTEND=1` is used for acceptance, build and serve the
same instrumented artifact explicitly:

```bash
VITE_BENCHMARK_ENABLED=true \
VITE_BASEMAP_ENABLED=false \
VITE_API_PROXY_TARGET=http://127.0.0.1:8000 \
  npm run build
npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
```

Do not use `npm run dev` to collect timing, memory, or long-task acceptance
evidence.

For software-rendered functional integration evidence:

```bash
MAPPER_BENCH_EVIDENCE_MODE=functional \
  MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
  npm run benchmark
```

Functional reports identify `configuration.evidenceMode` as `functional` and
still record timings, frame/long-task samples, and observed memory. The suite
does not assert hardware-dependent timing, memory, or long-task gates in this
mode. Functional picking demonstrates the interaction path, but is not hardware
picking acceptance. Functional reports must never be presented as GTX 1070
performance acceptance. The deprecated `MAPPER_BENCH_FUNCTIONAL_ONLY=1`
spelling remains an alias.

Acceptance mode records the unmasked WebGL renderer in every standard report
and rejects SwiftShader, other known software rasterizers, and unavailable
WebGL contexts. Run acceptance in a headed session attached to the target GPU;
the JSON report remains on disk when this renderer gate rejects a run.

### Real fresh unaligned scenes

The standard seven-scenario suite remains limited to deterministic aligned
fixtures. A separate spec exercises the copied Phase 1 fresh catalog without
mutating its external packages:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-fresh-catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.api:create_app --factory \
  --host 127.0.0.1 --port 8000

MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=functional \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_UNALIGNED_REPORT=/tmp/mapper-phase2-unaligned-functional-report.json \
  npm run test:unaligned
```

Use `npm run list:unaligned` to list this focused suite. Its default artifact
IDs are the fresh 200,228-point VGGT scene and 11,207,020-point MUSt3R scene;
`MAPPER_UNALIGNED_SMALL_ARTIFACT` and
`MAPPER_UNALIGNED_LARGE_ARTIFACT` can override them. See
`UNALIGNED_FUNCTIONAL_EVIDENCE.md` for the recorded run.

### Canonical republished mp7

A second separate functional spec exercises the canonical 146,911,634-point
mp7 package without changing the standard seven-scenario thresholds:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.api:create_app --factory \
  --host 127.0.0.1 --port 8000

MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=functional \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_MP7_REPORT=/tmp/mapper-phase2-mp7-functional-report.json \
MAPPER_MP7_SCREENSHOT=/tmp/mapper-phase2-mp7-source-colored.png \
  npm run test:mp7
```

Use `npm run list:mp7` to list the focused test.
`MAPPER_MP7_ARTIFACT` can override its canonical artifact ID. The test asserts
overview isolation, cold/warm ranged COPC traffic, source and trajectory
availability, a real post-reopen point pick with source/capture resolution,
point budgeting, detail disposal, and a source-colored screenshot. It also
records orbit/dive frame and long-task samples plus raw CPU/GPU geometry bytes.
Timings, memory, and the hardware-renderer gate are recorded but not asserted
in functional mode. See `MP7_FUNCTIONAL_EVIDENCE.md`.

For strict GTX 1070 acceptance, use a headed Chrome process on the hardware
display (after confirming ports 5173 and 8000 are free and starting the API and
frontend separately):

```bash
DISPLAY=:1 \
XAUTHORITY=/path/to/the/display/Xauthority \
MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=acceptance \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_MP7_REPORT=/tmp/mapper-phase2-mp7-acceptance-report.json \
MAPPER_MP7_SCREENSHOT=/tmp/mapper-phase2-mp7-source-colored.png \
  npm run test:mp7 -- --headed
```

This focused acceptance run requires hardware WebGL, cold/warm first geometry
at most 2.5 s/1 s, orbit and dive p95 at most 20 ms, at most one long task over
50 ms in each steady-state sample, CPU and GPU geometry at most 256 MiB each,
and at most 2,000,000 visible points. It also requires trajectory readiness to
survive the reopen and interactions, source mode after reopen, and a resolved
source/capture from a real point pick.

`benchmark:validate` and `benchmark:list` require neither a browser nor GPU.
The full run requires a functioning Chromium installation, live API, valid
COPC data, and hardware WebGL. NVIDIA acceptance evidence must be gathered on
the GTX 1070 target; software rendering is not valid performance evidence.

## Scenarios and evidence

The serial suite covers:

- cold and warm first geometry, with CDP cache control;
- steady-state orbit and dive;
- two-COPC same-site comparison;
- overview → site A → overview → site B without document reload, followed by
  geometry/trajectory readiness and point picking/source inspection;
- throttled site-A supersession by site B.

Every standard report follows `report.schema.json` and includes Git commit,
Chromium version, machine/CPU and WebGL renderer identity, performance marks,
HTTP range traffic, failures,
frame samples and p95, long tasks, visible points, CPU/GPU geometry memory, and
renderer disposal/active-artifact state.

The suite asserts:

- overview issues zero `/api/v1/assets/` requests;
- an inactive distant site receives zero COPC requests;
- at most 2,000,000 visible points;
- CPU and GPU geometry use at most 256 MiB each;
- no repeated long tasks over 50 ms;
- orbit/dive frame-time p95 at most 20 ms;
- cold/warm first geometry at most 2.5 s/1 s;
- site transitions use one document navigation;
- superseded requests abort, cannot commit stale state, and cannot successfully
  complete against an inactive disposed site.

Keep raw reports, Playwright traces, browser console output, GPU driver
identity, and the fixture summary together when recording acceptance evidence.
