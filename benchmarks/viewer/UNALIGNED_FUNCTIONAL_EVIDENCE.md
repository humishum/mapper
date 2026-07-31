# Fresh unaligned-scene functional evidence

## 2026-07-26 — copied Phase 1 fresh catalog

This is focused **functional evidence only**. The source package directories
were external and read-only. The API used only the disposable catalog copy:

```text
/tmp/mapper-phase2-fresh-catalog.sqlite3
```

Artifacts:

- `art_ec9f614413b54073b2e12ef0ffda17ec`: 200,228-point VGGT scene.
- `art_78a0f4e44d7a464db78dd95b6d9fe86c`: 11,207,020-point MUSt3R scene.

Start the services:

```bash
MAPPER_CATALOG_PATH=/tmp/mapper-phase2-fresh-catalog.sqlite3 \
  .venv/bin/uvicorn viewer.backend.api:create_app --factory \
  --host 127.0.0.1 --port 8000

VITE_BENCHMARK_ENABLED=true \
VITE_API_PROXY_TARGET=http://127.0.0.1:8000 \
  npm --prefix viewer/frontend run dev -- --host 127.0.0.1
```

Run the separate functional spec:

```bash
cd benchmarks/viewer
MAPPER_BENCH_EXTERNAL_FRONTEND=1 \
MAPPER_BENCH_EVIDENCE_MODE=functional \
MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
MAPPER_UNALIGNED_REPORT=/tmp/mapper-phase2-unaligned-functional-report.json \
  npm run test:unaligned
```

Result: **1 passed in 4.7 seconds** using Chrome `119.0.6045.199` and
SwiftShader.

The test proved:

- both artifacts appeared under **Unaligned local scenes**, displayed their
  exact point counts and `Local only`, and were absent from Geographic sites;
- overview issued zero `/api/v1/assets/` requests;
- both manifests reported `alignment.status=unaligned`,
  `rejection_reason=gps_unavailable`, and null `origin_wgs84`,
  `transform_to_ecef`, and footprint;
- detail showed `Local coordinates only: gps_unavailable` for each scene;
- small → overview → large completed with one document navigation;
- the small scene displayed 5,805 points and the large scene displayed 3,286
  points at the sampled readiness point, each under the shared 2,000,000 budget.

The raw focused report is
`/tmp/mapper-phase2-unaligned-functional-report.json`. These SwiftShader
observations do not constitute GTX 1070 performance, memory, or visual
acceptance.
