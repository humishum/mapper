# Phase 2 GTX 1070 hardware acceptance

Date: 2026-07-26  
Result: **PASS**

This is hardware acceptance evidence for the current Phase 2 worktree. The
JSON reports identify repository HEAD
`705f4b0bbea4b51aaffb49b8d3527b14786c31c2`; the Phase 2 implementation was
still uncommitted when the runs were made.

## Reference environment

- Host: `blade`
- CPU: Intel Core i5-9600K, 6 logical CPUs
- Memory: 16,706,269,184 bytes
- GPU: NVIDIA GeForce GTX 1070, 8 GB
- Driver: 560.35.03
- Browser: Google Chrome 119.0.6045.199
- WebGL:
  `ANGLE (NVIDIA Corporation, NVIDIA GeForce GTX 1070/PCIe/SSE2, OpenGL 4.5.0)`
- Display: X11 `:1`, 2560×1440 at 59.95 Hz
- Viewer: production Vite preview, basemap disabled, one Playwright worker

Acceptance mode rejects SwiftShader, llvmpipe, unavailable WebGL, and other
software-renderer identities. The reference display's DPMS state must remain
on during measurement: an off monitor makes Chromium intentionally pace
animation frames at 1 Hz. The final commands therefore disable DPMS only for
the run, force the display on, and restore DPMS in the exit trap.

Continuous Playwright trace screenshots and video are disabled during
performance runs because synchronous WebGL canvas capture perturbed this
machine by one to two seconds. They remain opt-in with
`MAPPER_BENCH_DIAGNOSTIC_RECORDING=1`. Scenario JSON, browser identity,
performance entries, network records, and scenario-owned screenshots remain
enabled.

## Standard seven-scenario suite

Command, after starting the package API on port 8000 with the deterministic
multi-site catalog:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  MAPPER_BENCH_EVIDENCE_MODE=acceptance \
  MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
  MAPPER_BENCH_REPORT_DIR=/tmp/mapper-phase2-gtx1070-standard-v15 \
  npm --prefix benchmarks/viewer test -- --headed
```

Result: **7/7 passed in 40.6 seconds**. All seven reports validate against
`report.schema.json`.

| Scenario | Relevant result |
| --- | --- |
| Cold first geometry | 576.4 ms; inactive-site requests 0 |
| Warm first geometry | 79.5 ms |
| Orbit steady state | 16.68 ms p95; 0 long tasks over 50 ms |
| Dive steady state | 16.68 ms p95; 0 long tasks over 50 ms |
| Two-COPC comparison | 11,610 visible points; 209,052 CPU/GPU bytes |
| A → overview → B | one navigation; stable trajectory and resolved source pick |
| Throttled supersession | 2 aborted requests; no stale disposed-state commit |

The largest standard steady-state geometry observation was 62,029 visible
points and 1,116,543 bytes in each CPU/GPU geometry pool. Overview and inactive
distant sites issued zero detailed asset requests.

## Canonical 146.9M-point mp7

The API used the disposable, independently validated republished package at
`/tmp/mapper-phase2-mp7.NNlC2W/catalog.sqlite3`.

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  MAPPER_BENCH_EVIDENCE_MODE=acceptance \
  MAPPER_BENCH_BROWSER_EXECUTABLE=/usr/bin/google-chrome \
  MAPPER_MP7_REPORT=/tmp/mapper-phase2-gtx1070-mp7-v4.json \
  MAPPER_MP7_SCREENSHOT=/tmp/mapper-phase2-gtx1070-mp7-v4.png \
  npm --prefix benchmarks/viewer run test:mp7 -- --headed
```

Result: **1/1 passed in 16.6 seconds**.

| Gate | Observed | Budget |
| --- | ---: | ---: |
| Cold first geometry | 643.8 ms | ≤2,500 ms |
| Warm first geometry | 184.1 ms | ≤1,000 ms |
| Orbit p95 | 16.68 ms | ≤20 ms |
| Dive p95 | 16.68 ms | ≤20 ms |
| Orbit/dive long tasks over 50 ms | 0 / 0 | ≤1 / ≤1 |
| Steady visible points | 309,925 | ≤2,000,000 |
| Steady CPU geometry | 4,738,688 bytes | ≤268,435,456 bytes |
| Steady GPU geometry | 4,738,688 bytes | ≤268,435,456 bytes |
| Overview asset requests | 0 | 0 |

The test also proved cache-disabled and cache-enabled ranged COPC traffic,
trajectory readiness before and after the transition, renderer disposal,
stable geometry after orbit/dive, Source mode after reopen, and a real Giro3D
pick resolving Source 0 and capture
`cap_2a31903de58f5c4ebecf3b3b1ffeadfe`.

The final full-page screenshot was inspected at original resolution. It shows
distinct high-contrast source regions, the magenta camera trajectory, 309,925
visible points, the two-million-point budget, and the inspection panel with
node/index, local coordinate, source kind, capture, run, frame range,
ContributorCount=1, missing confidence, and unaligned status. This visual
acceptance is the gate that permitted deletion of the Dash folder visualizer.

The same strict headed scenario was rerun after promotion directly against
the live five-artifact catalog and persistent `mp7` package. It passed
**1/1 in 16.3 seconds** with the GTX 1070 renderer: 641.9 ms cold, 483.9 ms
warm, 16.68 ms orbit/dive p95, zero steady-state long tasks over 50 ms,
309,925 visible points, 4,607,129 bytes in each CPU/GPU geometry pool, and
zero overview asset requests. The inspected screenshot shows all five catalog
scenes together with Source coloring, the trajectory, and resolved provenance.
The report and screenshot are
`/tmp/mapper-phase2-gtx1070-mp7-persistent-final.json` and
`/tmp/mapper-phase2-gtx1070-mp7-persistent-final.png`.

## Real fresh unaligned packages

A headed `acceptance`-mode run against the disposable catalog copy passed
**1/1 in 12.5 seconds**. The test rejects software WebGL and recorded the same
GTX 1070 renderer as the standard and `mp7` suites. It opened the fresh
200,228-point VGGT and 11,207,020-point MUSt3R packages, retained local-only
placement and `gps_unavailable` warnings, kept origin/ECEF/footprint null,
made zero overview asset requests, and used one document navigation across
both detail transitions.

## Acceptance conclusion

The GTX 1070 first-geometry, moving-view, long-task, visible-point, CPU/GPU
pool, inactive-request, transition, trajectory, source-coloring, and picking
gates all pass. Functional SwiftShader reports remain useful diagnostics but
are not part of this conclusion.
