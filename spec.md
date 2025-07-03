# Mapper: The Modern Cartography Project 
--- 

## Setup 
``` pip install -r requirements.txt ```



##  Usage
  Run the full pipeline:
  python -m pipeline.cli run path/to/video.mp4

  Or run individual stages:
  python -m pipeline.cli stage demux path/to/video.mp4
  python -m pipeline.cli stage filter data/frames_tmp/[video_id]/frames
  python -m pipeline.cli stage sfm data/frames_tmp/[video_id]/frames_filtered

  Check system status:
  python -m pipeline.cli status
# Proposal

# Hiking 3‑D Reconstruction System – Local Pipeline Architecture & Technical Specification

**Version:** 1.0   **Date:** 30 May 2025

---

## 1 Purpose & Scope

Design a fully local, drop‑folder–driven pipeline that converts GoPro/phone hike videos (with embedded GPS) into geo‑referenced 3‑D point‑cloud tiles viewable in a lightweight browser front‑end.  The system must be:

* **Modular** – each stage independently replaceable/testable.
* **Deterministic & debuggable** – clear inputs, outputs, logs.
* **Python‑first back‑end**, **JavaScript front‑end** (React + CesiumJS).
* **Offline‑friendly** – no cloud dependencies; runs on a single workstation.

---

## 2 High‑Level Architecture

```mermaid
graph TD
  subgraph Pipeline (Python)
    A[Watcher] --> B[Demux & GPS Extract]
    B --> C[Frame Filter]
    C --> D[SfM / MVS (COLMAP)]
    D --> E[Geo‑Align & Merge]
    E --> F[Tile Converter (Potree→3DTiles)]
    F --> G[Site Generator]
  end
  subgraph Front‑End (JavaScript)
    H[Static Site
(React + Cesium)]
  end
  G -->|manifest.json + tiles| H
```

---

## 3 Folder Layout

```
repo/
├── pipeline/
│   ├── watcher.py
│   ├── demux.py
│   ├── filter.py
│   ├── sfm.py
│   ├── geoalign.py
│   ├── convert.py
│   ├── sitegen.py
│   └── config.yaml
├── videos_new/          # Watched drop folder
├── data/
│   ├── frames_tmp/
│   ├── colmap_ws/
│   ├── ept_master/      # Entwine Point Tiles store
│   └── tiles_out/       # 3D Tiles ready for web
├── web/
│   ├── public/
│   │   └── 3d/          # tiles copied here
│   └── src/             # React front‑end
└── requirements.txt
```

---

## 4 Module Specifications

### 4.1 Watcher (`watcher.py`)

| Aspect             | Spec                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| **Responsibility** | Monitor `videos_new/` for new `*.mp4`/`*.mov` files and enqueue pipeline run. |
| **Tech**           | `watchdog` (Python ≥3.10).                                                    |
| **Interface**      | Emits a `VideoJob` JSON onto an internal queue (`multiprocessing.Queue`).     |
| **Config keys**    | `watch_path`, `pattern`, `skip_existing_on_start`.                            |

### 4.2 Demux & GPS Extract (`demux.py`)

* **Input:** path to video file.
* **Process:**

  1. `ffmpeg` → JPEG frames (`frames_tmp/<vid_id>/frame_%07d.jpg`).
  2. `exiftool` or `gopro2gpx` → `gps.csv` (timestamp, lat, lon, alt).
* **Output:** frame folder + CSV path written to `demux.json` metadata.
* **Notes:** frame timestamps aligned via filename index.

### 4.3 Frame Filter (`filter.py`)

* Drop blurry/redudant frames using **variance of Laplacian** & simple time‑based stride.
* CLI args: `--blur-thresh`, `--fps-target`.

### 4.4 SfM / MVS (`sfm.py`)

| Item       | Detail                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| Engine     | **COLMAP 3.9** run via CLI wrappers.                                    |
| Mode       | Sequential matcher + `--SiftMatching.use_gpu true`.                     |
| GPS priors | Pass `gps.csv` to COLMAP `--ImportPath` & similarity transform scripts. |
| Output     | `[vid_id].db`, sparse/ dense models, `model.ply`.                       |

### 4.5 Geo‑Align & Merge (`geoalign.py`)

* Converts COLMAP camera centers to **ECEF (EPSG:4978)** using `pyproj`.
* Runs Iterative‑Closest‑Point (Open3D) against `ept_master` for drift correction (optional flag `--icp`).
* Writes LAS 1.4 file with WGS‑84 coords; appends to **Entwine** dataset:
  `pdal pipeline entwine_append.json`.

### 4.6 Tile Converter (`convert.py`)

* **Input:** updated `ept_master` directory.
* **Tool:** `potreeconverter` v2 (`--generate-page false --output-format 3dtiles`).
* **Output:** tiles in `tiles_out/tileset.json`.
* **Runtime:** incremental – only retiles new points (`--incremental`).

### 4.7 Site Generator (`sitegen.py`)

* Copies tiles to `web/public/3d/<timestamp>/`.
* Updates `manifest.json`:

  ```json
  [{
    "title": "Hike 2025‑05‑28",
    "path": "3d/2025‑05‑28/tileset.json",
    "polyline": "gpx/2025‑05‑28.gpx"
  }, …]
  ```
* Touches `<web>/public/reload.txt` to trigger hot‑reload in dev server.

---

## 5 Configuration (`config.yaml`)

```yaml
paths:
  watch: "videos_new"
  frames: "data/frames_tmp"
  colmap: "data/colmap_ws"
  ept: "data/ept_master"
  tiles: "data/tiles_out"
  web: "web/public/3d"

colmap:
  matcher: sequential
  max_num_matches: 20
  gps_noise_m: 2.5        # σ for bundle adj.

filter:
  blur_thresh: 100.0
  fps_target: 5
```

---

## 6 Logging & Debugging

* Standard **Python `logging`**; each module gets its own logger.
* Log file per job: `logs/<vid_id>.log` plus stream to console.
* `--debug` flag forces retention of intermediate folders; otherwise cleaned after success.
* `pipeline/cli.py status` shows job queue + last 50 lines of each log.

---

## 7 Testing Strategy

| Layer       | Tool                                         | What is tested                                      |
| ----------- | -------------------------------------------- | --------------------------------------------------- |
| Unit        | `pytest` + fixtures                          | Each module with stub inputs.                       |
| Integration | `pytest` + `subprocess`                      | End‑to‑end on a 30‑sec sample video (\~200 MB max). |
| E2E manual  | Run `npm run dev` and load tiles in browser. |                                                     |

---

## 8 Dependency Versions

```
Python 3.10+
ffmpeg 6.0
exiftool 12.7+
COLMAP 3.9 (CUDA optional)
PDAL 2.6
Entwine 2.2
PotreeConverter 2.1
Open3D 0.18
pyproj 3.6
watchdog 4.0
React 18 (Vite)
CesiumJS 1.118
```

*Lock versions via `requirements.txt` and `package.json`.*

---

## 9 Build & Run

```bash
# 1 Back‑end env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2 Front‑end
cd web && npm i && npm run dev &

# 3 Start pipeline (terminal 2)
python -m pipeline.cli run --config pipeline/config.yaml

# 4 Drop video
mv ~/Downloads/video.MP4 videos_new/
```

Browser at [http://localhost:5173](http://localhost:5173) will hot‑reload when tiles appear.

---

## 10 Interfaces & Data Contracts

* **VideoJob JSON** – enqueued by Watcher → consumed by Demux.
* **demux.json** – list of frame files & gps.csv path.
* **model\_info.json** – SfM outputs: scale factor, cameras, stats.
* **entwine‐info.json** – metadata checksum for incremental tiling.
* **manifest.json** – consumed by React front‑end.

All JSON schemas kept in `schema/` and validated with `pydantic`.

---

## 11 Front‑End Blueprint (React)

```tsx
// App.tsx (pseudo)
import { Viewer } from "cesium";
import hikes from "../manifest.json";

export default function App() {
  const [selected, setSelected] = useState(hikes[0]);
  return (
    <Layout>
      <Sidebar hikes={hikes} onSelect={setSelected} />
      <CesiumViewer tilesetUrl={selected.path} gpx={selected.polyline} />
    </Layout>
  );
}
```

* **CesiumViewer** loads 3D Tiles, overlays GPX polyline, provides measurement & fly‑through tools.
* **Sidebar** lists hikes; uses simple Tailwind UI.

---

## 12 Extensibility Hooks

| Future need      | Hook                                                                   |
| ---------------- | ---------------------------------------------------------------------- |
| Cloud processing | Swap Watcher → S3 event trigger; same module interface.                |
| Nerf/Mesh        | Add new `nerf.py` module; outputs go through Geo‑Align step unchanged. |
| Multi‑user site  | Replace `sitegen.py` with GitHub Actions deploy.                       |

---

## 13 Troubleshooting Checklist

1. **No GPS?** – Check `gps.csv`; run `exiftool -ee -p "$GPSLatitude,$GPSLongitude" <file>`.
2. **COLMAP fails matching** – increase `max_num_matches`; ensure frames are in focus.
3. **Viewer blank** – open DevTools, verify tileset URL loads (200).  Check CORS headers if serving via local HTTP.

---

## 14 Appendix – CLI Command Cheat‑Sheet

```bash
# Single‑video test without watcher
demux.py --video path.mp4 --out tmp
filter.py --frames tmp/frames --blur-thresh 80 --fps 4
sfm.py --workspace colmap_ws --gps tmp/gps.csv
geoalign.py --model colmap_ws --ept data/ept_master
convert.py --ept data/ept_master --out data/tiles_out
sitegen.py --tiles data/tiles_out --gpx tmp/gps.gpx
```

## 15 Phased Implementation Roadmap

| Phase                            | Focus                   | Key Tasks                                                                                                                                                                         | Deliverables                                               | Verification / Gate Criteria                         |
| -------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| **0 Bootstrap**                  | Local dev env           | • Install Python/Node toolchains<br>• Clone repo, create `.venv`, `npm i`<br>• Verify `ffmpeg`, `colmap`, `pdal` on `$PATH`                                                       | Working `make doctor` script prints ✔︎ for all deps        | All checks green in terminal                         |
| **1 Video I/O MVP**              | Ingestion only          | • Implement `watcher.py` + CLI stub<br>• `demux.py` extracts JPEGs + rough FPS filter<br>• Unit tests (`pytest -k demux`)                                                         | Folder `data/frames_tmp/…` populated from sample 10‑s clip | · 100 % test pass<br>· Manual visual check of frames |
| **2 Single‑method 3‑D Core**     | SfM (COLMAP)            | • `sfm.py --engine colmap` end‑to‑end<br>• Skip GPS, skip tiling—just output `model.ply`<br>• Minimal desktop viewer: auto‑open MeshLab                                           | `model.ply` looks roughly correct                          | · Mesh visible<br>· Pipeline exits 0                 |
| **3 Geo‑Alignment**              | Scale + WGS‑84          | • `geoalign.py` converts to ECEF LAS<br>• Adds ICP refinement flag<br>• Save LAS to `ept_master/`                                                                                 | LAS opens in CloudCompare at right lat/lon                 | · Coordinates within 5 m of GPX                      |
| **4 Basic Web Viewer**           | Static Tiles            | • `convert.py` → Potree 3D Tiles<br>• `sitegen.py` writes `manifest.json`<br>• React viewer loads tiles & polyline                                                                | Browser shows scene; camera centres align with basemap     | · Tile downloads <50 MB<br>· 30 FPS on desktop       |
| **5 Pluggable Vid→3D Interface** | Multi‑backend           | • Abstract `Reconstructor` base‑class (`abc`)<br>• Register `ColmapReconstructor`, add stub `NerfstudioReconstructor`<br>• `sfm.py` selects via `--engine`                        | Unit tests cover factory pattern                           | · `pytest -k reconstructor` pass                     |
| **6 Performance & UX**           | Incremental & UI polish | • Incremental tiling (`--incremental`)<br>• React UI: measurement tool, fly‑through, progress bar from websockets<br>• Wrap CLI in `pipeline/cli.py` with `run/status/clean` cmds | Live reload + job dashboard                                | · End‑to‑end demo w/ two hikes <10 min total compute |

### Iteration Rhythm

1. **Implement** phase branch ➜ PR.
2. **Run** sample clip (`tests/data/sample.mp4`).
3. **Review** logs & viewer; fix issues.
4. **Merge** to `main` after gate passes.

Each phase is expected to take **\~1 week** of part‑time work; earlier phases often complete in a single evening.

---

**End of Document**
