# Architecture Overview

This document describes the architecture of the 3D Pointcloud Visualizer.

## Design Principles

1. **Separation of Concerns**: Clear separation between data processing (Python) and visualization (JavaScript)
2. **Modularity**: Each component has a single, well-defined responsibility
3. **Scalability**: Can handle many locations and large pointclouds efficiently
4. **Performance**: Uses WebGL-based rendering for smooth interaction with millions of points
5. **Maintainability**: Clean code structure with minimal dependencies

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                            │
│  (Pointcloud PLY files + Metadata JSON in folders)         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Python Backend (Data Processing)              │
│                                                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ DataLoader   │  │ Pointcloud      │  │ GPS           │ │
│  │              │  │ Processor       │  │ Converter     │ │
│  │ - Find files │  │ - Parse PLY     │  │ - GPS to XYZ  │ │
│  │ - Load meta  │  │ - Downsample    │  │ - Cartesian   │ │
│  │ - Filter     │  │ - Normalize     │  │   projection  │ │
│  └──────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│         │                   │                    │          │
│         └───────────────────┼────────────────────┘          │
│                             ▼                                │
│                    ┌─────────────────┐                      │
│                    │  DataExporter   │                      │
│                    │  - Combine data │                      │
│                    │  - Export JSON  │                      │
│                    └────────┬────────┘                      │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼ data.json
┌─────────────────────────────────────────────────────────────┐
│              JavaScript Frontend (Visualization)             │
│                                                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ DataLoader   │  │ Pointcloud      │  │ UI            │ │
│  │              │  │ Viewer          │  │ Controller    │ │
│  │ - Fetch JSON │  │ - deck.gl setup │  │ - Controls    │ │
│  │ - Validate   │  │ - Layer mgmt    │  │ - Panels      │ │
│  │ - Calculate  │  │ - Camera        │  │ - Events      │ │
│  │   bounds     │  │ - Rendering     │  │               │ │
│  └──────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│         │                   │                    │          │
│         └───────────────────┼────────────────────┘          │
│                             ▼                                │
│                      ┌─────────────┐                        │
│                      │   Browser   │                        │
│                      │   (WebGL)   │                        │
│                      └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Backend Components

#### 1. DataLoader (`backend/data_loader.py`)
**Responsibility**: Find and load metadata and pointcloud files

**Key Methods**:
- `load_all_metadata()`: Scan directories and load metadata.json files
- `find_pointcloud_file()`: Locate PLY files matching threshold and sequence
- `get_available_thresholds()`: List available threshold values

**Dependencies**: Python stdlib (json, pathlib, re)

#### 2. PointcloudProcessor (`backend/pointcloud_processor.py`)
**Responsibility**: Parse and process PLY files

**Key Methods**:
- `load_ply()`: Load and process a PLY file
- `_parse_ply_file()`: Parse both binary and ASCII PLY formats
- `_parse_binary_ply()`: Handle binary PLY data
- `_parse_ascii_ply()`: Handle ASCII PLY data

**Features**:
- Automatic downsampling for performance
- Support for both binary and ASCII formats
- Color data extraction (RGB/RGBA)
- Bounding box calculation

**Dependencies**: NumPy

#### 3. GPSConverter (`backend/gps_converter.py`)
**Responsibility**: Convert GPS coordinates to Cartesian coordinates

**Key Methods**:
- `gps_to_cartesian()`: Convert lat/lon/alt to x/y/z
- `set_reference_point()`: Set the origin for relative positioning
- `batch_convert()`: Convert multiple coordinates

**Algorithm**: Equirectangular projection (suitable for small areas)

**Dependencies**: Python stdlib (math)

#### 4. DataExporter (`backend/export_data.py`)
**Responsibility**: Orchestrate data processing and export to JSON

**Key Methods**:
- `export_to_json()`: Main export function
- Combines all backend components
- Generates structured JSON output

**Output Format**:
```json
{
  "metadata": {
    "version": "1.0",
    "threshold": 2.0,
    "sequence_id": 1,
    "max_points_per_cloud": 50000,
    "total_locations": 3
  },
  "locations": [
    {
      "id": "location_name",
      "name": "location_name",
      "gps": { "lat": 37.5, "lon": -121.9, "alt": 0.0 },
      "position": { "x": 0, "y": 0, "z": 0 },
      "metadata": { "frames": 77, "video": "video.MOV", "fps": 10 },
      "pointcloud": {
        "points": [[x, y, z], ...],
        "colors": [[r, g, b], ...],
        "center": [cx, cy, cz],
        "bbox_min": [minx, miny, minz],
        "bbox_max": [maxx, maxy, maxz],
        "original_count": 7298401,
        "display_count": 50000,
        "file_name": "scene_thr2.0.ply"
      }
    }
  ]
}
```

### Frontend Components

#### 1. DataLoader (`frontend/src/dataLoader.js`)
**Responsibility**: Load and validate data from backend

**Key Functions**:
- `loadData()`: Fetch and parse JSON
- `calculateBounds()`: Calculate scene bounding box

**Dependencies**: Browser Fetch API

#### 2. PointcloudViewer (`frontend/src/PointcloudViewer.js`)
**Responsibility**: Main visualization engine

**Key Methods**:
- `constructor()`: Initialize deck.gl and scene
- `createPointcloudLayers()`: Create deck.gl layers
- `updateLayers()`: Refresh visualization
- `setPointSize()`, `setScale()`, `setOpacity()`: Visual controls
- `setCameraPreset()`: Camera positioning

**Rendering**:
- Uses deck.gl's `PointCloudLayer`
- WebGL-based rendering for performance
- Supports millions of points smoothly

**Dependencies**: deck.gl

#### 3. UIController (`frontend/src/UIController.js`)
**Responsibility**: Manage user interface and interactions

**Key Methods**:
- `initLocationsList()`: Populate location panel
- `initControls()`: Set up control event handlers

**UI Elements**:
- Point size slider
- Scale slider
- Opacity slider
- Visibility toggle
- Camera presets

**Dependencies**: DOM API

## Data Flow

### Export Phase (Python)

1. User runs `python -m backend.export_data`
2. DataLoader scans directories for metadata.json files
3. For each location:
   - Load GPS coordinates and metadata
   - Find appropriate PLY file (by threshold/sequence)
   - PointcloudProcessor loads and processes PLY
   - GPSConverter converts GPS to Cartesian
4. DataExporter combines all data
5. Write to `frontend/public/data.json`

### Visualization Phase (JavaScript)

1. User opens browser / runs `npm run dev`
2. DataLoader fetches data.json
3. PointcloudViewer initializes deck.gl
4. For each location with pointcloud:
   - Create PointCloudLayer
   - Position at GPS-derived coordinates
   - Apply colors and styling
5. UIController sets up controls
6. Render loop starts (deck.gl handles this)
7. User interactions update layers/camera

## Technology Stack

### Backend
- **Python 3.8+**: Main language
- **NumPy**: Efficient array operations for pointcloud data
- **Standard Library**: json, pathlib, re for file operations

### Frontend
- **Vite**: Modern build tool with fast HMR
- **deck.gl 8.9**: WebGL-based visualization framework
- **ES6 Modules**: Modern JavaScript module system
- **Vanilla JS**: No heavy frameworks (React, Vue, etc.)

### Why deck.gl?
- High performance WebGL rendering
- Built for large geospatial datasets
- Excellent pointcloud support
- Mature and well-maintained
- Good documentation
- Works great without React (unlike some alternatives)

## Performance Considerations

### Backend
1. **Downsampling**: Limit points per cloud (default 50k)
2. **Binary PLY**: Much faster to parse than ASCII
3. **Lazy Loading**: Only load requested threshold/sequence

### Frontend
1. **WebGL Rendering**: GPU-accelerated
2. **Efficient Data Structure**: Float32Array for positions/colors
3. **Layer Caching**: deck.gl caches layers when unchanged
4. **Optimized Bundle**: Vite code-splitting and tree-shaking

### Scalability Limits
- **Backend**: Can handle hundreds of locations, GBs of PLY files
- **Frontend**: Tested with 3-5 locations × 50k points = 150k-250k points
- **Practical Limit**: ~500k total points for smooth 60fps
- **Beyond That**: Increase downsampling or implement LOD

## Future Enhancements

### Short Term
- [ ] Error boundary for better error handling
- [ ] Loading progress bar
- [ ] Save/restore camera position
- [ ] Export screenshots

### Medium Term
- [ ] Multiple sequence support
- [ ] Dynamic threshold switching
- [ ] Point picking / inspection
- [ ] Measurement tools
- [ ] Animation/playback

### Long Term
- [ ] Real-time streaming (no export step)
- [ ] Server-side spatial indexing (octree)
- [ ] Level-of-detail (LOD) rendering
- [ ] Collaborative viewing
- [ ] VR/AR support

## Development Workflow

### Adding Features

1. **Backend Changes**:
   - Modify appropriate module
   - Update export_data.py if needed
   - Test with actual data
   - Re-export JSON

2. **Frontend Changes**:
   - Modify appropriate module
   - Hot reload updates automatically
   - Test in browser
   - No re-export needed (unless data format changed)

### Testing

```bash
# Backend
cd visualizer
source ../.venv/bin/activate
python -m backend.export_data --data-dir ../data/output_100425

# Frontend
cd frontend
npm run dev
# Open http://localhost:3000 in browser
```

### Building for Production

```bash
cd frontend
npm run build
# Outputs to dist/
# Serve with: python -m http.server 8000 --directory dist
```

## File Organization

```
visualizer/
├── backend/                 # Python data processing
│   ├── __init__.py         # Package exports
│   ├── data_loader.py      # Find and load files
│   ├── pointcloud_processor.py  # PLY parsing
│   ├── gps_converter.py    # GPS conversion
│   ├── export_data.py      # Main export script
│   └── cli.py              # Command-line interface
│
├── frontend/               # JavaScript visualization
│   ├── src/
│   │   ├── main.js        # Entry point
│   │   ├── dataLoader.js  # Data loading
│   │   ├── PointcloudViewer.js  # Main viewer
│   │   └── UIController.js      # UI management
│   ├── public/            # Static assets
│   │   └── data.json      # Generated data
│   ├── index.html         # HTML entry
│   ├── package.json       # Dependencies
│   ├── vite.config.js     # Build config
│   └── .nvmrc            # Node version
│
├── scripts/               # Helper scripts
│   ├── export_data.sh    # Export wrapper
│   └── dev.sh            # Dev server wrapper
│
├── README.md             # Full documentation
├── QUICKSTART.md         # Quick start guide
└── ARCHITECTURE.md       # This file
```

## Design Decisions

### Why JSON instead of direct PLY loading?

**Pros**:
- Simpler frontend code
- Pre-processed data is faster to load
- Can apply backend optimizations once
- Easy to cache and CDN

**Cons**:
- Two-step process (export then visualize)
- Large JSON files
- Can't dynamically change parameters

**Decision**: JSON for now, but designed for future streaming

### Why separate Python/JavaScript instead of Python-only (e.g., PyVista)?

**Pros**:
- Web-based = accessible anywhere
- Better performance (WebGL)
- Modern UI possibilities
- Easier to share (just send a link)

**Cons**:
- More complex setup
- Two languages to maintain

**Decision**: Web-based for flexibility and accessibility

### Why deck.gl instead of Three.js?

**Pros**:
- Built for geospatial data
- Excellent large dataset performance
- Clean API for pointclouds
- Less boilerplate than Three.js

**Cons**:
- Larger bundle size
- More opinionated

**Decision**: deck.gl is perfect for this use case

## Conclusion

This architecture provides a clean, maintainable, and scalable solution for visualizing multiple 3D pointcloud reconstructions. The separation between data processing and visualization allows each component to be optimized independently, and the modular design makes it easy to extend and customize.

