# 3D Pointcloud Visualizer

A modular and scalable 3D pointcloud visualizer that loads reconstructed scenes from different locations and displays them in an interactive 3D environment using deck.gl.

## Architecture

The visualizer is split into two main components:

### Backend (Python)
- **Data Loader**: Reads metadata and finds pointcloud files
- **Pointcloud Processor**: Parses PLY files and processes point data
- **GPS Converter**: Converts GPS coordinates to Cartesian coordinates
- **Data Exporter**: Exports all data to JSON for frontend consumption

### Frontend (JavaScript)
- **Vite**: Modern build tool for fast development
- **deck.gl**: High-performance WebGL-based visualization library
- **Modular Architecture**: Separate modules for data loading, rendering, and UI

## Directory Structure

```
visualizer/
├── backend/
│   ├── __init__.py
│   ├── data_loader.py           # Load metadata and find PLY files
│   ├── pointcloud_processor.py  # Parse and process PLY files
│   ├── gps_converter.py         # GPS to Cartesian conversion
│   ├── export_data.py           # Export data to JSON
│   └── cli.py                   # Command-line interface
├── frontend/
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.js           # Vite configuration
│   ├── index.html               # HTML entry point
│   ├── public/                  # Static assets (data.json goes here)
│   └── src/
│       ├── main.js              # Application entry point
│       ├── PointcloudViewer.js  # Main viewer class
│       ├── dataLoader.js        # Data loading utilities
│       └── UIController.js      # UI interaction handling
└── scripts/
    ├── export_data.sh           # Export data from Python
    └── dev.sh                   # Start development server
```

## Setup

### Prerequisites

- Python 3.8+ with virtual environment
- Node.js 18+ and npm
- uv (Python package installer)

### Installation

1. **Install JavaScript dependencies:**

```bash
cd visualizer/frontend
npm install
```

2. **Python dependencies should already be in your virtual environment**

## Usage

### Step 1: Export Data

First, export your pointcloud data to JSON format:

```bash
cd visualizer
source ../.venv/bin/activate

# Export with default settings (threshold=2.0, sequence_id=1)
python -m backend.export_data --data-dir ../data/output_100425

# Or with custom settings
python -m backend.export_data \
  --data-dir ../data/output_100425 \
  --output frontend/public/data.json \
  --threshold 2.0 \
  --max-points 50000 \
  --sequence-id 1
```

**Options:**
- `--data-dir`: Directory containing pointcloud data folders (required)
- `--output`: Output JSON file path (default: frontend/public/data.json)
- `--threshold`: Threshold for pointcloud selection (default: 2.0)
- `--max-points`: Maximum points per pointcloud (default: 50000)
- `--sequence-id`: Sequence ID to use (default: 1)

### Step 2: Start Development Server

```bash
cd visualizer/frontend
npm run dev
```

This will:
- Start the Vite development server on http://localhost:3000
- Enable hot module reloading
- Open your browser automatically

### Step 3: Build for Production

```bash
cd visualizer/frontend
npm run build
```

The built files will be in `frontend/dist/`. You can serve them with any static file server.

## Features

### Interactive Controls

- **Point Size**: Adjust the size of individual points (1-20 pixels)
- **Scale**: Scale all pointclouds together (0.1x - 5x)
- **Opacity**: Control point transparency (0.1 - 1.0)
- **Show/Hide**: Toggle pointcloud visibility
- **Camera Presets**: Quick views (Overview, Top, Side, Front)

### Camera Controls

- **Rotate**: Click and drag to rotate the camera around the scene
- **Zoom**: Scroll to zoom in/out
- **Pan**: Right-click and drag to pan

### Location Panel

Shows all loaded locations with:
- Location name
- GPS coordinates and altitude
- Number of points loaded
- Video source information
- Visual indicators for data availability

## Configuration

### Backend Configuration

Edit `backend/export_data.py` to customize:
- Point downsampling strategy
- GPS conversion parameters
- Data export format

### Frontend Configuration

Edit `frontend/vite.config.js` to customize:
- Server port
- Build output directory
- Module optimization

Edit `frontend/src/PointcloudViewer.js` to customize:
- Initial camera position
- Rendering parameters
- Point visualization style

## Development

### Backend Development

The backend is organized into modular components:

```python
from visualizer.backend import DataLoader, PointcloudProcessor, GPSConverter

# Load metadata
loader = DataLoader('data/output_100425')
metadata = loader.load_all_metadata()

# Process pointclouds
processor = PointcloudProcessor(max_points=50000)
pc_data = processor.load_ply(ply_file_path)

# Convert GPS coordinates
gps_converter = GPSConverter()
x, y, z = gps_converter.gps_to_cartesian(lat, lon, alt)
```

### Frontend Development

The frontend uses ES6 modules:

```javascript
import { PointcloudViewer } from './PointcloudViewer.js';
import { loadData } from './dataLoader.js';

const data = await loadData('/public/data.json');
const viewer = new PointcloudViewer('container-id', data);
viewer.setPointSize(5);
```

## Troubleshooting

### "Failed to load data" error

Make sure you've run the export script first:
```bash
python -m backend.export_data --data-dir ../data/output_100425
```

### Points not visible

1. Check that your data has valid pointcloud files
2. Try adjusting the point size slider
3. Check the location panel to see which locations have data

### Performance issues

1. Reduce `--max-points` when exporting data
2. Increase the threshold value to get sparser pointclouds
3. Adjust opacity to reduce overdraw

## Technology Stack

- **Backend**: Python 3, NumPy
- **Frontend**: Vite, deck.gl, WebGL
- **Data Format**: JSON, PLY (binary and ASCII)

## Future Enhancements

- [ ] Support for multiple sequence selection
- [ ] Real-time data loading (no export step)
- [ ] Advanced filtering and selection tools
- [ ] Measurement tools
- [ ] Animation and playback features
- [ ] Export capabilities (screenshots, video)
- [ ] Multiple colorization schemes
- [ ] Point cloud comparison mode

## License

This is part of the mapper project.
