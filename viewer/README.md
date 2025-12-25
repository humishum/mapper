# 3D Pointcloud Map Viewer

A web-native pointcloud viewer that displays 3D reconstruction data on an interactive world map using deck.gl.

## Overview

This viewer allows you to:
- View a world map with markers showing pointcloud data locations
- Click on markers to load and display 3D pointclouds
- Navigate and explore multiple pointclouds simultaneously
- Control point size, rendering options, and camera views

## Architecture

- **Backend (Python)**: FastAPI server that loads PLY files, downsamples points, and transforms coordinates
- **Frontend (React + deck.gl)**: Interactive map with WebGL rendering of pointclouds

## Quick Start

### 1. Setup Backend

```bash
cd viewer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and adjust if needed:
```bash
DATA_DIR=/home/ape/mapper_output/122225-must3r
PORT=8000
```

### 3. Start Backend

```bash
cd viewer
source venv/bin/activate
python -m backend.server
```

The API will be available at http://localhost:8000

### 4. Setup Frontend

```bash
cd viewer/frontend
npm install
```

### 5. Start Frontend

```bash
cd viewer/frontend
npm run dev
```

The app will be available at http://localhost:5173

## API Endpoints

- `GET /api/locations` - Get all available locations with GPS coordinates
- `GET /api/pointcloud/{location_name}` - Load pointcloud data for a location
  - Query params: `sequence`, `threshold`, `max_points`, `use_gps_coords`

## Development

### Backend Structure

```
backend/
├── server.py              # FastAPI application
├── data_service.py        # Pointcloud loading and processing
├── coordinate_transform.py # GPS coordinate transformations
├── config.py              # Configuration management
└── __init__.py
```

### Frontend Structure

```
frontend/
└── src/
    ├── App.jsx            # Main application
    ├── components/        # React components
    ├── services/          # API clients
    └── utils/             # Utility functions
```

## Technology Stack

### Backend
- FastAPI - Web framework
- plyfile - PLY file parsing
- NumPy - Point processing
- uvicorn - ASGI server

### Frontend
- React - UI framework
- deck.gl - WebGL visualization
- Vite - Build tool
- MapLibre GL - Base map

## Performance

- Points are downsampled on the backend (default: 100k points per location)
- Results are cached in memory
- deck.gl provides hardware-accelerated rendering
- Frustum culling automatically hides off-screen points

## Future Enhancements

- Progressive loading (low-res first, then full resolution)
- Distance-based LOD
- Redis caching for production
- Point picking and measurements
- Animation and playback

## License

Part of the mapper project.

