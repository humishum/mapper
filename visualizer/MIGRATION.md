# Migration from Old Visualizer

This document explains the changes from the old `pointcloud_visualizer.py` to the new modular architecture.

## What Changed?

### Old Architecture
- Single monolithic Python file (`pointcloud_visualizer.py`)
- Three.js embedded in Python strings
- HTML generated from Python
- Mixed concerns (data loading + HTML generation + JS code)
- Difficult to maintain and extend

### New Architecture
- **Modular Backend**: Separate Python modules for each concern
- **Separate Frontend**: Proper JavaScript project with Vite
- **Clean Separation**: Data processing vs visualization
- **Modern Stack**: deck.gl for better performance
- **Scalable**: Easy to extend and maintain

## File Mapping

| Old File | New Files |
|----------|-----------|
| `pointcloud_visualizer.py` (735 lines) | **Backend:** `data_loader.py`, `pointcloud_processor.py`, `gps_converter.py`, `export_data.py` |
| | **Frontend:** `main.js`, `PointcloudViewer.js`, `dataLoader.js`, `UIController.js` |

## Key Improvements

### 1. Modularity
**Before**: Everything in one file
```python
class SimplePointcloudVisualizer:
    def load_metadata(self): ...
    def find_pointcloud_file(self): ...
    def load_pointcloud_data(self): ...
    def _parse_ply_file(self): ...
    def gps_to_cartesian(self): ...
    def generate_html(self): ...
    def _create_html_template(self): ...  # 400+ lines of HTML/JS string
```

**After**: Clear separation
```python
# backend/data_loader.py
class DataLoader:
    def load_all_metadata(self): ...
    def find_pointcloud_file(self): ...

# backend/pointcloud_processor.py
class PointcloudProcessor:
    def load_ply(self): ...
    def _parse_ply_file(self): ...

# backend/gps_converter.py
class GPSConverter:
    def gps_to_cartesian(self): ...

# backend/export_data.py
class DataExporter:
    def export_to_json(self): ...
```

### 2. Better Visualization
**Before**: Three.js with manual camera controls
- Basic orbit controls
- No proper interaction handling
- Performance issues with many points

**After**: deck.gl with professional features
- Smooth orbit controller
- Better performance with large datasets
- Professional rendering quality
- Easier to extend

### 3. Development Experience
**Before**:
```bash
python pointcloud_visualizer.py --data-dir data/
# Generates HTML file
# Open in browser
# To make changes: edit Python, regenerate HTML, reload
```

**After**:
```bash
# One-time export
python -m backend.export_data --data-dir data/

# Development with hot reload
cd frontend && npm run dev
# Edit JS files → automatic reload in browser
```

### 4. Code Quality
**Before**:
- 735 lines in one file
- Embedded HTML/JS strings
- Hard to test
- Hard to debug

**After**:
- Multiple focused modules (100-200 lines each)
- Proper file separation
- Easy to test
- Easy to debug

## Migration Steps

If you have existing code using the old visualizer:

### 1. Export your data
```bash
# Old way
python pointcloud_visualizer.py --data-dir data/ --output viz.html

# New way
python -m backend.export_data --data-dir data/ --output frontend/public/data.json
```

### 2. View the visualization
```bash
# Old way
firefox viz.html

# New way
cd frontend && npm run dev
# Opens http://localhost:3000 automatically
```

### 3. Update any automation
```bash
# Old automation script
python pointcloud_visualizer.py --data-dir $DATA --threshold 2.0 --output result.html
cp result.html /var/www/html/

# New automation script
python -m backend.export_data --data-dir $DATA --threshold 2.0
cd frontend && npm run build
cp -r dist/* /var/www/html/
```

## Configuration Changes

### Threshold Selection
**Before**:
```bash
python pointcloud_visualizer.py --threshold 2.0
```

**After**:
```bash
python -m backend.export_data --threshold 2.0
```

### Max Points
**Before**: Hardcoded in Python (`MAX_POINTS = 50000`)

**After**: Command-line argument
```bash
python -m backend.export_data --max-points 100000
```

### Scale Factor
**Before**: Hardcoded (`SCALE_FACTOR = 1.0`)

**After**: Interactive slider in UI (adjust in real-time)

## API Changes

### Python API

**Before**:
```python
from pointcloud_visualizer import SimplePointcloudVisualizer

viz = SimplePointcloudVisualizer('data/')
viz.generate_html('output.html')
```

**After**:
```python
from visualizer.backend import DataExporter

exporter = DataExporter('data/', max_points=50000, sequence_id=1)
exporter.export_to_json('output.json', threshold=2.0)
```

### JavaScript API

**Before**: No JavaScript API (everything in HTML string)

**After**: Clean modular API
```javascript
import { PointcloudViewer } from './PointcloudViewer.js';
import { loadData } from './dataLoader.js';

const data = await loadData('/data.json');
const viewer = new PointcloudViewer('container', data);

// Programmatic control
viewer.setPointSize(5);
viewer.setScale(2.0);
viewer.setCameraPreset('top');
```

## What to Keep

The old `pointcloud_visualizer.py` can remain for backward compatibility if needed, but new development should use the modular architecture.

### Keep Old If:
- You have scripts depending on it
- You need the single-file HTML output
- You're not ready to migrate yet

### Use New If:
- Starting fresh
- Need better performance
- Want to customize visualization
- Need modern development workflow

## Deprecation Timeline

1. **Now**: Both old and new coexist
2. **After testing**: Recommend new architecture
3. **Future**: May deprecate old file

## Questions?

See:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `ARCHITECTURE.md` - Architecture details

