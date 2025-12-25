# Quick Start & Troubleshooting Guide

Get the 3D Pointcloud Viewer running in 5 minutes.

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm

### Step 1: Install Backend Dependencies

```bash
cd viewer
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3: Start Backend

In one terminal:

```bash
cd viewer
source venv/bin/activate
python -m backend.server
```

The backend will start on http://localhost:8000

### Step 4: Start Frontend

In another terminal:

```bash
cd viewer/frontend
npm run dev
```

The frontend will start on http://localhost:5173

### Step 5: Open in Browser

Open http://localhost:5173 in your browser and you should see:
- A map with orange/green markers for each location
- Click a marker to load that pointcloud (turns green when loaded)
- Use the panels on the sides to control rendering
- Open browser console (F12) to see detailed logging

### Alternative: Use the Start Script

```bash
cd viewer
./start.sh
```

This will start both backend and frontend in one command.

## Using the Viewer

### Controls
- **Left drag:** Rotate camera
- **Scroll:** Zoom in/out
- **Right drag:** Pan camera
- **Point Size slider:** Adjust point size (1-10px)
- **Opacity slider:** Control transparency
- **Camera Presets:** Quick views (Overview, Top Down, Side View)

### Loading Pointclouds
1. Check a location in the left panel to load its pointcloud
2. Click "Fly To" to navigate to that location
3. Uncheck to unload and free memory
4. Load multiple locations simultaneously to compare

### Tips
- Use zoom level 16-18 for best pointcloud viewing
- Increase point size to 3-5px for better visibility
- Watch browser console for detailed logging (F12)

---

## Troubleshooting

### Issue 1: Node.js Version Error

**Symptom:** `Vite requires Node.js version 20.19+ or 22.12+`

**Solution:** You're using Node 18. Either:
- Upgrade Node to v20: `nvm install 20 && nvm use 20`
- Or the package.json has been updated with compatible versions - just reinstall:
  ```bash
  cd viewer/frontend
  rm -rf node_modules package-lock.json
  npm install
  ```

### Issue 2: Backend Won't Start

**Symptoms:**
- Port 8000 already in use
- Module not found errors
- Data directory doesn't exist

**Solutions:**

**A. Port in use:**
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

**B. Module errors:**
```bash
cd viewer
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

**C. Data directory:**
```bash
# Verify data exists
ls ~/mapper_output/122225-must3r
# If not, update viewer/backend/config.py with correct path
```

### Issue 3: Frontend Won't Start

**Symptoms:**
- Port 5173 already in use
- npm errors
- Module not found

**Solutions:**

**A. Port in use:**
```bash
lsof -i :5173
kill -9 <PID>
```

**B. Clean reinstall:**
```bash
cd viewer/frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue 4: WebGL Errors (`maxTextureDimension2D`)

**Symptom:** Console shows `TypeError: Cannot read properties of undefined (reading 'maxTextureDimension2D')`

**Solution:** This is generally harmless and can be ignored. The error occurs during initial WebGL context creation but doesn't affect functionality. If it causes problems:
1. Try a different browser (Chrome/Firefox recommended)
2. Update your graphics drivers
3. Check if hardware acceleration is enabled in browser settings

### Issue 5: Pointcloud Appears Too Small or Aliased

**Symptoms:**
- Points appear in straight lines (aliasing pattern)
- Pointcloud is tiny compared to world view
- Hard to see individual points

**Causes & Solutions:**

**A. Uniform Downsampling (Fixed)**
- The backend now uses random sampling instead of uniform stride
- Restart the backend to apply changes:
  ```bash
  # Stop backend (Ctrl+C)
  cd viewer
  source venv/bin/activate
  python -m backend.server
  ```

**B. Coordinate System Issues**
The PLY files contain local Cartesian coordinates that need proper GPS transformation.

**Current approach:** Points are transformed from local XYZ to GPS lat/lon/alt using flat-earth approximation.

**To improve:**
1. Check if PLY coordinates are in meters or other units
2. Verify the GPS origin point is correct in metadata.json
3. Consider the scale - if points span 100m in real life, they should span ~0.001° in GPS coords

**C. Point Size & View Settings**
- Adjust point size in Render Controls panel (try 3-5px)
- Zoom in closer (zoom level 16-18)
- Increase opacity if points are faint
- Check console logs for coordinate values

### Issue 6: Fly To Button Doesn't Work

**Symptom:** Clicking "Fly To" does nothing

**Cause:** Invalid GPS coordinates (0.0, 0.0)

**Solution:** 
1. Open browser console (F12) and look for `[App] Flying to location`
2. If you see "Invalid coordinates", that location has no GPS data in metadata.json
3. Some locations in your dataset have (0, 0) coordinates - these need GPS data added

**Check which locations have invalid coords:**
```bash
cd ~/mapper_output/122225-must3r
for dir in */; do
  coords=$(jq -r '.initial_gps_coordinates' "$dir/metadata.json" 2>/dev/null)
  if [[ "$coords" == *"0,0"* ]] || [[ "$coords" == *"0.0"* ]]; then
    echo "Invalid/missing coords in: $dir"
  fi
done
```

### Issue 7: Points Not Visible After Loading

**Debugging steps:**
1. Open browser console (F12)
2. Load a pointcloud and look for these log messages:
   ```
   [App] Loading pointcloud for <name>
   [API] Fetching pointcloud for <name>...
   [API] Parsed <N> points
   [MapView] Rendering pointcloud with <N> points
   ```
3. Check the sample point coordinates - they should be near the location's GPS coords

**Common issues:**
- **Points at (0, 0, 0):** Coordinate transformation failed - check metadata
- **Points very far from camera:** Zoom in or use "Fly To" button
- **All points behind terrain:** Increase altitude offset in coordinate transformation
- **Wrong color (black points on dark map):** Check if PLY file has color data

### Issue 8: Flat Earth / No 3D Terrain

**Current state:** Using 2D OpenStreetMap tiles (flat map)

**Why:** 3D terrain adds complexity and performance overhead. For now, pointclouds are positioned at their GPS altitude above a flat map.

**To add 3D terrain in the future:**

Option 1: Mapbox Terrain (requires free API key)
```javascript
import { TerrainLayer } from '@deck.gl/geo-layers';

// Add to layers in MapView.jsx
new TerrainLayer({
  elevationData: 'https://api.mapbox.com/v4/mapbox.terrain-rgb/...',
  texture: 'https://api.mapbox.com/v4/mapbox.satellite/...',
  elevationDecoder: {
    rScaler: 6553.6,
    gScaler: 25.6,
    bScaler: 0.1,
    offset: -10000
  }
})
```

Option 2: Free terrain tiles
```javascript
new TerrainLayer({
  elevationData: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  elevationDecoder: {
    rScaler: 256,
    gScaler: 1,
    bScaler: 1 / 256,
    offset: -32768
  }
})
```

### Issue 9: Slow Loading / Performance

**Symptoms:** 
- Takes >5 seconds to load a pointcloud
- Browser becomes sluggish
- High memory usage

**Solutions:**

**A. Reduce points per location:**
Edit `App.jsx` and change `maxPoints`:
```javascript
maxPoints: 50000,  // Instead of 100000
```

**B. Use lower threshold PLY files:**
```javascript
threshold: 5.0,  // Instead of 2.0 - fewer points, lower quality
```

**C. Check backend performance:**
- Look at backend terminal for processing times
- Verify caching is working (should see "Using cached pointcloud" on reload)
- First load is slower (reads PLY from disk), subsequent loads use cache

**D. Unload pointclouds when done:**
- Uncheck locations in the panel to free memory
- Frontend keeps all loaded pointclouds until manually unloaded

**E. Browser optimization:**
- Close unnecessary tabs
- Use Chrome/Firefox (better WebGL performance)
- Check GPU acceleration is enabled in browser settings

### Issue 10: CORS Errors

**Symptom:** Console shows "CORS policy" errors when fetching from API

**Cause:** Backend CORS is misconfigured or not running

**Solution:**
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check CORS is enabled in `backend/server.py` (it should be)
3. Ensure frontend is using correct API URL in `.env`:
   ```
   VITE_API_URL=http://localhost:8000
   ```

## Debugging Tips

### Enable Verbose Logging

All components now log with prefixes - open console (F12):
- `[App]` - Main application state
- `[MapView]` - Rendering and layers
- `[API]` - Network requests and parsing
- Backend logs appear in the terminal

### Check Backend Health

```bash
# Health check
curl http://localhost:8000/health

# List all locations
curl http://localhost:8000/api/locations | jq

# Test loading a specific pointcloud (small sample)
curl "http://localhost:8000/api/pointcloud/bridge_crossing?max_points=1000" | jq '.num_points'
```

### Browser Performance Analysis

1. Open DevTools (F12)
2. Go to **Performance** tab
3. Click Record (red circle)
4. Load a pointcloud
5. Stop recording and analyze

Look for:
- Long parsing times (hex → binary conversion)
- GPU rendering bottlenecks  
- Memory allocation spikes
- Frame rate drops

### Memory Usage

Press F12 → **Memory** tab → Take heap snapshot
- Look for large ArrayBuffers (pointcloud data)
- Each 100k point cloud ≈ 1.5MB positions + 300KB colors
- Multiple loaded pointclouds add up quickly

## Getting Help

When reporting issues, please include:
1. **Browser console logs** (F12 → Console, especially [App], [API], [MapView] messages)
2. **Backend terminal output** (the terminal where you ran `python -m backend.server`)
3. **Screenshots** if visual issues
4. **Steps to reproduce** the problem
5. **Which location/pointcloud** causes the issue
6. **Browser and OS** (Chrome/Firefox/Safari, Windows/Mac/Linux)

## Next Steps

Once everything is working:
- Adjust point size and opacity for best visualization
- Load multiple pointclouds to compare different locations
- Try camera presets for different viewing angles
- Explore locations with valid GPS coordinates
- Check TROUBLESHOOTING section for advanced configuration

