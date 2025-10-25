#!/usr/bin/env python3
"""
Simple 3D Pointcloud Visualizer

A lightweight pointcloud visualizer that loads pointclouds from folders,
positions them using GPS coordinates, and displays them in a 3D scene.
Uses only essential libraries for simplicity and scalability.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import webbrowser
import tempfile
import base64
from io import BytesIO
import math

# Global configuration
THRESHOLD = 2.0  # Global threshold for pointcloud selection
MAX_POINTS = 50000  # Maximum points to display per pointcloud
SCALE_FACTOR = 1.0  # Global scale factor for all pointclouds


class SimplePointcloudVisualizer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.metadata_files = []
        self.pointcloud_data = {}
        
    def load_metadata(self) -> List[Dict]:
        """Load all metadata files and extract GPS coordinates and altitude"""
        print("Loading metadata files...")
        
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                metadata_file = subdir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        gps_coords = metadata.get('initial_gps_coordinates', [0.0, 0.0])
                        altitude = metadata.get('altitude', 0.0)
                        
                        # Only include locations with actual GPS data (not 0,0)
                        if gps_coords != [0.0, 0.0]:
                            metadata['folder_name'] = subdir.name
                            metadata['gps_coords'] = gps_coords
                            metadata['altitude'] = altitude
                            self.metadata_files.append(metadata)
                            print(f"Loaded: {subdir.name} at {gps_coords}, altitude: {altitude}")
                    except Exception as e:
                        print(f"Error loading {metadata_file}: {e}")
        
        return self.metadata_files
    
    def find_pointcloud_file(self, folder_name: str, threshold: float) -> Optional[Path]:
        """Find the pointcloud file for a given threshold in the first sequence"""
        folder_path = self.data_dir / folder_name / "pointclouds"
        
        # First try to find in sequence_1 folder
        sequence_1_path = folder_path / "sequence_1"
        if sequence_1_path.exists():
            ply_files = list(sequence_1_path.glob("*.ply"))
            for ply_file in ply_files:
                if f"thr{threshold}" in ply_file.name:
                    return ply_file
        
        # If not found in sequence_1, try the main pointclouds folder
        ply_files = list(folder_path.glob("*.ply"))
        for ply_file in ply_files:
            if f"thr{threshold}" in ply_file.name:
                return ply_file
        
        # If still not found, try to find the closest threshold
        all_ply_files = []
        for ply_file in ply_files:
            match = re.search(r"thr([0-9.]+)", ply_file.name)
            if match:
                file_threshold = float(match.group(1))
                all_ply_files.append((file_threshold, ply_file))
        
        if all_ply_files:
            # Sort by threshold and find closest
            all_ply_files.sort(key=lambda x: x[0])
            closest_threshold, closest_file = min(all_ply_files, 
                                                key=lambda x: abs(x[0] - threshold))
            print(f"Using closest threshold {closest_threshold} for {folder_name}")
            return closest_file
        
        return None
    
    def load_pointcloud_data(self, folder_name: str, threshold: float = None) -> Optional[Dict]:
        """Load and process point cloud data for a specific folder and threshold"""
        if threshold is None:
            threshold = THRESHOLD
            
        folder_path = self.data_dir / folder_name
        pointcloud_file = self.find_pointcloud_file(folder_name, threshold)
        
        if not pointcloud_file:
            print(f"No pointcloud file found for {folder_name} with threshold {threshold}")
            return None
        
        print(f"Loading point cloud from {pointcloud_file}...")
        
        try:
            # Simple PLY file parser (without open3d dependency)
            points, colors = self._parse_ply_file(pointcloud_file)
            
            if points.shape[0] == 0:
                print(f"No points found in {pointcloud_file}")
                return None
            
            # Downsample if too many points
            if points.shape[0] > MAX_POINTS:
                indices = np.linspace(0, points.shape[0]-1, MAX_POINTS, dtype=int)
                points = points[indices]
                if colors is not None and hasattr(colors, 'shape') and colors.shape[0] > 0:
                    colors = colors[indices]
            
            # Scale points to reasonable size (meters) but keep them centered around origin
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                scale_factor = 10.0 / max_dist  # Scale to 10m max dimension
                points_scaled = points * scale_factor
            else:
                points_scaled = points
                scale_factor = 1.0
                
            return {
                'points': points_scaled.tolist(),
                'colors': colors.tolist() if colors is not None and hasattr(colors, 'shape') and colors.shape[0] > 0 else None,
                'center': [0, 0, 0],  # Points are already centered
                'scale_factor': scale_factor,
                'original_count': points.shape[0],
                'display_count': points_scaled.shape[0],
                'file_path': str(pointcloud_file)
            }
            
        except Exception as e:
            print(f"Error loading pointcloud {pointcloud_file}: {e}")
            return None
    
    def _parse_ply_file(self, file_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Simple PLY file parser that returns points and colors"""
        points = []
        colors = []
        
        with open(file_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # Parse header to get vertex count
            vertex_count = 0
            has_colors = False
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif 'red' in line or 'green' in line or 'blue' in line:
                    has_colors = True
            
            # Read binary data
            if 'binary' in ' '.join(header_lines):
                # Binary format
                data = f.read()
                
                if has_colors:
                    # Check if alpha channel is present
                    has_alpha = 'alpha' in ' '.join(header_lines)
                    
                    if has_alpha:
                        # 3 floats for position + 4 uchar for color (RGBA)
                        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])
                    else:
                        # 3 floats for position + 3 uchar for color (RGB)
                        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
                    
                    data_array = np.frombuffer(data, dtype=dtype)
                    points = np.column_stack([data_array['x'], data_array['y'], data_array['z']])
                    colors = np.column_stack([data_array['r'], data_array['g'], data_array['b']]) / 255.0
                else:
                    # Only position data
                    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                    data_array = np.frombuffer(data, dtype=dtype)
                    points = np.column_stack([data_array['x'], data_array['y'], data_array['z']])
                    colors = None
            else:
                # ASCII format
                for i in range(vertex_count):
                    line = f.readline().decode('utf-8').strip().split()
                    if len(line) >= 3:
                        x, y, z = float(line[0]), float(line[1]), float(line[2])
                        points.append([x, y, z])
                        
                        if has_colors and len(line) >= 6:
                            r, g, b = int(line[3]), int(line[4]), int(line[5])
                            colors.append([r/255.0, g/255.0, b/255.0])
        
        points = np.array(points)
        colors = np.array(colors) if len(colors) > 0 else None
        
        return points, colors
    
    def gps_to_cartesian(self, lat: float, lon: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """Convert GPS coordinates to Cartesian coordinates for 3D positioning"""
        # Simple conversion for small areas - scale down significantly
        # Use a much smaller scale factor for visualization
        scale_factor = 1000  # 1 degree = 1000 units
        
        # Convert to relative coordinates (subtract a reference point)
        # Use the first location as reference
        if not hasattr(self, 'reference_lat'):
            self.reference_lat = lat
            self.reference_lon = lon
        
        # Calculate relative position
        x = (lon - self.reference_lon) * scale_factor
        y = alt  # Use altitude as Y coordinate
        z = (lat - self.reference_lat) * scale_factor
        
        return x, y, z
    
    def generate_html(self, output_file: str = "pointcloud_visualizer.html"):
        """Generate the HTML file with 3D point cloud visualization"""
        
        # Load all metadata
        self.load_metadata()
        
        # Load point cloud data for each location
        for metadata in self.metadata_files:
            folder_name = metadata['folder_name']
            pointcloud_data = self.load_pointcloud_data(folder_name, THRESHOLD)
            if pointcloud_data:
                self.pointcloud_data[folder_name] = pointcloud_data
        
        # Generate the HTML content
        html_content = self._create_html_template()
        
        # Write to file
        output_path = self.data_dir / output_file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Point cloud visualizer saved to: {output_path}")
        return output_path
    
    def _create_html_template(self) -> str:
        """Create the HTML template with Three.js for 3D visualization"""
        
        # Prepare the data for JavaScript
        locations_js = []
        for metadata in self.metadata_files:
            folder_name = metadata['folder_name']
            gps_coords = metadata['gps_coords']
            altitude = metadata['altitude']
            
            # Convert GPS to Cartesian coordinates
            x, y, z = self.gps_to_cartesian(gps_coords[0], gps_coords[1], altitude)
            
            location_data = {
                'id': folder_name,
                'name': folder_name,
                'gps_coords': gps_coords,
                'altitude': altitude,
                'position': [x, y, z],
                'metadata': {
                    'frames': metadata.get('frames', 0),
                    'video': metadata.get('video_name', 'Unknown'),
                }
            }
            
            # Add point cloud data if available
            if folder_name in self.pointcloud_data:
                pc_data = self.pointcloud_data[folder_name]
                location_data['pointcloud'] = {
                    'points': pc_data['points'],
                    'colors': pc_data['colors'],
                    'center': pc_data['center'],
                    'scale_factor': pc_data['scale_factor'],
                    'original_count': pc_data['original_count'],
                    'display_count': pc_data['display_count'],
                    'file_path': pc_data['file_path']
                }
            
            locations_js.append(location_data)
        
        locations_json = json.dumps(locations_js, indent=2)
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simple Point Cloud Visualizer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #000;
            overflow: hidden;
        }}
        
        #container {{
            position: relative;
            width: 100vw;
            height: 100vh;
        }}
        
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
            max-width: 300px;
        }}
        
        #controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
        }}
        
        .control-group {{
            margin-bottom: 10px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        
        .control-group input, .control-group select {{
            width: 100%;
            padding: 5px;
            border: 1px solid #555;
            border-radius: 3px;
            background: #333;
            color: white;
        }}
        
        .location-info {{
            margin-bottom: 10px;
            padding: 5px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }}
        
        .location-info h4 {{
            margin: 0 0 5px 0;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="info">
            <h3>Point Cloud Visualizer</h3>
            <div id="location-list"></div>
        </div>
        
        <div id="controls">
            <div class="control-group">
                <label>Point Size:</label>
                <input type="range" id="sizeSlider" min="1" max="20" step="1" value="3">
            </div>
            <div class="control-group">
                <label>Global Scale:</label>
                <input type="range" id="scaleSlider" min="0.1" max="5" step="0.1" value="1">
            </div>
            <div class="control-group">
                <label>Show Point Clouds:</label>
                <input type="checkbox" id="showPointClouds" checked>
            </div>
            <div class="control-group">
                <label>Camera Speed:</label>
                <input type="range" id="cameraSpeed" min="0.1" max="10" step="0.1" value="2">
            </div>
        </div>
    </div>

    <script>
        // Data
        const locations = {locations_json};
        console.log('Raw locations data:', locations);
        console.log('Locations length:', locations.length);
        
        // Three.js setup
        let scene, camera, renderer;
        let pointCloudObjects = {{}};
        let controls = {{
            mouseX: 0,
            mouseY: 0,
            isMouseDown: false,
            cameraSpeed: 1
        }};
        
        // Initialize Three.js
        function init() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
            
            // Position camera to view the point clouds
            camera.position.set(50, 50, 50);
            camera.lookAt(0, 0, 0);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Add point clouds
            addPointClouds();
            
            // Position camera to view all point clouds
            positionCameraToViewAll();
            
            // Setup controls
            setupControls();
            
            // Setup mouse controls
            setupMouseControls();
            
            // Update location info
            updateLocationInfo();
            
            // Start animation loop
            animate();
        }}
        
        function addPointClouds() {{
            console.log('Adding point clouds for', locations.length, 'locations');
            locations.forEach(location => {{
                if (location.pointcloud) {{
                    console.log('Adding point cloud for', location.name, 'with', location.pointcloud.points.length, 'points');
                    addPointCloud(location);
                }} else {{
                    console.log('No point cloud data for', location.name);
                }}
            }});
        }}
        
        function addPointCloud(location) {{
            const {{ pointcloud, position }} = location;
            const {{ points, colors }} = pointcloud;
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            
            // Convert points to Three.js format
            const positions = new Float32Array(points.length * 3);
            const pointColors = new Float32Array(points.length * 3);
            
            for (let i = 0; i < points.length; i++) {{
                positions[i * 3] = points[i][0];
                positions[i * 3 + 1] = points[i][1];
                positions[i * 3 + 2] = points[i][2];
                
                if (colors && colors[i]) {{
                    pointColors[i * 3] = colors[i][0];
                    pointColors[i * 3 + 1] = colors[i][1];
                    pointColors[i * 3 + 2] = colors[i][2];
                }} else {{
                    // Default color based on height
                    const height = points[i][2];
                    const normalizedHeight = (height + 5) / 10;
                    pointColors[i * 3] = normalizedHeight;
                    pointColors[i * 3 + 1] = 1 - normalizedHeight;
                    pointColors[i * 3 + 2] = 0.5;
                }}
            }}
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3));
            
            // Create material
            const material = new THREE.PointsMaterial({{
                size: 3,
                vertexColors: true,
                transparent: true,
                opacity: 0.8
            }});
            
            // Create points object
            const pointsObject = new THREE.Points(geometry, material);
            
            // Position the point cloud at the GPS coordinates
            pointsObject.position.set(position[0], position[1], position[2]);
            pointsObject.userData = {{ 
                locationId: location.id, 
                scale_factor: pointcloud.scale_factor,
                original_count: pointcloud.original_count,
                display_count: pointcloud.display_count
            }};
            
            // Add to scene
            scene.add(pointsObject);
            pointCloudObjects[location.id] = pointsObject;
        }}
        
        function positionCameraToViewAll() {{
            // Calculate bounding box of all point clouds
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let minZ = Infinity, maxZ = -Infinity;
            
            console.log('Positioning camera for', Object.keys(pointCloudObjects).length, 'point cloud objects');
            
            Object.values(pointCloudObjects).forEach(obj => {{
                const position = obj.position;
                const scale = obj.scale.x; // Assuming uniform scaling
                
                console.log('Point cloud at position:', position.x, position.y, position.z, 'scale:', scale);
                
                // Estimate bounds (this is approximate)
                const size = 10; // Approximate size of normalized point cloud
                minX = Math.min(minX, position.x - size * scale);
                maxX = Math.max(maxX, position.x + size * scale);
                minY = Math.min(minY, position.y - size * scale);
                maxY = Math.max(maxY, position.y + size * scale);
                minZ = Math.min(minZ, position.z - size * scale);
                maxZ = Math.max(maxZ, position.z + size * scale);
            }});
            
            // Calculate center and size
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;
            const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
            
            console.log('Bounding box:', minX, maxX, minY, maxY, minZ, maxZ);
            console.log('Center:', centerX, centerY, centerZ, 'Size:', size);
            
            // Position camera at a good distance
            const distance = Math.max(size * 2, 50); // Ensure minimum distance
            camera.position.set(centerX + distance, centerY + distance, centerZ + distance);
            camera.lookAt(centerX, centerY, centerZ);
            
            console.log('Camera positioned at:', camera.position.x, camera.position.y, camera.position.z);
        }}
        
        function setupControls() {{
            const sizeSlider = document.getElementById('sizeSlider');
            const scaleSlider = document.getElementById('scaleSlider');
            const showCheckbox = document.getElementById('showPointClouds');
            const cameraSpeedSlider = document.getElementById('cameraSpeed');
            
            sizeSlider.addEventListener('input', (e) => {{
                const size = parseInt(e.target.value);
                Object.values(pointCloudObjects).forEach(obj => {{
                    obj.material.size = size;
                }});
            }});
            
            scaleSlider.addEventListener('input', (e) => {{
                const scale = parseFloat(e.target.value);
                Object.values(pointCloudObjects).forEach(obj => {{
                    obj.scale.setScalar(scale);
                }});
            }});
            
            showCheckbox.addEventListener('change', (e) => {{
                const visible = e.target.checked;
                Object.values(pointCloudObjects).forEach(obj => {{
                    obj.visible = visible;
                }});
            }});
            
            cameraSpeedSlider.addEventListener('input', (e) => {{
                controls.cameraSpeed = parseFloat(e.target.value);
            }});
        }}
        
        function setupMouseControls() {{
            const canvas = renderer.domElement;
            
            canvas.addEventListener('mousedown', (e) => {{
                controls.isMouseDown = true;
                controls.mouseX = e.clientX;
                controls.mouseY = e.clientY;
            }});
            
            canvas.addEventListener('mouseup', () => {{
                controls.isMouseDown = false;
            }});
            
            canvas.addEventListener('mousemove', (e) => {{
                if (controls.isMouseDown) {{
                    const deltaX = e.clientX - controls.mouseX;
                    const deltaY = e.clientY - controls.mouseY;
                    
                    // Rotate camera around the scene
                    const radius = Math.sqrt(camera.position.x**2 + camera.position.y**2 + camera.position.z**2);
                    const theta = (deltaX * 0.002) * controls.cameraSpeed;
                    const phi = (deltaY * 0.002) * controls.cameraSpeed;
                    
                    // Update camera position
                    camera.position.x = radius * Math.cos(theta) * Math.cos(phi);
                    camera.position.y = radius * Math.sin(phi);
                    camera.position.z = radius * Math.sin(theta) * Math.cos(phi);
                    
                    camera.lookAt(0, 0, 0);
                    
                    controls.mouseX = e.clientX;
                    controls.mouseY = e.clientY;
                }}
            }});
            
            canvas.addEventListener('wheel', (e) => {{
                const scale = e.deltaY > 0 ? 1.05 : 0.95;
                camera.position.multiplyScalar(scale);
            }});
        }}
        
        function updateLocationInfo() {{
            const locationList = document.getElementById('location-list');
            locationList.innerHTML = '';
            
            locations.forEach(location => {{
                const div = document.createElement('div');
                div.className = 'location-info';
                
                const hasPointCloud = location.pointcloud ? '✓' : '✗';
                const pointCount = location.pointcloud ? location.pointcloud.display_count : 0;
                
                div.innerHTML = `
                    <h4>${{hasPointCloud}} ${{location.name}}</h4>
                    <p>GPS: ${{location.gps_coords[0].toFixed(4)}}, ${{location.gps_coords[1].toFixed(4)}}</p>
                    <p>Alt: ${{location.altitude}}m</p>
                    <p>Points: ${{pointCount}}</p>
                `;
                
                locationList.appendChild(div);
            }});
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        
        // Initialize when page loads
        init();
    </script>
</body>
</html>"""


def main():
    """Main function to generate the point cloud visualizer"""
    import argparse
    
    # Update global threshold first
    global THRESHOLD
    
    parser = argparse.ArgumentParser(description='Generate Simple Point Cloud Visualizer')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Directory containing the data folders')
    parser.add_argument('--output', type=str, default='pointcloud_visualizer.html',
                       help='Output HTML file name')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                       help='Threshold for pointcloud selection (default: {})'.format(THRESHOLD))
    
    args = parser.parse_args()
    
    # Update global threshold
    THRESHOLD = args.threshold
    
    # Create visualizer
    visualizer = SimplePointcloudVisualizer(args.data_dir)
    
    # Generate the visualizer
    output_path = visualizer.generate_html(args.output)
    
    print(f"Point Cloud Visualizer generated successfully!")
    print(f"Open the file in your browser: {output_path}")
    print(f"Using threshold: {THRESHOLD}")
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{output_path.absolute()}")
    except:
        print("Could not automatically open browser. Please open the file manually.")


if __name__ == "__main__":
    main()
