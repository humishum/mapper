/**
 * UI Controller for managing user interface interactions
 */

export class UIController {
  constructor(viewer, data) {
    this.viewer = viewer;
    this.data = data;
    
    this.initLocationsList();
    this.initControls();
  }
  
  initLocationsList() {
    const listContainer = document.getElementById('location-list');
    
    // Only show locations with pointcloud data
    const locationsWithData = this.data.locations.filter(loc => 
      loc.pointcloud && loc.pointcloud.points && loc.pointcloud.points.length > 0
    );
    
    if (locationsWithData.length === 0) {
      listContainer.innerHTML = '<p style="color: #f44336; padding: 10px;">No pointcloud data available. Make sure to run the export script with the correct threshold and sequence ID.</p>';
      return;
    }
    
    locationsWithData.forEach(location => {
      const item = document.createElement('div');
      item.className = 'location-item';
      
      const pointCount = location.pointcloud.display_count.toLocaleString();
      const originalCount = location.pointcloud.original_count.toLocaleString();
      
      item.innerHTML = `
        <h4>
          ${location.name}
          <span class="status-badge success">✓ Loaded</span>
        </h4>
        <p><strong>GPS:</strong> ${location.gps.lat.toFixed(5)}, ${location.gps.lon.toFixed(5)}</p>
        <p><strong>Alt:</strong> ${location.gps.alt}m</p>
        <p><strong>Points:</strong> ${pointCount} <span style="opacity: 0.6;">(of ${originalCount})</span></p>
        ${location.metadata.video ? `<p><strong>Video:</strong> ${location.metadata.video}</p>` : ''}
        <p><strong>File:</strong> ${location.pointcloud.file_name}</p>
      `;
      
      listContainer.appendChild(item);
    });
  }
  
  initControls() {
    // Point size control
    const pointSizeSlider = document.getElementById('point-size');
    const pointSizeValue = document.getElementById('point-size-value');
    
    pointSizeSlider.addEventListener('input', (e) => {
      const size = parseInt(e.target.value);
      pointSizeValue.textContent = size;
      this.viewer.setPointSize(size);
    });
    
    // Scale control
    const scaleSlider = document.getElementById('scale');
    const scaleValue = document.getElementById('scale-value');
    
    scaleSlider.addEventListener('input', (e) => {
      const scale = parseFloat(e.target.value);
      scaleValue.textContent = scale.toFixed(1);
      this.viewer.setScale(scale);
    });
    
    // Opacity control
    const opacitySlider = document.getElementById('opacity');
    const opacityValue = document.getElementById('opacity-value');
    
    opacitySlider.addEventListener('input', (e) => {
      const opacity = parseFloat(e.target.value);
      opacityValue.textContent = opacity.toFixed(2);
      this.viewer.setOpacity(opacity);
    });
    
    // Show/hide toggle
    const showPointsCheckbox = document.getElementById('show-points');
    showPointsCheckbox.addEventListener('change', (e) => {
      this.viewer.setShowPoints(e.target.checked);
    });
    
    // Camera preset
    const cameraPreset = document.getElementById('camera-preset');
    cameraPreset.addEventListener('change', (e) => {
      this.viewer.setCameraPreset(e.target.value);
    });
  }
}

