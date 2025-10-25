/**
 * Main pointcloud viewer using deck.gl
 */

import { Deck, OrbitView, COORDINATE_SYSTEM } from '@deck.gl/core';
import { PointCloudLayer } from '@deck.gl/layers';
import { calculateBounds, filterLocationsWithData } from './dataLoader.js';

export class PointcloudViewer {
  constructor(containerId, data) {
    this.container = document.getElementById(containerId);
    this.data = data;
    this.locations = data.locations || [];
    
    // Filter to only locations with pointcloud data
    this.locationsWithData = filterLocationsWithData(this.locations);
    console.log(`Loaded ${this.locationsWithData.length} locations with pointcloud data`);
    
    // Normalize positions to be centered at origin
    this.normalizePositions();
    
    // Calculate scene bounds based only on locations with data
    this.bounds = calculateBounds(this.locations);
    
    // View state
    this.viewState = this.calculateInitialViewState();
    
    // Rendering options
    this.options = {
      pointSize: 10,  // Increased from 3 to 10 for better visibility
      scale: 1.0,
      opacity: 0.8,
      showPoints: true
    };
    
    // Initialize deck.gl
    this.initDeck();
    
    // Create layers
    this.updateLayers();
  }
  
  normalizePositions() {
    // Find the center of all locations with data
    if (this.locationsWithData.length === 0) return;
    
    let sumX = 0, sumY = 0, sumZ = 0;
    this.locationsWithData.forEach(loc => {
      sumX += loc.position.x;
      sumY += loc.position.y;
      sumZ += loc.position.z;
    });
    
    const centerX = sumX / this.locationsWithData.length;
    const centerY = sumY / this.locationsWithData.length;
    const centerZ = sumZ / this.locationsWithData.length;
    
    console.log(`Normalizing positions around center: (${centerX.toFixed(2)}, ${centerY.toFixed(2)}, ${centerZ.toFixed(2)})`);
    
    // Shift all positions to be centered at origin
    this.locations.forEach(loc => {
      loc.position.x -= centerX;
      loc.position.y -= centerY;
      loc.position.z -= centerZ;
    });
    
    console.log('After normalization, first location position:', this.locationsWithData[0].position);
  }
  
  calculateInitialViewState() {
    const { minX, maxX, minY, maxY, minZ, maxZ } = this.bounds;
    
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    const rangeZ = maxZ - minZ;
    const maxRange = Math.max(rangeX, rangeY, rangeZ, 100);
    
    // Calculate appropriate zoom for the scene
    // For orbit view, zoom controls the distance from target
    const zoom = 0; // Start at neutral zoom since we normalized coordinates
    
    console.log(`Initial view: center=(${centerX.toFixed(2)}, ${centerY.toFixed(2)}, ${centerZ.toFixed(2)}), range=${maxRange.toFixed(2)}, zoom=${zoom.toFixed(2)}`);
    
    return {
      target: [centerX, centerY, centerZ],
      zoom: zoom,
      minZoom: -10,
      maxZoom: 10,
      rotationX: -30,
      rotationOrbit: 30
    };
  }
  
  initDeck() {
    console.log('=== Initializing deck.gl ===');
    console.log('Container:', this.container);
    console.log('Initial view state:', this.viewState);
    
    this.deck = new Deck({
      container: this.container,
      views: [new OrbitView()],
      initialViewState: this.viewState,
      controller: true,
      parameters: {
        clearColor: [0.1, 0.1, 0.1, 1],  // Slightly gray to verify rendering
        depthTest: true
      },
      onViewStateChange: ({ viewState }) => {
        this.viewState = viewState;
        console.log('View state changed:', {
          target: viewState.target,
          zoom: viewState.zoom?.toFixed(2),
          rotationX: viewState.rotationX?.toFixed(2),
          rotationOrbit: viewState.rotationOrbit?.toFixed(2)
        });
      },
      onWebGLInitialized: (gl) => {
        console.log('WebGL initialized:', gl);
        console.log('WebGL version:', gl.getParameter(gl.VERSION));
        console.log('WebGL vendor:', gl.getParameter(gl.VENDOR));
        console.log('WebGL renderer:', gl.getParameter(gl.RENDERER));
      },
      onLoad: () => {
        console.log('Deck.gl loaded successfully');
      },
      onError: (error) => {
        console.error('Deck.gl error:', error);
      },
      _animate: true
    });
    
    console.log('Deck instance created:', this.deck);
  }
  
  createPointcloudLayers() {
    const layers = [];
    
    // Only create layers for locations with data
    this.locationsWithData.forEach(location => {
      const { points, colors } = location.pointcloud;
      const position = location.position;
      
      console.log(`Creating layer for ${location.name}:`);
      console.log(`  - Points: ${points.length}`);
      console.log(`  - Position: (${position.x.toFixed(2)}, ${position.y.toFixed(2)}, ${position.z.toFixed(2)})`);
      console.log(`  - Point size: ${this.options.pointSize}`);
      console.log(`  - Scale: ${this.options.scale}`);
      console.log(`  - Visible: ${this.options.showPoints}`);
      
      // Sample first few points for debugging
      console.log(`  - Sample points (first 3):`);
      for (let i = 0; i < Math.min(3, points.length); i++) {
        const absPos = [
          points[i][0] * this.options.scale + position.x,
          points[i][1] * this.options.scale + position.y,
          points[i][2] * this.options.scale + position.z
        ];
        console.log(`    [${i}]: relative=(${points[i][0].toFixed(2)}, ${points[i][1].toFixed(2)}, ${points[i][2].toFixed(2)}) -> absolute=(${absPos[0].toFixed(2)}, ${absPos[1].toFixed(2)}, ${absPos[2].toFixed(2)})`);
      }
      
      // Convert points to absolute positions
      const data = points.map((point, i) => ({
        position: [
          point[0] * this.options.scale + position.x,
          point[1] * this.options.scale + position.y,
          point[2] * this.options.scale + position.z
        ],
        color: colors && colors[i] 
          ? [colors[i][0] * 255, colors[i][1] * 255, colors[i][2] * 255, 255]
          : [255, 255, 255, 255]  // White for visibility
      }));
      
      console.log(`  - Sample colors (first 3):`);
      for (let i = 0; i < Math.min(3, data.length); i++) {
        console.log(`    [${i}]: RGBA(${data[i].color.join(', ')})`);
      }
      
      const layer = new PointCloudLayer({
        id: `pointcloud-${location.id}`,
        data: data,
        getPosition: d => d.position,
        getColor: d => d.color,
        pointSize: this.options.pointSize,
        sizeUnits: 'pixels',
        opacity: 1.0,  // Force full opacity for debugging
        visible: this.options.showPoints,
        pickable: true,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        material: {
          ambient: 1.0,
          diffuse: 0.8
        },
        onHover: info => {
          if (info.picked) {
            console.log('Point hovered:', info.object.position);
          }
        }
      });
      
      console.log(`  - Layer created:`, layer);
      layers.push(layer);
    });
    
    console.log(`Total layers created: ${layers.length}`);
    return layers;
  }
  
  updateLayers() {
    console.log('=== Updating layers ===');
    const layers = this.createPointcloudLayers();
    console.log('Setting layers on deck:', layers);
    this.deck.setProps({ layers });
    
    // Check deck state after update
    setTimeout(() => {
      const deckLayers = this.deck.props.layers;
      console.log('Deck layers after update:', deckLayers);
      console.log('Deck view state:', this.deck.viewState);
    }, 100);
  }
  
  setPointSize(size) {
    this.options.pointSize = size;
    this.updateLayers();
  }
  
  setScale(scale) {
    this.options.scale = scale;
    this.updateLayers();
  }
  
  setOpacity(opacity) {
    this.options.opacity = opacity;
    this.updateLayers();
  }
  
  setShowPoints(show) {
    this.options.showPoints = show;
    this.updateLayers();
  }
  
  setCameraPreset(preset) {
    const { minX, maxX, minY, maxY, minZ, maxZ } = this.bounds;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const maxRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 100);
    
    let newViewState = {
      target: [centerX, centerY, centerZ],
      zoom: Math.log2(100 / maxRange),
      rotationX: 0,
      rotationOrbit: 0
    };
    
    switch (preset) {
      case 'overview':
        newViewState.rotationX = -45;
        newViewState.rotationOrbit = 30;
        break;
      case 'top':
        newViewState.rotationX = -90;
        newViewState.rotationOrbit = 0;
        break;
      case 'side':
        newViewState.rotationX = 0;
        newViewState.rotationOrbit = 90;
        break;
      case 'front':
        newViewState.rotationX = 0;
        newViewState.rotationOrbit = 0;
        break;
    }
    
    // Animate to new view
    this.deck.setProps({
      initialViewState: newViewState
    });
  }
  
  getStats() {
    let totalPoints = 0;
    
    this.locationsWithData.forEach(location => {
      totalPoints += location.pointcloud.display_count;
    });
    
    return {
      totalLocations: this.locations.length,
      locationsWithData: this.locationsWithData.length,
      totalPoints
    };
  }
}

