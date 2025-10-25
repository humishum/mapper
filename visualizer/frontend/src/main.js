/**
 * Main entry point for the pointcloud visualizer
 */

import { PointcloudViewer } from './PointcloudViewer.js';
import { loadData } from './dataLoader.js';
import { UIController } from './UIController.js';

async function init() {
  const loadingEl = document.getElementById('loading');
  
  try {
    console.log('=== Starting Visualizer ===');
    console.log('Window size:', window.innerWidth, 'x', window.innerHeight);
    
    // Load pointcloud data
    console.log('Loading data...');
    const data = await loadData('/data.json');
    console.log('Data loaded successfully');
    console.log('Metadata:', data.metadata);
    console.log('Locations:', data.locations.length);
    
    // Initialize viewer
    console.log('Creating viewer...');
    const viewer = new PointcloudViewer('canvas-container', data);
    
    // Initialize UI controller
    console.log('Creating UI controller...');
    const uiController = new UIController(viewer, data);
    
    // Hide loading screen
    loadingEl.classList.add('hidden');
    
    console.log('=== Visualizer initialized successfully ===');
    console.log('Viewer stats:', viewer.getStats());
    
    // Expose viewer to console for debugging
    window.debugViewer = viewer;
    console.log('Debug: viewer available as window.debugViewer');
    console.log('Try: debugViewer.deck.viewState');
    console.log('Try: debugViewer.deck.props.layers');
    
  } catch (error) {
    console.error('Failed to initialize visualizer:', error);
    console.error('Stack trace:', error.stack);
    loadingEl.innerHTML = `
      <div class="spinner"></div>
      <div>Error loading data: ${error.message}</div>
      <div style="margin-top: 10px; font-size: 12px; opacity: 0.7;">
        Check console for details. Make sure to run the Python export script first.
      </div>
    `;
  }
}

// Start the application
console.log('Initializing application...');
init();

