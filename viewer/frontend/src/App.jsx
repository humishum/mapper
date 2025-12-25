/**
 * Main App component for the 3D Pointcloud Viewer
 */
import { useState, useEffect } from 'react';
import MapView from './components/MapView';
import LocationPanel from './components/LocationPanel';
import RenderControls from './components/RenderControls';
import { fetchLocations, fetchPointcloud, parsePointcloudData } from './services/api';
import './App.css';

const INITIAL_VIEW_STATE = {
  longitude: -122,
  latitude: 37.5,
  zoom: 8,
  pitch: 60,
  bearing: 0
};

function App() {
  const [locations, setLocations] = useState([]);
  const [loadedPointclouds, setLoadedPointclouds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [pointSize, setPointSize] = useState(2);
  const [pointOpacity, setPointOpacity] = useState(1.0);

  // Load locations on mount
  useEffect(() => {
    loadLocations();
  }, []);

  const loadLocations = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchLocations();
      setLocations(data);
      console.log(`Loaded ${data.length} locations`);
    } catch (err) {
      console.error('Failed to load locations:', err);
      setError('Failed to load locations. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadLocation = async (location) => {
    console.log('[App] handleLoadLocation called for:', location.name);
    
    // Check if already loaded
    if (loadedPointclouds.some(pc => pc.location === location.name)) {
      console.log(`[App] Location ${location.name} already loaded, skipping`);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      console.log(`[App] Loading pointcloud for ${location.name}...`);
      console.log(`[App] Location details:`, {
        lat: location.latitude,
        lon: location.longitude,
        alt: location.altitude,
        sequences: location.sequences
      });
      
      const rawData = await fetchPointcloud(location.name, {
        sequence: 'sequence_1',
        threshold: 2.0,
        maxPoints: 100000,
        useGpsCoords: true
      });
      
      console.log('[App] Raw data received:', {
        location: rawData.location,
        numPoints: rawData.num_points
      });
      
      const parsedData = parsePointcloudData(rawData);
      console.log(`[App] Loaded ${parsedData.numPoints} points for ${location.name}`);
      console.log('[App] First few positions:', parsedData.positions.slice(0, 15));
      
      setLoadedPointclouds(prev => {
        const updated = [...prev, parsedData];
        console.log('[App] Total loaded pointclouds:', updated.length);
        return updated;
      });
      
      // Fly to location
      handleFlyToLocation(location);
    } catch (err) {
      console.error(`[App] Failed to load pointcloud for ${location.name}:`, err);
      setError(`Failed to load pointcloud for ${location.name}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUnloadLocation = (locationName) => {
    console.log(`[App] Unloading pointcloud for ${locationName}`);
    setLoadedPointclouds(prev => {
      const updated = prev.filter(pc => pc.location !== locationName);
      console.log('[App] Remaining loaded pointclouds:', updated.length);
      return updated;
    });
  };

  const handleMarkerClick = (location) => {
    // Toggle load/unload
    const isLoaded = loadedPointclouds.some(pc => pc.location === location.name);
    if (isLoaded) {
      handleUnloadLocation(location.name);
    } else {
      handleLoadLocation(location);
    }
  };

  const handleFlyToLocation = (location) => {
    console.log('[App] Flying to location:', {
      name: location.name,
      lat: location.latitude,
      lon: location.longitude,
      alt: location.altitude
    });
    
    // Check if coordinates are valid
    if (!location.latitude || !location.longitude || 
        location.latitude === 0 && location.longitude === 0) {
      console.warn('[App] Invalid coordinates for location:', location.name);
      setError(`Location ${location.name} has invalid GPS coordinates (0, 0)`);
      return;
    }
    
    const newViewState = {
      longitude: location.longitude,
      latitude: location.latitude,
      zoom: 16,
      pitch: 60,
      bearing: 0,
      transitionDuration: 1500,
      transitionInterpolator: null
    };
    
    console.log('[App] Setting new view state:', newViewState);
    setViewState(newViewState);
  };

  const handleViewPreset = (preset) => {
    setViewState(prev => ({
      ...prev,
      ...preset,
      transitionDuration: 500
    }));
  };

  return (
    <div className="app">
      <MapView
        locations={locations}
        loadedPointclouds={loadedPointclouds}
        onMarkerClick={handleMarkerClick}
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        pointSize={pointSize}
        pointOpacity={pointOpacity}
      />
      
      <LocationPanel
        locations={locations}
        loadedPointclouds={loadedPointclouds}
        onLoadLocation={handleLoadLocation}
        onUnloadLocation={handleUnloadLocation}
        onFlyToLocation={handleFlyToLocation}
        loading={loading}
      />
      
      <RenderControls
        pointSize={pointSize}
        onPointSizeChange={setPointSize}
        pointOpacity={pointOpacity}
        onPointOpacityChange={setPointOpacity}
        onViewPreset={handleViewPreset}
      />

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="error-toast">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}
    </div>
  );
}

export default App;
