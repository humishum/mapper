/**
 * LocationPanel component - Side panel for location management
 */
import React from 'react';
import './LocationPanel.css';

export default function LocationPanel({
  locations = [],
  loadedPointclouds = [],
  onLoadLocation,
  onUnloadLocation,
  onFlyToLocation,
  loading = false
}) {
  const isLoaded = (locationName) => {
    return loadedPointclouds.some(pc => pc.location === locationName);
  };

  return (
    <div className="location-panel">
      <div className="panel-header">
        <h2>Locations ({locations.length})</h2>
        <p className="panel-subtitle">
          {loadedPointclouds.length} loaded
        </p>
      </div>

      <div className="location-list">
        {locations.map(location => {
          const loaded = isLoaded(location.name);
          
          return (
            <div key={location.name} className="location-item">
              <div className="location-header">
                <input
                  type="checkbox"
                  checked={loaded}
                  onChange={(e) => {
                    if (e.target.checked) {
                      onLoadLocation(location);
                    } else {
                      onUnloadLocation(location.name);
                    }
                  }}
                  disabled={loading}
                />
                <span className="location-name">{location.name}</span>
              </div>
              
              <div className="location-details">
                <div className="location-coords">
                  📍 {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
                </div>
                <div className="location-altitude">
                  ⛰️ {location.altitude.toFixed(1)}m
                </div>
                {location.sequences && location.sequences.length > 0 && (
                  <div className="location-sequences">
                    🎬 {location.sequences.length} sequence(s)
                  </div>
                )}
              </div>

              <button
                className="fly-to-button"
                onClick={() => onFlyToLocation(location)}
                disabled={loading}
              >
                Fly To
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

