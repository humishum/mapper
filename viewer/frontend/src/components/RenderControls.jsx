/**
 * RenderControls component - Controls for rendering options
 */
import React from 'react';
import './RenderControls.css';

export default function RenderControls({
  pointSize = 2,
  onPointSizeChange,
  pointOpacity = 1.0,
  onPointOpacityChange,
  onViewPreset,
  collapsed = false,
  onToggleCollapse
}) {
  const presets = [
    { name: 'Overview', zoom: 8, pitch: 60, bearing: 0 },
    { name: 'Top Down', zoom: 12, pitch: 0, bearing: 0 },
    { name: 'Side View', zoom: 12, pitch: 80, bearing: 0 }
  ];

  return (
    <div className={`render-controls ${collapsed ? 'collapsed' : ''}`}>
      <div className="controls-header">
        <button
          className="hamburger-button"
          onClick={onToggleCollapse}
          aria-label={collapsed ? "Expand render controls" : "Collapse render controls"}
        >
          <div className="hamburger-icon">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </button>
        {!collapsed && <h3>Render Controls</h3>}
      </div>

      {!collapsed && (
        <>
          <div className="control-group">
        <label htmlFor="point-size">
          Point Size: {pointSize}px
        </label>
        <input
          id="point-size"
          type="range"
          min="1"
          max="10"
          step="0.5"
          value={pointSize}
          onChange={(e) => onPointSizeChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label htmlFor="point-opacity">
          Opacity: {Math.round(pointOpacity * 100)}%
        </label>
        <input
          id="point-opacity"
          type="range"
          min="0.1"
          max="1"
          step="0.1"
          value={pointOpacity}
          onChange={(e) => onPointOpacityChange(parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>Camera Presets</label>
        <div className="preset-buttons">
          {presets.map(preset => (
            <button
              key={preset.name}
              className="preset-button"
              onClick={() => onViewPreset(preset)}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div className="control-info">
        <p>🖱️ <strong>Controls:</strong></p>
        <ul>
          <li>Left drag: Rotate</li>
          <li>Scroll: Zoom</li>
          <li>Right drag: Pan</li>
        </ul>
          </div>
        </>
      )}
    </div>
  );
}

