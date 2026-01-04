/**
 * MapView component - Main deck.gl map with markers and pointclouds
 * Using deck.gl 8.x (WebGL only, no WebGPU)
 */
import React from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, PointCloudLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';

import { BitmapLayer } from '@deck.gl/layers';
import { COORDINATE_SYSTEM } from '@deck.gl/core';
import 'maplibre-gl/dist/maplibre-gl.css';

const INITIAL_VIEW_STATE = {
  longitude: -122,
  latitude: 37.5,
  zoom: 8,
  pitch: 45,
  bearing: 0,
  minZoom: 2,
  maxZoom: 22
};

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

export default function MapView({
  locations = [],
  loadedPointclouds = [],
  onMarkerClick,
  viewState,
  onViewStateChange,
  pointSize = 2,
  pointOpacity = 1.0
}) {
  const layers = [
    // Base map tiles
    new TileLayer({
      id: 'base-map',
      data: 'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: props => {
        const { boundingBox } = props.tile;
        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [boundingBox[0][0], boundingBox[0][1], boundingBox[1][0], boundingBox[1][1]]
        });
      }
    }),

    // Location markers
    new ScatterplotLayer({
      id: 'location-markers',
      data: locations,
      getPosition: d => [d.longitude, d.latitude, d.altitude || 0],
      getFillColor: d => {
        // Check if this location is loaded
        const isLoaded = loadedPointclouds.some(pc => pc.location === d.name);
        return isLoaded ? [0, 255, 0, 255] : [255, 140, 0, 255];
      },
      getRadius: 200,
      radiusScale: 1,
      radiusMinPixels: 10,
      radiusMaxPixels: 30,
      pickable: true,
      onClick: info => {
        if (info.object && onMarkerClick) {
          console.log('[MapView] Marker clicked:', info.object.name);
          onMarkerClick(info.object);
        }
      },
      updateTriggers: {
        getFillColor: [loadedPointclouds]
      }
    }),

    // Pointclouds
    ...loadedPointclouds.map(pc => {
      console.log(`[MapView] Rendering pointcloud for ${pc.location} with ${pc.numPoints} points`);
      return new PointCloudLayer({
        id: `pointcloud-${pc.location}`,
        data: {
          length: pc.numPoints,
          attributes: {
            getPosition: { value: pc.positions, size: 3 },
            getColor: { value: pc.colors, size: 3 }
          }
        },
        pointSize: pointSize,
        opacity: pointOpacity,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        pickable: false,
        // Add better point rendering
        sizeUnits: 'pixels',
        // Enable billboard mode so points face camera
        billboard: true
      });
    })
  ];

  return (
    <DeckGL
      initialViewState={viewState || INITIAL_VIEW_STATE}
      controller={true}
      layers={layers}
      onViewStateChange={onViewStateChange}
      getTooltip={({ object }) => {
        if (object && object.name) {
          return {
            html: `
              <div style="padding: 8px; background: white; border-radius: 4px;">
                <strong>${object.name}</strong><br/>
                ${object.latitude.toFixed(6)}, ${object.longitude.toFixed(6)}<br/>
                Altitude: ${object.altitude?.toFixed(1)}m
              </div>
            `,
            style: {
              backgroundColor: 'transparent',
              fontSize: '12px'
            }
          };
        }
      }}
    />
  );
}

