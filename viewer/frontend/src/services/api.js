/**
 * API client for communicating with the backend server.
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Fetch all available locations with GPS coordinates.
 * @returns {Promise<Array>} Array of location objects
 */
export async function fetchLocations() {
  try {
    console.log('[API] Fetching locations from:', `${API_BASE_URL}/api/locations`);
    const response = await axios.get(`${API_BASE_URL}/api/locations`);
    console.log('[API] Received locations:', response.data.count);
    return response.data.locations;
  } catch (error) {
    console.error('[API] Error fetching locations:', error);
    throw error;
  }
}

/**
 * Fetch pointcloud data for a specific location.
 * @param {string} locationName - Name of the location
 * @param {Object} options - Optional parameters
 * @param {string} options.sequence - Sequence name (default: "sequence_1")
 * @param {number} options.threshold - Threshold value (default: 2.0)
 * @param {number} options.maxPoints - Maximum points to load (default: 100000)
 * @param {boolean} options.useGpsCoords - Use GPS coordinates (default: true)
 * @returns {Promise<Object>} Pointcloud data
 */
export async function fetchPointcloud(locationName, options = {}) {
  try {
    const params = {
      sequence: options.sequence || 'sequence_1',
      threshold: options.threshold || 2.0,
      max_points: options.maxPoints || 100000,
      use_gps_coords: options.useGpsCoords !== undefined ? options.useGpsCoords : true
    };
    
    console.log(`[API] Fetching pointcloud for ${locationName} with params:`, params);
    const startTime = performance.now();
    
    const response = await axios.get(
      `${API_BASE_URL}/api/pointcloud/${encodeURIComponent(locationName)}`,
      { params }
    );
    
    const duration = performance.now() - startTime;
    console.log(`[API] Pointcloud fetched in ${duration.toFixed(0)}ms, size: ${JSON.stringify(response.data).length} bytes`);
    
    return response.data;
  } catch (error) {
    console.error(`[API] Error fetching pointcloud for ${locationName}:`, error);
    throw error;
  }
}

/**
 * Parse hex-encoded pointcloud data into typed arrays.
 * @param {Object} data - Raw pointcloud data from API
 * @returns {Object} Parsed pointcloud with Float32Array positions and Uint8Array colors
 */
export function parsePointcloudData(data) {
  console.log(`[API] Parsing pointcloud data for ${data.location}`);
  console.log(`[API] Raw data - positions hex length: ${data.positions.length}, colors hex length: ${data.colors.length}`);
  
  // Convert hex strings to binary data
  const positionsHex = data.positions;
  const colorsHex = data.colors;
  
  const startTime = performance.now();
  
  // Convert hex to Uint8Array
  const positionsBytes = new Uint8Array(positionsHex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
  const colorsBytes = new Uint8Array(colorsHex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
  
  // Create typed arrays
  const positions = new Float32Array(positionsBytes.buffer);
  const colors = new Uint8Array(colorsBytes.buffer);
  
  const duration = performance.now() - startTime;
  console.log(`[API] Parsing took ${duration.toFixed(0)}ms`);
  console.log(`[API] Parsed ${positions.length / 3} points (${positions.length} floats, ${colors.length} bytes)`);
  
  // Log sample of first few points
  if (positions.length >= 9) {
    console.log('[API] Sample points:', {
      point1: [positions[0], positions[1], positions[2]],
      point2: [positions[3], positions[4], positions[5]],
      point3: [positions[6], positions[7], positions[8]]
    });
  }
  
  return {
    location: data.location,
    numPoints: data.num_points,
    positions,
    colors
  };
}

