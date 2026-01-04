/**
 * API client for communicating with the backend server.
 */
import axios from 'axios';

// Auto-detect API URL based on current hostname
// When accessed via port forwarding, use the same hostname but backend port
// Or use relative URLs if Vite proxy is configured
function getApiBaseUrl() {
  // Check for explicit environment variable first
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // In development with Vite proxy, use relative URLs (proxy handles forwarding)
  if (import.meta.env.DEV) {
    return ''; // Relative URL - Vite proxy will forward to backend
  }
  
  // Auto-detect: use same hostname as current page, but backend port (8000)
  const currentHost = window.location.hostname;
  const currentProtocol = window.location.protocol;
  
  // If accessing via IP address (port forwarding), use that IP for backend
  if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
    return `${currentProtocol}//${currentHost}:8000`;
  }
  
  // Default to localhost for local development
  return 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();
console.log('[API] Using API base URL:', API_BASE_URL || '(relative - using Vite proxy)');

// Configure axios with timeout and interceptors for debugging
axios.defaults.timeout = 30000; // 30 second timeout

// Request interceptor
axios.interceptors.request.use(
  (config) => {
    console.log('[API] Request:', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      fullURL: `${config.baseURL || ''}${config.url}`
    });
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
axios.interceptors.response.use(
  (response) => {
    console.log('[API] Response received:', {
      status: response.status,
      statusText: response.statusText,
      url: response.config.url,
      hasData: !!response.data
    });
    return response;
  },
  (error) => {
    console.error('[API] Response error:', {
      message: error.message,
      code: error.code,
      status: error.response?.status,
      statusText: error.response?.statusText,
      url: error.config?.url
    });
    return Promise.reject(error);
  }
);

/**
 * Fetch all available locations with GPS coordinates.
 * @returns {Promise<Array>} Array of location objects
 */
export async function fetchLocations() {
  try {
    console.log('[API] Fetching locations from:', `${API_BASE_URL}/api/locations`);
    const response = await axios.get(`${API_BASE_URL}/api/locations`);
    console.log('[API] Response status:', response.status);
    console.log('[API] Response headers:', response.headers);
    console.log('[API] Response data:', response.data);
    console.log('[API] Response data type:', typeof response.data);
    console.log('[API] Received locations count:', response.data?.count);
    console.log('[API] Received locations array:', response.data?.locations);
    return response.data.locations;
  } catch (error) {
    console.error('[API] Error fetching locations:', error);
    console.error('[API] Error details:', {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status,
      headers: error.response?.headers
    });
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

