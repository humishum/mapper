/**
 * Data loading utilities
 */

/**
 * Load pointcloud data from JSON file
 * @param {string} url - URL to data JSON file
 * @returns {Promise<Object>} Parsed data object
 */
export async function loadData(url) {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to load data: ${response.statusText}`);
  }
  
  const data = await response.json();
  
  // Validate data structure
  if (!data.locations || !Array.isArray(data.locations)) {
    throw new Error('Invalid data format: missing locations array');
  }
  
  console.log(`Loaded ${data.locations.length} locations`);
  console.log(`Threshold: ${data.metadata?.threshold}`);
  console.log(`Sequence: ${data.metadata?.sequence_id}`);
  
  return data;
}

/**
 * Filter locations to only include those with pointcloud data
 * @param {Array} locations - Array of location objects
 * @returns {Array} Filtered locations with pointcloud data
 */
export function filterLocationsWithData(locations) {
  return locations.filter(location => 
    location.pointcloud && 
    location.pointcloud.points && 
    location.pointcloud.points.length > 0
  );
}

/**
 * Calculate bounding box for all locations with pointcloud data
 * @param {Array} locations - Array of location objects
 * @returns {Object} Bounding box with min/max coordinates
 */
export function calculateBounds(locations) {
  // Only consider locations with actual pointcloud data
  const locationsWithData = filterLocationsWithData(locations);
  
  console.log(`Calculating bounds for ${locationsWithData.length} locations with data`);
  
  if (locationsWithData.length === 0) {
    console.warn('No locations with pointcloud data found');
    return { minX: -100, maxX: 100, minY: -100, maxY: 100, minZ: -100, maxZ: 100 };
  }
  
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  
  locationsWithData.forEach(location => {
    const pos = location.position;
    const pc = location.pointcloud;
    
    // Calculate actual bounds considering pointcloud size
    const bboxSize = Math.max(
      Math.abs(pc.bbox_max[0] - pc.bbox_min[0]),
      Math.abs(pc.bbox_max[1] - pc.bbox_min[1]),
      Math.abs(pc.bbox_max[2] - pc.bbox_min[2]),
      10  // Minimum size
    );
    
    minX = Math.min(minX, pos.x - bboxSize);
    maxX = Math.max(maxX, pos.x + bboxSize);
    minY = Math.min(minY, pos.y - bboxSize);
    maxY = Math.max(maxY, pos.y + bboxSize);
    minZ = Math.min(minZ, pos.z - bboxSize);
    maxZ = Math.max(maxZ, pos.z + bboxSize);
    
    console.log(`Location ${location.name}: pos=(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}), bbox_size=${bboxSize.toFixed(2)}`);
  });
  
  console.log(`Bounds: X=[${minX.toFixed(2)}, ${maxX.toFixed(2)}], Y=[${minY.toFixed(2)}, ${maxY.toFixed(2)}], Z=[${minZ.toFixed(2)}, ${maxZ.toFixed(2)}]`);
  
  return { minX, maxX, minY, maxY, minZ, maxZ };
}

