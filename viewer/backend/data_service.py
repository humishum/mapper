
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from plyfile import PlyData

from .config import DATA_DIR, DEFAULT_THRESHOLD, MAX_POINTS, CACHE_SIZE
from .coordinate_transform import transform_pointcloud_to_gps
from .cache import LRUCache

logger = logging.getLogger(__name__)
np.random.seed(42)

# could be a dataclass? works for now
class LocationInfo:
    def __init__(self, name: str, metadata: Dict, data_dir: Path):
        self.name = name
        self.lat = metadata.get("initial_gps_coordinates", [0, 0])[0]
        self.lon = metadata.get("initial_gps_coordinates", [0, 0])[1]
        self.altitude = metadata.get("altitude", 0)
        self.frames = metadata.get("frames", 0)
        self.video_name = metadata.get("video_name", "")
        self.data_dir = data_dir

        # TODO: remove sequences once we get sub-global aligment/bundle adjustment working
        pointcloud_dir = data_dir / "pointclouds"
        self.sequences = []
        if pointcloud_dir.exists():
            for seq_dir in sorted(pointcloud_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.startswith("sequence_"):
                    self.sequences.append(seq_dir.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "latitude": self.lat,
            "longitude": self.lon,
            "altitude": self.altitude,
            "frames": self.frames,
            "video_name": self.video_name,
            "sequences": self.sequences,
        }


class PointCloudDataService:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.cache = LRUCache(max_size=CACHE_SIZE)
        logger.info(f"Initialized PointCloudDataService with data_dir: {data_dir}")

    def get_all_locations(self) -> List[LocationInfo]:
        """
        Scan data directory and return information about all locations.
        """
        locations = []

        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return locations

        for location_dir in sorted(self.data_dir.iterdir()):
            if not location_dir.is_dir():
                continue

            metadata_file = location_dir / "metadata.json"
            if not metadata_file.exists():
                logger.warning(f"No metadata.json found in {location_dir}")
                continue

            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                location_info = LocationInfo(location_dir.name, metadata, location_dir)
                locations.append(location_info)
                logger.info(
                    f"Found location: {location_info.name} at ({location_info.lat}, {location_info.lon})"
                )
            except Exception as e:
                logger.error(f"Error reading metadata from {metadata_file}: {e}")
                continue

        return locations

    def load_pointcloud(
        self,
        location: str,
        sequence: str = "sequence_1",
        threshold: float = DEFAULT_THRESHOLD,
        max_points: int = MAX_POINTS,
        use_gps_coords: bool = True,
    ) -> Optional[Dict]:
        """
        Load and process a pointcloud from a specific location.

        Args:
            location: Name of the location directory
            sequence: Sequence name (e.g., "sequence_1")
            threshold: Threshold value for PLY file selection (e.g., 2.0)
            max_points: Maximum number of points to return
            use_gps_coords: If True, convert to GPS coordinates; if False, keep local coords

        Returns:
            Dictionary with pointcloud data or None if not found
        """
        cache_key = f"{location}_{sequence}_{threshold}_{max_points}_{use_gps_coords}"

        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached pointcloud: {cache_key}")
            positions, colors = cached_data
            return self._format_response(positions, colors, location)

        # Find PLY file
        location_dir = self.data_dir / location
        if not location_dir.exists():
            logger.error(f"Location directory not found: {location_dir}")
            return None

        ply_file = location_dir / "pointclouds" / sequence / f"scene_thr{threshold}.ply"
        if not ply_file.exists():
            logger.error(f"PLY file not found: {ply_file}")
            return None

        # Load metadata for GPS coordinates
        metadata_file = location_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return None

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        origin_lat = metadata.get("initial_gps_coordinates", [0, 0])[0]
        origin_lon = metadata.get("initial_gps_coordinates", [0, 0])[1]
        origin_alt = metadata.get("altitude", 0)

        # Load and process PLY file
        logger.info(f"Loading PLY file: {ply_file}")
        positions, colors = self._load_ply_file(ply_file, max_points)

        if positions is None or len(positions) == 0:
            logger.error(f"No points loaded from {ply_file}")
            return None

        # Transform to GPS coordinates if requested
        if use_gps_coords:
            logger.info(f"Transforming {len(positions)} points to GPS coordinates")
            positions = transform_pointcloud_to_gps(
                positions, origin_lat, origin_lon, origin_alt
            )

        # Cache the result
        self.cache.put(cache_key, (positions, colors))
        logger.info(
            f"Cached pointcloud: {cache_key} with {len(positions)} points (cache size: {self.cache.size()})"
        )

        return self._format_response(positions, colors, location)

    def _load_ply_file(
        self, ply_file: Path, max_points: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a PLY file and extract positions and colors.

        Args:
            ply_file: Path to PLY file
            max_points: Maximum number of points to load

        Returns:
            Tuple of (positions, colors) as numpy arrays, or (None, None) on error
        """
        try:
            ply_data = PlyData.read(str(ply_file))
            vertex = ply_data["vertex"]

            # Extract positions
            x = vertex["x"]
            y = vertex["y"]
            z = vertex["z"]
            positions = np.column_stack([x, y, z])

            # Extract colors
            if "red" in vertex.data.dtype.names:
                r = vertex["red"]
                g = vertex["green"]
                b = vertex["blue"]
                colors = np.column_stack([r, g, b])
            else:
                # Default to white if no colors
                colors = np.full((len(positions), 3), 255, dtype=np.uint8)

            # Downsample if needed using random sampling
            if len(positions) > max_points:
                logger.info(
                    f"Downsampling from {len(positions)} to {max_points} points using random sampling"
                )
                
                indices = np.random.choice(len(positions), max_points, replace=False)
                # Sort indices to maintain some spatial coherence
                indices = np.sort(indices)
                positions = positions[indices]
                colors = colors[indices]
                logger.info(f"Downsampling complete: {len(positions)} points retained")

            logger.info(f"Loaded {len(positions)} points from {ply_file.name}")
            return positions, colors

        except Exception as e:
            logger.error(f"Error loading PLY file {ply_file}: {e}")
            return None, None

    def _format_response(
        self, positions: np.ndarray, colors: np.ndarray, location: str
    ) -> Dict:
        """
        Format pointcloud data for API response.

        Args:
            positions: Nx3 array of positions
            colors: Nx3 array of colors
            location: Location name

        Returns:
            Dictionary with formatted data
        """
        return {
            "location": location,
            "num_points": len(positions),
            "positions": positions.astype(np.float32).tobytes().hex(),
            "colors": colors.astype(np.uint8).tobytes().hex(),
        }
