
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


class LocationInfo:
    """
    Information about a pointcloud location.

    Supports two directory structures:

    1. Legacy/Constructor format:
       {location}/
         metadata.json
         images/
         pointclouds/
           sequence_1/
             scene_thr5.0.ply
           sequence_2/
             ...

    2. Experiment runner format:
       {experiment}/outputs/{location}/
         images/
         scene_thr5.0.ply  (directly in folder)
         aligned_pointcloud.ply
    """

    def __init__(self, name: str, metadata: Dict, data_dir: Path):
        self.name = name
        self.lat = metadata.get("initial_gps_coordinates", [0, 0])[0]
        self.lon = metadata.get("initial_gps_coordinates", [0, 0])[1]
        self.altitude = metadata.get("altitude", 0)
        self.frames = metadata.get("frames", 0)
        self.video_name = metadata.get("video_name", "")
        self.data_dir = data_dir

        # Detect directory structure and find sequences/pointclouds
        self.sequences = []
        self.ply_files = []  # Direct PLY files (new format)

        # Check for legacy structure: pointclouds/sequence_N/
        pointcloud_dir = data_dir / "pointclouds"
        if pointcloud_dir.exists():
            for seq_dir in sorted(pointcloud_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.startswith("sequence_"):
                    self.sequences.append(seq_dir.name)

        # Check for new structure: PLY files directly in folder
        for ply_file in sorted(data_dir.glob("*.ply")):
            self.ply_files.append(ply_file.name)

        # If no sequences found but we have direct PLY files, create a virtual "default" sequence
        if not self.sequences and self.ply_files:
            self.sequences = ["default"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "latitude": self.lat,
            "longitude": self.lon,
            "altitude": self.altitude,
            "frames": self.frames,
            "video_name": self.video_name,
            "sequences": self.sequences,
            "ply_files": self.ply_files,  # Include direct PLY files
        }


class PointCloudDataService:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.cache = LRUCache(max_size=CACHE_SIZE)
        logger.info(f"Initialized PointCloudDataService with data_dir: {data_dir}")

    def get_all_locations(self) -> List[LocationInfo]:
        """
        Scan data directory and return information about all locations.

        Supports both direct location folders and experiment output folders.
        """
        locations = []

        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return locations

        # Check for experiment runner format: look in outputs/ subfolder
        outputs_dir = self.data_dir / "outputs"
        if outputs_dir.exists():
            # This is an experiment output folder
            locations.extend(self._scan_directory(outputs_dir))
        else:
            # This is a direct data folder
            locations.extend(self._scan_directory(self.data_dir))

        return locations

    def _scan_directory(self, scan_dir: Path) -> List[LocationInfo]:
        """Scan a directory for location folders."""
        locations = []

        for location_dir in sorted(scan_dir.iterdir()):
            if not location_dir.is_dir():
                continue

            # Look for metadata in the location folder or images subfolder
            metadata = self._find_metadata(location_dir)
            if metadata is None:
                # No metadata found, skip unless we have PLY files
                if not list(location_dir.glob("*.ply")):
                    logger.warning(f"No metadata.json or PLY files in {location_dir}")
                    continue
                # Create minimal metadata for PLY-only folders
                metadata = {"initial_gps_coordinates": [0, 0], "altitude": 0}

            try:
                location_info = LocationInfo(location_dir.name, metadata, location_dir)
                locations.append(location_info)
                logger.info(
                    f"Found location: {location_info.name} at ({location_info.lat}, {location_info.lon})"
                )
            except Exception as e:
                logger.error(f"Error creating LocationInfo from {location_dir}: {e}")
                continue

        return locations

    def _find_metadata(self, location_dir: Path) -> Optional[Dict]:
        """Find and load metadata from various possible locations."""
        # Check direct metadata.json
        metadata_file = location_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)

        # Check in images subfolder (experiment runner caches metadata there)
        images_metadata = location_dir / "images" / ".cache_info.json"
        if images_metadata.exists():
            # This is cache info, not full metadata - try to supplement
            with open(images_metadata, "r") as f:
                cache_info = json.load(f)
            return {
                "video_name": cache_info.get("video_name", ""),
                "frames": cache_info.get("frame_count", 0),
                "initial_gps_coordinates": [0, 0],
                "altitude": 0,
            }

        return None

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
            sequence: Sequence name (e.g., "sequence_1") or "default" for direct PLY files
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

        # Find location directory (check both direct and outputs/ paths)
        location_dir = self.data_dir / location
        if not location_dir.exists():
            location_dir = self.data_dir / "outputs" / location
        if not location_dir.exists():
            logger.error(f"Location directory not found: {location}")
            return None

        # Find PLY file based on structure
        ply_file = self._find_ply_file(location_dir, sequence, threshold)
        if ply_file is None or not ply_file.exists():
            logger.error(f"PLY file not found for {location}/{sequence}/thr{threshold}")
            return None

        # Load metadata for GPS coordinates
        metadata = self._find_metadata(location_dir) or {}
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

    def _find_ply_file(
        self, location_dir: Path, sequence: str, threshold: float
    ) -> Optional[Path]:
        """
        Find PLY file supporting both directory structures.

        1. Legacy: pointclouds/sequence_N/scene_thrX.X.ply
        2. New: scene_thrX.X.ply or aligned_pointcloud.ply directly in folder
        """
        # Try legacy structure first
        legacy_path = location_dir / "pointclouds" / sequence / f"scene_thr{threshold}.ply"
        if legacy_path.exists():
            return legacy_path

        # Try new structure - direct PLY in folder
        if sequence == "default" or sequence == "sequence_1":
            # Try threshold-specific file
            direct_path = location_dir / f"scene_thr{threshold}.ply"
            if direct_path.exists():
                return direct_path

            # Try aligned pointcloud (from experiment runner)
            aligned_path = location_dir / "aligned_pointcloud.ply"
            if aligned_path.exists():
                return aligned_path

            # Try any threshold file
            for thr_file in location_dir.glob("scene_thr*.ply"):
                logger.info(f"Using available threshold file: {thr_file.name}")
                return thr_file

        return None

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
