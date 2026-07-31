# Experiment Framework Plan: Modern Cartography Pipeline

## Executive Summary

A simplified, modular framework for testing 3D reconstruction models and building a crowd-sourced trail mapping system.

**Target Models (Phase 1):** MASt3R, VGGT, DA3-Streaming, ORB-SLAM (Python reimplementation)

---

## Part 1: Core Problems & Solutions

### Problem 1: Metric Scale

| Model | Metric Scale? | Notes |
|-------|--------------|-------|
| MASt3R | No | Relative scale only - need GPS for scale recovery |
| VGGT | Likely Yes | Outputs camera params + depth + point maps |
| DA3-Streaming | Likely Yes | Outputs poses + depth + combined_pcd.ply |
| ORB-SLAM | No | Needs IMU or GPS for scale |

### Problem 2: Global Alignment
Use GPS trajectory from GoPro telemetry (via gopro-py) to:
1. Recover metric scale (compare trajectory lengths)
2. Align to ENU (East-North-Up) coordinate frame
3. Rotate using gravity vector from IMU

### Problem 3: Multi-Video Fusion
Only needed when point clouds overlap. Strategy:
1. GPS-based coarse alignment
2. Feature matching in overlap regions
3. ICP refinement
4. Point deduplication

---

## Part 2: Simplified Architecture

```
mapper/
├── src/
│   ├── core/                      # Core types only
│   │   ├── __init__.py
│   │   ├── types.py               # PointCloud, VideoMetadata, etc.
│   │   └── config.py              # Simple dataclass configs
│   │
│   ├── models/                    # Model wrappers (plug-and-play)
│   │   ├── __init__.py            # Simple get_model() function
│   │   ├── base.py                # BaseModel class
│   │   ├── must3r.py              # MASt3R (existing, refactored)
│   │   ├── vggt.py                # VGGT
│   │   ├── da3_streaming.py       # DA3-Streaming
│   │   └── orb_slam.py            # ORB-SLAM (Python implementation)
│   │
│   ├── preprocessing/             # Input processing
│   │   ├── __init__.py
│   │   ├── video_processor.py     # Frame extraction (smart caching)
│   │   └── telemetry.py           # GoPro telemetry via gopro-py
│   │
│   ├── alignment/                 # GPS/IMU alignment
│   │   ├── __init__.py
│   │   ├── gps_aligner.py         # Scale recovery + positioning
│   │   └── icp_refiner.py         # Fine alignment for fusion
│   │
│   ├── experiments/               # Lightweight experiment runner
│   │   ├── __init__.py
│   │   ├── runner.py              # Run experiments from config
│   │   └── metrics.py             # Compute comparison metrics
│   │
│   ├── preprocessor.py            # (existing)
│   ├── constructor.py             # (existing, will be refactored)
│   └── process_folder.py          # (existing)
│
├── configs/                       # YAML configs
│   ├── default.yaml               # Default settings
│   └── models/
│       ├── must3r.yaml
│       ├── vggt.yaml
│       ├── da3_streaming.yaml
│       └── orb_slam.yaml
│
├── experiments/                   # Experiment outputs
│   └── {experiment_name}/
│       ├── config.yaml            # Copy of config used
│       ├── metrics.json           # Results
│       └── outputs/               # Model outputs
│
└── viewer/                        # (existing, enhanced)
```

---

## Part 3: Core Types

```python
# src/core/types.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np

@dataclass
class PointCloud:
    """Unified point cloud output"""
    points: np.ndarray              # (N, 3) XYZ
    colors: Optional[np.ndarray] = None  # (N, 3) RGB 0-255

    # Optional fields - models may or may not provide these
    confidence: Optional[np.ndarray] = None  # (N,) per-point confidence
    normals: Optional[np.ndarray] = None     # (N, 3) surface normals

    # Geo-referencing (filled in by alignment step)
    origin_gps: Optional[tuple] = None       # (lat, lon, alt)
    scale: float = 1.0                       # Scale factor applied
    is_metric: bool = False                  # Whether scale is metric

    def save_ply(self, path: Path) -> None:
        """Save to PLY file"""
        from plyfile import PlyData, PlyElement
        # ... implementation

@dataclass
class CameraPoses:
    """Camera pose output (optional)"""
    poses: np.ndarray               # (M, 4, 4) world-to-camera transforms
    timestamps: Optional[np.ndarray] = None  # (M,) frame timestamps
    intrinsics: Optional[np.ndarray] = None  # (3, 3) or (M, 3, 3)

@dataclass
class ReconstructionResult:
    """What every model returns"""
    pointcloud: PointCloud
    poses: Optional[CameraPoses] = None
    metadata: dict = field(default_factory=dict)  # Model-specific info

@dataclass
class GPSTrack:
    """GPS trajectory from GoPro telemetry"""
    latitudes: np.ndarray
    longitudes: np.ndarray
    altitudes: np.ndarray
    timestamps: np.ndarray
    accuracies: Optional[np.ndarray] = None

@dataclass
class IMUData:
    """IMU data from GoPro telemetry"""
    accelerometer: np.ndarray      # (N, 3) m/s^2
    gyroscope: np.ndarray          # (N, 3) rad/s
    gravity_vectors: Optional[np.ndarray] = None  # (N, 3)
    orientations: Optional[np.ndarray] = None     # (N, 4) quaternions wxyz
    timestamps: np.ndarray

@dataclass
class VideoInput:
    """Processed video input"""
    image_dir: Path                # Directory of extracted frames
    fps: float                     # Extraction frame rate
    frame_count: int
    gps_track: Optional[GPSTrack] = None
    imu_data: Optional[IMUData] = None
    metadata: dict = field(default_factory=dict)
```

---

## Part 4: Base Model Interface

```python
# src/models/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from ..core.types import ReconstructionResult, VideoInput

class BaseModel(ABC):
    """
    Simple base class for reconstruction models.

    Each model just needs to implement:
    - load() - load weights
    - reconstruct() - run reconstruction
    - get_default_config() - return default settings
    """

    name: str = "base"

    # Declare capabilities (for informational purposes)
    outputs_metric_scale: bool = False
    outputs_poses: bool = False
    outputs_confidence: bool = False
    supports_video_input: bool = False

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    @abstractmethod
    def load(self, weights_path: Optional[Path] = None) -> None:
        """Load model weights"""
        pass

    @abstractmethod
    def reconstruct(self, video_input: VideoInput, output_dir: Path) -> ReconstructionResult:
        """
        Run reconstruction.

        Args:
            video_input: Processed video with frames and optional telemetry
            output_dir: Where to save outputs

        Returns:
            ReconstructionResult with at minimum a PointCloud
        """
        pass

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default configuration for this model"""
        return {}


# Simple model getter - no fancy registry needed
def get_model(name: str) -> type:
    """Get model class by name"""
    models = {
        "must3r": "src.models.must3r.MASt3RModel",
        "vggt": "src.models.vggt.VGGTModel",
        "da3_streaming": "src.models.da3_streaming.DA3StreamingModel",
        "orb_slam": "src.models.orb_slam.ORBSLAMModel",
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    # Lazy import
    module_path, class_name = models[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

**Why no decorator registry?**

The `@register_model` decorator pattern is useful when you have many plugins that auto-register on import. For your case with 4 models that you'll manually maintain, a simple dict is clearer and easier to understand. You can see exactly which models exist by looking at `get_model()`.

---

## Part 5: Model Implementations

### 5.1 MASt3R (Refactored from existing)

```python
# src/models/must3r.py

from .base import BaseModel
from ..core.types import ReconstructionResult, PointCloud, CameraPoses, VideoInput

class MASt3RModel(BaseModel):
    name = "must3r"
    outputs_metric_scale = False
    outputs_poses = True
    outputs_confidence = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.window_size = config.get("window_size", 500)
        self.window_overlap = config.get("window_overlap", 20)
        self.confidence_thresholds = config.get("confidence_thresholds", [5.0, 2.0, 1.5, 1.05])
        self.num_mem_imgs = config.get("num_mem_imgs", 50)
        self.image_size = config.get("image_size", 512)

    def load(self, weights_path=None):
        from must3r.model import load_model
        self.model = load_model(
            weights_path or self.config.get("weights_path"),
            device='cuda',
            img_size=self.image_size
        )

    def reconstruct(self, video_input: VideoInput, output_dir: Path) -> ReconstructionResult:
        from must3r.demo.gradio import get_reconstructed_scene, get_3D_model_from_scene

        # Load images
        images = self._load_images(video_input.image_dir)

        # Run reconstruction (handles windowing internally or we do it)
        scene = get_reconstructed_scene(
            self.model, images,
            num_mem_imgs=self.num_mem_imgs,
            # ... other params
        )

        # Extract point cloud at best threshold
        points, colors, confidence = self._extract_pointcloud(scene, self.confidence_thresholds[0])
        poses = self._extract_poses(scene)

        # Save PLY files at multiple thresholds
        for thr in self.confidence_thresholds:
            ply_path = output_dir / f"scene_thr{thr}.ply"
            get_3D_model_from_scene(scene, ply_path, thr)

        return ReconstructionResult(
            pointcloud=PointCloud(
                points=points,
                colors=colors,
                confidence=confidence,
                is_metric=False
            ),
            poses=CameraPoses(poses=poses) if poses is not None else None,
            metadata={
                "thresholds": self.confidence_thresholds,
                "window_size": self.window_size
            }
        )

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "window_size": 500,
            "window_overlap": 20,
            "confidence_thresholds": [5.0, 2.0, 1.5, 1.05],
            "num_mem_imgs": 50,
            "image_size": 512,
        }
```

### 5.2 VGGT

```python
# src/models/vggt.py

class VGGTModel(BaseModel):
    """
    VGGT: Visual Geometry Grounded Transformer

    Outputs:
    - Camera extrinsics and intrinsics
    - Depth maps with confidence
    - Point maps with confidence
    - 3D point clouds (via unprojection)

    Likely metric scale (uses geometry grounding)
    """

    name = "vggt"
    outputs_metric_scale = True  # To verify
    outputs_poses = True
    outputs_confidence = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_frames = config.get("max_frames", 100)

    def load(self, weights_path=None):
        # VGGT auto-downloads from HuggingFace
        # pip install torch torchvision numpy Pillow huggingface_hub
        from vggt.model import VGGT
        self.model = VGGT.from_pretrained("facebook/vggt-1b")
        self.model.cuda().eval()

    def reconstruct(self, video_input: VideoInput, output_dir: Path) -> ReconstructionResult:
        # Load and preprocess images
        images = self._load_images(video_input.image_dir, max_frames=self.max_frames)

        # Run VGGT
        with torch.no_grad():
            outputs = self.model.predict(images)

        # Extract outputs
        # outputs contains: camera_params, depth_maps, point_maps, confidence
        pointcloud = self._build_pointcloud(outputs)
        poses = self._extract_poses(outputs)

        # Save to COLMAP format for compatibility
        self._export_colmap(outputs, output_dir / "colmap")

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={"model": "vggt", "frames_processed": len(images)}
        )
```

### 5.3 DA3-Streaming

```python
# src/models/da3_streaming.py

class DA3StreamingModel(BaseModel):
    """
    Depth Anything 3 - Streaming variant

    Outputs:
    - camera_poses.txt - extrinsic matrices
    - intrinsic.txt - camera intrinsics
    - combined_pcd.ply - fused point cloud
    - Per-frame depth maps (optional)

    Designed for video with temporal consistency
    """

    name = "da3_streaming"
    outputs_metric_scale = True  # To verify
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.save_per_frame = config.get("save_per_frame_depth", False)

    def load(self, weights_path=None):
        # DA3 streaming setup
        # See: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/
        pass

    def reconstruct(self, video_input: VideoInput, output_dir: Path) -> ReconstructionResult:
        # DA3-Streaming expects image directory
        # It outputs: camera_poses.txt, intrinsic.txt, combined_pcd.ply

        # Run DA3 streaming
        self._run_da3_streaming(video_input.image_dir, output_dir)

        # Load outputs
        pointcloud = self._load_combined_ply(output_dir / "combined_pcd.ply")
        poses = self._load_poses(output_dir / "camera_poses.txt")
        intrinsics = self._load_intrinsics(output_dir / "intrinsic.txt")

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=CameraPoses(poses=poses, intrinsics=intrinsics),
            metadata={"model": "da3_streaming"}
        )
```

### 5.4 ORB-SLAM (Python Implementation)

```python
# src/models/orb_slam.py

class ORBSLAMModel(BaseModel):
    """
    ORB-SLAM reimplemented in Python

    Core components:
    1. ORB feature extraction (OpenCV)
    2. Feature matching
    3. Essential matrix / PnP pose estimation
    4. Local bundle adjustment
    5. Loop closure detection

    Does NOT output metric scale without IMU
    """

    name = "orb_slam"
    outputs_metric_scale = False
    outputs_poses = True
    outputs_confidence = False

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_features = config.get("num_features", 2000)
        self.use_imu = config.get("use_imu", False)

    def load(self, weights_path=None):
        # No weights needed - uses OpenCV ORB
        import cv2
        self.orb = cv2.ORB_create(nfeatures=self.num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def reconstruct(self, video_input: VideoInput, output_dir: Path) -> ReconstructionResult:
        """
        Basic ORB-SLAM pipeline:
        1. Extract ORB features from each frame
        2. Match features between consecutive frames
        3. Estimate relative pose (Essential matrix)
        4. Triangulate 3D points
        5. Run local BA
        6. Detect loop closures
        """

        images = self._load_images(video_input.image_dir)

        # Initialize
        poses = [np.eye(4)]  # First frame at origin
        map_points = []

        for i in range(1, len(images)):
            # Extract features
            kp1, desc1 = self.orb.detectAndCompute(images[i-1], None)
            kp2, desc2 = self.orb.detectAndCompute(images[i], None)

            # Match
            matches = self.matcher.match(desc1, desc2)

            # Estimate pose
            E, mask = cv2.findEssentialMat(pts1, pts2, K)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

            # Update pose
            pose = poses[-1] @ self._make_transform(R, t)
            poses.append(pose)

            # Triangulate new points
            new_points = self._triangulate(kp1, kp2, matches, poses[-2], poses[-1])
            map_points.extend(new_points)

        # Optional: use IMU for scale if available
        if self.use_imu and video_input.imu_data is not None:
            scale = self._estimate_scale_from_imu(poses, video_input.imu_data)
            map_points = [p * scale for p in map_points]

        return ReconstructionResult(
            pointcloud=PointCloud(points=np.array(map_points), is_metric=self.use_imu),
            poses=CameraPoses(poses=np.array(poses)),
            metadata={"model": "orb_slam", "num_features": self.num_features}
        )
```

---

## Part 6: Smart Frame Extraction (Caching)

```python
# src/preprocessing/video_processor.py

from pathlib import Path
import json
import subprocess

class VideoProcessor:
    """
    Extract frames from video with smart caching.

    Reuses existing frames if:
    - Same video file (by name + size)
    - Same FPS
    - Frames already exist
    """

    def __init__(self, fps: float = 10, jpeg_quality: int = 10):
        self.fps = fps
        self.jpeg_quality = jpeg_quality

    def process(self, video_path: Path, output_dir: Path) -> Path:
        """
        Extract frames, reusing cache if valid.

        Returns: Path to image directory
        """
        image_dir = output_dir / "images"
        cache_file = image_dir / ".cache_info.json"

        # Check if we can reuse existing frames
        if self._can_reuse_cache(video_path, cache_file):
            print(f"Reusing cached frames from {image_dir}")
            return image_dir

        # Extract frames
        image_dir.mkdir(parents=True, exist_ok=True)
        self._extract_frames(video_path, image_dir)

        # Save cache info
        self._save_cache_info(video_path, cache_file)

        return image_dir

    def _can_reuse_cache(self, video_path: Path, cache_file: Path) -> bool:
        """Check if cached frames are valid for this video"""
        if not cache_file.exists():
            return False

        with open(cache_file) as f:
            cache = json.load(f)

        # Verify video matches
        if cache.get("video_name") != video_path.name:
            return False
        if cache.get("video_size") != video_path.stat().st_size:
            return False
        if cache.get("fps") != self.fps:
            return False

        # Verify frames exist
        expected_frames = cache.get("frame_count", 0)
        actual_frames = len(list(cache_file.parent.glob("frame_*.jpg")))
        if actual_frames < expected_frames * 0.95:  # Allow 5% tolerance
            return False

        return True

    def _extract_frames(self, video_path: Path, output_dir: Path):
        """Extract frames using ffmpeg"""
        # Use hardware acceleration if available
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={self.fps}",
            "-q:v", str(self.jpeg_quality),
            str(output_dir / "frame_%04d.jpg")
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _save_cache_info(self, video_path: Path, cache_file: Path):
        """Save cache metadata"""
        frame_count = len(list(cache_file.parent.glob("frame_*.jpg")))
        cache = {
            "video_name": video_path.name,
            "video_size": video_path.stat().st_size,
            "fps": self.fps,
            "frame_count": frame_count,
        }
        with open(cache_file, "w") as f:
            json.dump(cache, f)
```

---

## Part 7: GoPro Telemetry Integration

```python
# src/preprocessing/telemetry.py

from pathlib import Path
from ..core.types import GPSTrack, IMUData
import numpy as np

class TelemetryExtractor:
    """
    Extract telemetry from GoPro videos using gopro-py library.

    Provides:
    - GPS track (lat, lon, alt, timestamps)
    - IMU data (accelerometer, gyroscope)
    - Camera orientations (quaternions)
    - Gravity vectors
    """

    def extract(self, video_path: Path) -> tuple[GPSTrack, IMUData]:
        """Extract all telemetry from GoPro video"""
        import gopropy

        telemetry = gopropy.load(str(video_path))

        gps_track = self._extract_gps(telemetry)
        imu_data = self._extract_imu(telemetry)

        return gps_track, imu_data

    def _extract_gps(self, telemetry) -> GPSTrack:
        """Extract GPS track"""
        try:
            gps = telemetry.get_stream("GPS")
            df = gps.to_dataframe()

            return GPSTrack(
                latitudes=df["GPS_lat"].values,
                longitudes=df["GPS_lon"].values,
                altitudes=df["GPS_alt"].values if "GPS_alt" in df else None,
                timestamps=df["timestamp"].values,
            )
        except KeyError:
            return None

    def _extract_imu(self, telemetry) -> IMUData:
        """Extract IMU data"""
        try:
            accel = telemetry.get_stream("Accelerometer")
            gyro = telemetry.get_stream("Gyroscope")

            # Get gravity and orientation if available (Hero8+)
            gravity = None
            orientations = None
            try:
                grav_stream = telemetry.get_stream("Gravity Vector")
                gravity = grav_stream.data
            except KeyError:
                pass

            try:
                cori_stream = telemetry.get_stream("CameraOrientation")
                orientations = cori_stream.data  # Quaternions (w, x, y, z)
            except KeyError:
                pass

            return IMUData(
                accelerometer=accel.data,
                gyroscope=gyro.data,
                gravity_vectors=gravity,
                orientations=orientations,
                timestamps=accel.timestamps,
            )
        except KeyError:
            return None
```

---

## Part 8: Alignment

### 8.1 GPS Aligner (Scale Recovery + Positioning)

```python
# src/alignment/gps_aligner.py

import numpy as np
from ..core.types import PointCloud, CameraPoses, GPSTrack

class GPSAligner:
    """
    Align reconstruction to GPS coordinate system.

    Does two things:
    1. Scale recovery - match trajectory length to GPS distance
    2. Position alignment - place point cloud at GPS location
    """

    def align(
        self,
        pointcloud: PointCloud,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> PointCloud:
        """
        Align point cloud to GPS coordinates.

        Steps:
        1. Compute scale from GPS vs pose trajectory lengths
        2. Compute rotation to align trajectories
        3. Compute translation to GPS origin
        4. Apply transformation
        """

        # Step 1: Scale recovery
        gps_length = self._compute_gps_trajectory_length(gps_track)
        pose_length = self._compute_pose_trajectory_length(poses)
        scale = gps_length / pose_length if pose_length > 0 else 1.0

        # Step 2: Compute alignment rotation (Kabsch algorithm)
        # Match camera trajectory to GPS trajectory
        pose_positions = poses.poses[:, :3, 3]  # Extract translation vectors
        gps_positions = self._gps_to_local_enu(gps_track)

        # Subsample to matching count
        pose_positions, gps_positions = self._align_sample_counts(
            pose_positions, gps_positions
        )

        rotation, translation = self._kabsch_align(
            pose_positions * scale, gps_positions
        )

        # Step 3: Apply transformation to point cloud
        transform = np.eye(4)
        transform[:3, :3] = rotation * scale
        transform[:3, 3] = translation

        aligned_points = (transform[:3, :3] @ pointcloud.points.T).T + transform[:3, 3]

        return PointCloud(
            points=aligned_points,
            colors=pointcloud.colors,
            confidence=pointcloud.confidence,
            origin_gps=(gps_track.latitudes[0], gps_track.longitudes[0],
                       gps_track.altitudes[0] if gps_track.altitudes is not None else 0),
            scale=scale,
            is_metric=True,
        )

    def _compute_gps_trajectory_length(self, gps: GPSTrack) -> float:
        """Compute total GPS trajectory length in meters (haversine)"""
        from math import radians, sin, cos, sqrt, atan2

        total = 0.0
        R = 6371000  # Earth radius in meters

        for i in range(1, len(gps.latitudes)):
            lat1, lat2 = radians(gps.latitudes[i-1]), radians(gps.latitudes[i])
            lon1, lon2 = radians(gps.longitudes[i-1]), radians(gps.longitudes[i])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            total += R * c

        return total

    def _gps_to_local_enu(self, gps: GPSTrack) -> np.ndarray:
        """Convert GPS to local ENU (East-North-Up) coordinates"""
        # Use first point as origin
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")  # WGS84 to ECEF

        # Convert to ECEF then to local ENU
        # ... implementation
        pass
```

### 8.2 Why Multiple Alignment Files?

The alignment module has these responsibilities:

1. **gps_aligner.py** - Primary alignment
   - Scale recovery from GPS trajectory
   - Initial positioning using GPS
   - This is what makes non-metric models (MASt3R, ORB-SLAM) usable

2. **icp_refiner.py** - Fine alignment for fusion
   - Only used when merging multiple overlapping point clouds
   - Takes two roughly-aligned point clouds and refines the alignment
   - Uses ICP (Iterative Closest Point) algorithm

You're right that fusion is only needed for overlapping point clouds. For single-video processing, only `gps_aligner.py` is used.

---

## Part 9: Metrics Calculation

```python
# src/experiments/metrics.py

import numpy as np
from ..core.types import PointCloud, GPSTrack, CameraPoses

class MetricsCalculator:
    """
    Compute evaluation metrics for reconstruction quality.

    Current metrics:
    1. Scale accuracy - how close to metric scale
    2. GPS alignment error - RMSE of camera positions vs GPS
    3. Point density - points per cubic meter
    4. Trajectory consistency - smoothness of estimated path
    """

    def compute_all(
        self,
        pointcloud: PointCloud,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> dict:
        """Compute all metrics"""
        return {
            "scale_accuracy": self.scale_accuracy(poses, gps_track),
            "gps_rmse_meters": self.gps_alignment_rmse(poses, gps_track),
            "point_count": len(pointcloud.points),
            "point_density": self.point_density(pointcloud),
            "trajectory_length_meters": self.trajectory_length(poses),
        }

    def scale_accuracy(self, poses: CameraPoses, gps_track: GPSTrack) -> float:
        """
        Compute scale accuracy as ratio of trajectory lengths.

        Returns: ratio (1.0 = perfect, <1 = underestimate, >1 = overestimate)
        """
        pose_length = self._pose_trajectory_length(poses)
        gps_length = self._gps_trajectory_length(gps_track)

        if gps_length == 0:
            return None

        return pose_length / gps_length

    def gps_alignment_rmse(self, poses: CameraPoses, gps_track: GPSTrack) -> float:
        """
        Compute RMSE between camera positions and GPS track (in meters).

        Assumes poses are already scaled and aligned.
        """
        # Extract pose positions
        pose_positions = poses.poses[:, :3, 3]

        # Convert GPS to same coordinate system
        gps_positions = self._gps_to_local(gps_track)

        # Align sample counts via interpolation
        aligned_gps = self._interpolate_to_match(gps_positions, gps_track.timestamps,
                                                  poses.timestamps)

        # Compute RMSE
        errors = np.linalg.norm(pose_positions - aligned_gps, axis=1)
        return np.sqrt(np.mean(errors ** 2))

    def point_density(self, pointcloud: PointCloud) -> float:
        """Compute points per cubic meter"""
        if len(pointcloud.points) == 0:
            return 0

        # Compute bounding box volume
        mins = pointcloud.points.min(axis=0)
        maxs = pointcloud.points.max(axis=0)
        volume = np.prod(maxs - mins)

        if volume == 0:
            return float('inf')

        return len(pointcloud.points) / volume
```

---

## Part 10: Experiment Runner

```python
# src/experiments/runner.py

from pathlib import Path
from dataclasses import dataclass
import yaml
import json
from datetime import datetime

from ..models import get_model
from ..preprocessing.video_processor import VideoProcessor
from ..preprocessing.telemetry import TelemetryExtractor
from ..alignment.gps_aligner import GPSAligner
from .metrics import MetricsCalculator

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str
    model: str
    model_config: dict
    input_folder: Path
    output_folder: Path
    fps: float = 10
    align_to_gps: bool = True

class ExperimentRunner:
    """
    Run experiments on a folder of videos.

    Simple workflow:
    1. Load config
    2. Process each video with the model
    3. Compute metrics
    4. Save results
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.video_processor = VideoProcessor(fps=config.fps)
        self.telemetry_extractor = TelemetryExtractor()
        self.gps_aligner = GPSAligner()
        self.metrics_calculator = MetricsCalculator()

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'ExperimentRunner':
        """Load experiment from YAML config"""
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        config = ExperimentConfig(
            name=raw["name"],
            model=raw["model"],
            model_config=raw.get("model_config", {}),
            input_folder=Path(raw["input_folder"]),
            output_folder=Path(raw["output_folder"]),
            fps=raw.get("fps", 10),
            align_to_gps=raw.get("align_to_gps", True),
        )
        return cls(config)

    def run(self):
        """Run the experiment"""
        print(f"Starting experiment: {self.config.name}")

        # Setup
        exp_dir = self._setup_experiment_dir()
        model_cls = get_model(self.config.model)
        model = model_cls(self.config.model_config)
        model.load()

        # Find videos
        videos = list(self.config.input_folder.glob("*.MP4")) + \
                 list(self.config.input_folder.glob("*.MOV")) + \
                 list(self.config.input_folder.glob("*.mp4"))

        results = []

        for video_path in videos:
            print(f"Processing: {video_path.name}")

            try:
                result = self._process_video(video_path, model, exp_dir)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"video": video_path.name, "error": str(e)})

        # Save summary
        self._save_results(exp_dir, results)

        print(f"Experiment complete. Results in: {exp_dir}")
        return results

    def _process_video(self, video_path: Path, model, exp_dir: Path) -> dict:
        """Process a single video"""
        video_name = video_path.stem
        output_dir = exp_dir / "outputs" / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract frames (with caching)
        image_dir = self.video_processor.process(video_path, output_dir)

        # Step 2: Extract telemetry
        gps_track, imu_data = self.telemetry_extractor.extract(video_path)

        # Step 3: Create video input
        from ..core.types import VideoInput
        video_input = VideoInput(
            image_dir=image_dir,
            fps=self.config.fps,
            frame_count=len(list(image_dir.glob("frame_*.jpg"))),
            gps_track=gps_track,
            imu_data=imu_data,
        )

        # Step 4: Run reconstruction
        result = model.reconstruct(video_input, output_dir)

        # Step 5: Align to GPS (if available and enabled)
        if self.config.align_to_gps and gps_track is not None and result.poses is not None:
            result.pointcloud = self.gps_aligner.align(
                result.pointcloud, result.poses, gps_track
            )

        # Step 6: Compute metrics
        metrics = {}
        if gps_track is not None and result.poses is not None:
            metrics = self.metrics_calculator.compute_all(
                result.pointcloud, result.poses, gps_track
            )

        # Step 7: Save outputs
        result.pointcloud.save_ply(output_dir / "aligned_pointcloud.ply")

        return {
            "video": video_path.name,
            "metrics": metrics,
            "output_dir": str(output_dir),
            "point_count": len(result.pointcloud.points),
            "is_metric": result.pointcloud.is_metric,
        }

    def _setup_experiment_dir(self) -> Path:
        """Create experiment output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.config.output_folder / f"{self.config.name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config copy
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump({
                "name": self.config.name,
                "model": self.config.model,
                "model_config": self.config.model_config,
                "input_folder": str(self.config.input_folder),
                "fps": self.config.fps,
            }, f)

        return exp_dir

    def _save_results(self, exp_dir: Path, results: list):
        """Save experiment results"""
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
```

### Example Config

```yaml
# configs/experiments/vggt_test.yaml

name: "vggt_trail_test"
model: "vggt"
model_config:
  max_frames: 200

input_folder: "/home/ape/gopro_videos/tahoe_trails"
output_folder: "/home/ape/mapper_output/experiments"

fps: 10
align_to_gps: true
```

### CLI Usage

```bash
# Run experiment
python -m src.experiments.runner --config configs/experiments/vggt_test.yaml

# Or via main CLI
python -m src.cli experiment --config configs/experiments/vggt_test.yaml
```

---

## Part 11: Viewer Enhancements

### 11.1 Free 3D Terrain Options

Based on research, here are the best free options for 3D terrain:

| Option | Pros | Cons |
|--------|------|------|
| **MapLibre GL + Terrain RGB** | Free, open source, native 3D terrain | Need to find/host terrain tiles |
| **MapLibre Demo Tiles** | Free demo tiles available | Lower resolution, demo only |
| **MapTiler Free Tier** | High quality, easy setup | Limited requests/month |
| **Cesium Ion** | Full 3D tiles, terrain | Free tier limited |
| **Self-hosted DEM** | Full control | Need to process/host tiles |

**Recommended approach**: Use MapLibre GL JS with free terrain tiles, then generate your own elevation from point clouds.

### 11.2 Viewer Updates

```jsx
// viewer/frontend/src/App.jsx enhancements

// 1. Add 3D terrain support
const TERRAIN_SOURCE = {
  type: 'raster-dem',
  url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json',
  tileSize: 256
};

// 2. Enable terrain in map style
map.setTerrain({
  source: 'terrain',
  exaggeration: 1.5  // Adjust for visibility
});

// 3. Add sky layer for better 3D effect
map.addLayer({
  id: 'sky',
  type: 'sky',
  paint: {
    'sky-type': 'atmosphere',
    'sky-atmosphere-sun': [0.0, 90.0],
    'sky-atmosphere-sun-intensity': 15
  }
});
```

### 11.3 New API Endpoints

```python
# viewer/backend/server.py additions

@app.get("/api/experiments")
async def list_experiments():
    """List all experiments in the experiments folder"""
    exp_dir = Path("experiments")
    experiments = []

    for exp_path in exp_dir.iterdir():
        if exp_path.is_dir():
            config_path = exp_path / "config.yaml"
            results_path = exp_path / "results.json"

            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                results = None
                if results_path.exists():
                    with open(results_path) as f:
                        results = json.load(f)

                experiments.append({
                    "name": exp_path.name,
                    "model": config.get("model"),
                    "created": exp_path.stat().st_mtime,
                    "video_count": len(results) if results else 0,
                })

    return experiments

@app.get("/api/experiments/{exp_name}/pointclouds")
async def get_experiment_pointclouds(exp_name: str):
    """Get all pointclouds from an experiment"""
    exp_dir = Path("experiments") / exp_name / "outputs"

    pointclouds = []
    for video_dir in exp_dir.iterdir():
        if video_dir.is_dir():
            ply_path = video_dir / "aligned_pointcloud.ply"
            if ply_path.exists():
                pointclouds.append({
                    "video": video_dir.name,
                    "path": str(ply_path),
                })

    return pointclouds

@app.get("/api/elevation/generate")
async def generate_elevation_tiles(
    location_name: str,
    zoom: int = 15,
):
    """
    Generate elevation tiles from our point cloud data.

    This creates terrain-rgb tiles from the point cloud,
    allowing us to show our own elevation data.
    """
    # Load point cloud
    # Rasterize to DEM grid
    # Encode as terrain-rgb
    # Return as tile
    pass
```

### 11.4 Custom Elevation from Point Clouds

One exciting possibility: generate your own terrain from the point cloud data!

```python
# viewer/backend/elevation_generator.py

import numpy as np
from scipy.interpolate import griddata

def pointcloud_to_dem(
    points: np.ndarray,
    resolution: float = 1.0,  # meters per pixel
) -> tuple[np.ndarray, dict]:
    """
    Convert point cloud to Digital Elevation Model.

    Returns: (dem_array, bounds_dict)
    """
    # Get bounds
    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)

    # Create grid
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)

    xi = np.linspace(x_min, x_max, x_bins)
    yi = np.linspace(y_min, y_max, y_bins)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate elevation
    dem = griddata(
        points[:, :2],  # x, y
        points[:, 2],   # z (elevation)
        (xi, yi),
        method='linear'
    )

    return dem, {"x_min": x_min, "y_min": y_min, "resolution": resolution}

def dem_to_terrain_rgb(dem: np.ndarray) -> np.ndarray:
    """
    Encode DEM as terrain-RGB tiles.

    Formula: height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
    """
    # Offset and scale
    encoded = (dem + 10000) / 0.1

    # Encode to RGB
    r = (encoded // (256 * 256)).astype(np.uint8)
    g = ((encoded // 256) % 256).astype(np.uint8)
    b = (encoded % 256).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)
```

---

## Part 12: Implementation Priority

### Phase 1: Core Infrastructure
- [ ] Create `src/core/types.py` with data classes
- [ ] Create `src/models/base.py` with BaseModel
- [ ] Refactor existing MASt3R to new interface
- [ ] Create `src/preprocessing/video_processor.py` with caching
- [ ] Integrate gopro-py for telemetry extraction

### Phase 2: GPS Alignment
- [ ] Implement `src/alignment/gps_aligner.py`
- [ ] Add scale recovery from GPS trajectory
- [ ] Test with MASt3R output

### Phase 3: Additional Models
- [ ] Integrate VGGT
- [ ] Integrate DA3-Streaming
- [ ] Create ORB-SLAM Python implementation (basic version)

### Phase 4: Experiment Runner
- [ ] Create simple experiment runner
- [ ] Add metrics calculation
- [ ] Create comparison output

### Phase 5: Viewer Enhancements
- [ ] Add 3D terrain to MapLibre
- [ ] Add experiment API endpoints
- [ ] Create elevation generation from point clouds

---

## Answers to Your Questions

### Q: What if models don't return confidence or poses?
**A:** The `ReconstructionResult` has optional fields. Models just return what they have:
```python
return ReconstructionResult(
    pointcloud=PointCloud(points=pts, colors=colors),
    poses=None,  # Not available
    metadata={}
)
```
The alignment step checks for poses before trying to align.

### Q: What's the purpose of register_model?
**A:** Removed it! For 4 models, a simple dict in `get_model()` is clearer. The decorator pattern is useful for plugin architectures with many auto-registering modules, but overkill here.

### Q: Fusion only for overlapping pointclouds?
**A:** Correct. The ICP refiner is only used when you want to merge multiple point clouds from different videos of the same trail. For single videos, you just use GPS alignment.

### Q: What are all the alignment files for?
**A:** Simplified to two:
- `gps_aligner.py` - Scale recovery + GPS positioning (always used)
- `icp_refiner.py` - Only for merging overlapping point clouds

### Q: How do we calculate metrics?
**A:** Using GPS as ground truth:
- **Scale accuracy** = pose_trajectory_length / gps_trajectory_length
- **GPS RMSE** = sqrt(mean(|pose_position - gps_position|²))
- **Point density** = point_count / bounding_box_volume

### Q: Viewer 3D terrain?
**A:** Use MapLibre GL JS with free terrain tiles. Can also generate our own elevation tiles from the point cloud data!

---

## Summary

This simplified plan:

1. **No external tracking** - Just JSON files, no W&B/MLflow
2. **Folder-based I/O** - Input folder of videos, output folder for results
3. **Smart caching** - Reuses extracted frames if video/fps matches
4. **Simple model interface** - No fancy registry, just a dict
5. **GoPro integration** - Full telemetry via your gopro-py library
6. **Practical metrics** - GPS-based validation
7. **Incremental viewer updates** - 3D terrain + experiment APIs
