"""External MASt3R-SLAM adapter for trajectory and loop-closure controls."""

from pathlib import Path
from typing import Optional
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from .base import BaseModel
from ..core.types import (
    CameraPoses,
    PointCloud,
    PoseConvention,
    ReconstructionResult,
    VideoInput,
)

logger = logging.getLogger(__name__)


class MASt3RSLAMModel(BaseModel):
    """Run the official MASt3R-SLAM executable in headless mode.

    MASt3R-SLAM exports optimized keyframe poses rather than one pose per input
    frame.  The adapter preserves those keyframe identities and imports the
    accompanying dense keyframe reconstruction.

    Reference: https://github.com/rmurai0610/MASt3R-SLAM
    """

    name = "mast3r_slam"
    outputs_metric_scale = False
    outputs_poses = True
    # Confidence is used to filter the PLY but is not retained as an attribute.
    outputs_confidence = False
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.mast3r_slam_path = self.config.get("mast3r_slam_path")
        self.mast3r_slam_config_path = self.config.get("mast3r_slam_config_path")
        self.weights_dir = self.config.get("weights_dir", "weights/mast3r_slam")
        self.calibration_path = self.config.get("calibration_path")
        self.python_executable = self.config.get("python_executable", sys.executable)
        self.rgb_folder_timebase_hz = float(
            self.config.get("rgb_folder_timebase_hz", 30.0)
        )
        self.upstream_revision = self.config.get(
            "upstream_revision",
            "e6f4e3d474fad0e11f561482012be864ba8c3f17",
        )
        self.verify_upstream_revision = self.config.get(
            "verify_upstream_revision", True
        )
        self.mast3r_slam_dir: Optional[Path] = None

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "mast3r_slam_path": None,
            "mast3r_slam_config_path": None,
            "weights_dir": "weights/mast3r_slam",
            "calibration_path": None,
            "python_executable": sys.executable,
            # Official RGBFiles assigns timestamps as ordinal / 30.
            "rgb_folder_timebase_hz": 30.0,
            "upstream_revision": "e6f4e3d474fad0e11f561482012be864ba8c3f17",
            "verify_upstream_revision": True,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        self.mast3r_slam_dir = self._resolve_repo_dir()
        config_path = self._resolve_config_path()
        if not config_path.is_file():
            raise FileNotFoundError(f"MASt3R-SLAM config not found: {config_path}")
        if self.calibration_path is not None:
            calibration = self._resolve_optional_path(self.calibration_path)
            if not calibration.is_file():
                raise FileNotFoundError(
                    f"MASt3R-SLAM calibration not found: {calibration}"
                )
        self._verify_revision(self.mast3r_slam_dir)
        self._is_loaded = True

    def reconstruct(
        self, video_input: VideoInput, output_dir: Path
    ) -> ReconstructionResult:
        self.ensure_loaded()
        image_paths = video_input.get_frame_paths()
        if not image_paths:
            raise ValueError(f"No images found in {video_input.image_dir}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="mast3r_slam_input_", dir=output_dir
        ) as temporary:
            input_dir = Path(temporary)
            self._prepare_png_input(image_paths, input_dir)
            self._run_mast3r_slam(input_dir, output_dir)
            sequence_name = input_dir.stem
        trajectory_path = output_dir / f"{sequence_name}.txt"
        pointcloud_path = output_dir / f"{sequence_name}.ply"
        if not pointcloud_path.is_file():
            raise FileNotFoundError(
                f"MASt3R-SLAM reconstruction not found: {pointcloud_path}"
            )
        pointcloud = PointCloud.from_ply(pointcloud_path)
        pointcloud.is_metric = False
        poses = self._load_trajectory(
            trajectory_path, video_input, self.rgb_folder_timebase_hz
        )

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": self.name,
                "input_frames": len(image_paths),
                "keyframes": len(poses),
                "output_dir": str(output_dir),
                "config_path": str(self._resolve_config_path()),
                "upstream_revision": self.upstream_revision,
                "scale_status": "relative",
                "pose_convention": PoseConvention.CAMERA_TO_WORLD.value,
            },
        )

    def _run_mast3r_slam(self, image_dir: Path, output_dir: Path) -> None:
        repo_dir = self.mast3r_slam_dir or self._resolve_repo_dir()
        cmd = [
            str(self.python_executable),
            "main.py",
            "--dataset",
            str(Path(image_dir).resolve()),
            "--config",
            str(self._resolve_config_path()),
            "--save-as",
            str(output_dir),
            "--no-viz",
        ]
        if self.calibration_path is not None:
            cmd.extend(
                ["--calib", str(self._resolve_optional_path(self.calibration_path))]
            )
        logger.info("Running MASt3R-SLAM: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=repo_dir, check=True)

    @staticmethod
    def _prepare_png_input(image_paths: list[Path], input_dir: Path) -> None:
        """Expose selected Mapper JPEGs under the PNG names upstream accepts.

        OpenCV detects image encoding from file contents, so symlinking retains
        the source bytes and avoids a lossy/redundant JPEG-to-PNG transcode.
        """

        for ordinal, source in enumerate(image_paths):
            destination = input_dir / f"frame_{ordinal:08d}.png"
            destination.symlink_to(Path(source).resolve())

    def _resolve_repo_dir(self) -> Path:
        configured = self.mast3r_slam_path or os.environ.get("MAST3R_SLAM_PATH")
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path(__file__).resolve().parents[3] / "MASt3R-SLAM")
        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "main.py").is_file():
                return candidate
        raise FileNotFoundError(
            "MASt3R-SLAM checkout not found. Set mast3r_slam_path or "
            "MAST3R_SLAM_PATH."
        )

    def _resolve_config_path(self) -> Path:
        repo_dir = self.mast3r_slam_dir or self._resolve_repo_dir()
        configured = self.mast3r_slam_config_path
        if configured is None:
            return (repo_dir / "config" / "base.yaml").resolve()
        return self._resolve_optional_path(configured)

    def _resolve_optional_path(self, value: str | Path) -> Path:
        repo_dir = self.mast3r_slam_dir or self._resolve_repo_dir()
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        for base in (Path.cwd(), Path(__file__).resolve().parents[2], repo_dir):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (repo_dir / path).resolve()

    def _verify_revision(self, repo_dir: Path) -> None:
        if not self.verify_upstream_revision:
            return
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Cannot verify MASt3R-SLAM revision in {repo_dir}"
            ) from exc
        actual = completed.stdout.strip()
        if actual != self.upstream_revision:
            raise RuntimeError(
                "MASt3R-SLAM checkout revision mismatch: "
                f"expected {self.upstream_revision}, found {actual}"
            )

    @staticmethod
    def _load_trajectory(
        path: Path,
        video_input: VideoInput,
        rgb_folder_timebase_hz: float = 30.0,
    ) -> CameraPoses:
        if not path.is_file():
            raise FileNotFoundError(f"MASt3R-SLAM trajectory not found: {path}")
        rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if rows.shape[1] != 8:
            raise ValueError(
                f"Invalid MASt3R-SLAM trajectory {path}: expected 8 columns, "
                f"got {rows.shape[1]}"
            )

        source_times = video_input.get_frame_timestamps()
        source_indices = video_input.get_source_frame_indices()
        slam_times = rows[:, 0]
        ordinal_values = slam_times * rgb_folder_timebase_hz
        rounded = np.rint(ordinal_values).astype(np.int64)
        ordinal_encoded = (
            len(source_times) > 0
            and np.allclose(ordinal_values, rounded, atol=1e-4)
            and np.all((rounded >= 0) & (rounded < len(source_times)))
        )
        if ordinal_encoded:
            selected = rounded
        else:
            if len(source_times) == 0:
                raise ValueError("Cannot map MASt3R-SLAM keyframes without timestamps")
            selected = np.abs(
                source_times[:, None] - slam_times[None, :]
            ).argmin(axis=0)

        matrices = np.repeat(
            np.eye(4, dtype=np.float32)[None], len(rows), axis=0
        )
        matrices[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
        matrices[:, :3, 3] = rows[:, 1:4]
        return CameraPoses(
            poses=matrices,
            timestamps=source_times[selected],
            frame_indices=source_indices[selected],
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
        )
