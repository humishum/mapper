"""External VGGT-Long adapter for kilometer-scale image sequences."""

from pathlib import Path
from typing import Optional
import logging
import os
import subprocess
import sys

import numpy as np

from .base import BaseModel
from ..core.types import (
    CameraPoses,
    PointCloud,
    PoseConvention,
    ReconstructionResult,
    VideoInput,
)

logger = logging.getLogger(__name__)


class VGGTLongModel(BaseModel):
    """Run the official VGGT-Long pipeline and import its durable outputs.

    The upstream pipeline performs chunk alignment and loop correction itself.
    Its default configuration uses Sim(3), so output scale remains relative.

    Reference: https://github.com/DengKaiCQ/VGGT-Long
    """

    name = "vggt_long"
    outputs_metric_scale = False
    outputs_poses = True
    # Upstream confidence filters the exported PLY but is not stored in it.
    outputs_confidence = False
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.vggt_long_path = self.config.get("vggt_long_path")
        self.vggt_long_config_path = self.config.get("vggt_long_config_path")
        self.weights_dir = self.config.get("weights_dir", "weights/vggt_long")
        self.python_executable = self.config.get("python_executable", sys.executable)
        self.upstream_revision = self.config.get(
            "upstream_revision",
            "c160869d1d99c96bb227f414afb3bc68c29c9a76",
        )
        self.verify_upstream_revision = self.config.get(
            "verify_upstream_revision", True
        )
        self.vggt_long_dir: Optional[Path] = None

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "vggt_long_path": None,
            "vggt_long_config_path": None,
            "weights_dir": "weights/vggt_long",
            "python_executable": sys.executable,
            "upstream_revision": "c160869d1d99c96bb227f414afb3bc68c29c9a76",
            "verify_upstream_revision": True,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        self.vggt_long_dir = self._resolve_repo_dir()
        config_path = self._resolve_config_path()
        if not config_path.is_file():
            raise FileNotFoundError(f"VGGT-Long config not found: {config_path}")
        self._verify_revision(self.vggt_long_dir)
        self._is_loaded = True

    def reconstruct(
        self, video_input: VideoInput, output_dir: Path
    ) -> ReconstructionResult:
        self.ensure_loaded()
        image_paths = video_input.get_frame_paths()
        if not image_paths:
            raise ValueError(f"No images found in {video_input.image_dir}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir = self._run_vggt_long(video_input.image_dir)

        pointcloud_path = result_dir / "pcd" / "combined_pcd.ply"
        if not pointcloud_path.is_file():
            raise FileNotFoundError(
                f"VGGT-Long combined point cloud not found: {pointcloud_path}"
            )
        pointcloud = PointCloud.from_ply(pointcloud_path)
        pointcloud.is_metric = False

        intrinsics = self._load_intrinsics(result_dir / "intrinsic.txt")
        poses = self._load_poses(
            result_dir / "camera_poses.txt",
            video_input.get_frame_timestamps(),
            intrinsics,
            video_input.get_source_frame_indices(),
        )
        if len(poses) != len(image_paths):
            raise ValueError(
                "VGGT-Long output count does not match the input: "
                f"{len(poses)} poses for {len(image_paths)} images"
            )

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": self.name,
                "frames": len(image_paths),
                "upstream_output_dir": str(result_dir),
                "config_path": str(self._resolve_config_path()),
                "upstream_revision": self.upstream_revision,
                "scale_status": "relative",
                "pose_convention": PoseConvention.CAMERA_TO_WORLD.value,
            },
        )

    def _run_vggt_long(self, image_dir: Path) -> Path:
        repo_dir = self.vggt_long_dir or self._resolve_repo_dir()
        exp_root = repo_dir / "exps"
        previous = {
            path.resolve(): path.stat().st_mtime_ns
            for path in exp_root.rglob("camera_poses.txt")
        } if exp_root.exists() else {}

        cmd = [
            str(self.python_executable),
            "vggt_long.py",
            "--image_dir",
            str(Path(image_dir).resolve()),
            "--config",
            str(self._resolve_config_path()),
        ]
        logger.info("Running VGGT-Long: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=repo_dir, check=True)

        candidates = []
        if exp_root.exists():
            for pose_path in exp_root.rglob("camera_poses.txt"):
                resolved = pose_path.resolve()
                mtime = pose_path.stat().st_mtime_ns
                if resolved not in previous or previous[resolved] != mtime:
                    candidates.append(pose_path.parent)
        if not candidates:
            raise FileNotFoundError(
                "VGGT-Long completed without a new output under "
                f"{exp_root}; upstream currently has no --output_dir option"
            )
        return max(
            candidates,
            key=lambda candidate: (candidate / "camera_poses.txt").stat().st_mtime_ns,
        )

    def _resolve_repo_dir(self) -> Path:
        configured = self.vggt_long_path or os.environ.get("VGGT_LONG_PATH")
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path(__file__).resolve().parents[3] / "VGGT-Long")
        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "vggt_long.py").is_file():
                return candidate
        raise FileNotFoundError(
            "VGGT-Long checkout not found. Set vggt_long_path or VGGT_LONG_PATH."
        )

    def _resolve_config_path(self) -> Path:
        repo_dir = self.vggt_long_dir or self._resolve_repo_dir()
        configured = self.vggt_long_config_path
        if configured is None:
            return (repo_dir / "configs" / "base_config.yaml").resolve()
        path = Path(configured).expanduser()
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
                f"Cannot verify VGGT-Long revision in {repo_dir}"
            ) from exc
        actual = completed.stdout.strip()
        if actual != self.upstream_revision:
            raise RuntimeError(
                "VGGT-Long checkout revision mismatch: "
                f"expected {self.upstream_revision}, found {actual}"
            )

    @staticmethod
    def _load_intrinsics(path: Path) -> Optional[np.ndarray]:
        if not path.is_file():
            return None
        rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            values = [float(value) for value in line.split()]
            if len(values) < 4:
                raise ValueError(
                    f"Invalid intrinsics line {line_number} in {path}"
                )
            fx, fy, cx, cy = values[:4]
            rows.append(
                np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            )
        return np.stack(rows) if rows else None

    @staticmethod
    def _load_poses(
        path: Path,
        timestamps: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        frame_indices: Optional[np.ndarray],
    ) -> CameraPoses:
        if not path.is_file():
            raise FileNotFoundError(f"VGGT-Long camera poses not found: {path}")
        rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            values = [float(value) for value in line.split()]
            if len(values) != 16:
                raise ValueError(
                    f"Invalid pose line {line_number} in {path}: "
                    f"expected 16 values, got {len(values)}"
                )
            rows.append(np.asarray(values, dtype=np.float32).reshape(4, 4))
        if not rows:
            raise ValueError(f"VGGT-Long pose file is empty: {path}")
        matrices = np.stack(rows)
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if len(timestamps) != len(matrices):
                raise ValueError(
                    "VGGT-Long timestamps and poses have different lengths"
                )
        if intrinsics is not None and len(intrinsics) != len(matrices):
            raise ValueError(
                "VGGT-Long intrinsics and poses have different lengths"
            )
        if frame_indices is None:
            frame_indices = np.arange(len(matrices), dtype=np.int64)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if len(frame_indices) != len(matrices):
            raise ValueError("VGGT-Long frame indices and poses have different lengths")
        return CameraPoses(
            poses=matrices,
            timestamps=timestamps,
            intrinsics=intrinsics,
            frame_indices=frame_indices,
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
        )
