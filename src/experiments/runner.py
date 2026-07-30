"""Experiment runner for testing reconstruction models/workflows."""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import json
import shutil
import yaml
import logging
import sys
import numpy as np

from viewer.backend.domain.package import CaptureMetadata, Producer
from viewer.backend.services.catalog import new_opaque_id

from ..models import get_model
from ..preprocessing import VideoProcessor, TelemetryExtractor
from ..alignment import AlignmentResult, GPSAligner, WindowAligner
from ..core.types import PointCloud, ReconstructionResult, VideoInput
from ..publisher import CopcPublisher, CopcPublisherConfig
from ..publisher.package import (
    PackageIdentity,
    PackageSource,
    ReconstructionPackagePublisher,
    capture_id_for_file,
    package_source_from_window,
)
from .metrics import MetricsCalculator
from .utils import get_git_info


DEFAULT_FRAME_RATE = 10
DEFAULT_VIDEO_EXTENSIONS = [".MP4", ".MOV", ".mp4", ".mov"]


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    model: str
    input_folder: Path
    output_folder: Path

    # Model configuration (passed to model constructor)
    model_config: dict = field(default_factory=dict)

    # Processing options
    fps: float = DEFAULT_FRAME_RATE
    align_to_gps: bool = True
    force_reprocess: bool = False
    frame_cache_dir: Optional[Path] = None
    alignment_config: dict = field(default_factory=dict)
    package_config: dict = field(default_factory=dict)
    video_names: List[str] = field(default_factory=list)
    max_videos: Optional[int] = None

    # Video filtering (optional)
    video_extensions: List[str] = field(
        default_factory=lambda: DEFAULT_VIDEO_EXTENSIONS
    )

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)
        self.output_folder = Path(self.output_folder)
        if self.frame_cache_dir is not None:
            self.frame_cache_dir = Path(self.frame_cache_dir)
        if self.max_videos is not None and self.max_videos <= 0:
            raise ValueError("max_videos must be positive")


class ExperimentRunner:
    """
    Run experiments on a folder of videos.

    Workflow:
    1. Load config
    2. Process each video:
       a. Extract frames (with caching)
       b. Extract telemetry (GPS, IMU)
       c. Run reconstruction model
       d. Align to GPS (if enabled)
       e. Compute metrics
    3. Save results
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config

        self.video_processor = VideoProcessor(fps=config.fps)
        self.telemetry_extractor = TelemetryExtractor()
        gps_option_names = {
            "min_correspondences",
            "min_gps_trajectory_length_m",
            "min_gps_std_dev_m",
            "max_rmse_m",
            "max_scale",
            "min_scale",
            "min_inlier_fraction",
            "max_clock_offset_s",
            "clock_step_s",
            "min_clock_peak_quality",
            "max_gps_interpolation_gap_s",
        }
        gps_options = {
            key: value
            for key, value in config.alignment_config.items()
            if key in gps_option_names
        }
        self.gps_aligner = GPSAligner(**gps_options)
        self.window_aligner = WindowAligner(config=self.config.alignment_config)
        self.metrics_calculator = MetricsCalculator()
        copc_options = {
            "memory_limit": self.config.package_config.get("copc_memory_limit", "4G"),
            "threads": self.config.package_config.get("copc_threads"),
            "temp_dir": self.config.package_config.get("temp_dir"),
        }
        copc_options = {
            key: (Path(value) if key == "temp_dir" and value is not None else value)
            for key, value in copc_options.items()
        }
        self.package_publisher = ReconstructionPackagePublisher(
            CopcPublisher(CopcPublisherConfig(**copc_options))
        )

        # lazy load model
        self._model = None

        # Setup experiment directory
        self.exp_dir = self._setup_experiment_dir()

        # Setup logger with custom handlers to avoid duplication with root logger
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = logging.FileHandler(self.exp_dir / "experiment.log", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        *,
        video_names: Optional[List[str]] = None,
        max_videos: Optional[int] = None,
        input_folder: Optional[Path] = None,
        output_folder: Optional[Path] = None,
    ) -> "ExperimentRunner":
        """
        Load experiment configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Configured ExperimentRunner instance
        """
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        config = ExperimentConfig(
            name=raw["name"],
            model=raw["model"],
            input_folder=(
                input_folder if input_folder is not None else raw["input_folder"]
            ),
            output_folder=(
                output_folder if output_folder is not None else raw["output_folder"]
            ),
            model_config=raw.get("model_config", {}),
            fps=raw.get("fps", DEFAULT_FRAME_RATE),
            align_to_gps=raw.get("align_to_gps", True),
            force_reprocess=raw.get("force_reprocess", False),
            frame_cache_dir=raw.get("frame_cache_dir"),
            video_extensions=raw.get("video_extensions", DEFAULT_VIDEO_EXTENSIONS),
            alignment_config=raw.get("alignment_config", {}),
            package_config=raw.get("package_config", {}),
            video_names=(
                video_names if video_names is not None else raw.get("video_names", [])
            ),
            max_videos=(
                max_videos if max_videos is not None else raw.get("max_videos")
            ),
        )

        return cls(config)

    def run(self) -> List[dict]:
        """
        Run the experiment.

        Returns:
            List of results for each video processed
        """
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Input folder: {self.config.input_folder}")

        # Find videos
        videos = self._find_videos()
        self.logger.info(f"Found {len(videos)} videos to process")

        if len(videos) == 0:
            self.logger.warning("No videos found!")
            return []

        # Load model
        self._load_model()

        results = []

        for i, video_path in enumerate(videos, 1):
            self.logger.info(f"[{i}/{len(videos)}] Processing: {video_path.name}")

            try:
                result = self._process_video(video_path, self.exp_dir)
                results.append(result)
                self.logger.info(f"  Success: {result.get('point_count', 0)} points")
            except Exception as e:
                self.logger.error(f"  Error: {e}")
                results.append(
                    {
                        "video": video_path.name,
                        "error": str(e),
                        "success": False,
                    }
                )

        # Save results
        self._save_results(self.exp_dir, results)

        # Print summary
        self._print_summary(results)

        self.logger.info(f"Experiment complete. Results in: file://{self.exp_dir}")
        return results

    def _load_model(self) -> None:
        """Load the reconstruction model."""
        self.logger.info(f"Loading model: {self.config.model}")

        model_cls = get_model(self.config.model)
        self._model = model_cls(self.config.model_config)
        self._model.load()

        self.logger.info(
            f"Model loaded. Capabilities: {self._model.get_capabilities()}"
        )

    def _find_videos(self) -> List[Path]:
        """Find all video files in input folder, return sorted by file size(smallest first)"""
        videos = []

        for ext in self.config.video_extensions:
            videos.extend(self.config.input_folder.glob(f"*{ext}"))

        selected = sorted(set(videos), key=lambda p: p.stat().st_size)
        if self.config.video_names:
            requested = {Path(name).stem for name in self.config.video_names}
            selected = [path for path in selected if path.stem in requested]
        if self.config.max_videos is not None:
            selected = selected[: self.config.max_videos]
        return selected

    def _setup_experiment_dir(self) -> Path:
        """Create experiment output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.config.output_folder / f"{self.config.name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Get git info
        git_commit, git_status = get_git_info()

        # Save config copy
        config_dict = {
            "name": self.config.name,
            "model": self.config.model,
            "model_config": self.config.model_config,
            "input_folder": str(self.config.input_folder),
            "output_folder": str(self.config.output_folder),
            "fps": self.config.fps,
            "align_to_gps": self.config.align_to_gps,
            "force_reprocess": self.config.force_reprocess,
            "frame_cache_dir": (
                str(self.config.frame_cache_dir)
                if self.config.frame_cache_dir is not None
                else None
            ),
            "alignment_config": self.config.alignment_config,
            "package_config": self.config.package_config,
            "video_names": self.config.video_names,
            "max_videos": self.config.max_videos,
            "timestamp": timestamp,
            "git_commit": git_commit,
            "git_status": git_status,
        }

        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return exp_dir

    def _process_video(self, video_path: Path, exp_dir: Path) -> dict:
        """Process a single video."""
        video_name = video_path.stem
        identity = PackageIdentity(
            capture_id=capture_id_for_file(video_path),
            run_id=new_opaque_id("run"),
            artifact_id=new_opaque_id("art"),
        )
        package_dir = exp_dir / "reconstructions" / identity.run_id
        work_dir = exp_dir / ".scratch" / video_name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames with ffmpeg, skip if already cached in output folder
        self.logger.info("  Extracting frames...")
        frame_cache_root = self._get_frame_cache_root()
        frame_cache_root.mkdir(parents=True, exist_ok=True)
        frame_output_dir = frame_cache_root / video_name
        image_dir = self.video_processor.process(
            video_path,
            frame_output_dir,
            force=self.config.force_reprocess,
        )
        frame_count = self.video_processor.get_frame_count(image_dir)
        self.logger.info(f"  {frame_count} frames")

        # Extract initial GPS first (GoPro telemetry or EXIF fallback).
        self.logger.info("  Extracting initial GPS...")
        initial_gps = self.telemetry_extractor.extract_initial_gps(video_path)
        if initial_gps is not None:
            self.logger.info(
                "  Initial GPS: %.6f, %.6f, %.1f",
                initial_gps[0],
                initial_gps[1],
                initial_gps[2],
            )
        else:
            self.logger.info("  Initial GPS: not available")

        # Extract telemetry from video, only works for gopro videos.
        self.logger.info("  Extracting telemetry...")
        gps_track, imu_data = self.telemetry_extractor.extract_gps_imu(video_path)

        self.logger.info(
            f"  GPS: {len(gps_track)} points"
            if gps_track is not None
            else "  GPS: not available"
        )
        self.logger.info(
            f"  IMU: {len(imu_data)} samples"
            if imu_data is not None
            else "  IMU: not available"
        )

        # Create video input
        video_input = VideoInput(
            video_path=video_path,
            image_dir=image_dir,
            fps=self.config.fps,
            frame_count=frame_count,
            gps_track=gps_track,
            imu_data=imu_data,
            metadata={
                "video_path": str(video_path),
                "video_name": video_name,
                "initial_gps": initial_gps,
            },
        )

        # Step 4: Run reconstruction
        self.logger.info("  Running reconstruction...")
        result = self._model.reconstruct(video_input, work_dir)

        # Step 4b: Align source units without concatenating point buffers. Fixed
        # windows are today's VRAM boundary; the package contract also accepts
        # SLAM submaps, keyframe groups, and generic batches.
        if result.chunks:
            self.logger.info(
                "  Aligning %d reconstruction windows...", len(result.chunks)
            )
            source_results, poses, alignment_metadata = (
                self.window_aligner.align_chunks(
                    result.chunks, result.pointcloud.is_metric
                )
            )
            for source, source_alignment in zip(
                source_results,
                alignment_metadata.get("chunks", []),
                strict=False,
            ):
                source.window_metadata = {
                    **source.window_metadata,
                    "alignment_to_common_frame": source_alignment,
                }
            result.metadata = {
                **result.metadata,
                "window_alignment": alignment_metadata,
            }
            # Release model-native chunks after their transforms and metadata
            # have been copied into the publisher source units.
            result.chunks = None
        else:
            source_results = [result]
            poses = result.poses

        # Step 5: Align to GPS. Every path produces an explicit result so an
        # unaligned artifact can never be mistaken for a georeferenced one.
        alignment_result: AlignmentResult
        if self.config.align_to_gps and gps_track is not None and poses is not None:
            self.logger.info("  Aligning to GPS...")
            # GPS alignment depends on the camera trajectory, not a materialized
            # merged cloud. A zero-point placeholder lets the aligner solve the
            # transform, which is then applied independently to every source.
            alignment_input = PointCloud(
                points=np.empty((0, 3), dtype=np.float32),
                scale=source_results[0].pointcloud.scale,
                is_metric=source_results[0].pointcloud.is_metric,
            )
            alignment_result = self.gps_aligner.align(
                alignment_input,
                poses,
                gps_track,
                imu_data,
                allow_scale=not source_results[0].pointcloud.is_metric,
            )
            if alignment_result.accepted:
                # Replace one source at a time so GPS alignment needs only one
                # additional chunk-sized point buffer, not a second full run.
                for index, source in enumerate(source_results):
                    source_results[index] = self._apply_global_alignment(
                        source, alignment_result
                    )
                poses = alignment_result.transform_poses(poses)
            else:
                self.logger.warning(
                    "  GPS alignment rejected: %s",
                    alignment_result.reason or "quality gate failed",
                )
        elif not self.config.align_to_gps:
            alignment_result = AlignmentResult.unaligned("gps_alignment_disabled")
        elif gps_track is None:
            alignment_result = AlignmentResult.unaligned("gps_unavailable")
        else:
            alignment_result = AlignmentResult.unaligned("camera_poses_unavailable")

        result.metadata = {
            **result.metadata,
            "alignment": alignment_result.to_dict(),
        }

        # Step 6: Compute metrics
        self.logger.info("  Computing metrics...")
        metrics = self.metrics_calculator.compute_chunks(
            [source.pointcloud for source in source_results],
            poses,
            gps_track,
        )

        # Step 7: Publish the canonical package. The package publisher writes
        # manifest.json last, validates all checksums/contracts, then registers
        # it in SQLite when catalog publication is enabled.
        self.logger.info("  Publishing reconstruction package...")
        package_sources = self._build_package_sources(
            source_results, frame_count=frame_count
        )
        git_commit, git_status = get_git_info()
        catalog_path = self._catalog_path()
        model_capabilities = {
            name: bool(value)
            for name, value in self._model.get_capabilities().items()
            if isinstance(value, (bool, np.bool_))
        }
        published = self.package_publisher.publish(
            package_dir,
            identity=identity,
            sources=package_sources,
            alignment=alignment_result,
            poses=poses,
            gps_track=gps_track,
            imu_data=imu_data,
            capture=CaptureMetadata(
                video_name=video_path.name,
                source_uri=video_path.resolve().as_uri(),
                frame_count=frame_count,
                fps=self.config.fps,
                gps_sample_count=len(gps_track) if gps_track is not None else 0,
                imu_sample_count=len(imu_data) if imu_data is not None else 0,
            ),
            producer=Producer(
                model_name=self.config.model,
                model_config=self.config.model_config,
                capabilities=model_capabilities,
                git_commit=git_commit,
                git_status=git_status,
                adapter_name=f"{self.config.model}_adapter",
                adapter_version="1.0.0",
                publisher_name="copc-converter",
                publisher_version="0.11.0",
            ),
            reconstruction_metrics=metrics,
            model_metadata=result.metadata,
            catalog_path=catalog_path,
            voxel_size_m=float(self.config.package_config.get("voxel_size_m", 0.02)),
        )
        self.logger.info("  Published: %s", published.package_root)

        # Model work products are disposable after the validated package commit.
        shutil.rmtree(work_dir)

        # Build result dict
        return {
            "video": video_path.name,
            "success": True,
            "output_dir": str(published.package_root),
            "manifest": str(published.package.manifest_path),
            "run_id": identity.run_id,
            "capture_id": identity.capture_id,
            "artifact_id": identity.artifact_id,
            "point_count": published.copc.point_count,
            "is_metric": all(source.pointcloud.is_metric for source in source_results),
            "origin_gps": alignment_result.anchor_wgs84,
            "metrics": metrics,
            "alignment": alignment_result.to_dict(),
            "model_metadata": result.metadata,
        }

    def _get_frame_cache_root(self) -> Path:
        """Resolve the root directory used for cached frame extraction."""
        if self.config.frame_cache_dir is not None:
            return self.config.frame_cache_dir
        return self.config.output_folder / "_frame_cache"

    @staticmethod
    def _apply_global_alignment(
        source: ReconstructionResult,
        alignment: AlignmentResult,
    ) -> ReconstructionResult:
        """Apply one accepted model-to-local transform to a source unit."""

        transformed = source.pointcloud.transform(alignment.transform)
        transformed.origin_gps = alignment.anchor_wgs84
        transformed.scale = float(source.pointcloud.scale * alignment.scale)
        transformed.is_metric = True
        return ReconstructionResult(
            pointcloud=transformed,
            poses=(
                alignment.transform_poses(source.poses)
                if source.poses is not None
                else None
            ),
            metadata=dict(source.metadata),
            window_metadata=dict(source.window_metadata),
        )

    @staticmethod
    def _build_package_sources(
        source_results: List[ReconstructionResult],
        *,
        frame_count: int,
    ) -> List[PackageSource]:
        """Build generic provenance while preserving current window metadata."""

        if len(source_results) == 1:
            source = source_results[0]
            timestamps = source.poses.timestamps if source.poses is not None else None
            return [
                PackageSource(
                    pointcloud=source.pointcloud,
                    kind="capture",
                    name="capture",
                    frame_start=0 if frame_count else None,
                    frame_end=frame_count - 1 if frame_count else None,
                    frame_indices=(list(range(frame_count)) if frame_count else None),
                    timestamp_start_s=(
                        float(timestamps[0])
                        if timestamps is not None and len(timestamps)
                        else None
                    ),
                    timestamp_end_s=(
                        float(timestamps[-1])
                        if timestamps is not None and len(timestamps)
                        else None
                    ),
                    metadata={
                        "source_contract": "entire_capture",
                        "model_metadata": source.metadata,
                    },
                )
            ]
        return [
            package_source_from_window(
                source.pointcloud,
                {
                    **source.window_metadata,
                    "timestamp_start_s": (
                        float(source.poses.timestamps[0])
                        if source.poses is not None
                        and source.poses.timestamps is not None
                        and len(source.poses.timestamps)
                        else None
                    ),
                    "timestamp_end_s": (
                        float(source.poses.timestamps[-1])
                        if source.poses is not None
                        and source.poses.timestamps is not None
                        and len(source.poses.timestamps)
                        else None
                    ),
                    "model_metadata": source.metadata,
                },
            )
            for source in source_results
        ]

    def _catalog_path(self) -> Path | None:
        if not self.config.package_config.get("register_catalog", True):
            return None
        configured = self.config.package_config.get("catalog_path")
        if configured is None:
            return self.config.output_folder / "catalog.sqlite3"
        path = Path(configured)
        if not path.is_absolute():
            path = self.config.output_folder / path
        return path

    def _save_results(self, exp_dir: Path, results: List[dict]) -> None:
        """Save experiment results to JSON."""
        results_path = exp_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {results_path}")

    def _print_summary(self, results: List[dict]) -> None:
        """Print experiment summary."""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        self.logger.info("\n" + "=" * 50)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total videos: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")

        if successful:
            total_points = sum(r.get("point_count", 0) for r in successful)
            self.logger.info(f"Total points: {total_points:,}")

            # Average metrics
            all_metrics = [r.get("metrics", {}) for r in successful]
            if all_metrics:
                self.logger.info("\nAverage Metrics:")
                metric_keys = set()
                for m in all_metrics:
                    metric_keys.update(m.keys())

                for key in sorted(metric_keys):
                    values = [m.get(key) for m in all_metrics if key in m]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        avg = sum(values) / len(values)
                        self.logger.info(f"  {key}: {avg:.4f}")

        if failed:
            self.logger.info("\nFailed videos:")
            for r in failed:
                self.logger.info(
                    f"  - {r.get('video')}: {r.get('error', 'Unknown error')}"
                )

        self.logger.info("=" * 50)


def main():
    """CLI entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run reconstruction experiment")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help="Process only this video filename or stem; may be supplied more than once",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process at most this many matching videos, smallest first",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=None,
        help="Override the config input folder for this run",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Override the config output folder for this run",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiment
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    runner = ExperimentRunner.from_yaml(
        args.config,
        video_names=args.video,
        max_videos=args.max_videos,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )
    runner.run()


if __name__ == "__main__":
    main()
