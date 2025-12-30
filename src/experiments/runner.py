"""Experiment runner for testing reconstruction models."""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import json
import yaml
import logging

from ..models import get_model
from ..preprocessing import VideoProcessor, TelemetryExtractor
from ..alignment import GPSAligner
from ..core.types import VideoInput
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


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
    fps: float = 10.0
    align_to_gps: bool = True
    force_reprocess: bool = False

    # Video filtering (optional)
    video_extensions: List[str] = field(
        default_factory=lambda: [".MP4", ".MOV", ".mp4", ".mov"]
    )

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)
        self.output_folder = Path(self.output_folder)


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

        # Initialize components
        self.video_processor = VideoProcessor(fps=config.fps)
        self.telemetry_extractor = TelemetryExtractor()
        self.gps_aligner = GPSAligner()
        self.metrics_calculator = MetricsCalculator()

        # Model will be loaded lazily
        self._model = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentRunner":
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
            input_folder=raw["input_folder"],
            output_folder=raw["output_folder"],
            model_config=raw.get("model_config", {}),
            fps=raw.get("fps", 10.0),
            align_to_gps=raw.get("align_to_gps", True),
            force_reprocess=raw.get("force_reprocess", False),
            video_extensions=raw.get(
                "video_extensions", [".MP4", ".MOV", ".mp4", ".mov"]
            ),
        )

        return cls(config)

    def run(self) -> List[dict]:
        """
        Run the experiment.

        Returns:
            List of results for each video processed
        """
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Input folder: {self.config.input_folder}")

        # Setup experiment directory
        exp_dir = self._setup_experiment_dir()

        # Load model
        self._load_model()

        # Find videos
        videos = self._find_videos()
        logger.info(f"Found {len(videos)} videos to process")

        if len(videos) == 0:
            logger.warning("No videos found!")
            return []

        results = []

        for i, video_path in enumerate(videos, 1):
            logger.info(f"[{i}/{len(videos)}] Processing: {video_path.name}")

            try:
                result = self._process_video(video_path, exp_dir)
                results.append(result)
                logger.info(f"  Success: {result.get('point_count', 0)} points")
            except Exception as e:
                logger.error(f"  Error: {e}")
                results.append({
                    "video": video_path.name,
                    "error": str(e),
                    "success": False,
                })

        # Save results
        self._save_results(exp_dir, results)

        # Print summary
        self._print_summary(results)

        logger.info(f"Experiment complete. Results in: {exp_dir}")
        return results

    def _load_model(self) -> None:
        """Load the reconstruction model."""
        logger.info(f"Loading model: {self.config.model}")

        model_cls = get_model(self.config.model)
        self._model = model_cls(self.config.model_config)
        self._model.load()

        logger.info(f"Model loaded. Capabilities: {self._model.get_capabilities()}")

    def _find_videos(self) -> List[Path]:
        """Find all video files in input folder."""
        videos = []

        for ext in self.config.video_extensions:
            videos.extend(self.config.input_folder.glob(f"*{ext}"))

        return sorted(videos)

    def _setup_experiment_dir(self) -> Path:
        """Create experiment output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.config.output_folder / f"{self.config.name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config copy
        config_dict = {
            "name": self.config.name,
            "model": self.config.model,
            "model_config": self.config.model_config,
            "input_folder": str(self.config.input_folder),
            "output_folder": str(self.config.output_folder),
            "fps": self.config.fps,
            "align_to_gps": self.config.align_to_gps,
            "timestamp": timestamp,
        }

        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return exp_dir

    def _process_video(self, video_path: Path, exp_dir: Path) -> dict:
        """Process a single video."""
        video_name = video_path.stem
        output_dir = exp_dir / "outputs" / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract frames (with caching)
        logger.info("  Extracting frames...")
        image_dir = self.video_processor.process(
            video_path,
            output_dir,
            force=self.config.force_reprocess,
        )
        frame_count = self.video_processor.get_frame_count(image_dir)
        logger.info(f"  {frame_count} frames")

        # Step 2: Extract telemetry
        logger.info("  Extracting telemetry...")
        gps_track, imu_data = self.telemetry_extractor.extract(video_path)

        if gps_track is not None:
            logger.info(f"  GPS: {len(gps_track)} points")
        else:
            logger.info("  GPS: not available")

        if imu_data is not None:
            logger.info(f"  IMU: {len(imu_data)} samples")
        else:
            logger.info("  IMU: not available")

        # Step 3: Create video input
        video_input = VideoInput(
            image_dir=image_dir,
            fps=self.config.fps,
            frame_count=frame_count,
            gps_track=gps_track,
            imu_data=imu_data,
            metadata={
                "video_path": str(video_path),
                "video_name": video_name,
            },
        )

        # Step 4: Run reconstruction
        logger.info("  Running reconstruction...")
        result = self._model.reconstruct(video_input, output_dir)

        # Step 5: Align to GPS (if enabled and data available)
        if (
            self.config.align_to_gps
            and gps_track is not None
            and result.poses is not None
        ):
            logger.info("  Aligning to GPS...")
            result.pointcloud = self.gps_aligner.align(
                result.pointcloud,
                result.poses,
                gps_track,
                imu_data,
            )

        # Step 6: Compute metrics
        logger.info("  Computing metrics...")
        metrics = self.metrics_calculator.compute_all(
            result.pointcloud,
            result.poses,
            gps_track,
        )

        # Step 7: Save aligned point cloud
        aligned_ply_path = output_dir / "aligned_pointcloud.ply"
        result.pointcloud.save_ply(aligned_ply_path)
        logger.info(f"  Saved: {aligned_ply_path}")

        # Step 8: Save metadata.json for viewer compatibility
        metadata = {
            "video_name": video_name,
            "initial_gps_coordinates": [
                result.pointcloud.origin_gps[0] if result.pointcloud.origin_gps else 0,
                result.pointcloud.origin_gps[1] if result.pointcloud.origin_gps else 0,
            ],
            "altitude": result.pointcloud.origin_gps[2] if result.pointcloud.origin_gps else 0,
            "frames": frame_count,
            "is_metric": result.pointcloud.is_metric,
            "point_count": len(result.pointcloud),
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved: {metadata_path}")

        # Build result dict
        return {
            "video": video_path.name,
            "success": True,
            "output_dir": str(output_dir),
            "point_count": len(result.pointcloud),
            "is_metric": result.pointcloud.is_metric,
            "origin_gps": result.pointcloud.origin_gps,
            "metrics": metrics,
            "model_metadata": result.metadata,
        }

    def _save_results(self, exp_dir: Path, results: List[dict]) -> None:
        """Save experiment results to JSON."""
        results_path = exp_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_path}")

    def _print_summary(self, results: List[dict]) -> None:
        """Print experiment summary."""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Total videos: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            total_points = sum(r.get("point_count", 0) for r in successful)
            print(f"Total points: {total_points:,}")

            # Average metrics
            all_metrics = [r.get("metrics", {}) for r in successful]
            if all_metrics:
                print("\nAverage Metrics:")
                metric_keys = set()
                for m in all_metrics:
                    metric_keys.update(m.keys())

                for key in sorted(metric_keys):
                    values = [m.get(key) for m in all_metrics if key in m]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        avg = sum(values) / len(values)
                        print(f"  {key}: {avg:.4f}")

        if failed:
            print("\nFailed videos:")
            for r in failed:
                print(f"  - {r.get('video')}: {r.get('error', 'Unknown error')}")

        print("=" * 50)


def main():
    """CLI entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run reconstruction experiment")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiment
    runner = ExperimentRunner.from_yaml(args.config)
    runner.run()


if __name__ == "__main__":
    main()
