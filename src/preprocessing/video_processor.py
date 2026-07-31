"""Video normalization and keyframe extraction."""

from pathlib import Path
from dataclasses import dataclass, asdict
import json
import subprocess
import logging
import re
import shutil
import tempfile
import time
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..core.types import GPSTrack, IMUData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyframeSelectionConfig:
    """Configuration for deterministic frame selection.

    ``candidate_fps`` is the decode/sampling ceiling. ``target_fps`` controls
    fixed-rate output and is also used as the default maximum interval for the
    adaptive selector.
    """

    mode: str = "fixed_rate"
    candidate_fps: float = 10.0
    target_fps: float = 10.0
    min_interval_s: float = 0.25
    max_interval_s: Optional[float] = None
    min_sharpness: float = 25.0
    min_mean_luma: float = 12.0
    max_mean_luma: float = 243.0
    max_clipped_fraction: float = 0.85
    min_motion_fraction: float = 0.015
    min_angular_speed_rad_s: float = 0.35

    @classmethod
    def from_mapping(
        cls, fps: float, values: Optional[dict[str, Any]] = None
    ) -> "KeyframeSelectionConfig":
        values = dict(values or {})
        values.setdefault("candidate_fps", fps)
        values.setdefault("target_fps", fps)
        config = cls(**values)
        if config.mode not in {"fixed_rate", "quality_motion"}:
            raise ValueError("keyframe mode must be 'fixed_rate' or 'quality_motion'")
        if config.candidate_fps <= 0 or config.target_fps <= 0:
            raise ValueError("candidate_fps and target_fps must be positive")
        if config.target_fps > config.candidate_fps:
            raise ValueError("target_fps cannot exceed candidate_fps")
        if config.min_interval_s < 0:
            raise ValueError("min_interval_s cannot be negative")
        if config.max_interval_s is not None and config.max_interval_s <= 0:
            raise ValueError("max_interval_s must be positive")
        return config


class VideoProcessor:
    """
    Extract frames from video with smart caching.

    Reuses existing frames if:
    - Same video file (by name + size)
    - Same FPS
    - Frames already exist

    This avoids re-running ffmpeg when experimenting with different models
    on the same video.
    """

    def __init__(
        self,
        fps: float = 10,
        jpeg_quality: int = 10,
        keyframe_config: Optional[dict[str, Any]] = None,
        capture_metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize video processor.

        Args:
            fps: Backwards-compatible candidate extraction rate.
            jpeg_quality: JPEG quality (1-31, lower = higher quality)
            keyframe_config: Fixed-rate or quality/motion selection settings.
            capture_metadata: Known capture/calibration settings to persist.
        """
        self.keyframe_config = KeyframeSelectionConfig.from_mapping(
            fps, keyframe_config
        )
        self.fps = self.keyframe_config.candidate_fps
        self.jpeg_quality = jpeg_quality
        self.configured_capture_metadata = dict(capture_metadata or {})
        self._hwaccel: Optional[str] = None
        self._hwaccel_checked = False

        # Verify ffmpeg is installed
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is installed."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg is not installed. Please install FFmpeg and try again."
            )

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        force: bool = False,
        *,
        gps_track: Optional[GPSTrack] = None,
        imu_data: Optional[IMUData] = None,
    ) -> Path:
        """
        Extract frames from video, reusing cache if valid.

        Args:
            video_path: Path to input video file
            output_dir: Directory for output (images will be in output_dir/images/)
            force: If True, ignore cache and re-extract
            gps_track: GPS already extracted from the source capture.
            imu_data: IMU already extracted from the source capture.

        Returns:
            Path to directory containing extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        image_dir = output_dir / "images"
        cache_file = image_dir / ".cache_info.json"

        # Check if we can reuse existing frames
        if not force and self._can_reuse_cache(
            video_path,
            cache_file,
            gps_track=gps_track,
            imu_data=imu_data,
        ):
            logger.info(f"Reusing cached frames from {image_dir}")
            return image_dir

        # Probe before extraction so capture geometry and source timing are part
        # of both the selection table and the cache identity.
        capture_metadata = self._probe_capture_metadata(video_path)
        capture_metadata["telemetry"] = self._telemetry_summary(gps_track, imu_data)
        source_timestamps = self._probe_source_timestamps(video_path)

        # Extract candidate frames to a temporary directory, then publish only
        # selected keyframes into the stable image directory.
        image_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".keyframe_candidates_", dir=output_dir
        ) as candidate_dir_name:
            candidate_dir = Path(candidate_dir_name)
            self._extract_frames(video_path, candidate_dir)
            records = self._select_keyframes(
                candidate_dir,
                source_timestamps=source_timestamps,
                source_fps=capture_metadata["video"].get("source_fps"),
                imu_data=imu_data,
                gps_track=gps_track,
            )
            self._publish_selected_frames(candidate_dir, image_dir, records)

        # Save cache info
        self._save_capture_metadata(image_dir, capture_metadata)
        self._save_keyframe_manifest(image_dir, records)
        self._save_cache_info(video_path, cache_file, capture_metadata)

        return image_dir

    def _can_reuse_cache(
        self,
        video_path: Path,
        cache_file: Path,
        *,
        gps_track: Optional[GPSTrack] = None,
        imu_data: Optional[IMUData] = None,
    ) -> bool:
        """Check if cached frames are valid for this video."""
        if not cache_file.exists():
            return False

        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        # Verify video matches
        if cache.get("video_name") != video_path.name:
            logger.debug("Cache mismatch: video name")
            return False

        if cache.get("video_size") != video_path.stat().st_size:
            logger.debug("Cache mismatch: video size")
            return False

        if cache.get("fps") != self.fps:
            logger.debug(f"Cache mismatch: fps ({cache.get('fps')} != {self.fps})")
            return False
        if cache.get("jpeg_quality") != self.jpeg_quality:
            logger.debug("Cache mismatch: JPEG quality")
            return False
        if cache.get("keyframe_config") != asdict(self.keyframe_config):
            logger.debug("Cache mismatch: keyframe selection configuration")
            return False
        if cache.get("configured_capture_metadata") != self.configured_capture_metadata:
            logger.debug("Cache mismatch: configured capture metadata")
            return False
        if cache.get("telemetry_summary") != self._telemetry_summary(
            gps_track, imu_data
        ):
            logger.debug("Cache mismatch: telemetry availability")
            return False
        if not (cache_file.parent / "keyframes.json").exists():
            logger.debug("Cache mismatch: keyframe manifest missing")
            return False
        if not (cache_file.parent / "capture_metadata.json").exists():
            logger.debug("Cache mismatch: capture metadata missing")
            return False

        # Verify frames exist
        expected_frames = cache.get("frame_count", 0)
        actual_frames = len(list(cache_file.parent.glob("frame_*.jpg")))

        if actual_frames < expected_frames * 0.95:  # Allow 5% tolerance
            logger.debug(
                f"Cache mismatch: frame count ({actual_frames} < {expected_frames * 0.95})"
            )
            return False

        return True

    def _probe_capture_metadata(self, video_path: Path) -> dict[str, Any]:
        """Return stable capture and calibration metadata from ffprobe."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,"
                "time_base,nb_frames,duration:stream_tags=rotate,make,model,"
                "lens_model,encoder:format=duration,format_name"
            ),
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            payload = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            logger.warning("Could not probe capture metadata")
            payload = {}

        stream = (payload.get("streams") or [{}])[0]
        format_metadata = payload.get("format") or {}
        source_fps = self._parse_frame_rate(
            stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        )
        probed = {
            "schema_version": 1,
            "source": {
                "video_name": video_path.name,
                "video_size_bytes": video_path.stat().st_size,
            },
            "video": {
                "codec": stream.get("codec_name"),
                "width_px": stream.get("width"),
                "height_px": stream.get("height"),
                "pixel_format": stream.get("pix_fmt"),
                "source_fps": source_fps,
                "average_frame_rate": stream.get("avg_frame_rate"),
                "real_frame_rate": stream.get("r_frame_rate"),
                "time_base": stream.get("time_base"),
                "source_frame_count": self._optional_int(stream.get("nb_frames")),
                "duration_s": self._optional_float(
                    stream.get("duration") or format_metadata.get("duration")
                ),
                "format": format_metadata.get("format_name"),
                "tags": dict(sorted((stream.get("tags") or {}).items())),
            },
            "selection": asdict(self.keyframe_config),
            "calibration": self.configured_capture_metadata,
        }
        return probed

    def _probe_source_timestamps(self, video_path: Path) -> np.ndarray:
        """Read original decoded-frame timestamps in source-frame order."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            frames = json.loads(result.stdout).get("frames", [])
            timestamps = [
                float(frame["best_effort_timestamp_time"])
                for frame in frames
                if frame.get("best_effort_timestamp_time") is not None
            ]
            return np.asarray(timestamps, dtype=np.float64)
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
            logger.warning("Could not probe source frame timestamps")
            return np.empty(0, dtype=np.float64)

    @staticmethod
    def _telemetry_summary(
        gps_track: Optional[GPSTrack], imu_data: Optional[IMUData]
    ) -> dict[str, Any]:
        gps_timestamps = (
            np.asarray(gps_track.timestamps, dtype=np.float64)
            if gps_track is not None and gps_track.timestamps is not None
            else np.empty(0, dtype=np.float64)
        )
        imu_timestamps = (
            np.asarray(imu_data.timestamps, dtype=np.float64)
            if imu_data is not None
            else np.empty(0, dtype=np.float64)
        )
        return {
            "gps": {
                "available": gps_track is not None,
                "sample_count": len(gps_track) if gps_track is not None else 0,
                "timestamp_start_s": (
                    float(gps_timestamps[0]) if len(gps_timestamps) else None
                ),
                "timestamp_end_s": (
                    float(gps_timestamps[-1]) if len(gps_timestamps) else None
                ),
                "has_fix_type": (
                    gps_track is not None and gps_track.fixes is not None
                ),
                "has_horizontal_accuracy": (
                    gps_track is not None and gps_track.accuracies is not None
                ),
            },
            "imu": {
                "available": imu_data is not None,
                "sample_count": len(imu_data) if imu_data is not None else 0,
                "timestamp_start_s": (
                    float(imu_timestamps[0]) if len(imu_timestamps) else None
                ),
                "timestamp_end_s": (
                    float(imu_timestamps[-1]) if len(imu_timestamps) else None
                ),
                "has_gravity": (
                    imu_data is not None and imu_data.gravity_vectors is not None
                ),
                "has_orientation": (
                    imu_data is not None and imu_data.orientations is not None
                ),
            },
        }

    @staticmethod
    def _parse_frame_rate(value: Any) -> Optional[float]:
        if value in {None, "", "0/0"}:
            return None
        try:
            numerator, denominator = str(value).split("/", maxsplit=1)
            denominator_value = float(denominator)
            return (
                float(numerator) / denominator_value
                if denominator_value != 0
                else None
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _candidate_source_identity(
        self,
        candidate_index: int,
        source_timestamps: np.ndarray,
        source_fps: Optional[float],
    ) -> tuple[int, float]:
        """Map a sampled candidate back to an original frame and timestamp."""
        target_time = candidate_index / self.keyframe_config.candidate_fps
        if len(source_timestamps):
            insertion = int(np.searchsorted(source_timestamps, target_time))
            choices = [
                index
                for index in (insertion - 1, insertion)
                if 0 <= index < len(source_timestamps)
            ]
            source_index = min(
                choices, key=lambda index: abs(source_timestamps[index] - target_time)
            )
            return source_index, float(source_timestamps[source_index])
        if source_fps and source_fps > 0:
            source_index = int(round(target_time * source_fps))
            return source_index, source_index / source_fps
        return candidate_index, target_time

    def _select_keyframes(
        self,
        candidate_dir: Path,
        *,
        source_timestamps: np.ndarray,
        source_fps: Optional[float],
        imu_data: Optional[IMUData],
        gps_track: Optional[GPSTrack],
    ) -> list[dict[str, Any]]:
        """Score candidates and deterministically select reconstruction frames."""
        paths = sorted(candidate_dir.glob("frame_*.jpg"))
        records: list[dict[str, Any]] = []
        last_selected_gray: Optional[np.ndarray] = None
        last_selected_time: Optional[float] = None

        for candidate_index, path in enumerate(paths):
            source_index, timestamp_s = self._candidate_source_identity(
                candidate_index, source_timestamps, source_fps
            )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                records.append(
                    {
                        "candidate_index": candidate_index,
                        "source_frame_index": source_index,
                        "timestamp_s": timestamp_s,
                        "selected": False,
                        "selection_reasons": ["decode_failed"],
                    }
                )
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean_luma = float(np.mean(gray))
            clipped_fraction = float(
                np.mean((gray <= 3) | (gray >= 252))
            )
            visual_motion = (
                self._visual_motion_fraction(last_selected_gray, gray)
                if last_selected_gray is not None
                else None
            )
            angular_speed = self._interpolated_vector_norm(
                imu_data.timestamps if imu_data is not None else None,
                imu_data.gyroscope if imu_data is not None else None,
                timestamp_s,
            )
            reasons: list[str] = []
            selected = False
            elapsed = (
                timestamp_s - last_selected_time
                if last_selected_time is not None
                else None
            )

            quality_rejections = []
            if sharpness < self.keyframe_config.min_sharpness:
                quality_rejections.append("blur_rejected")
            if not (
                self.keyframe_config.min_mean_luma
                <= mean_luma
                <= self.keyframe_config.max_mean_luma
            ):
                quality_rejections.append("exposure_rejected")
            if clipped_fraction > self.keyframe_config.max_clipped_fraction:
                quality_rejections.append("clipping_rejected")

            if not quality_rejections:
                if self.keyframe_config.mode == "fixed_rate":
                    interval = 1.0 / self.keyframe_config.target_fps
                    if last_selected_time is None:
                        selected, reasons = True, ["initial"]
                    elif elapsed is not None and elapsed + 1e-9 >= interval:
                        selected, reasons = True, ["fixed_rate"]
                    else:
                        reasons = ["interval_not_reached"]
                else:
                    min_interval = self.keyframe_config.min_interval_s
                    max_interval = self.keyframe_config.max_interval_s
                    if max_interval is None:
                        max_interval = 1.0 / self.keyframe_config.target_fps
                    if last_selected_time is None:
                        selected, reasons = True, ["initial"]
                    elif elapsed is not None and elapsed + 1e-9 >= min_interval:
                        if (
                            visual_motion is not None
                            and visual_motion
                            >= self.keyframe_config.min_motion_fraction
                        ):
                            reasons.append("visual_motion")
                        if (
                            angular_speed is not None
                            and angular_speed
                            >= self.keyframe_config.min_angular_speed_rad_s
                        ):
                            reasons.append("angular_motion")
                        if elapsed + 1e-9 >= max_interval:
                            reasons.append("max_interval")
                        selected = bool(reasons)
                        if not selected:
                            reasons = ["motion_below_threshold"]
                    else:
                        reasons = ["interval_not_reached"]
            else:
                reasons.extend(quality_rejections)

            record = {
                "candidate_index": candidate_index,
                "source_frame_index": source_index,
                "timestamp_s": timestamp_s,
                "selected": selected,
                "selection_reasons": reasons,
                "quality": {
                    "sharpness": sharpness,
                    "mean_luma": mean_luma,
                    "clipped_fraction": clipped_fraction,
                    "visual_motion_fraction": visual_motion,
                },
                "telemetry": self._telemetry_at(
                    timestamp_s, imu_data=imu_data, gps_track=gps_track
                ),
            }
            records.append(record)
            if selected:
                last_selected_gray = gray
                last_selected_time = timestamp_s

        if paths and not any(record["selected"] for record in records):
            logger.warning("No candidate passed the configured keyframe quality gates")
        return records

    @staticmethod
    def _visual_motion_fraction(
        previous: np.ndarray, current: np.ndarray
    ) -> float:
        """Median tracked-pixel displacement normalized by image diagonal."""
        features = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if features is None:
            return 0.0
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, features, None
        )
        if tracked is None or status is None:
            return 0.0
        valid = status.reshape(-1).astype(bool)
        if not np.any(valid):
            return 0.0
        displacement = np.linalg.norm(
            tracked.reshape(-1, 2)[valid] - features.reshape(-1, 2)[valid],
            axis=1,
        )
        diagonal = float(np.hypot(*previous.shape[:2]))
        return float(np.median(displacement) / diagonal) if diagonal > 0 else 0.0

    @staticmethod
    def _interpolated_vector_norm(
        timestamps: Optional[np.ndarray],
        values: Optional[np.ndarray],
        timestamp_s: float,
    ) -> Optional[float]:
        if timestamps is None or values is None or len(timestamps) == 0:
            return None
        timestamps = np.asarray(timestamps, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 3:
            return None
        sample = np.asarray(
            [
                np.interp(timestamp_s, timestamps, values[:, axis])
                for axis in range(3)
            ]
        )
        return float(np.linalg.norm(sample))

    def _telemetry_at(
        self,
        timestamp_s: float,
        *,
        imu_data: Optional[IMUData],
        gps_track: Optional[GPSTrack],
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {}
        if imu_data is not None and len(imu_data.timestamps):
            telemetry["angular_speed_rad_s"] = self._interpolated_vector_norm(
                imu_data.timestamps, imu_data.gyroscope, timestamp_s
            )
            telemetry["acceleration_m_s2"] = self._interpolated_vector_norm(
                imu_data.timestamps, imu_data.accelerometer, timestamp_s
            )
        if (
            gps_track is not None
            and gps_track.timestamps is not None
            and len(gps_track.timestamps)
        ):
            nearest = int(
                np.argmin(np.abs(np.asarray(gps_track.timestamps) - timestamp_s))
            )
            telemetry["gps"] = {
                "sample_timestamp_s": float(gps_track.timestamps[nearest]),
                "latitude_deg": float(gps_track.latitudes[nearest]),
                "longitude_deg": float(gps_track.longitudes[nearest]),
                "ellipsoidal_height_m": (
                    float(gps_track.altitudes[nearest])
                    if gps_track.altitudes is not None
                    else None
                ),
                "fix_type": (
                    float(gps_track.fixes[nearest])
                    if gps_track.fixes is not None
                    else None
                ),
                "horizontal_accuracy_m": (
                    float(gps_track.accuracies[nearest])
                    if gps_track.accuracies is not None
                    else None
                ),
            }
        return telemetry

    @staticmethod
    def _publish_selected_frames(
        candidate_dir: Path,
        image_dir: Path,
        records: list[dict[str, Any]],
    ) -> None:
        for old_frame in image_dir.glob("frame_*.jpg"):
            old_frame.unlink()
        output_index = 1
        for record in records:
            if not record["selected"]:
                continue
            source = candidate_dir / f"frame_{record['candidate_index'] + 1:04d}.jpg"
            destination = image_dir / f"frame_{output_index:04d}.jpg"
            shutil.copy2(source, destination)
            record["output_frame_index"] = output_index - 1
            record["output_filename"] = destination.name
            output_index += 1

    @staticmethod
    def _save_capture_metadata(
        image_dir: Path, capture_metadata: dict[str, Any]
    ) -> None:
        with open(image_dir / "capture_metadata.json", "w") as file:
            json.dump(capture_metadata, file, indent=2, sort_keys=True)

    def _save_keyframe_manifest(
        self, image_dir: Path, records: list[dict[str, Any]]
    ) -> None:
        manifest = {
            "schema_version": 1,
            "selection_config": asdict(self.keyframe_config),
            "candidates": records,
            "selected_count": sum(bool(record["selected"]) for record in records),
        }
        with open(image_dir / "keyframes.json", "w") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)

    @staticmethod
    def load_keyframe_manifest(image_dir: Path) -> dict[str, Any]:
        with open(Path(image_dir) / "keyframes.json") as file:
            return json.load(file)

    @staticmethod
    def load_capture_metadata(image_dir: Path) -> dict[str, Any]:
        with open(Path(image_dir) / "capture_metadata.json") as file:
            return json.load(file)

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            logger.warning("Could not determine video duration")
            return 0.0

    def _check_hardware_acceleration(self) -> Optional[str]:
        """Return a usable decode accelerator, cached for the process lifetime."""
        if self._hwaccel_checked:
            return self._hwaccel

        self._hwaccel_checked = True
        self._hwaccel = self._detect_hardware_acceleration()
        return self._hwaccel

    def _detect_hardware_acceleration(self) -> Optional[str]:
        """Prefer NVIDIA NVDEC when ffmpeg and a live GPU are both present."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        available = {
            line.strip()
            for line in result.stdout.splitlines()[1:]
            if line.strip()
        }

        if "cuda" in available and self._nvidia_runtime_available():
            logger.info("NVIDIA NVDEC (cuda) decode available")
            return "cuda"

        if "videotoolbox" in available:
            logger.info("VideoToolbox hardware acceleration available")
            return "videotoolbox"

        if "vaapi" in available:
            logger.info("VAAPI hardware acceleration available")
            return "vaapi"

        return None

    @staticmethod
    def _nvidia_runtime_available() -> bool:
        """True when the NVIDIA driver reports at least one GPU."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0 and "GPU " in result.stdout

    def _build_extract_command(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        mode: str,
    ) -> list[str]:
        """Build an ffmpeg JPEG extract command.

        ``cuda`` keeps NVDEC frames in GPU memory so the ``fps`` filter can
        drop unneeded frames before the PCIe download. That matters a lot for
        high-rate 4K HEVC (e.g. 60 fps source → 10 fps candidates). JPEG encode
        remains on the CPU; newer NVIDIA cards still help via faster NVDEC.
        """
        cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner"]

        if mode == "cuda":
            cmd.extend(
                [
                    "-hwaccel",
                    "cuda",
                    "-hwaccel_device",
                    "0",
                    "-hwaccel_output_format",
                    "cuda",
                    "-extra_hw_frames",
                    "8",
                ]
            )
            video_filter = f"fps={self.fps},hwdownload,format=nv12"
        elif mode == "videotoolbox":
            cmd.extend(["-hwaccel", "videotoolbox"])
            video_filter = f"fps={self.fps}"
        elif mode == "vaapi":
            cmd.extend(["-hwaccel", "vaapi"])
            video_filter = f"fps={self.fps}"
        else:
            video_filter = f"fps={self.fps}"

        cmd.extend(
            [
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                video_filter,
                "-q:v",
                str(self.jpeg_quality),
                "-threads",
                "0",
                "-stats",
                str(output_dir / "frame_%04d.jpg"),
            ]
        )
        return cmd

    def _extract_frames(self, video_path: Path, output_dir: Path) -> None:
        """Extract frames from video using ffmpeg with progress bar."""
        logger.info(f"Extracting frames from {video_path}")

        duration = self._get_video_duration(video_path)
        estimated_frames = int(duration * self.fps) if duration > 0 else None

        hwaccel = self._check_hardware_acceleration()
        modes: list[str] = []
        if hwaccel == "cuda":
            modes.append("cuda")
        elif hwaccel in {"videotoolbox", "vaapi"}:
            modes.append(hwaccel)
        modes.append("software")

        last_error: Optional[subprocess.CalledProcessError] = None
        for index, mode in enumerate(modes):
            cmd = self._build_extract_command(video_path, output_dir, mode=mode)
            logger.info("ffmpeg extract mode=%s", mode)
            try:
                self._run_ffmpeg_extract(cmd, estimated_frames)
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                # Clear any partial JPEGs so a fallback retry starts clean.
                for partial in output_dir.glob("frame_*.jpg"):
                    partial.unlink()
                if index + 1 >= len(modes):
                    raise
                logger.warning(
                    "ffmpeg extract with mode=%s failed; falling back (%s)",
                    mode,
                    error.stderr[-500:] if isinstance(error.stderr, str) else error,
                )
                if mode == "cuda":
                    self._hwaccel = None
                    self._hwaccel_checked = True
        else:
            if last_error is not None:
                raise last_error

        actual_frames = len(list(output_dir.glob("frame_*.jpg")))
        logger.info(f"Extracted {actual_frames} frames")

    def _run_ffmpeg_extract(
        self, cmd: list[str], estimated_frames: Optional[int]
    ) -> None:
        """Run one ffmpeg extract command and stream progress into tqdm."""
        pbar = tqdm(
            total=estimated_frames,
            desc="Extracting frames",
            unit="frame",
            dynamic_ncols=True,
        )
        stderr_chunks: list[str] = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            current_frame = 0
            last_update = time.time()

            assert process.stderr is not None
            while True:
                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                stderr_chunks.append(line)

                if "frame=" in line and "fps=" in line:
                    match = re.search(r"frame=\s*(\d+)", line)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num > current_frame:
                            pbar.update(frame_num - current_frame)
                            current_frame = frame_num
                            last_update = time.time()

                if time.time() - last_update > 30:
                    logger.warning("No progress for 30s, continuing...")
                    last_update = time.time()

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code, cmd, output=None, stderr="".join(stderr_chunks)
                )

        finally:
            pbar.close()

    def _save_cache_info(
        self,
        video_path: Path,
        cache_file: Path,
        capture_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save cache metadata."""
        frame_count = len(list(cache_file.parent.glob("frame_*.jpg")))

        cache = {
            "video_name": video_path.name,
            "video_size": video_path.stat().st_size,
            "fps": self.fps,
            "frame_count": frame_count,
            "jpeg_quality": self.jpeg_quality,
            "keyframe_config": asdict(self.keyframe_config),
            "configured_capture_metadata": self.configured_capture_metadata,
            "telemetry_summary": (capture_metadata or {}).get("telemetry"),
            "capture_metadata_schema_version": (
                capture_metadata or {}
            ).get("schema_version"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    def get_frame_count(self, image_dir: Path) -> int:
        """Get number of extracted frames in a directory."""
        return len(list(Path(image_dir).glob("frame_*.jpg")))
