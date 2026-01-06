"""Video frame extraction with smart caching."""

from pathlib import Path
import json
import subprocess
import logging
import re
import time
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize video processor.

        Args:
            fps: Frame extraction rate (frames per second)
            jpeg_quality: JPEG quality (1-31, lower = higher quality)
        """
        self.fps = fps
        self.jpeg_quality = jpeg_quality

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
    ) -> Path:
        """
        Extract frames from video, reusing cache if valid.

        Args:
            video_path: Path to input video file
            output_dir: Directory for output (images will be in output_dir/images/)
            force: If True, ignore cache and re-extract

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
        if not force and self._can_reuse_cache(video_path, cache_file):
            logger.info(f"Reusing cached frames from {image_dir}")
            return image_dir

        # Extract frames
        image_dir.mkdir(parents=True, exist_ok=True)
        self._extract_frames(video_path, image_dir)

        # Save cache info
        self._save_cache_info(video_path, cache_file)

        return image_dir

    def _can_reuse_cache(self, video_path: Path, cache_file: Path) -> bool:
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
            logger.debug(f"Cache mismatch: video name")
            return False

        if cache.get("video_size") != video_path.stat().st_size:
            logger.debug(f"Cache mismatch: video size")
            return False

        if cache.get("fps") != self.fps:
            logger.debug(f"Cache mismatch: fps ({cache.get('fps')} != {self.fps})")
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
        """Check if hardware acceleration is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hwaccels"],
                capture_output=True,
                text=True,
                check=True,
            )

            hwaccels = result.stdout.strip().split("\n")[1:]  # Skip header
            available = [accel.strip() for accel in hwaccels if accel.strip()]

            if any(accel in available for accel in ["cuda", "nvdec", "nvenc"]):
                logger.info("NVIDIA GPU acceleration available")
                return "cuda"
            elif any(accel in available for accel in ["vaapi", "videotoolbox"]):
                logger.info("Hardware acceleration available")
                return "auto"

        except subprocess.CalledProcessError:
            pass

        return None

    def _extract_frames(self, video_path: Path, output_dir: Path) -> None:
        """Extract frames from video using ffmpeg with progress bar."""
        logger.info(f"Extracting frames from {video_path}")

        # Get video duration for progress estimation
        duration = self._get_video_duration(video_path)
        estimated_frames = int(duration * self.fps) if duration > 0 else None

        # Check for hardware acceleration
        hwaccel = self._check_hardware_acceleration()

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]

        if hwaccel:
            cmd.extend(["-hwaccel", hwaccel])

        cmd.extend(
            [
                "-i",
                str(video_path),
                "-vf",
                f"fps={self.fps}",
                "-q:v",
                str(self.jpeg_quality),
                "-threads",
                "0",
                "-stats",
                str(output_dir / "frame_%04d.jpg"),
            ]
        )

        # Run with progress tracking
        pbar = tqdm(
            total=estimated_frames,
            desc="Extracting frames",
            unit="frame",
            dynamic_ncols=True,
        )

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

            while True:
                if process.poll() is not None:
                    break

                try:
                    line = process.stderr.readline()
                    if not line:
                        break

                    # Parse frame number from ffmpeg output
                    if "frame=" in line and "fps=" in line:
                        match = re.search(r"frame=\s*(\d+)", line)
                        if match:
                            frame_num = int(match.group(1))
                            if frame_num > current_frame:
                                pbar.update(frame_num - current_frame)
                                current_frame = frame_num
                                last_update = time.time()

                    # Timeout warning
                    if time.time() - last_update > 30:
                        logger.warning("No progress for 30s, continuing...")
                        last_update = time.time()

                except Exception:
                    continue

            return_code = process.wait()

            if return_code != 0:
                stderr = process.stderr.read()
                raise subprocess.CalledProcessError(return_code, cmd, stderr)

        finally:
            pbar.close()

        actual_frames = len(list(output_dir.glob("frame_*.jpg")))
        logger.info(f"Extracted {actual_frames} frames")

    def _save_cache_info(self, video_path: Path, cache_file: Path) -> None:
        """Save cache metadata."""
        frame_count = len(list(cache_file.parent.glob("frame_*.jpg")))

        cache = {
            "video_name": video_path.name,
            "video_size": video_path.stat().st_size,
            "fps": self.fps,
            "frame_count": frame_count,
            "jpeg_quality": self.jpeg_quality,
        }

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    def get_frame_count(self, image_dir: Path) -> int:
        """Get number of extracted frames in a directory."""
        return len(list(Path(image_dir).glob("frame_*.jpg")))
