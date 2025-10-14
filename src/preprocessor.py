# this file takes in a video and outputs the images 
# does filtering and other preprocessing 

from pathlib import Path 
import subprocess
from typing import Tuple
import json
import os
import logging
import shutil
import re
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters 
FPS = 10
JPEG_QUALITY = 5  # JPEG quality for frame extraction (1-31, lower = higher quality)


class Preprocessor:
    def __init__(self, video_path:Path, output_path:Path, overwrite:bool = False):
        self.video_path = video_path # input video 
        self.output_path = output_path # output folder for frames 
        self.metadata_path = self.output_path.parent / "metadata.json"
        self.overwrite = overwrite # whether to overwrite existing frames 
        self.fps = FPS
        self._cached = self._check_cached()
        self._check_paths()
        self._check_ffmpeg_installed()
        

    def __call__(self, save_metadata: bool = True)->Path:
        """
        Process the video and extract frames.
        
        Args:
            save_metadata: Whether to save metadata JSON file
        """
        logger.info(f"Processing {self.video_path}")
        
        if self._cached:
            logger.info(f"Frames already cached at {self.output_path}")
            return
        
        self._get_frames()

        if save_metadata:
            logger.info(f"Saving metadata to {self.output_path}")
            self._save_metadata()
        return self.output_path

    def _check_cached(self): 
        """ Check if frames are already written to output path with the same parameters and we can just skip"""

        # Check if metadata file exists 
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata["fps"] == self.fps and metadata["video_name"] == self.video_path.name:
                logger.info(f"Frames already cached at {self.output_path}")
                return True
        else:
            logger.info(f"No metadata file found at {self.metadata_path}, will need to process video")
            return False

    def _check_paths(self): 
        # Check video is valid 
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        else:
            logger.info(f"Video file found: {self.video_path}")

        # Check output path is valid    
        if not self.output_path.exists():
            os.makedirs(self.output_path)
        
        # if overwrite is True, delete existing frames 
        if self.overwrite:
            logger.warning(f"Overwriting existing frames at {self.output_path}")
            shutil.rmtree(self.output_path)
            os.makedirs(self.output_path)
    
    def _check_ffmpeg_installed(self):
        try:
            logger.info(f"Checking if FFmpeg is installed")
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise RuntimeError("FFmpeg is not installed. Please install FFmpeg and try again.")
    
    def _check_hardware_acceleration(self):
        """Check if hardware acceleration is available."""
        try:
            result = subprocess.run([
                "ffmpeg", "-hwaccels"
            ], capture_output=True, text=True, check=True)
            
            hwaccels = result.stdout.strip().split('\n')[1:]  # Skip header line
            available_hwaccels = [accel.strip() for accel in hwaccels if accel.strip()]
            
            logger.info(f"Available hardware accelerations: {available_hwaccels}")
            
            # Check for common hardware accelerations
            if any(accel in available_hwaccels for accel in ['cuda', 'nvdec', 'nvenc']):
                logger.info("NVIDIA GPU acceleration available")
                return 'cuda'
            elif any(accel in available_hwaccels for accel in ['vaapi', 'videotoolbox']):
                logger.info("Other hardware acceleration available")
                return 'auto'
            else:
                logger.info("No hardware acceleration detected, using CPU")
                return None
                
        except subprocess.CalledProcessError:
            logger.warning("Could not check hardware acceleration support")
            return None
    
    def _get_video_duration(self) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(self.video_path)
            ], capture_output=True, text=True, check=True)
            
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError):
            logger.warning("Could not determine video duration, progress bar will be less accurate")
            return 0.0

    def _get_frames(self):
        """Extract frames from video with progress bar - optimized for speed."""
        logger.info(f"Getting frames from {self.video_path}")
        
        # Get video duration to estimate total frames
        duration = self._get_video_duration()
        estimated_frames = int(duration * FPS) if duration > 0 else None
        
        # Check for hardware acceleration
        hwaccel = self._check_hardware_acceleration()
        
        cmd = ["ffmpeg", "-y"]  # -y to overwrite existing files
        
        # Add hardware acceleration if available
        if hwaccel:
            cmd.extend(["-hwaccel", hwaccel])
            
        cmd.extend([
            "-i", str(self.video_path),
            "-vf", f"fps={FPS}",  # Use fps filter for better performance than -r
            "-c:v", "mjpeg",  # Use MJPEG encoder for faster JPEG encoding
            "-q:v", str(JPEG_QUALITY),  # JPEG quality (1-31, lower = higher quality)
            "-preset", "ultrafast",  # Fastest encoding preset
            "-threads", "0",  # Use all available CPU threads
            "-stats",  # Enable statistics output
            str(self.output_path / "frame_%04d.jpg")
        ])
        
        # Initialize progress bar
        pbar = tqdm(
            total=estimated_frames,
            desc="Extracting frames",
            unit="frame",
            dynamic_ncols=True
        )
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            current_frame = 0
            last_update_time = time.time()
            
            # Read from stderr where ffmpeg outputs progress with -stats
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    break
                
                try:
                    # Read stderr with timeout to avoid hanging
                    stderr_line = process.stderr.readline()
                    if not stderr_line:
                        break
                        
                    line = stderr_line.strip()
                    
                    # Parse frame number from ffmpeg stats output
                    # Format: frame= 1234 fps=25.0 q=28.0 size=1234kB time=00:00:49.36 bitrate= 205.0kbits/s speed=1.01x
                    if 'frame=' in line and 'fps=' in line:
                        try:
                            # Extract frame number using regex
                            frame_match = re.search(r'frame=\s*(\d+)', line)
                            if frame_match:
                                frame_num = int(frame_match.group(1))
                                if frame_num > current_frame:
                                    pbar.update(frame_num - current_frame)
                                    current_frame = frame_num
                                    last_update_time = time.time()
                        except (ValueError, AttributeError):
                            pass
                    
                    # Timeout check - if no updates for 30 seconds, something might be wrong
                    if time.time() - last_update_time > 30:
                        logger.warning("No progress updates for 30 seconds, continuing to wait...")
                        last_update_time = time.time()
                        
                except Exception as e:
                    logger.debug(f"Error reading progress: {e}")
                    continue
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                stderr_output = process.stderr.read()
                logger.error(f"FFmpeg failed with return code {return_code}")
                logger.error(f"Error output: {stderr_output}")
                raise subprocess.CalledProcessError(return_code, cmd, stderr_output)
                
        finally:
            pbar.close()
        
        # Log final frame count
        actual_frames = len(list(self.output_path.glob("frame_*.jpg")))
        logger.info(f"Extracted {actual_frames} frames")

    def _get_initial_gps_coordinates(self)->Tuple[float, float]:
        # Use ffmpeg to get GPS coordinates, if present 
        try:
            # Try to extract GPS coordinates using ffprobe (part of ffmpeg)
            logger.info(f"Extracting GPS coordinates from {self.video_path}")
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-select_streams", "v:0", 
                "-show_entries", "format_tags=location", "-of", "csv=p=0", 
                str(self.video_path)
            ], capture_output=True, text=True, check=True)
            
            location = result.stdout.strip()
            if location and location != "N/A":
                # Parse location string (format: +37.5090+127.0243/)
                import re
                match = re.match(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', location)
                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    return lat, lon
        except (subprocess.CalledProcessError, ValueError, AttributeError):
            pass
        
        return 0.0, 0.0
    
    def _get_altitude(self)->float:
        # Use ffmpeg to get altitude, if present 
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-select_streams", "v:0", 
                "-show_entries", "format_tags=altitude", "-of", "csv=p=0", 
                str(self.video_path)
            ], capture_output=True, text=True, check=True)
            
            altitude = result.stdout.strip()
            if altitude and altitude != "N/A":
                return float(altitude)
        except (subprocess.CalledProcessError, ValueError, AttributeError):
            pass
        
        return 0.0
    
    def _save_metadata(self)->dict:
        metadata = {
            "fps": FPS,
            "video_name": self.video_path.name,
            "video_type": self.video_path.suffix,
            "video_size": self.video_path.stat().st_size,
            "output_path": str(self.output_path),
            "initial_gps_coordinates": self._get_initial_gps_coordinates(),
            "altitude": self._get_altitude(),
            "frames": len(list(self.output_path.glob("*.jpg")))
        }
        # Save to json at output path 
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)
        return metadata


if __name__ == "__main__":
    preprocessor = Preprocessor(video_path=Path("/home/ape/Documents/MapperGoProVids/tahoe_ridge_1.MP4"), output_path=Path("/tmp/images"))
    preprocessor()