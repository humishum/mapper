#!/usr/bin/env python3
"""
Demux & GPS Extract module - extracts frames and GPS data from video files
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import cv2
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction and GPS data parsing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.frames_base_path = Path(config['paths']['frames'])
        
    def extract_frames_and_gps(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Extract frames and GPS data from video file
        
        Args:
            video_path: Path to input video file
            output_dir: Optional custom output directory
            
        Returns:
            Dict with frame_dir, gps_csv_path, and metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Generate unique video ID from filename and timestamp
        vid_id = f"{video_path.stem}_{int(datetime.now().timestamp())}"
        
        if output_dir:
            frame_dir = Path(output_dir) / vid_id
        else:
            frame_dir = self.frames_base_path / vid_id
            
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output directory: {frame_dir}")
        
        # Extract frames using ffmpeg
        frames_path = self._extract_frames(video_path, frame_dir)
        
        # Extract GPS data
        gps_csv_path = self._extract_gps_data(video_path, frame_dir)
        
        # Create metadata file
        metadata = {
            "video_id": vid_id,
            "source_video": str(video_path),
            "frame_dir": str(frames_path),
            "gps_csv": str(gps_csv_path) if gps_csv_path else None,
            "timestamp": datetime.now().isoformat(),
            "frame_count": len(list(frames_path.glob("*.jpg")))
        }
        
        metadata_path = frame_dir / "demux.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Extracted {metadata['frame_count']} frames")
        return metadata
        
    def _extract_frames(self, video_path: Path, output_dir: Path) -> Path:
        """Extract frames using ffmpeg"""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Use ffmpeg to extract frames at 1 FPS initially
        frame_pattern = frames_dir / "frame_%07d.jpg"
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=1",  # Extract 1 frame per second initially
            "-q:v", "2",     # High quality JPEG
            str(frame_pattern),
            "-y"             # Overwrite existing files
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Frame extraction completed successfully")
            return frames_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise
            
    def _extract_gps_data(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """Extract GPS data using exiftool"""
        gps_csv_path = output_dir / "gps.csv"
        
        # Try to extract GPS data using exiftool
        cmd = [
            "exiftool", "-ee", "-csv",
            "-GPSDateTime", "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout and "GPS" in result.stdout:
                # Parse the CSV output and save to file
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # More than just header
                    with open(gps_csv_path, 'w') as f:
                        f.write(result.stdout)
                    logger.info(f"GPS data extracted to {gps_csv_path}")
                    return gps_csv_path
                else:
                    logger.warning("No GPS data found in video")
                    return None
            else:
                logger.warning("No GPS data found in video")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"exiftool failed, trying alternative method: {e}")
            return self._extract_gps_alternative(video_path, output_dir)
        except FileNotFoundError:
            logger.warning("exiftool not found, trying alternative method")
            return self._extract_gps_alternative(video_path, output_dir)
            
    def _extract_gps_alternative(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """Alternative GPS extraction method using ffprobe"""
        gps_csv_path = output_dir / "gps.csv"
        
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", 
            "stream_tags:format_tags", "-of", "csv=p=0",
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Look for GPS-related metadata
            if "location" in result.stdout.lower() or "gps" in result.stdout.lower():
                # This is a simplified approach - real implementation would need
                # more sophisticated parsing based on the video format
                logger.info("Found location metadata, but parsing not implemented yet")
                return None
            else:
                logger.warning("No GPS metadata found with ffprobe")
                return None
                
        except subprocess.CalledProcessError:
            logger.warning("ffprobe failed - no GPS data extracted")
            return None
    
    def get_frame_timestamps(self, frames_dir: Path) -> List[float]:
        """Generate timestamps for frames based on filename indices"""
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        timestamps = []
        
        for frame_file in frame_files:
            # Extract frame number from filename (e.g., frame_0000001.jpg -> 1)
            frame_num = int(frame_file.stem.split('_')[1])
            # Assume 1 FPS extraction, so timestamp = frame_number - 1
            timestamp = frame_num - 1
            timestamps.append(timestamp)
            
        return timestamps


def main():
    """CLI interface for demux module"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Extract frames and GPS from video")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--out", help="Output directory (optional)")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process video
    processor = VideoProcessor(config)
    try:
        metadata = processor.extract_frames_and_gps(args.video, args.out)
        print(f"✓ Processing complete. Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()