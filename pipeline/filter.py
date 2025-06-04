#!/usr/bin/env python3
"""
Frame Filter module - filters blurry and redundant frames
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FrameFilter:
    """Filters frames based on blur detection and temporal sampling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.blur_threshold = config['filter']['blur_thresh']
        self.fps_target = config['filter']['fps_target']
        
    def filter_frames(self, frames_dir: str, output_dir: str = None) -> Dict:
        """
        Filter frames based on blur and temporal criteria
        
        Args:
            frames_dir: Directory containing input frames
            output_dir: Directory for filtered frames (optional)
            
        Returns:
            Dict with filtering statistics and output paths
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_path}")
            
        # Get all frame files
        frame_files = sorted(frames_path.glob("frame_*.jpg"))
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_path}")
            
        logger.info(f"Found {len(frame_files)} frames to filter")
        
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = frames_path.parent / "frames_filtered"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Blur detection
        sharp_frames = self._filter_blurry_frames(frame_files)
        logger.info(f"After blur filtering: {len(sharp_frames)} frames")
        
        # Step 2: Temporal sampling
        sampled_frames = self._temporal_sampling(sharp_frames)
        logger.info(f"After temporal sampling: {len(sampled_frames)} frames")
        
        # Step 3: Copy filtered frames to output directory
        copied_files = self._copy_filtered_frames(sampled_frames, output_path)
        
        # Create filter metadata
        metadata = {
            "input_dir": str(frames_path),
            "output_dir": str(output_path),
            "original_count": len(frame_files),
            "after_blur_filter": len(sharp_frames),
            "final_count": len(copied_files),
            "blur_threshold": self.blur_threshold,
            "fps_target": self.fps_target,
            "filtered_files": [str(f) for f in copied_files]
        }
        
        # Save metadata
        metadata_path = output_path / "filter_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
        
    def _filter_blurry_frames(self, frame_files: List[Path]) -> List[Tuple[Path, float]]:
        """
        Filter out blurry frames using variance of Laplacian
        
        Returns:
            List of tuples (frame_path, sharpness_score)
        """
        sharp_frames = []
        
        logger.info("Analyzing frame sharpness...")
        for frame_file in tqdm(frame_files, desc="Blur detection"):
            sharpness = self._calculate_sharpness(frame_file)
            
            if sharpness >= self.blur_threshold:
                sharp_frames.append((frame_file, sharpness))
                
        return sharp_frames
        
    def _calculate_sharpness(self, image_path: Path) -> float:
        """
        Calculate sharpness using variance of Laplacian
        
        Args:
            image_path: Path to image file
            
        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Read image in grayscale
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return 0.0
                
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
            
        except Exception as e:
            logger.error(f"Error calculating sharpness for {image_path}: {e}")
            return 0.0
            
    def _temporal_sampling(self, sharp_frames: List[Tuple[Path, float]]) -> List[Path]:
        """
        Perform temporal sampling to achieve target FPS
        
        Args:
            sharp_frames: List of (frame_path, sharpness_score) tuples
            
        Returns:
            List of selected frame paths
        """
        if not sharp_frames:
            return []
            
        # Sort by frame number (extracted from filename)
        sharp_frames.sort(key=lambda x: self._extract_frame_number(x[0]))
        
        # Calculate sampling interval
        # Assuming original extraction was at 1 FPS, adjust for target FPS
        original_fps = 1.0  # Frames were extracted at 1 FPS
        if self.fps_target >= original_fps:
            # No need to downsample
            return [frame[0] for frame in sharp_frames]
            
        sample_interval = original_fps / self.fps_target
        
        # Sample frames
        sampled_frames = []
        last_selected_idx = -sample_interval
        
        for i, (frame_path, sharpness) in enumerate(sharp_frames):
            if i - last_selected_idx >= sample_interval:
                sampled_frames.append(frame_path)
                last_selected_idx = i
                
        return sampled_frames
        
    def _extract_frame_number(self, frame_path: Path) -> int:
        """Extract frame number from filename like frame_0000001.jpg"""
        try:
            return int(frame_path.stem.split('_')[1])
        except (IndexError, ValueError):
            logger.warning(f"Could not extract frame number from {frame_path}")
            return 0
            
    def _copy_filtered_frames(self, selected_frames: List[Path], output_dir: Path) -> List[Path]:
        """
        Copy selected frames to output directory with sequential naming
        
        Args:
            selected_frames: List of frame paths to copy
            output_dir: Destination directory
            
        Returns:
            List of copied file paths
        """
        copied_files = []
        
        logger.info("Copying filtered frames...")
        for i, frame_path in enumerate(tqdm(selected_frames, desc="Copying frames")):
            # Create sequential filename
            output_filename = f"frame_{i+1:07d}.jpg"
            output_path = output_dir / output_filename
            
            try:
                shutil.copy2(frame_path, output_path)
                copied_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to copy {frame_path} to {output_path}: {e}")
                
        return copied_files
        
    def analyze_frame_quality(self, frames_dir: str) -> Dict:
        """
        Analyze frame quality without filtering (for debugging)
        
        Args:
            frames_dir: Directory containing frames
            
        Returns:
            Dict with quality statistics
        """
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob("frame_*.jpg"))
        
        sharpness_scores = []
        
        logger.info("Analyzing frame quality...")
        for frame_file in tqdm(frame_files, desc="Quality analysis"):
            sharpness = self._calculate_sharpness(frame_file)
            sharpness_scores.append(sharpness)
            
        if sharpness_scores:
            stats = {
                "total_frames": len(sharpness_scores),
                "mean_sharpness": float(np.mean(sharpness_scores)),
                "std_sharpness": float(np.std(sharpness_scores)),
                "min_sharpness": float(np.min(sharpness_scores)),
                "max_sharpness": float(np.max(sharpness_scores)),
                "frames_above_threshold": int(np.sum(np.array(sharpness_scores) >= self.blur_threshold)),
                "blur_threshold": self.blur_threshold
            }
        else:
            stats = {"error": "No frames found or analyzed"}
            
        return stats


def main():
    """CLI interface for filter module"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Filter video frames")
    parser.add_argument("--frames", required=True, help="Input frames directory")
    parser.add_argument("--out", help="Output directory for filtered frames")
    parser.add_argument("--blur-thresh", type=float, help="Blur threshold override")
    parser.add_argument("--fps", type=float, help="Target FPS override")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Config file")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze quality, don't filter")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args if provided
    if args.blur_thresh is not None:
        config['filter']['blur_thresh'] = args.blur_thresh
    if args.fps is not None:
        config['filter']['fps_target'] = args.fps
    
    # Create filter
    frame_filter = FrameFilter(config)
    
    try:
        if args.analyze_only:
            stats = frame_filter.analyze_frame_quality(args.frames)
            print("Frame Quality Analysis:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            metadata = frame_filter.filter_frames(args.frames, args.out)
            print(f"✓ Filtering complete. Results:")
            print(f"  Original frames: {metadata['original_count']}")
            print(f"  After blur filter: {metadata['after_blur_filter']}")
            print(f"  Final count: {metadata['final_count']}")
            print(f"  Output directory: {metadata['output_dir']}")
            
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()