#!/usr/bin/env python3
"""
Main CLI interface for the hiking 3D reconstruction pipeline
"""

import os
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict

from .demux import VideoProcessor
from .filter import FrameFilter
from .sfm import ColmapReconstructor

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.video_processor = VideoProcessor(self.config)
        self.frame_filter = FrameFilter(self.config)
        self.reconstructor = ColmapReconstructor(self.config)
        
    def run_full_pipeline(self, video_path: str, output_dir: str = None) -> Dict:
        """
        Run the complete pipeline from video to 3D model
        
        Args:
            video_path: Path to input video file
            output_dir: Optional custom output directory
            
        Returns:
            Dict with pipeline results
        """
        logger.info(f"Starting full pipeline for video: {video_path}")
        
        try:
            # Step 1: Extract frames and GPS
            logger.info("Step 1: Extracting frames and GPS data...")
            demux_result = self.video_processor.extract_frames_and_gps(video_path, output_dir)
            frames_dir = demux_result['frame_dir']
            gps_csv = demux_result['gps_csv']
            
            # Step 2: Filter frames
            logger.info("Step 2: Filtering frames...")
            filter_result = self.frame_filter.filter_frames(frames_dir)
            filtered_frames_dir = filter_result['output_dir']
            
            # Step 3: 3D Reconstruction
            logger.info("Step 3: Running 3D reconstruction...")
            sfm_result = self.reconstructor.reconstruct(filtered_frames_dir, gps_csv)
            
            # Compile final results
            pipeline_result = {
                "status": "success",
                "video_id": demux_result['video_id'],
                "source_video": video_path,
                "stages": {
                    "demux": demux_result,
                    "filter": filter_result,
                    "sfm": sfm_result
                },
                "final_outputs": {
                    "point_cloud": sfm_result['ply_path'],
                    "model_path": sfm_result['model_path'],
                    "filtered_frames": filtered_frames_dir
                }
            }
            
            logger.info("✓ Pipeline completed successfully!")
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
            
    def run_single_stage(self, stage: str, **kwargs) -> Dict:
        """Run a single pipeline stage"""
        
        if stage == "demux":
            return self.video_processor.extract_frames_and_gps(**kwargs)
        elif stage == "filter":
            return self.frame_filter.filter_frames(**kwargs)
        elif stage == "sfm":
            return self.reconstructor.reconstruct(**kwargs)
        else:
            raise ValueError(f"Unknown stage: {stage}")


def setup_logging(debug: bool = False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )


def cmd_run(args):
    """Run the full pipeline"""
    setup_logging(args.debug)
    
    runner = PipelineRunner(args.config)
    
    try:
        result = runner.run_full_pipeline(args.video, args.output)
        
        print("\n✓ Pipeline completed successfully!")
        print(f"Video ID: {result['video_id']}")
        print(f"Point cloud: {result['final_outputs']['point_cloud']}")
        print(f"Model path: {result['final_outputs']['model_path']}")
        
        if args.visualize:
            runner.reconstructor.visualize_model(result['final_outputs']['model_path'])
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)


def cmd_stage(args):
    """Run a single pipeline stage"""
    setup_logging(args.debug)
    
    runner = PipelineRunner(args.config)
    
    # Prepare kwargs based on stage
    kwargs = {}
    if args.stage == "demux":
        kwargs = {"video_path": args.input, "output_dir": args.output}
    elif args.stage == "filter":
        kwargs = {"frames_dir": args.input, "output_dir": args.output}
    elif args.stage == "sfm":
        kwargs = {"frames_dir": args.input, "gps_csv_path": args.gps, "workspace_dir": args.output}
    
    try:
        result = runner.run_single_stage(args.stage, **kwargs)
        print(f"✓ Stage '{args.stage}' completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        logger.error(f"Stage '{args.stage}' failed: {e}")
        exit(1)


def cmd_status(args):
    """Show pipeline status and logs"""
    setup_logging(args.debug)
    
    # Check dependencies
    print("Checking dependencies...")
    
    dependencies = {
        "ffmpeg": "ffmpeg -version",
        "exiftool": "exiftool -ver",
        "colmap": "colmap -h"
    }
    
    for dep, cmd in dependencies.items():
        try:
            import subprocess
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {dep}: Available")
            else:
                print(f"  ✗ {dep}: Error")
        except FileNotFoundError:
            print(f"  ✗ {dep}: Not found")
    
    # Show recent logs
    log_file = Path("pipeline.log")
    if log_file.exists():
        print(f"\nRecent log entries from {log_file}:")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:  # Last 20 lines
                print(f"  {line.rstrip()}")
    else:
        print("\nNo log file found")


def cmd_clean(args):
    """Clean up temporary files"""
    setup_logging(args.debug)
    
    runner = PipelineRunner(args.config)
    
    # Define cleanup paths
    cleanup_paths = [
        Path(runner.config['paths']['frames']),
        Path(runner.config['paths']['colmap'])
    ]
    
    for path in cleanup_paths:
        if path.exists():
            try:
                import shutil
                shutil.rmtree(path)
                print(f"✓ Cleaned: {path}")
            except Exception as e:
                print(f"✗ Failed to clean {path}: {e}")
        else:
            print(f"  {path}: Not found")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Hiking 3D Reconstruction Pipeline")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("video", help="Input video file")
    run_parser.add_argument("--output", help="Output directory")
    run_parser.add_argument("--visualize", action="store_true", help="Open COLMAP GUI after completion")
    run_parser.set_defaults(func=cmd_run)
    
    # Stage command (single stage)
    stage_parser = subparsers.add_parser("stage", help="Run single pipeline stage")
    stage_parser.add_argument("stage", choices=["demux", "filter", "sfm"], help="Stage to run")
    stage_parser.add_argument("input", help="Input path")
    stage_parser.add_argument("--output", help="Output path")
    stage_parser.add_argument("--gps", help="GPS CSV file (for sfm stage)")
    stage_parser.set_defaults(func=cmd_stage)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.set_defaults(func=cmd_status)
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean temporary files")
    clean_parser.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()