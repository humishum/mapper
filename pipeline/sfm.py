#!/usr/bin/env python3
"""
Structure from Motion module using COLMAP
"""

import os
import json
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ColmapReconstructor:
    """COLMAP-based 3D reconstruction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.colmap_config = config['colmap']
        self.workspace_base = Path(config['paths']['colmap'])
        
    def reconstruct(self, frames_dir: str, gps_csv_path: Optional[str] = None, 
                   workspace_dir: Optional[str] = None) -> Dict:
        """
        Perform 3D reconstruction using COLMAP
        
        Args:
            frames_dir: Directory containing filtered frames
            gps_csv_path: Path to GPS data CSV (optional)
            workspace_dir: Custom workspace directory (optional)
            
        Returns:
            Dict with reconstruction results and metadata
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_path}")
            
        frame_files = list(frames_path.glob("*.jpg"))
        if not frame_files:
            raise ValueError(f"No image files found in {frames_path}")
            
        logger.info(f"Starting COLMAP reconstruction with {len(frame_files)} images")
        
        # Set up workspace
        if workspace_dir:
            workspace = Path(workspace_dir)
        else:
            workspace = self.workspace_base / f"recon_{frames_path.parent.name}"
            
        workspace.mkdir(parents=True, exist_ok=True)
        
        # COLMAP workflow
        database_path = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        dense_dir = workspace / "dense"
        
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Feature extraction
            self._feature_extraction(frames_path, database_path)
            
            # Step 2: Feature matching
            self._feature_matching(database_path)
            
            # Step 3: Sparse reconstruction
            self._sparse_reconstruction(database_path, sparse_dir, frames_path)
            
            # Step 4: Check if reconstruction succeeded
            model_files = list(sparse_dir.glob("*/"))
            if not model_files:
                raise RuntimeError("COLMAP sparse reconstruction failed - no models generated")
                
            # Select the largest model (most images)
            best_model = self._select_best_model(model_files)
            logger.info(f"Selected model: {best_model}")
            
            # Step 5: Dense reconstruction (optional, can be skipped for speed)
            # self._dense_reconstruction(frames_path, sparse_dir / best_model.name, dense_dir)
            
            # Step 6: Export point cloud
            model_path = sparse_dir / best_model.name
            ply_path = self._export_point_cloud(model_path, workspace)
            
            # Step 7: Create metadata
            metadata = self._create_metadata(
                frames_path, workspace, model_path, ply_path, gps_csv_path
            )
            
            # Save metadata
            metadata_path = workspace / "model_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            logger.error(f"COLMAP reconstruction failed: {e}")
            raise
            
    def _feature_extraction(self, images_path: Path, database_path: Path):
        """Extract SIFT features from images"""
        logger.info("Extracting features...")
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--ImageReader.single_camera", "true",
            "--SiftExtraction.use_gpu", "true" if self._check_gpu_available() else "false"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Feature extraction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature extraction failed: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError("COLMAP not found. Please install COLMAP and ensure it's in PATH")
            
    def _feature_matching(self, database_path: Path):
        """Match features between images"""
        logger.info("Matching features...")
        
        matcher_type = self.colmap_config.get('matcher', 'sequential')
        
        if matcher_type == 'sequential':
            cmd = [
                "colmap", "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "true" if self._check_gpu_available() else "false"
            ]
        else:
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "true" if self._check_gpu_available() else "false"
            ]
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Feature matching completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Feature matching failed: {e.stderr}")
            raise
            
    def _sparse_reconstruction(self, database_path: Path, sparse_dir: Path, images_path: Path):
        """Perform sparse 3D reconstruction"""
        logger.info("Running sparse reconstruction...")
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--output_path", str(sparse_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Sparse reconstruction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Sparse reconstruction failed: {e.stderr}")
            raise
            
    def _dense_reconstruction(self, images_path: Path, sparse_model_path: Path, dense_dir: Path):
        """Perform dense 3D reconstruction (MVS)"""
        logger.info("Running dense reconstruction...")
        
        # This is computationally intensive and optional for the MVP
        # Undistort images
        undistorted_dir = dense_dir / "undistorted"
        undistorted_dir.mkdir(exist_ok=True)
        
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(images_path),
            "--input_path", str(sparse_model_path),
            "--output_path", str(undistorted_dir)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Dense reconstruction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Dense reconstruction failed: {e.stderr}")
            raise
            
    def _select_best_model(self, model_dirs: List[Path]) -> Path:
        """Select the model with the most reconstructed images"""
        best_model = None
        max_images = 0
        
        for model_dir in model_dirs:
            images_file = model_dir / "images.bin"
            if images_file.exists():
                # Count images in this model
                try:
                    # Simple approximation: file size correlates with number of images
                    image_count = images_file.stat().st_size // 100  # Rough estimate
                    if image_count > max_images:
                        max_images = image_count
                        best_model = model_dir
                except Exception:
                    continue
                    
        if best_model is None and model_dirs:
            # Fallback: use the first model
            best_model = model_dirs[0]
            
        return best_model
        
    def _export_point_cloud(self, model_path: Path, workspace: Path) -> Path:
        """Export point cloud as PLY file"""
        logger.info("Exporting point cloud...")
        
        ply_path = workspace / "model.ply"
        
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(model_path),
            "--output_path", str(ply_path),
            "--output_type", "PLY"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Point cloud exported to {ply_path}")
            return ply_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Point cloud export failed: {e.stderr}")
            raise
            
    def _create_metadata(self, frames_path: Path, workspace: Path, 
                        model_path: Path, ply_path: Path, 
                        gps_csv_path: Optional[str]) -> Dict:
        """Create reconstruction metadata"""
        
        # Count reconstructed points and cameras
        points_file = model_path / "points3D.bin"
        cameras_file = model_path / "cameras.bin"
        images_file = model_path / "images.bin"
        
        metadata = {
            "workspace": str(workspace),
            "model_path": str(model_path),
            "ply_path": str(ply_path),
            "frames_path": str(frames_path),
            "gps_csv_path": gps_csv_path,
            "files_exist": {
                "points3D": points_file.exists(),
                "cameras": cameras_file.exists(),
                "images": images_file.exists(),
                "ply": ply_path.exists()
            }
        }
        
        # Try to get basic statistics
        try:
            if ply_path.exists():
                file_size = ply_path.stat().st_size
                metadata["ply_size_mb"] = round(file_size / (1024 * 1024), 2)
        except Exception:
            pass
            
        return metadata
        
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for COLMAP"""
        try:
            # Try to run nvidia-smi to check for NVIDIA GPU
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def visualize_model(self, model_path: str):
        """Open COLMAP GUI to visualize the model"""
        cmd = ["colmap", "gui", "--import_path", str(model_path)]
        
        try:
            subprocess.Popen(cmd)
            logger.info("COLMAP GUI launched")
        except Exception as e:
            logger.error(f"Failed to launch COLMAP GUI: {e}")


def main():
    """CLI interface for SfM module"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Perform 3D reconstruction with COLMAP")
    parser.add_argument("--frames", required=True, help="Input frames directory")
    parser.add_argument("--workspace", help="COLMAP workspace directory")
    parser.add_argument("--gps", help="GPS CSV file")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Config file")
    parser.add_argument("--visualize", action="store_true", help="Open COLMAP GUI after reconstruction")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create reconstructor
    reconstructor = ColmapReconstructor(config)
    
    try:
        metadata = reconstructor.reconstruct(args.frames, args.gps, args.workspace)
        print("✓ 3D reconstruction complete!")
        print(f"  Model path: {metadata['model_path']}")
        print(f"  Point cloud: {metadata['ply_path']}")
        
        if metadata.get('ply_size_mb'):
            print(f"  PLY file size: {metadata['ply_size_mb']} MB")
            
        if args.visualize:
            reconstructor.visualize_model(metadata['model_path'])
            
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()