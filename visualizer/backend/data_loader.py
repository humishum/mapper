"""
Data loader for reading metadata and finding pointcloud files
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import re


class DataLoader:
    """Handles loading metadata and finding pointcloud files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_all_metadata(self) -> List[Dict]:
        """
        Load metadata from all subdirectories
        
        Returns:
            List of metadata dictionaries with added folder_name field
        """
        metadata_list = []
        
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            metadata_file = subdir / "metadata.json"
            if not metadata_file.exists():
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Add folder name for reference
                metadata['folder_name'] = subdir.name
                
                # Extract GPS coordinates
                gps_coords = metadata.get('initial_gps_coordinates', [0.0, 0.0])
                altitude = metadata.get('altitude', 0.0)
                
                # Only include locations with actual GPS data
                if gps_coords != [0.0, 0.0]:
                    metadata['gps_coords'] = gps_coords
                    metadata['altitude'] = altitude
                    metadata_list.append(metadata)
                    
            except Exception as e:
                print(f"Error loading {metadata_file}: {e}")
        
        return metadata_list
    
    def find_pointcloud_file(
        self, 
        folder_name: str, 
        threshold: float, 
        sequence_id: int = 1
    ) -> Optional[Path]:
        """
        Find pointcloud file for given parameters
        
        Args:
            folder_name: Name of the folder containing the pointcloud
            threshold: Threshold value to look for (e.g., 2.0 for thr2.0)
            sequence_id: Sequence ID (default: 1)
            
        Returns:
            Path to pointcloud file or None if not found
        """
        # Look in sequence folder first
        sequence_path = (
            self.data_dir / folder_name / "pointclouds" / f"sequence_{sequence_id}"
        )
        
        if sequence_path.exists():
            ply_files = list(sequence_path.glob("*.ply"))
            
            # Look for exact threshold match
            for ply_file in ply_files:
                if f"thr{threshold}" in ply_file.name:
                    return ply_file
            
            # If exact match not found, find closest threshold
            return self._find_closest_threshold(ply_files, threshold)
        
        # Fallback to main pointclouds folder
        main_path = self.data_dir / folder_name / "pointclouds"
        if main_path.exists():
            ply_files = list(main_path.glob("*.ply"))
            
            for ply_file in ply_files:
                if f"thr{threshold}" in ply_file.name:
                    return ply_file
            
            return self._find_closest_threshold(ply_files, threshold)
        
        return None
    
    def _find_closest_threshold(
        self, 
        ply_files: List[Path], 
        target_threshold: float
    ) -> Optional[Path]:
        """
        Find PLY file with closest threshold to target
        
        Args:
            ply_files: List of PLY file paths
            target_threshold: Target threshold value
            
        Returns:
            Path to closest matching file or None
        """
        threshold_files = []
        
        for ply_file in ply_files:
            match = re.search(r"thr([0-9.]+)", ply_file.name)
            if match:
                file_threshold = float(match.group(1))
                threshold_files.append((file_threshold, ply_file))
        
        if not threshold_files:
            return None
        
        # Find closest threshold
        closest = min(threshold_files, key=lambda x: abs(x[0] - target_threshold))
        
        if closest[0] != target_threshold:
            print(
                f"Using threshold {closest[0]} "
                f"(closest to requested {target_threshold})"
            )
        
        return closest[1]
    
    def get_available_thresholds(
        self, 
        folder_name: str, 
        sequence_id: int = 1
    ) -> List[float]:
        """
        Get list of available thresholds for a folder
        
        Args:
            folder_name: Name of the folder
            sequence_id: Sequence ID
            
        Returns:
            Sorted list of available thresholds
        """
        sequence_path = (
            self.data_dir / folder_name / "pointclouds" / f"sequence_{sequence_id}"
        )
        
        if not sequence_path.exists():
            sequence_path = self.data_dir / folder_name / "pointclouds"
        
        if not sequence_path.exists():
            return []
        
        ply_files = list(sequence_path.glob("*.ply"))
        thresholds = []
        
        for ply_file in ply_files:
            match = re.search(r"thr([0-9.]+)", ply_file.name)
            if match:
                thresholds.append(float(match.group(1)))
        
        return sorted(thresholds)

