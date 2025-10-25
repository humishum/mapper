"""
Pointcloud processor for loading and processing PLY files
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class PointcloudProcessor:
    """Handles loading and processing pointcloud data from PLY files"""
    
    def __init__(self, max_points: int = 50000):
        self.max_points = max_points
    
    def load_ply(self, file_path: Path) -> Optional[Dict]:
        """
        Load and process a PLY file
        
        Args:
            file_path: Path to PLY file
            
        Returns:
            Dictionary with processed pointcloud data or None on error
        """
        try:
            points, colors = self._parse_ply_file(file_path)
            
            if points.shape[0] == 0:
                print(f"No points found in {file_path}")
                return None
            
            original_count = points.shape[0]
            
            # Downsample if needed
            if points.shape[0] > self.max_points:
                indices = np.linspace(0, points.shape[0] - 1, self.max_points, dtype=int)
                points = points[indices]
                if colors is not None and colors.shape[0] > 0:
                    colors = colors[indices]
            
            # Calculate bounding box for normalization
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            center = (bbox_min + bbox_max) / 2
            
            # Center points around origin
            points_centered = points - center
            
            return {
                'points': points_centered.tolist(),
                'colors': colors.tolist() if colors is not None else None,
                'center': center.tolist(),
                'bbox_min': bbox_min.tolist(),
                'bbox_max': bbox_max.tolist(),
                'original_count': original_count,
                'display_count': points.shape[0],
                'file_path': str(file_path)
            }
            
        except Exception as e:
            print(f"Error loading pointcloud {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_ply_file(self, file_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Parse PLY file and extract points and colors
        
        Args:
            file_path: Path to PLY file
            
        Returns:
            Tuple of (points array, colors array or None)
        """
        with open(file_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # Parse header
            vertex_count = 0
            has_colors = False
            has_alpha = False
            is_binary = False
            
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('format'):
                    is_binary = 'binary' in line
                elif any(color in line for color in ['red', 'green', 'blue']):
                    has_colors = True
                elif 'alpha' in line:
                    has_alpha = True
            
            # Read vertex data
            if is_binary:
                return self._parse_binary_ply(
                    f, vertex_count, has_colors, has_alpha
                )
            else:
                return self._parse_ascii_ply(
                    f, vertex_count, has_colors
                )
    
    def _parse_binary_ply(
        self,
        f,
        vertex_count: int,
        has_colors: bool,
        has_alpha: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse binary format PLY data"""
        data = f.read()
        
        if has_colors:
            if has_alpha:
                # Position (3 floats) + RGBA (4 bytes)
                dtype = np.dtype([
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')
                ])
            else:
                # Position (3 floats) + RGB (3 bytes)
                dtype = np.dtype([
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                ])
            
            data_array = np.frombuffer(data, dtype=dtype, count=vertex_count)
            points = np.column_stack([
                data_array['x'],
                data_array['y'],
                data_array['z']
            ])
            colors = np.column_stack([
                data_array['r'],
                data_array['g'],
                data_array['b']
            ]) / 255.0
        else:
            # Only position
            dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            data_array = np.frombuffer(data, dtype=dtype, count=vertex_count)
            points = np.column_stack([
                data_array['x'],
                data_array['y'],
                data_array['z']
            ])
            colors = None
        
        return points, colors
    
    def _parse_ascii_ply(
        self,
        f,
        vertex_count: int,
        has_colors: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse ASCII format PLY data"""
        points = []
        colors = []
        
        for _ in range(vertex_count):
            line = f.readline().decode('utf-8').strip().split()
            if len(line) >= 3:
                x, y, z = float(line[0]), float(line[1]), float(line[2])
                points.append([x, y, z])
                
                if has_colors and len(line) >= 6:
                    r, g, b = int(line[3]), int(line[4]), int(line[5])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
        
        points = np.array(points)
        colors = np.array(colors) if len(colors) > 0 else None
        
        return points, colors

