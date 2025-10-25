#!/usr/bin/env python3
"""
Export pointcloud data to JSON for frontend visualization
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from .data_loader import DataLoader
from .pointcloud_processor import PointcloudProcessor
from .gps_converter import GPSConverter


class DataExporter:
    """Export pointcloud data to JSON format for frontend"""
    
    def __init__(
        self, 
        data_dir: str,
        max_points: int = 50000,
        sequence_id: int = 1
    ):
        self.data_loader = DataLoader(data_dir)
        self.processor = PointcloudProcessor(max_points=max_points)
        self.gps_converter = GPSConverter()
        self.sequence_id = sequence_id
    
    def export_to_json(
        self, 
        output_file: str,
        threshold: float = 2.0
    ) -> Path:
        """
        Export all pointcloud data to JSON file
        
        Args:
            output_file: Output JSON file path
            threshold: Threshold value for pointcloud selection
            
        Returns:
            Path to output file
        """
        print(f"Loading metadata from {self.data_loader.data_dir}...")
        metadata_list = self.data_loader.load_all_metadata()
        print(f"Found {len(metadata_list)} locations with GPS data")
        
        locations = []
        
        for metadata in metadata_list:
            folder_name = metadata['folder_name']
            gps_coords = metadata['gps_coords']
            altitude = metadata['altitude']
            
            print(f"\nProcessing {folder_name}...")
            
            # Convert GPS to Cartesian
            x, y, z = self.gps_converter.gps_to_cartesian(
                gps_coords[0], gps_coords[1], altitude
            )
            
            location_data = {
                'id': folder_name,
                'name': folder_name,
                'gps': {
                    'lat': gps_coords[0],
                    'lon': gps_coords[1],
                    'alt': altitude
                },
                'position': {'x': x, 'y': y, 'z': z},
                'metadata': {
                    'frames': metadata.get('frames', 0),
                    'video': metadata.get('video_name', 'Unknown'),
                    'fps': metadata.get('fps', 0)
                }
            }
            
            # Load pointcloud data
            pointcloud_file = self.data_loader.find_pointcloud_file(
                folder_name, threshold, self.sequence_id
            )
            
            if pointcloud_file:
                print(f"  Loading pointcloud from {pointcloud_file.name}...")
                pc_data = self.processor.load_ply(pointcloud_file)
                
                if pc_data:
                    location_data['pointcloud'] = {
                        'points': pc_data['points'],
                        'colors': pc_data['colors'],
                        'center': pc_data['center'],
                        'bbox_min': pc_data['bbox_min'],
                        'bbox_max': pc_data['bbox_max'],
                        'original_count': pc_data['original_count'],
                        'display_count': pc_data['display_count'],
                        'file_name': pointcloud_file.name
                    }
                    print(f"  Loaded {pc_data['display_count']} points "
                          f"(from {pc_data['original_count']} original)")
                else:
                    print(f"  Failed to load pointcloud")
            else:
                print(f"  No pointcloud file found for threshold {threshold}")
            
            locations.append(location_data)
        
        # Write to JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'metadata': {
                'version': '1.0',
                'threshold': threshold,
                'sequence_id': self.sequence_id,
                'max_points_per_cloud': self.processor.max_points,
                'total_locations': len(locations)
            },
            'locations': locations
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nExported data to {output_path}")
        print(f"Total size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Export pointcloud data to JSON for visualization'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing pointcloud data folders'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='frontend/public/data.json',
        help='Output JSON file path (default: frontend/public/data.json)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='Threshold for pointcloud selection (default: 2.0)'
    )
    parser.add_argument(
        '--max-points',
        type=int,
        default=50000,
        help='Maximum points per pointcloud (default: 50000)'
    )
    parser.add_argument(
        '--sequence-id',
        type=int,
        default=1,
        help='Sequence ID to use (default: 1)'
    )
    
    args = parser.parse_args()
    
    exporter = DataExporter(
        data_dir=args.data_dir,
        max_points=args.max_points,
        sequence_id=args.sequence_id
    )
    
    exporter.export_to_json(
        output_file=args.output,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()

