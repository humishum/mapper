#!/usr/bin/env python3
"""
Example usage of the Simple Pointcloud Visualizer
"""

from pointcloud_visualizer import SimplePointcloudVisualizer, THRESHOLD
import webbrowser
from pathlib import Path

def main():
    # Configuration
    data_dir = "../data/output_100425"  # Adjust path as needed
    output_file = "my_pointcloud_map.html"
    
    # You can change the global threshold here
    # THRESHOLD = 1.5  # Uncomment to use a different threshold
    
    print(f"Creating pointcloud visualizer with threshold: {THRESHOLD}")
    print(f"Data directory: {data_dir}")
    
    # Create visualizer
    visualizer = SimplePointcloudVisualizer(data_dir)
    
    # Generate the HTML file
    output_path = visualizer.generate_html(output_file)
    
    print(f"\nVisualizer created successfully!")
    print(f"Output file: {output_path}")
    print(f"Open this file in your web browser to view the 3D point clouds.")
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{output_path.absolute()}")
        print("Opening in browser...")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please open the HTML file manually in your browser.")

if __name__ == "__main__":
    main()
