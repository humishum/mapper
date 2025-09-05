import numpy as np
import subprocess
import json
import argparse
import os
from datetime import datetime, timedelta
import cv2


def extract_frames_ffmpeg(video_path, fps=None):
    """
    Extract frames from video using ffmpeg subprocess
    Returns list of tuples: (timestamp_seconds, frame_data)
    """
    frames_data = []
    
    # Create temporary directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Build ffmpeg command to extract frames
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}' if fps else 'fps=1',  # Default to 1 fps if not specified
            '-y',  # Overwrite output files
            f'{temp_dir}/frame_%06d.png'
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return frames_data
        
        # Read extracted frames
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                timestamp = i / (fps if fps else 1)  # Calculate timestamp based on fps
                frames_data.append((timestamp, frame))
            
            # Clean up frame file
            os.remove(frame_path)
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    return frames_data


def extract_gps_data(video_path):
    """
    Extract GPS data from video using ffprobe
    Returns dict with GPS coordinates and metadata
    """
    gps_data = {}
    
    try:
        # Use ffprobe to get metadata in JSON format
        cmd = [
            'ffprobe', '-i', video_path,
            '-print_format', 'json',
            '-show_format'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return gps_data
        
        # Parse JSON output
        metadata = json.loads(result.stdout)
        
        # Extract GPS data from format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            
            # Look for GPS coordinates in various tag formats
            gps_keys = ['location', 'com.apple.quicktime.location.ISO6709', 
                       'location-eng', 'GPS_COORDINATES']
            
            for key in gps_keys:
                if key in tags:
                    gps_data['coordinates'] = tags[key]
                    break
            
            # Extract creation time if available
            time_keys = ['creation_time', 'date']
            for key in time_keys:
                if key in tags:
                    gps_data['creation_time'] = tags[key]
                    break
        
        # Parse coordinates if in ISO6709 format (+DDMM.MMMM+DDDMM.MMMM/)
        if 'coordinates' in gps_data:
            coord_str = gps_data['coordinates']
            if coord_str.startswith('+') and '+' in coord_str[1:]:
                try:
                    # Simple parsing for ISO6709 format
                    parts = coord_str.replace('+', ' +').replace('-', ' -').split()
                    if len(parts) >= 2:
                        lat_str = parts[0].strip('+')
                        lon_str = parts[1].strip('+')
                        
                        gps_data['latitude'] = float(lat_str[:3]) + float(lat_str[3:]) / 60
                        gps_data['longitude'] = float(lon_str[:4]) + float(lon_str[4:]) / 60
                except (ValueError, IndexError):
                    pass
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"GPS extraction error: {e}")
    
    return gps_data


def detect_video_type(video_path):
    """
    Detect if video is GoPro format or regular video
    Returns 'gopro' or 'regular'
    """
    try:
        # Use ffprobe to get video metadata
        cmd = [
            'ffprobe', '-i', video_path,
            '-print_format', 'json',
            '-show_format', '-show_streams'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 'regular'  # Default to regular if probe fails
        
        metadata = json.loads(result.stdout)
        
        # Check for GoPro indicators in metadata
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            
            # Look for GoPro-specific tags
            gopro_indicators = ['com.apple.quicktime.make', 'encoder']
            for key in gopro_indicators:
                if key in tags and 'gopro' in tags[key].lower():
                    return 'gopro'
        
        # Check filename patterns
        filename = os.path.basename(video_path).lower()
        if filename.startswith(('gopr', 'gp', 'gh')) and filename.endswith(('.mp4', '.mov')):
            return 'gopro'
    
    except Exception:
        pass
    
    return 'regular'


def process_gopro_video(video_path, fps=None, output_dir=None):
    """
    Process GoPro video - extract frames and GPS data
    Returns dict with frames and GPS data
    """
    return process_video(video_path, fps, output_dir)


def process_video(video_path, fps=None, output_dir=None):
    """
    Process regular video - extract frames and GPS data
    Returns dict with timestamped frames and GPS data
    """
    print(f"Processing video: {video_path}")
    
    # Extract frames
    frames = extract_frames_ffmpeg(video_path, fps)
    print(f"Extracted {len(frames)} frames")
    
    # Extract GPS data
    gps_data = extract_gps_data(video_path)
    if gps_data:
        print(f"Found GPS data: {gps_data}")
    else:
        print("No GPS data found")
    
    # Prepare output
    result = {
        'video_path': video_path,
        'frames': frames,  # List of (timestamp, frame) tuples
        'gps_data': gps_data,
        'total_frames': len(frames),
        'fps_used': fps
    }
    
    # Save to output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save frames as images
        for i, (timestamp, frame) in enumerate(frames):
            frame_filename = f"frame_{i:06d}_t{timestamp:.2f}s.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
        
        # Save GPS data as JSON
        if gps_data:
            gps_filename = "gps_data.json"
            gps_path = os.path.join(output_dir, gps_filename)
            with open(gps_path, 'w') as f:
                json.dump(gps_data, f, indent=2)
        
        print(f"Output saved to: {output_dir}")
    
    return result


def main(video_path=None, fps=None, output_dir=None):
    """
    Main function to process video
    """
    if not video_path:
        print("No video path provided")
        return None
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    # Detect video type
    video_type = detect_video_type(video_path)
    print(f"Detected video type: {video_type}")
    
    # Process video based on type
    if video_type == 'gopro':
        result = process_gopro_video(video_path, fps, output_dir)
    else:
        result = process_video(video_path, fps, output_dir)
    
    return result


if __name__ == "__main__":
    # CLI argument parsing
    parser = argparse.ArgumentParser(description="Extract frames and GPS data from videos")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--output-dir", help="Output directory for frames and GPS data")
    
    args = parser.parse_args()
    
    # Run main function
    result = main(args.video, args.fps, args.output_dir)
    
    if result:
        print(f"\nProcessing complete!")
        print(f"Total frames: {result['total_frames']}")
        print(f"GPS data available: {'Yes' if result['gps_data'] else 'No'}")