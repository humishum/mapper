from constructor import Constructor
import argparse
from pathlib import Path
import json

file_types = ["MOV", "mp4", "MP4"]


def is_already_processed(video_file: Path, output_folder: Path) -> bool:
    """Check if a video has already been processed by looking for completed flag in metadata.json."""
    video_name = video_file.name.split(".")[0]
    video_output_dir = output_folder / video_name
    metadata_path = video_output_dir / "metadata.json"
    
    if not metadata_path.exists():
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get("completed", False)
    except (json.JSONDecodeError, IOError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="Force reprocessing of already processed videos")
    
    args = parser.parse_args()

    top_level_folder = Path(args.folder)
    output_folder = Path(args.output_folder)
    weights_path = Path(args.weights_path)
    retrieval_path = Path(args.retrieval_path)
    image_size = args.image_size
    print(f"Processing videos from  {top_level_folder}")
    for file_type in file_types:
        video_files = sorted(top_level_folder.glob(f"*.{file_type}"))
        print(f"Found {len(video_files)} {file_type} video files")
        print(video_files)
        for file in video_files:
            if not args.force and is_already_processed(file, output_folder):
                print(f"Skipping {file} - already processed (use --force to reprocess)")
                continue
            print(f"Running Constructor pipeline for {file} ")
            constructor = Constructor(file, output_folder)
            constructor.run(weights_path, retrieval_path, image_size)
            print(f"Completed Constructor pipeline for {file} ")
