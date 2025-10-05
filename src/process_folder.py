from constructor import Constructor
import argparse
from pathlib import Path

file_types = ["MOV", "mp4", "MP4"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    
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
            print(f"Running Constructor pipeline for {file} ")
            constructor = Constructor(file, output_folder)
            constructor.run(weights_path, retrieval_path, image_size)
            print(f"Completed Constructor pipeline for {file} ")
