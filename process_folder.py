from constructor import Constructor
import argparse
from pathlib import Path

file_types = ["MOV", "mp4", "MP4"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    top_level_folder = Path(args.folder)
    output_folder = Path(args.output_folder)
    print(f"Processing {top_level_folder}")
    for file_type in file_types:
        for file in top_level_folder.glob(f"*.{file_type}"):
            constructor = Constructor(file, output_folder)
            constructor.preprocess()
            print(f"Processed {file}")    
    # constructor = Constructor(args.folder