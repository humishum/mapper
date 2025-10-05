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
    print(f"Processing {top_level_folder}")
    for file_type in file_types:
        for file in top_level_folder.glob(f"*.{file_type}"):
            constructor = Constructor(file, output_folder)
            constructor.preprocess()
            print(f"Processed {file}")  
            constructor.run_3d_reconstruction(weights_path, retrieval_path, image_size)
            # constructor.align()
            # print(f"Aligned {file}")
    # constructor = Constructor(args.folder