# this will be the main top runner function for a single video file 
# can run an instance of this for each video, 
# input: video 
# output pointcloud and metadata 


from must3r_wrapper import MuSt3RWrapper
from preprocessor import Preprocessor
from aligner import Aligner
from pathlib import Path 
import argparse
import logging
import os
import shutil
import tempfile
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WINDOW_SIZE = 500
WINDOW_OVERLAP = 20 
assert WINDOW_SIZE > WINDOW_OVERLAP, "WINDOW_SIZE must be greater than WINDOW_OVERLAP"

class Constructor:
    def __init__(self, input_video_path:Path, output_path:Path):
        self.input_video_path = input_video_path
        self.output_path = output_path / self.input_video_path.name.split(".")[0]
        os.makedirs(self.output_path, exist_ok=True)
        self.image_dir_path = self.output_path / "images"
        os.makedirs(self.image_dir_path, exist_ok=True)
        self.pointcloud_dir_path = self.output_path / "pointclouds"
        os.makedirs(self.pointcloud_dir_path, exist_ok=True)

    def run(self, weights_path:Path, retrieval_path:Path, image_size:int):
        self._preprocess()
        self._run_3d_reconstruction(weights_path, retrieval_path, image_size)
        # self._align()
        self._mark_done()

    def _mark_done(self):
        """Update metadata.json with completed flag to indicate successful completion."""
        metadata_path = self.output_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["completed"] = True
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Marked as complete in: {metadata_path}")

    def _preprocess(self):
        # Preprocess video into individual frames and metadata 
        preprocessor = Preprocessor(self.input_video_path, self.image_dir_path)
        preprocessor()
      

    def _run_3d_reconstruction(self, weights_path:Path, retrieval_path:Path, image_size:int):

        images = sorted(list(self.image_dir_path.glob('*')))
        total_images = len(images)

        window_size = WINDOW_SIZE
        overlap = WINDOW_OVERLAP  

        # Calculate the start indices for each window, ensuring overlap
        window_starts = []
        idx = 0
        while idx < total_images:
            window_starts.append(idx)
            # Move the start index for the next window by (window_size - overlap)
            # This ensures 'overlap' images overlap between consecutive windows
            idx += window_size - overlap
            if idx <= 0:  # Safety check to avoid infinite loops if overlap >= window_size
                break
        
        logger.info(f"Total images found: {total_images} \n Splitting into total of {len(window_starts)} windows of size {window_size} with {overlap} overlap")

        # Load model once
        must3r_wrapper = MuSt3RWrapper(
            weights_path=weights_path, 
            retrieval_path=retrieval_path, 
            image_size=image_size
        )
        must3r_wrapper.load_model()

        for i, start_idx in enumerate(window_starts):
            end_idx = min(start_idx + window_size, total_images)
            window_images = images[start_idx:end_idx]

            with tempfile.TemporaryDirectory() as temp_img_dir:
                temp_img_dir_path = Path(temp_img_dir)

                # Copy images to temp directory
                for img_path in window_images:
                    shutil.copy(img_path, temp_img_dir_path / img_path.name)
                
                # Make window-specific pointcloud output directory
                sequence_dir_name = f"sequence_{i+1}"
                sequence_output_dir = self.pointcloud_dir_path / sequence_dir_name
                os.makedirs(sequence_output_dir, exist_ok=True)
                
                logger.info(f"Running 3D reconstruction for {sequence_dir_name} with {len(window_images)} images.")
                must3r_wrapper.run(temp_img_dir_path, sequence_output_dir)

    def _align(self): 
        alinged_output_path = self.output_path/ "aligned"
        aligner = Aligner(self.output_path, alinged_output_path)
        aligner.align() 
        

def main():
    parser = argparse.ArgumentParser(description="Construct pointcloud from video")

    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    args = parser.parse_args()
    input_video_path = Path(args.input_video_path)
    output_path = Path(args.output_path)
    logger.info(f"Input video path: {input_video_path}")
    logger.info(f"Output path: {output_path}")

    # Preprocess video into individual frames and metadata 
    constructor = Constructor(input_video_path, output_path)
    
    constructor.run(
        weights_path=Path(args.weights_path),
        retrieval_path=Path(args.retrieval_path),
        image_size=args.image_size
    )



if __name__ == "__main__":
    main()