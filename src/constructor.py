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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_LIMIT = 100 

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
        if len([image for image in self.image_dir_path.glob("*.jpg")]) >  IMAGE_LIMIT:
            logger.info(f"Skipping 3D reconstruction for {self.input_video_path} because it has more than {IMAGE_LIMIT} images")
            return
        self._run_3d_reconstruction(weights_path, retrieval_path, image_size)
        # self._align()

    def _preprocess(self):
        # Preprocess video into individual frames and metadata 
        preprocessor = Preprocessor(self.input_video_path, self.image_dir_path)
        preprocessor.process()

    def _run_3d_reconstruction(self, weights_path:Path, retrieval_path:Path, image_size:int):
        must3r_wrapper = MuSt3RWrapper(
            image_dir=self.image_dir_path, 
            output_dir=self.pointcloud_dir_path, 
            weights_path=weights_path, 
            retrieval_path=retrieval_path, 
            image_size=image_size
        )
        must3r_wrapper.load_model()
        must3r_wrapper.run()

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