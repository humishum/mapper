# this will be the main top runner function for a single video file 
# can run an instance of this for each video, 
# input: video 
# output pointcloud and metadata 

# make a call to sfm pipeline her

from preprocessor import Preprocessor
from aligner import Aligner
from pathlib import Path 
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Constructor:
    def __init__(self, input_video_path:Path, output_path:Path):
        self.input_video_path = input_video_path
        self.output_path = output_path

    def preprocess(self):
        # Preprocess video into individual frames and metadata 
        preprocessor = Preprocessor(self.input_video_path, self.output_path)
        preprocessor.process()


    def align(self): 
        alinged_output_path = self.output_path/ "aligned"
        aligner = Aligner(self.output_path, alinged_output_path)
        aligner.align() 
        




def main():
    parser = argparse.ArgumentParser(description="Construct pointcloud from video")

    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    input_video_path = Path(args.input_video_path)
    output_path = Path(args.output_path)
    logger.info(f"Input video path: {input_video_path}")
    logger.info(f"Output path: {output_path}")

    # Preprocess video into individual frames and metadata 
    constructor = Constructor(input_video_path, output_path)
    constructor.preprocess()
      


if __name__ == "__main__":
    main()