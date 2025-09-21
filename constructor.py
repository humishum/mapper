# this will be the main top runner function. 
# input: video 
# output pointcloud and metadata 

# make a call to sfm pipeline her

from preprocessor import Preprocessor

from pathlib import Path 
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description="Construct pointcloud from video")
    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    input_video_path = Path(args.input_video_path)
    output_path = Path(args.output_path)
    logger.info(f"Input video path: {input_video_path}")
    logger.info(f"Output path: {output_path}")
    preprocessor = Preprocessor(input_video_path, output_path)
    preprocessor.process()
      





if __name__ == "__main__":
    main()