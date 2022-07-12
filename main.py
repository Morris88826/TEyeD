import argparse
import glob
import numpy as np
import tqdm
from libs.processor import Processor
from libs.visualize import visualize
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="../dataset/Dikablis", help="The path to the TEyeD dataset.")
    args = parser.parse_args()
    
    videos_path = sorted(glob.glob(args.dataset_dir+'/VIDEOS/*.mp4'))

    for video_path in tqdm.tqdm(videos_path):
        processor = Processor(video_path, data_dir=args.dataset_dir)