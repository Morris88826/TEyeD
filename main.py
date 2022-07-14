import argparse
import glob
import numpy as np
import tqdm
from libs.processor import Processor
from libs.visualize import visualize
from PIL import Image
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="../dataset/Dikablis", help="The path to the TEyeD dataset.")
    args = parser.parse_args()
    
    videos_path = sorted(glob.glob(args.dataset_dir+'/VIDEOS/*.mp4'))

    broken = ["DikablisSS_10_1", "DikablisT_11_3"] # DikablisSS_10_1/iris_lm_2d encoding error

    for video_path in tqdm.tqdm(videos_path):

        output_path = video_path.replace('VIDEOS', 'processed').replace('.mp4', '')
        
        
        if not os.path.exists(output_path) and output_path.split('/')[-1].split('.')[0] not in broken:
            processor = Processor(video_path, data_dir=args.dataset_dir, target_width=320, target_height=240)