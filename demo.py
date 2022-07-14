import json
import numpy as np
from libs.processor import Processor
from libs.visualize import visualize
from PIL import Image

def process(video_path, data_dir="./sample_data"):
    processor = Processor(video_path, data_dir=data_dir, process_video=True, target_width=320, target_height=240)
    info, images_path = processor.get_data()
    return info, images_path

if __name__ == "__main__":
    
    video_path = "./sample_data/video/dikablisR_1_1.mp4"
    idx = 289

    config = {
        "iris_eli": True,
        "iris_lm_2D": True, 
        "lid_lm_2D": True,
        "pupil_eli": True,
        "pupil_in_iris_eli": True,
        "pupil_lm_2D": True
    }

    
    all_info, images_path = process(video_path)
    
    image = np.array(Image.open("./sample_data/processed/dikablisR_1_1/image/{:07d}.png".format(idx)))
    info = all_info["{}".format(idx)]

    visualize(image, info, config)
