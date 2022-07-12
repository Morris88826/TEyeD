import numpy as np
from libs.processor import Processor
from libs.visualize import visualize
from PIL import Image

def process(video_path, data_dir="./sample_data"):
    processor = Processor(video_path, data_dir=data_dir, process_video=False)
    info, images_path = processor.get_data()
    return info, images_path

if __name__ == "__main__":
    
    video_path = "./sample_data/video/dikablisR_1_1.mp4"
    idx = 0

    config = {
        "eye_ball": True,
        "gaze_vec": True, 
        "iris_eli": True,
        "iris_lm_2D": True, 
        "lid_lm_2D": True,
        "pupil_eli": True,
        "pupil_in_iris_eli": True,
        "pupil_lm_2D": True
    }

    
    all_info, images_path = process(video_path)
    
    image = np.array(Image.open(images_path[idx]))
    info = all_info["{}".format(idx+1)]

    visualize(image, info, config)
