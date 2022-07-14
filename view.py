import json
from libs.visualize import visualize
from PIL import Image

if __name__ == "__main__":

    data_root = "../dataset/Dikablis/processed"
    subject_name = "DikablisSA_10_2"
    image_idx = 81

    image_path = data_root + '/{}/image/{:07d}.png'.format(subject_name, image_idx)
    info_path = data_root + '/{}/info.json'.format(subject_name)
    image = Image.open(image_path)

    config = {
        "iris_eli": False,
        "iris_lm_2D": True, 
        "lid_lm_2D": True,
        "pupil_eli": False,
        "pupil_in_iris_eli": True,
        "pupil_lm_2D": True
    }

    with open(info_path, 'r') as jsonfile:
        all_info = json.load(jsonfile)
    info = all_info["{}".format(image_idx)]

    visualize(image, info, config)