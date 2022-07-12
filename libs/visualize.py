import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def visualize(image, info, config):

    plt.figure()
    ax = plt.gca()
    ax.imshow(image)
    
    if info["validity_iris"]:
        if config["iris_eli"]:
            center_x, center_y = info["iris_eli"]["center"]
            width = info["iris_eli"]["width"]
            height = info["iris_eli"]["height"]
            angle = info["iris_eli"]["angle"]
            ellipse = Ellipse(xy=[center_x, center_y], width=width, height=height, angle=angle, color='red', alpha=0.5)
            ax.add_patch(ellipse)

        if config["iris_lm_2D"]:
            landmarks = np.array(info["iris_lm_2D"]["landmarks"])
            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='yellow')

    if info["validity_iris"]:
        if config["pupil_eli"]:
            center_x, center_y = info["pupil_eli"]["center"]
            width = info["pupil_eli"]["width"]
            height = info["pupil_eli"]["height"]
            angle = info["pupil_eli"]["angle"]
            ellipse = Ellipse(xy=[center_x, center_y], width=width, height=height, angle=angle, color="blue", alpha=0.5)
            ax.add_patch(ellipse)
            ax.scatter(center_x, center_y, c='green')

        if config["pupil_lm_2D"]:
            landmarks = np.array(info["pupil_lm_2D"]["landmarks"])
            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='pink')

    if info["validity_lid"]:
        if config["lid_lm_2D"]:
            landmarks = np.array(info["lid_lm_2D"]["landmarks"])
            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='purple')
    
    plt.show()
