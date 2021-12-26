import numpy as np


class RoadDistanceSensors:

    SEGMENTATION_IMAGE_COLORS = {
        "GROUND": np.array([13, 255, 13]),
        "SKY": np.array([13, 255, 255]),
        "CAR": np.array([65, 16, 149]),
        "ROAD": np.array([204, 80, 109]),
    }

    def __init__(self, image):
        road_mask = np.all(
            np.equal(image, self.SEGMENTATION_IMAGE_COLORS["ROAD"]), axis=2
        )
        car_mask = np.all(
            np.equal(image, self.SEGMENTATION_IMAGE_COLORS["CAR"]), axis=2
        )
