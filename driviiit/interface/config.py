import numpy as np
import torch

from driviiit.interface.vectors import CoordinateTransform


CAMERA_FRONT_POSITION = CoordinateTransform(
    x=0, y=0, z=1.05, pitch=0.0, roll=0.0, yaw=0.0
)
FIELD_OF_VIEW = np.pi / 2
IMAGE_SHAPE = (512, 384)

SEGMENTATION_COLORS_MAP = {
    "GROUND": np.array([13, 255, 13]),
    "SKY": np.array([13, 255, 255]),
    "CAR": np.array([65, 16, 149]),
    "ROAD": np.array([204, 80, 109]),
}

DEVICE = torch.device('cuda')
