import numpy as np

from driviiit.interface.vectors import CoordinateTransform


CAMERA_FRONT_POSITION = CoordinateTransform(
    x=0, y=0, z=1.05, pitch=0.0, roll=0.0, yaw=0.0
)
FIELD_OF_VIEW = np.pi / 2
IMAGE_SHAPE = (512, 384)
