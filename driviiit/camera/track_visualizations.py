from matplotlib import pyplot as plt

from driviiit.interface.config import CAMERA_FRONT_POSITION, FIELD_OF_VIEW, IMAGE_SHAPE
from driviiit.sensors.imu import IMUSensorReading
from driviiit.camera.camera_utils import camera_details_to_intrinsic_matrix
from driviiit.camera.ground_transform import ground_points_to_camera


def plot_track_boundaries_on_camera(obs, env):
    x = ground_points_to_camera(
        env.reward.inner_track,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    y = ground_points_to_camera(
        env.reward.centre_path,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    z = ground_points_to_camera(
        env.reward.outside_track,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    plt.scatter(x[:, 0], 384 - x[:, 1], s=1, color='purple')
    plt.scatter(y[:, 0], 384 - y[:, 1], s=1, color='yellow')
    plt.scatter(z[:, 0], 384 - z[:, 1], s=1, color='black')
    plt.imshow(obs[1][1])
    plt.xlim(0, 512)
    plt.ylim(384, 0)
    plt.show()
    print("OBSERVATION POINT:", IMUSensorReading(obs[0]).position)
    print("COMPLETING THIS ITERATION")
