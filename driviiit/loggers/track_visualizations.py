import numpy as np
from matplotlib import pyplot as plt

from driviiit.interface.config import (
    CAMERA_FRONT_POSITION,
    FIELD_OF_VIEW,
    IMAGE_SHAPE,
    SEGMENTATION_COLORS_MAP,
)
from sensors.camera_ground import (
    ground_points_to_camera,
    camera_points_to_ground,
    camera_points_to_car,
)
from driviiit.sensors.imu_readings import IMUSensorReading
from sensors.camera_utils import camera_details_to_intrinsic_matrix


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


def plot_camera_points_on_map(obs, env):
    x = ground_points_to_camera(
        env.reward.inner_track,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    mask = np.logical_and(np.logical_and(x[:, 0] >= 0, x[:, 1] >= 0), np.logical_and(x[:, 0] < 512, x[:, 1] < 384))
    x = x[mask]

    road_mask = np.all(np.equal(obs[1][1], SEGMENTATION_COLORS_MAP["ROAD"]), axis=2)
    road_y, road_x = np.where(road_mask)
    road_points = np.stack([road_x, 384 - road_y], axis=1)

    recovered_track_points = camera_points_to_ground(
        road_points,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    recovered_center_points = camera_points_to_ground(
        x,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    plt.scatter(env.reward.inner_track[:, 0], env.reward.inner_track[:, 1], s=0.1, color='green')
    plt.scatter(recovered_track_points[:, 0], recovered_track_points[:, 1], s=0.2, color='red')
    plt.scatter(recovered_center_points[:, 0], recovered_center_points[:, 1], s=0.2, color='orange')
    plt.show()
    return 0


def plot_local_camera_map(obs):
    road_mask = np.all(np.equal(obs[1][1], SEGMENTATION_COLORS_MAP["ROAD"]), axis=2)
    road_y, road_x = np.where(road_mask)
    road_points = np.stack([road_x, 384 - road_y], axis=1)

    recovered_track_points = camera_points_to_car(
        road_points,
        IMUSensorReading(obs[0]),
        CAMERA_FRONT_POSITION,
        camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
    )
    plt.scatter(recovered_track_points[:, 0], recovered_track_points[:, 1], s=0.2, color='red')
    plt.show()
    return 0
