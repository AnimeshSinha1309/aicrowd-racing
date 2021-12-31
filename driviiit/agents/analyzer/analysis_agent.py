import typing as ty

import numpy as np

from driviiit.interface.metas import BaseAgent
from driviiit.loggers.tensor_logger import TensorLogger
from driviiit.sensors.imu import IMUSensorReading
from driviiit.interface.config import (
    SEGMENTATION_COLORS_MAP,
    IMAGE_SHAPE,
    FIELD_OF_VIEW,
    CAMERA_FRONT_POSITION,
)
from driviiit.camera.ground_transform import (
    camera_points_to_car,
    camera_details_to_intrinsic_matrix,
)
from driviiit.camera.track_visualizations import (
    plot_track_boundaries_on_camera,
    plot_camera_points_on_map,
    plot_local_camera_map,
)

if ty.TYPE_CHECKING:
    from l2r.envs.env import RacingEnv


class DriverAgent(BaseAgent):
    def __init__(self, log_data=True):
        super().__init__()

        self.loggers = (
            [
                TensorLogger(name="imu_otx_0001"),
                TensorLogger(name="camera_front_0001"),
                TensorLogger(name="segm_front_0001"),
                TensorLogger(name="camera_left_0001"),
                TensorLogger(name="segm_left_0001"),
                TensorLogger(name="camera_right_0001"),
                TensorLogger(name="segm_right_0001"),
                TensorLogger(name="camera_birds_0001"),
                TensorLogger(name="segm_birds_0001"),
            ]
            if log_data
            else []
        )

    def select_action(self, obs) -> np.array:
        imu = IMUSensorReading(obs[0])

        road_mask = np.all(np.equal(obs[1][1], SEGMENTATION_COLORS_MAP["ROAD"]), axis=2)
        road_y, road_x = np.where(road_mask)
        road_points = np.stack([road_x, 384 - road_y], axis=1)

        recovered_track_points = camera_points_to_car(
            road_points,
            IMUSensorReading(obs[0]),
            CAMERA_FRONT_POSITION,
            camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
        )
        nearby_points = recovered_track_points[recovered_track_points[:, 1] < 10]
        position_mean = np.mean(nearby_points[:, 0]) / 5

        steering_angle = np.clip(position_mean, -1.0, 1.0)
        acceleration = 1.0 if imu.speed < 15 else (-0.2 if imu.speed > 20 else 0.0)
        return np.array([steering_angle, acceleration])

    def register_reset(self, obs) -> np.array:
        pass

    def training(self, env: "RacingEnv"):
        """Train your agent here."""
        for _ in range(10):
            done = False
            obs, _ = env.reset()

            while not done:
                self.loggers[0].log(obs[0])
                for i in range(len(self.loggers) - 1):
                    self.loggers[i + 1].log(obs[1][i])

                # plot_camera_points_on_map(obs, env)
                # plot_track_boundaries_on_camera(obs, env)
                # plot_local_camera_map(obs)

                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)
