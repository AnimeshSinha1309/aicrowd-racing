import logging

import numpy as np

from driviiit.interface.metas import BaseAgent
from driviiit.loggers.tensor_logger import TensorLogger
from driviiit.sensors.imu import IMUSensorReading
from driviiit.camera.ground_transform import CameraGroundTransformer
from driviiit.interface.config import FIELD_OF_VIEW, IMAGE_SHAPE, CAMERA_FRONT_POSITION
from l2r.envs.env import RacingEnv


class DriverAgent(BaseAgent):
    def __init__(self, log_data=True):
        super().__init__()

        self.loggers = [
            TensorLogger(name="imu_otx_0001"),
            TensorLogger(name="camera_front_0001"),
            TensorLogger(name="segm_front_0001"),
            TensorLogger(name="camera_left_0001"),
            TensorLogger(name="segm_left_0001"),
            TensorLogger(name="camera_right_0001"),
            TensorLogger(name="segm_right_0001"),
            TensorLogger(name="camera_birds_0001"),
            TensorLogger(name="segm_birds_0001"),
        ] if log_data else []
        self.cg = CameraGroundTransformer(
            FIELD_OF_VIEW,
            IMAGE_SHAPE,
            CAMERA_FRONT_POSITION
        )

    def select_action(self, obs) -> np.array:
        imu = IMUSensorReading(obs[0])

        road_mask = np.all(np.equal(obs[1][1], np.array([204, 80, 109])), axis=2)
        road_points = self.cg.mask_camera_to_ground(road_mask)
        road_shift = np.mean(road_points[road_points[:, 0] > -2.2, 1]) + 1.1

        steering_angle = -np.clip(road_shift / 1.2, -1.0, 1.0)
        acceleration = 1.0 if imu.speed < 40 else (-0.2 if imu.speed > 50 else 0.0)
        return np.array([steering_angle, acceleration])

    def register_reset(self, obs) -> np.array:
        pass

    def training(self, env: RacingEnv):
        """Train your agent here."""
        for _ in range(1):
            done = False
            obs, _ = env.reset()

            while not done:
                self.loggers[0].log(obs[0])
                for i in range(len(self.loggers)):
                    self.loggers[i + 1].log(obs[1][i])
                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)
