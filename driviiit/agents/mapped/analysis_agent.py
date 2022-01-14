import typing as ty

import numpy as np
import cv2 as cv

from driviiit.interface.metas import BaseAgent
from driviiit.loggers.tensor_logger import TensorLogger
from driviiit.sensors.imu_readings import IMUSensorReading
from driviiit.agents.mapped.steering_calibration import SteeringCalibration

if ty.TYPE_CHECKING:
    from l2r.envs.env import RacingEnv


class DriverAgent(BaseAgent):
    def __init__(self, perform_logging=False, use_ground_truth=False):
        super().__init__()
        self.perform_logging = perform_logging
        self.use_ground_truth = use_ground_truth

        if perform_logging:
            self.image_logger = TensorLogger(
                name="trajectory_0001", fields=("camera_front", "segm_front", "imu")
            )
        self.steering_controller: ty.Optional[SteeringCalibration] = None

    def select_action(self, obs) -> np.array:
        current_speed = (
            obs[0] if isinstance(obs[0], int) else IMUSensorReading(obs[0]).speed
        )
        steering_angle = 1.0
        target_speed = 20 - 10 * np.abs(steering_angle)
        acceleration = np.tanh(target_speed - current_speed)
        return np.array([steering_angle, acceleration])

    def select_action_when_training(self, obs, env) -> np.array:
        current_speed = (
            obs[0] if isinstance(obs[0], int) else IMUSensorReading(obs[0]).speed
        )
        steering_angle = self.steering_controller.steering_angle(
            env.reward.current_segment,
            imu=IMUSensorReading(obs[0])
        )
        target_speed = 20 - 10 * np.abs(steering_angle)
        acceleration = np.tanh(target_speed - current_speed)
        return np.array([steering_angle, acceleration])

    def register_reset(self, obs) -> np.array:
        pass

    def training(self, env: "RacingEnv"):
        """Train your agent here."""
        self.steering_controller = SteeringCalibration(
            env.reward.centre_path, env.reward.inner_track, env.reward.outside_track
        )
        for _ in range(10):
            done = False
            obs, _ = env.reset()

            while not done:
                if self.perform_logging:
                    self.image_logger.log(
                        camera_front=obs[1][0], segm_front=obs[1][1], imu=obs[0]
                    )
                action = self.select_action_when_training(obs, env)
                obs, reward, done, info = env.step(action)

            if self.perform_logging:
                self.image_logger.save()
