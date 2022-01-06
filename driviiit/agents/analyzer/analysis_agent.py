import typing as ty

import numpy as np
import torch

from driviiit.interface.metas import BaseAgent
from driviiit.loggers.tensor_logger import TensorLogger
from driviiit.sensors.imu_readings import IMUSensorReading
from driviiit.interface.config import (
    SEGMENTATION_COLORS_MAP,
    IMAGE_SHAPE,
    FIELD_OF_VIEW,
    CAMERA_FRONT_POSITION,
    DEVICE,
)
from driviiit.sensors.camera_ground import (
    camera_points_to_car,
    camera_details_to_intrinsic_matrix,
)
from driviiit.models.segment import LiveSegmentationTrainer

if ty.TYPE_CHECKING:
    from l2r.envs.env import RacingEnv


class DriverAgent(BaseAgent):
    def __init__(self, perform_logging=False, use_ground_truth=False):
        super().__init__()
        self.perform_logging = perform_logging
        self.use_ground_truth = use_ground_truth

        if perform_logging:
            self.image_logger = TensorLogger(name="trajectory_0001", fields=('camera_front', 'segm_front', 'imu'))
        self.segmentation_model = LiveSegmentationTrainer(load=True)

    def select_action(self, obs) -> np.array:
        current_speed = obs[0] if isinstance(obs[0], int) else IMUSensorReading(obs[0]).speed

        if self.use_ground_truth:
            road_mask = np.all(np.equal(obs[1][1], SEGMENTATION_COLORS_MAP["ROAD"]), axis=2)
        else:
            tensor = torch.from_numpy(obs[1][0].transpose(2, 0, 1) / 255.).unsqueeze(0).float().to(DEVICE)
            road_mask = self.segmentation_model.model(tensor).squeeze().detach().cpu().numpy()

        road_y, road_x = np.where(road_mask > 0.5)
        road_points = np.stack([road_x, 384 - road_y], axis=1)

        recovered_track_points = camera_points_to_car(
            road_points,
            CAMERA_FRONT_POSITION,
            camera_details_to_intrinsic_matrix(FIELD_OF_VIEW, IMAGE_SHAPE)
        )
        nearby_points = recovered_track_points[recovered_track_points[:, 1] < 15]
        position_mean = np.mean(nearby_points[:, 0]) / 4

        steering_angle = np.clip(position_mean, -1.0, 1.0)
        target_speed = 20 - 10 * np.abs(steering_angle)
        acceleration = np.tanh(target_speed - current_speed)
        return np.array([steering_angle, acceleration])

    def register_reset(self, obs) -> np.array:
        pass

    def training(self, env: "RacingEnv"):
        """Train your agent here."""
        for _ in range(10):
            done = False
            obs, _ = env.reset()

            while not done:
                if self.perform_logging:
                    self.image_logger.log(camera_front=obs[1][0], segm_front=obs[1][1], imu=obs[0])
                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)

            if self.perform_logging:
                self.image_logger.save()
