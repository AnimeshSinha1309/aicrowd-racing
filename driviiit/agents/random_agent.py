import matplotlib.pyplot as plt
import numpy as np

from driviiit.interface.metas import BaseAgent
from driviiit.loggers.tensor_logger import TensorLogger
from driviiit.sensors.imu import IMUSensorReading


class DriverAgent(BaseAgent):
    def __init__(self):
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
        ]

    def select_action(self, obs) -> np.array:
        imu = IMUSensorReading(obs[0])

        steering_angle = 0
        acceleration = 1.0 if imu.speed < 40 else (-0.2 if imu.speed > 50 else 0.0)
        return np.array([steering_angle, acceleration])

    def register_reset(self, obs) -> np.array:
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def training(self, env):
        """Train your agent here."""
        for _ in range(1):
            done = False
            obs, _ = env.reset()

            while not done:
                self.loggers[0].log(obs[0])
                for i in range(len(obs[1])):
                    self.loggers[i + 1].log(obs[1][i])
                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)
