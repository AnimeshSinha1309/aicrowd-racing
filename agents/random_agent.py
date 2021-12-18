import json
import time

import matplotlib.pyplot as plt
import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, obs) -> np.array:
        return self.action_space.sample()

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

            for image in obs[1]:
                plt.imshow(image)
                plt.show()

            while not done:
                action = self.select_action(obs)
                obs, reward, done, info = env.step(action)

            # Update your agent
