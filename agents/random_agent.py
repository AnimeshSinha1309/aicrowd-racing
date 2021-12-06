import json
import time
import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, obs) -> np.array:
        return self.action_space.sample()

    def training(self, env):
        info = {}
        done = False
        obs = env.reset()

        for _ in range(300):
            action = self.select_action(obs)
            obs, reward, done, info = env.step(action)

            if done:
                obs = env.reset()
                done = False

            ## Update your agent
