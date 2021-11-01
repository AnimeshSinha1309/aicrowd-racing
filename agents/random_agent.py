import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        pass

    def compute_action(self, state):
        return np.random.random(2)

    def pre_evaluate(self, env):
        pass
