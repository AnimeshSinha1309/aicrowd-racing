import json
import time
import numpy as np
from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        model_path = "models/random_agent_bounds.json"
        self.model = None
        self.load_model(model_path)

    def compute_action(self, state):
        return np.random.random(self.model["dimensions"])

    def register_reset(self, state):
        pass

    def pre_evaluate(self, env):
        time.sleep(60*5)

    def load_model(self, path):
        with open(path) as fp:
            self.model = json.load(fp)

    def save_model(self, path):
        with open(path, "w") as fp:
            json.dump({"dimensions": 2}, fp)
