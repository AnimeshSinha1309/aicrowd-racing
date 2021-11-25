from abc import ABC


class BaseAgent(ABC):
    def compute_action(self, state):
        pass

    def register_reset(self, state):
        pass

    def pre_evaluate(self, env):
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass
