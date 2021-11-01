from abc import ABC


class BaseAgent(ABC):
    def compute_action(self, state):
        pass

    def pre_evaluate(self, env):
        pass
