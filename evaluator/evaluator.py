from typing import Type

from loguru import logger
import timeout_decorator

from l2r.envs.env import RacingEnv
from collections import defaultdict

from config import SubmissionConfig, EnvConfig, SimulatorConfig


class Learn2RaceEvaluator:
    """Evaluator class which consists of a 1-hour pre-evaluation phase followed by an evaluation phase."""

    def __init__(
        self,
        submission_config: Type[SubmissionConfig],
        env_config: Type[EnvConfig],
        sim_config: Type[SimulatorConfig],
    ):
        logger.info("Starting learn to race evaluator")
        self.submission_config = submission_config
        self.env_config = env_config
        self.sim_config = sim_config

        self.agent = None
        self.env = None
        self.metrics = defaultdict(list)

    def init_agent(self):
        """ """
        self.agent = self.submission_config.agent()

    def load_agent_model(self, path):
        self.agent.load_model(path)

    def save_agent_model(self, path):
        self.agent.save_model(path)

    @timeout_decorator.timeout(1 * 60 * 60)
    def train(self):
        logger.info("Starting one-hour 'practice' phase")
        self.agent.training(self.env)

    def evaluate(self):
        """Evaluate the episodes."""
        logger.info("Starting evaluation")

        for ep in range(self.submission_config.eval_episodes):
            state, _ = self.env.reset()
            self.agent.register_reset(state)
            done = False
            info = {}
            while not done:
                action = self.agent.select_action(state)
                state, reward, done, info = self.env.step(action)
            self._record_metrics(ep, info["metrics"])

    def _record_metrics(self, episode, metrics):
        logger.info(
            f"Completed evaluation episode {episode + 1} with metrics: {metrics}"
        )
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key] = [value]
            else:
                self.metrics[key].append(value)

    def get_average_metrics(self):
        avg_metrics = {}
        for key, values in self.metrics.items():
            avg_metrics[key] = sum(values) / len(values)
        return avg_metrics

    def create_env(self):
        """Your configuration yaml file must contain the keys below.
        """
        self.env = RacingEnv(self.env_config.__dict__, self.sim_config.__dict__)
        self.env.make()
