from loguru import logger
import timeout_decorator

from envs.env import RacingEnv
from collections import defaultdict
import os
from common.utils import setup_logging

from config import SubmissionConfig, EnvConfig, SimulatorConfig, SACConfig


class Learn2RaceEvaluator:
    """Evaluator class which consists of a 1-hour pre-evaluation phase followed by an evaluation phase."""

    def __init__(
        self,
        submission_config: SubmissionConfig,
        env_config: EnvConfig,
        sim_config: SimulatorConfig,
        sac_config: SACConfig
    ):
        logger.info("Starting learn to race evaluator")
        self.submission_config = submission_config
        self.env_config = env_config
        self.sim_config = sim_config
        self.sac_config = sac_config

        self.agent = None
        self.env = None
        self.metrics = defaultdict(list)

    def init_agent(self):
        """ """
        if self.agent is not None:
            return

        save_path = self.sac_config['save_path']
        if not os.path.exists(f'{save_path}/runlogs'):
            os.umask(0)
            os.makedirs(save_path, mode=0o777, exist_ok=True)
            os.makedirs(f"{save_path}/runlogs", mode=0o777, exist_ok=True)
            os.makedirs(f"{save_path}/tblogs", mode=0o777, exist_ok=True)

        loggers = setup_logging(save_path, self.sac_config['experiment_name'], True) 

        loggers[0]('Using random seed: {}'.format(0))
        self.agent = self.submission_config.agent(self.env, self.sac_config, loggers=loggers) ###

    def load_agent_model(self, path):
        self.agent.load_model(path)

    def save_agent_model(self, path):
        self.agent.save_model(path)

    @timeout_decorator.timeout(1 * 60 * 60)
    def train(self):
        logger.info("Starting one-hour 'practice' phase")
        self.agent.sac_train()

    def evaluate(self):
        """Evaluate the episodes."""
        logger.info("Starting evaluation")
        # self.env.eval()

        for ep in range(self.submission_config.eval_episodes):
            done = False
            state = self.env.reset()
            info = {}
            action = self.agent.register_reset(state)
            while not done:
                state, reward, done, info = self.env.step(action)
                action = self.agent.select_action(state)
            self._record_metrics(ep, info["metrics"])

    def _record_metrics(self, episode, metrics):
        """Do not modify."""
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

    def create_env(self, eval_track):
        """Do not modify.

        Your configuration yaml file must contain the keys below.
        """
        self.env = RacingEnv(
            max_timesteps=self.env_config.max_timesteps,
            obs_delay=self.env_config.obs_delay,
            not_moving_timeout=self.env_config.not_moving_timeout,
            controller_kwargs=self.env_config.controller_kwargs,
            reward_pol=self.env_config.reward_pol,
            reward_kwargs=self.env_config.reward_kwargs,
            action_if_kwargs=self.env_config.action_if_kwargs,
            pose_if_kwargs=self.env_config.pose_if_kwargs,
            camera_if_kwargs=self.env_config.camera_if_kwargs,
            sensors=self.sim_config.active_sensors
        )

        self.env.make(
            level=eval_track,
            multimodal=self.env_config.multimodal,
            driver_params=self.sim_config.driver_params,
            camera_params=self.sim_config.camera_params,
            sensors=self.sim_config.active_sensors
        )
