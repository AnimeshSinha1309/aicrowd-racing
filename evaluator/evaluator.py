from loguru import logger
import timeout_decorator

from envs.env import RacingEnv

from config import SubmissionConfig, EnvConfig, SimulatorConfig


class Learn2RaceEvaluator:
    """Evaluator class which consists of a 1-hour pre-evaluation phase followed by an evaluation phase."""

    def __init__(
        self,
        submission_config: SubmissionConfig,
        env_config: EnvConfig,
        sim_config: SimulatorConfig,
    ):
        logger.info("Starting learn to race evaluator")
        self.submission_config = submission_config
        self.env_config = env_config
        self.sim_config = sim_config

        self.agent = None
        self.env = None
        self.metrics = {}

    def init_agent(self):
        """ """
        if self.agent is not None:
            return
        self.agent = self.submission_config.agent()

    def load_agent_model(self, path):
        self.agent.load_model(path)

    def save_agent_model(self, path):
        self.agent.save_model(path)

    @timeout_decorator.timeout(1 * 60 * 60)
    def pre_evaluate(self):
        logger.info("Starting pre-evaluation phase")
        self.agent.pre_evaluate(self.env)

    def evaluate(self):
        """Evaluate the episodes."""
        logger.info("Starting evaluation")
        self.env.eval()

        for ep in range(self.submission_config.eval_episodes):
            done = False
            state = self.env.reset()
            info = {}
            action = self.agent.register_reset(state)
            while not done:
                state, reward, done, info = self.env.step(action)
                action = self.agent.compute_action(state)
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
            cameras=self.env_config.cameras,
        )

        self.env.make(
            level=eval_track,
            multimodal=self.env_config.multimodal,
            driver_params=self.sim_config.driver_params,
        )
