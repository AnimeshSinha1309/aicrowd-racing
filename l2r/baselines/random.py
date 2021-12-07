# ========================================================================= #
# Filename:                                                                 #
#    random.py                                                              #
#                                                                           #
# Description:                                                              #
#    an agent that randomly chooses actions                                 #
# ========================================================================= #

from l2r.core.templates import AbstractAgent
from l2r.envs.env import RacingEnv


class RandomActionAgent(AbstractAgent):
    """Reinforcement learning agent that simply chooses random actions.

    :param dict training_kwargs: training keyword arguments
    """

    def __init__(self, training_kwargs):
        self.num_episodes = training_kwargs["num_episodes"]

    def race(self):
        """Demonstrative training method."""
        for e in range(self.num_episodes):
            print(f"Episode {e+1} of {self.num_episodes}")
            ep_reward = 0
            state, done = self.env.reset(), False

            while not done:
                action = self.select_action()
                state, reward, done, info = self.env.step(action)
                ep_reward += reward

            print(f"Completed episode with total reward: {ep_reward}")
            print(f"Episode info: {info}\n")

    def select_action(self):
        """Select a random action from the action space.

        :return: random action to take
        :rtype: numpy array
        """
        return self.env.action_space.sample()

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param dict env_kwargs: environment keyword arguments
        :param dict sim_kwargs: simulator setting keyword arguments
        """
        self.env = RacingEnv(env_kwargs, sim_kwargs)
        self.env.make()

        print("Environment created with observation space: ")
        for k, v in self.env.observation_space.spaces.items():
            print(f"\t{k}: {v}")
