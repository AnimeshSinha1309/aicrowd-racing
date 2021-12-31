import numpy as np

from l2r.baselines.reward import GranTurismo


class AnalyzerReward(GranTurismo):

    def __init__(
        self,
        *_args,
        **_kwargs,
    ):
        super(AnalyzerReward, self).__init__()

    def get_reward(self, state, oob_flag=False):
        pose_data, race_idx = state
        velocity = np.linalg.norm(pose_data[3:6])
        oob_reward = self._reward_oob(velocity, oob_flag)
        progress_reward = self._reward_progress(race_idx)
        return oob_reward + progress_reward

