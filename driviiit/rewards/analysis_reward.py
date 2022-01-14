import numpy as np

from l2r.baselines.reward import GranTurismo


class AnalyzerReward(GranTurismo):
    def __init__(
        self,
        *_args,
        **_kwargs,
    ):
        super(AnalyzerReward, self).__init__()
        self.current_segment = 0

    def get_reward(self, state, oob_flag=False):
        pose_data, race_idx = state
        velocity = pose_data[3:6]
        self.current_segment = race_idx
        # Compute the reward based on speed along the track
        speed_along_road, speed_across_road = self._get_velocity_components(
            race_idx, velocity
        )
        speed_reward = speed_along_road
        # Check if we have gone out of bounds
        boundary_reward = -1000 if oob_flag else 0
        # Combine all the rewards and return
        return speed_reward + boundary_reward

    def _get_velocity_components(self, race_idx, velocity):
        velocity = velocity[:2]
        road_direction = (
            self.centre_path[(race_idx + 1) % len(self.centre_path)]
            - self.centre_path[race_idx]
        )
        road_direction = road_direction / np.linalg.norm(road_direction)
        velocity_along_road = np.dot(road_direction, velocity)
        velocity_across_road = velocity - velocity_along_road
        speed_along_road = np.linalg.norm(velocity_along_road)
        speed_across_road = np.linalg.norm(velocity_across_road)
        return speed_along_road, speed_across_road
