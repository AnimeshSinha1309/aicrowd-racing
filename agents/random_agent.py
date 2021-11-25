import json
import time
import numpy as np
from agents.base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
    def select_action(self, obs) -> np.array:
        return self.action_space.sample()
        
    def training(self, env):
        info = {}
        done = False
        obs = self.env.reset()
        
        for _ in range(300):
            action = self.agent.select_action(obs)
            obs, reward, done, info = self.env.step(action)
        
            if done:
                obs = self.env.reset()
                done = False
                
            ## Update your agent
