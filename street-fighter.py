import retro
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis",
                             use_restricted_actions=retro.Actions.FILTERED)
        self.previous_frame = None
        self.score = 0
    
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resized, (84, 84, 1))
        return channels
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        
        reward = info.get('score', 0) - self.score
        self.score = info.get('score', 0)
        
        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render(*args, **kwargs)
    
    def close(self):
        self.game.close()

# Run the game for 1 episode
env = StreetFighter()  # Fixed typo from StreetFigher to StreetFighter
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random actions
    obs, reward, done, info = env.step(action)
    env.render()
    
    # Optional: Print current reward and score
    if reward > 0:
        print(reward)
    
    # Slow down the game for visualization
    time.sleep(0.01)

env.close()

