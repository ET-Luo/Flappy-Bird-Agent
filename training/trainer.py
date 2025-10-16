from __future__ import annotations

import os
from typing import List

import gymnasium as gym

from agents.ppo_agent import PPOAgent
from configs.training_config import TrainingConfig
from training.callback import CheckpointCallback, BestModelCallback


class Trainer:
	def __init__(self, env_id: str = "FlappyBird-v0", config: TrainingConfig | None = None):
		self.env_id = env_id
		self.config = config or TrainingConfig()
		self.env = gym.make(self.env_id)

		callbacks = [
			CheckpointCallback(save_freq=self.config.save_interval, save_path=self.config.model_save_path, verbose=1),
			BestModelCallback(save_path=os.path.join(self.config.final_model_path, "best_model.zip"), verbose=1),
		]
		self.agent = PPOAgent(self.env, config=self.config, callback=callbacks)

	def train(self) -> None:
		os.makedirs(self.config.model_save_path, exist_ok=True)
		os.makedirs(self.config.final_model_path, exist_ok=True)
		self.agent.train(total_timesteps=self.config.total_timesteps)
		self.agent.save(os.path.join(self.config.final_model_path, "final_model.zip"))

	def evaluate(self) -> float:
		return self.agent.evaluate(n_episodes=self.config.eval_episodes)
