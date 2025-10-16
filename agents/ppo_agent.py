from __future__ import annotations

from typing import Optional, Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from configs.training_config import TrainingConfig

from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
	"""PPO agent wrapper using stable-baselines3."""

	def __init__(self, env: gym.Env, config: Optional[TrainingConfig] = None, callback: Optional[BaseCallback] = None):
		super().__init__(env)
		self.config = config or TrainingConfig()
		self.callback = callback
		# Wrap env for SB3
		self.vec_env = DummyVecEnv([lambda: env])
		self.model = PPO(
			"CnnPolicy" if hasattr(env, "observation_space") and len(env.observation_space.shape) == 3 else "MlpPolicy",
			self.vec_env,
			learning_rate=self.config.learning_rate,
			n_steps=self.config.n_steps,
			batch_size=self.config.batch_size,
			n_epochs=self.config.n_epochs,
			clip_range=self.config.clip_range,
			ent_coef=self.config.ent_coef,
			vf_coef=self.config.vf_coef,
			max_grad_norm=self.config.max_grad_norm,
			verbose=1,
			tensorboard_log=self.config.log_path,
		)

	def train(self, total_timesteps: int) -> None:
		self.model.learn(total_timesteps=total_timesteps, callback=self.callback, progress_bar=True)

	def save(self, path: str) -> None:
		self.model.save(path)

	def load(self, path: str) -> None:
		self.model = PPO.load(path, env=self.vec_env)

	def evaluate(self, n_episodes: int = 5) -> float:
		rets = []
		for _ in range(n_episodes):
			obs, info = self.env.reset()
			done = False
			total = 0.0
			while not done:
				action, _ = self.model.predict(obs, deterministic=True)
				obs, reward, terminated, truncated, info = self.env.step(int(action))
				total += float(reward)
				done = terminated or truncated
			rets.append(total)
		return float(sum(rets) / len(rets))
