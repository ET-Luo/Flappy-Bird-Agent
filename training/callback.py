from __future__ import annotations

import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
	"""Saves model every `save_freq` steps to `save_path`."""

	def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
		super().__init__(verbose)
		self.save_freq = int(save_freq)
		self.save_path = save_path
		os.makedirs(save_path, exist_ok=True)

	def _on_step(self) -> bool:
		if self.num_timesteps % self.save_freq == 0:
			path = os.path.join(self.save_path, f"ckpt_{self.num_timesteps}.zip")
			self.model.save(path)
			if self.verbose:
				print(f"Saved checkpoint: {path}")
		return True


class BestModelCallback(BaseCallback):
	"""Tracks best mean reward over last N episodes and saves model."""

	def __init__(self, save_path: str, window: int = 100, verbose: int = 0):
		super().__init__(verbose)
		self.save_path = save_path
		self.window = window
		self._episode_rewards = []
		self._best = None
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

	def _on_step(self) -> bool:
		# SB3 stores episode rewards in the logger, but easiest is to use `infos` with 'score' if provided
		infos = self.locals.get("infos") or []
		for info in infos:
			score = info.get("score")
			if score is not None:
				self._episode_rewards.append(score)
				if len(self._episode_rewards) > self.window:
					self._episode_rewards.pop(0)
				mean_r = sum(self._episode_rewards) / len(self._episode_rewards)
				if self._best is None or mean_r > self._best:
					self._best = mean_r
					self.model.save(self.save_path)
					if self.verbose:
						print(f"Saved best model with score {mean_r:.2f} -> {self.save_path}")
		return True
