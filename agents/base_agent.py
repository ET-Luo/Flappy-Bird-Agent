from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import gymnasium as gym


class BaseAgent(ABC):
	"""Base interface for RL agents."""

	def __init__(self, env: gym.Env):
		super().__init__()
		self.env = env

	@abstractmethod
	def train(self, total_timesteps: int) -> None:
		...

	@abstractmethod
	def save(self, path: str) -> None:
		...

	@abstractmethod
	def load(self, path: str) -> None:
		...

	@abstractmethod
	def evaluate(self, n_episodes: int = 5) -> float:
		"""Run evaluation episodes and return mean reward."""
		...
