from __future__ import annotations

import gymnasium as gym


class IdentityWrapper(gym.Wrapper):
	"""Placeholder wrapper that currently does nothing. Useful for future stacking."""

	def __init__(self, env: gym.Env):
		super().__init__(env)
