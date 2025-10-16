from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class FlappyConfig:
	gravity: float = 0.5
	flap_velocity: float = -7.5
	max_velocity: float = 10.0
	pipe_gap: int = 100
	pipe_distance: int = 180
	ground_y: int = 256
	ceiling_y: int = 0
	screen_width: int = 288
	screen_height: int = 512
	bird_x: int = 56
	start_y: int = 200
	start_velocity: float = 0.0
	max_episode_steps: int = 10000


class FlappyBirdEnv(gym.Env):
	"""A lightweight, vector-state Flappy Bird environment.

	Observation (Box(4,)):
		0: bird_y (pixels)
		1: bird_velocity (pixels per step)
		2: horizontal_distance_to_next_pipe (pixels)
		3: vertical_offset_to_pipe_gap_center (pixels)

	Action (Discrete(2)):
		0: do nothing
		1: flap (apply upward velocity)

	Rewards:
		+1.0 for each pipe successfully passed
		+0.01 per alive step (survival shaping)
		-1.0 on collision (episode termination)
	"""

	metadata = {"render_modes": ["rgb_array", "none"], "render_fps": 30}

	def __init__(self, render_mode: Optional[str] = None, config: Optional[FlappyConfig] = None, seed: Optional[int] = None):
		super().__init__()
		self.config = config or FlappyConfig()
		self.render_mode = render_mode or "none"
		self.np_random, _ = gym.utils.seeding.np_random(seed)

		# Observation: 4-dim continuous
		high = np.array([
			self.config.screen_height,  # bird_y
			self.config.max_velocity,   # bird_velocity
			self.config.screen_width,   # dist to pipe
			self.config.screen_height,  # vertical offset
		], dtype=np.float32)
		self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
		self.action_space = spaces.Discrete(2)

		self._episode_steps = 0
		self._score = 0
		self.state: Optional[np.ndarray] = None
		self._next_pipe: Optional[Tuple[int, int]] = None  # (x, gap_center_y)

	def seed(self, seed: Optional[int] = None):
		self.np_random, _ = gym.utils.seeding.np_random(seed)
		return [seed]

	def _spawn_next_pipe(self, last_pipe_x: Optional[int]) -> Tuple[int, int]:
		gap_center_min = 60
		gap_center_max = self.config.ground_y - 60
		gap_center = int(self.np_random.integers(gap_center_min, gap_center_max))
		if last_pipe_x is None:
			x = self.config.screen_width + 20
		else:
			x = last_pipe_x + self.config.pipe_distance
		return x, gap_center

	def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
		if seed is not None:
			self.seed(seed)
		self._episode_steps = 0
		self._score = 0
		bird_y = float(self.config.start_y)
		bird_v = float(self.config.start_velocity)
		first_pipe_x, first_pipe_gap = self._spawn_next_pipe(last_pipe_x=None)
		self._next_pipe = (first_pipe_x, first_pipe_gap)
		obs = self._build_obs(bird_y, bird_v)
		self.state = obs.copy()
		return obs, {}

	def _build_obs(self, bird_y: float, bird_v: float) -> np.ndarray:
		next_x, gap_center = self._next_pipe
		dist_x = float(next_x - self.config.bird_x)
		vert_off = float(gap_center - bird_y)
		return np.array([bird_y, bird_v, dist_x, vert_off], dtype=np.float32)

	def _collides(self, bird_y: float) -> bool:
		return bird_y >= self.config.ground_y or bird_y <= self.config.ceiling_y

	def step(self, action: int):
		assert self.action_space.contains(action)
		self._episode_steps += 1

		bird_y, bird_v, dist_x, vert_off = self.state.astype(np.float32)

		# Action -> velocity change
		if action == 1:
			bird_v = self.config.flap_velocity
		# gravity
		bird_v = float(np.clip(bird_v + self.config.gravity, -self.config.max_velocity, self.config.max_velocity))
		# integrate position
		bird_y = float(bird_y + bird_v)

		# Move pipe to left
		next_x, gap_center = self._next_pipe
		next_x -= 2  # pipe speed

		# Check if pipe passed
		reward = 0.01  # alive shaping
		if next_x + 26 < self.config.bird_x:  # assumed pipe width ~52, use half-width for center threshold
			self._score += 1
			reward += 1.0
			# spawn a new pipe ahead
			next_x, gap_center = self._spawn_next_pipe(last_pipe_x=self._next_pipe[0])

		self._next_pipe = (next_x, gap_center)

		terminated = False
		# Basic vertical collision with ground/ceiling
		if self._collides(bird_y):
			terminated = True
			reward -= 1.0

		# Horizontal and vertical collision with pipe gap
		pipe_left = next_x - 26
		pipe_right = next_x + 26
		bird_x = self.config.bird_x
		if pipe_left <= bird_x <= pipe_right:
			gap_top = gap_center - self.config.pipe_gap // 2
			gap_bottom = gap_center + self.config.pipe_gap // 2
			if bird_y < gap_top or bird_y > gap_bottom:
				terminated = True
				reward -= 1.0

		truncated = self._episode_steps >= self.config.max_episode_steps

		obs = self._build_obs(bird_y, bird_v)
		self.state = obs.copy()

		info = {"score": self._score}
		return obs, float(reward), bool(terminated), bool(truncated), info

	def _render_frame(self) -> np.ndarray:
		# Simple RGB frame with bird, pipes, and ground
		h = self.config.screen_height
		w = self.config.screen_width
		frame = np.zeros((h, w, 3), dtype=np.uint8)

		# Background sky
		frame[:, :] = (135, 206, 235)  # light sky blue (BGR-like but we're in RGB order)

		# Ground
		ground_y = self.config.ground_y
		frame[ground_y:h, :, :] = (222, 184, 135)  # burlywood

		# Pipe
		next_x, gap_center = self._next_pipe
		gap = self.config.pipe_gap
		top_end = max(0, gap_center - gap // 2)
		bottom_start = min(ground_y, gap_center + gap // 2)
		pipe_color = (34, 139, 34)  # forest green
		pipe_half_w = 26
		left = max(0, next_x - pipe_half_w)
		right = min(w, next_x + pipe_half_w)
		# top pipe
		frame[0:top_end, left:right, :] = pipe_color
		# bottom pipe
		frame[bottom_start:ground_y, left:right, :] = pipe_color

		# Bird
		bird_x = self.config.bird_x
		bird_y = int(self.state[0]) if self.state is not None else self.config.start_y
		bird_color = (255, 215, 0)  # gold
		bird_radius = 6
		by0 = max(0, bird_y - bird_radius)
		by1 = min(h, bird_y + bird_radius)
		bx0 = max(0, bird_x - bird_radius)
		bx1 = min(w, bird_x + bird_radius)
		frame[by0:by1, bx0:bx1, :] = bird_color

		return frame

	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()
		return None

	def close(self):
		pass
