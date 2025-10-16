"""
Environment package for Flappy Bird Agent.
Contains the Flappy Bird game environment implementation and wrappers.
"""

from gymnasium.envs.registration import register

# Register environment ID for use via gymnasium.make
register(
	id="FlappyBird-v0",
	entry_point="environments.flappy_bird_env:FlappyBirdEnv",
)
