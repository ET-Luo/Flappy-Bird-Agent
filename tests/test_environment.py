import environments  # ensure FlappyBird-v0 is registered
import gymnasium as gym


def test_flappy_env_basic():
	env = gym.make("FlappyBird-v0")
	obs, info = env.reset(seed=42)
	assert obs.shape == (4,)

	for _ in range(10):
		obs, reward, terminated, truncated, info = env.step(0)
		assert obs.shape == (4,)
		if terminated or truncated:
			break


def test_flappy_env_action_space():
	env = gym.make("FlappyBird-v0")
	assert env.action_space.n == 2
