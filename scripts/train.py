from __future__ import annotations

import argparse

import environments  # ensure env is registered
from training.trainer import Trainer
from configs.training_config import TrainingConfig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train PPO agent for Flappy Bird")
	parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
	return parser.parse_args()


def main():
	args = parse_args()
	config = TrainingConfig()
	if args.timesteps is not None:
		config.total_timesteps = int(args.timesteps)

	trainer = Trainer(config=config)
	trainer.train()
	mean_reward = trainer.evaluate()
	print(f"Evaluation mean reward: {mean_reward:.2f}")


if __name__ == "__main__":
	main()
