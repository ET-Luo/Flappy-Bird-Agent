from __future__ import annotations

import argparse
import os
import time

import environments  # ensure env registered
import gymnasium as gym
from stable_baselines3 import PPO

try:
	import cv2  # type: ignore
	has_cv2 = True
except Exception:
	has_cv2 = False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird PPO agent")
	parser.add_argument("--model", type=str, default="models/final/best_model.zip", help="Path to model .zip file")
	parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
	parser.add_argument("--render", action="store_true", help="Render using rgb_array and show score")
	parser.add_argument("--show", action="store_true", help="Use OpenCV window to display frames (requires opencv-python)")
	parser.add_argument("--save_video", type=str, default="", help="If set, save video to this file path (e.g., videos/run.mp4)")
	parser.add_argument("--fps", type=int, default=30, help="FPS for display/video saving")
	return parser.parse_args()


def main():
	args = parse_args()
	if not os.path.exists(args.model):
		raise FileNotFoundError(f"Model not found: {args.model}")

	# Use rgb_array render mode if rendering/showing/saving is requested
	need_frames = args.render or args.show or bool(args.save_video)
	render_mode = "rgb_array" if need_frames else None
	env = gym.make("FlappyBird-v0", render_mode=render_mode)
	model = PPO.load(args.model)

	# Video writer setup
	writer = None
	if args.save_video:
		os.makedirs(os.path.dirname(args.save_video), exist_ok=True)
		if not has_cv2:
			raise RuntimeError("OpenCV is required to save video. Please install opencv-python.")
		# Determine frame size lazily after first render

	scores = []
	for ep in range(args.episodes):
		obs, info = env.reset()
		done = False
		score = 0.0
		while not done:
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, terminated, truncated, info = env.step(int(action))
			done = terminated or truncated
			score += float(reward)

			if need_frames:
				frame = env.render()
				if has_cv2 and args.show and frame is not None:
					# Convert RGB -> BGR for OpenCV
					bgr = frame[..., ::-1]
					cv2.imshow("FlappyBird-Agent", bgr)
					cv2.waitKey(int(1000 / args.fps))

				if args.save_video:
					if writer is None:
						h, w = frame.shape[:2]
						fourcc = cv2.VideoWriter_fourcc(*"mp4v")
						writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, (w, h))
					writer.write(frame[..., ::-1])  # RGB->BGR

		print(f"Episode {ep+1}: score={score:.2f}")
		scores.append(score)

	if writer is not None:
		writer.release()
	if has_cv2:
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass

	mean_score = sum(scores) / len(scores)
	print(f"Mean score over {args.episodes} episodes: {mean_score:.2f}")


if __name__ == "__main__":
	main()
