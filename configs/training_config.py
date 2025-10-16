"""
Training configuration for Flappy Bird Agent.
Contains all hyperparameters and settings for PPO training.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    
    # Training parameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # PPO specific parameters
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Environment parameters
    env_name: str = "FlappyBird-v0"
    render_mode: str = "rgb_array"
    frame_skip: int = 4
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 100000
    eval_interval: int = 50000
    eval_episodes: int = 10
    
    # Paths
    model_save_path: str = "models/checkpoints"
    log_path: str = "logs/tensorboard"
    final_model_path: str = "models/final"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'total_timesteps': self.total_timesteps,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
        }
