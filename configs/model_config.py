"""
Model configuration for Flappy Bird Agent.
Contains network architecture and model-specific parameters.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Network architecture
    policy_net_arch: List[int] = None
    value_net_arch: List[int] = None
    
    # CNN parameters for image input
    cnn_features_dim: int = 512
    cnn_kernel_sizes: List[int] = None
    cnn_strides: List[int] = None
    cnn_filters: List[int] = None
    
    # Activation functions
    activation_fn: str = "relu"  # "relu", "tanh", "elu"
    
    # Dropout
    dropout_rate: float = 0.1
    
    # Image preprocessing
    image_size: Tuple[int, int] = (84, 84)
    grayscale: bool = True
    frame_stack: int = 4
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.policy_net_arch is None:
            self.policy_net_arch = [64, 64]
        
        if self.value_net_arch is None:
            self.value_net_arch = [64, 64]
        
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [8, 4, 3]
        
        if self.cnn_strides is None:
            self.cnn_strides = [4, 2, 1]
        
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 64]
