"""
Learning rate scheduler utilities
"""
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a learning rate scheduler with linear warmup and cosine decay

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step):
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase: cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

