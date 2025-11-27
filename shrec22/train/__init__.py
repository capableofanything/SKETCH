"""
SHREC22 Training Module

This package contains all necessary components for training gesture recognition models.

Modules:
    - config: Configuration settings
    - dataset: Custom dataset class
    - model: Model architecture
    - scheduler: Learning rate scheduler
    - train: Main training script
    - test: Evaluation utilities
"""

from .config import CONFIG, save_config
from .dataset import CustomImageDataset
from .model import SwinWithRegressor
from .scheduler import get_cosine_schedule_with_warmup
from .test import evaluate_model

__all__ = [
    'CONFIG',
    'save_config',
    'CustomImageDataset',
    'SwinWithRegressor',
    'get_cosine_schedule_with_warmup',
    'evaluate_model',
]

__version__ = '1.0.0'

