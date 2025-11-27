"""
Configuration file for SHREC22 training
"""
import os

# Window and stride settings
WINDOW_SIZE = 16
STRIDE = 1
AUG = 3

# Get the directory of this config file
script_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Random seed
    "SEED": 42,

    # Device settings
    "SET_DEVICE": 2,  # GPU number to use

    # Experiment number
    "exp_num": 4949,

    # Data paths
    "train_image_dir": os.path.join(script_dir, '..', f"draw/train_imgs_w{WINDOW_SIZE}_s{STRIDE}_aug{AUG}/-_0.5__1_3x1_384x384"),
    "test_image_dir": os.path.join(script_dir, '..', f"draw/test_imgs_w{WINDOW_SIZE}_s{STRIDE}/-_0.5__1_3x1_384x384"),
    "train_label_path": os.path.join(script_dir, f"../prepare_window/train_labels_w{WINDOW_SIZE}_s{STRIDE}_aug{AUG}.npy"),
    "test_label_path": os.path.join(script_dir, f"../prepare_window/test_labels_w{WINDOW_SIZE}_s{STRIDE}.npy"),
    "train_window_label_path": os.path.join(script_dir, f"../prepare_window/train_labels_window_w{WINDOW_SIZE}_s{STRIDE}_aug{AUG}.npy"),
    "test_window_label_path": os.path.join(script_dir, f"../prepare_window/test_labels_window_w{WINDOW_SIZE}_s{STRIDE}.npy"),
    "train_sequence_path": os.path.join(script_dir, f"../prepare_window/train_sequence_w{WINDOW_SIZE}_s{STRIDE}_aug{AUG}.npy"),
    "test_sequence_path": os.path.join(script_dir, f"../prepare_window/test_sequence_w{WINDOW_SIZE}_s{STRIDE}.npy"),

    # Training hyperparameters
    "batch_size": 32,
    "epochs": 20,
    "start_test_epoch": 9,
    "learning_rate": 4e-5,
    "lambda_mse": 1,  # MSE Loss weight
    "weight_decay": 1e-4,

    # Model settings
    "pretrained_model": "swin_large_patch4_window12_384",

    # Training settings
    "warmup_ratio": 0.3,
    "num_workers": 8,
    "threshold": 0.7,
}

def save_config(exp_folder_path, config):
    """
    Save configuration to a text file

    Args:
        exp_folder_path: Path to the experiment folder
        config: Configuration dictionary
    """
    if not os.path.exists(exp_folder_path):
        os.makedirs(exp_folder_path)

    config_filename = os.path.join(exp_folder_path, f"config_exp{config['exp_num']}.txt")
    with open(config_filename, "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print(f"Configuration saved to {config_filename}")
    return config_filename

