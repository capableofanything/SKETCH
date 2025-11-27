"""
Main training script for SHREC22 gesture recognition
"""
import os
import random
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CONFIG, save_config
from dataset import CustomImageDataset
from model import SwinWithRegressor
from scheduler import get_cosine_schedule_with_warmup
from test import evaluate_model


def setup_environment(config):
    """
    Setup random seeds, devices, and other environment settings

    Args:
        config: Configuration dictionary

    Returns:
        device: PyTorch device to use for training
    """
    print(f"Total CPUs available: {multiprocessing.cpu_count()}")

    # Set random seeds for reproducibility
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Setup GPU
    num_gpus = torch.cuda.device_count()
    print(f"Total available GPUs: {num_gpus}")

    torch.cuda.set_device(config["SET_DEVICE"])
    print(f"Using GPU {torch.cuda.current_device()}: "
          f"{torch.cuda.get_device_name(torch.cuda.current_device())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def create_dataloaders(config):
    """
    Create train and test dataloaders

    Args:
        config: Configuration dictionary

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        num_classes: Number of gesture classes
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create train dataset
    train_dataset = CustomImageDataset(
        config["train_image_dir"],
        config["train_label_path"],
        config["train_window_label_path"],
        config["train_sequence_path"],
        transform,
        threshold=config["threshold"]
    )

    # Use only pure samples for training
    train_pure_idx = train_dataset.pure_indices
    train_pure_dataset = Subset(train_dataset, train_pure_idx)

    train_loader = DataLoader(
        train_pure_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Create test dataset
    test_dataset = CustomImageDataset(
        config["test_image_dir"],
        config["test_label_path"],
        config["test_window_label_path"],
        config["test_sequence_path"],
        transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Get number of classes
    num_classes = len(np.unique(np.load(config["train_label_path"])))

    return train_loader, test_loader, num_classes


def train_epoch(model, train_loader, criterion_cls, criterion_reg,
                optimizer, scheduler, device, config):
    """
    Train for one epoch

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion_cls: Classification loss criterion
        criterion_reg: Regression loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration dictionary

    Returns:
        avg_loss: Average total loss
        avg_cls_loss: Average classification loss
        avg_start_loss: Average gesture start loss
        avg_end_loss: Average gesture end loss
        accuracy: Training accuracy
    """
    model.train()
    torch.cuda.empty_cache()  # Free cache memory

    train_loss = 0.0
    correct = 0
    total = 0
    total_loss_cls = 0.0
    total_loss_start = 0.0
    total_loss_end = 0.0

    train_loader_tqdm = tqdm(train_loader, desc="Training", unit="batch")

    for images, labels, gesture_start_gt, gesture_end_gt, seq in train_loader_tqdm:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        gesture_start_gt = gesture_start_gt.to(device).float()
        gesture_end_gt = gesture_end_gt.to(device).float()
        seq = seq.to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs_cls, outputs_start, outputs_end, _ = model(images, seq)

        # Calculate losses
        loss_cls = criterion_cls(outputs_cls, labels)
        loss_start = criterion_reg(outputs_start, gesture_start_gt)
        loss_end = criterion_reg(outputs_end, gesture_end_gt)
        loss = loss_cls + config["lambda_mse"] * (loss_start + loss_end)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate losses
        train_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_start += loss_start.item()
        total_loss_end += loss_end.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs_cls, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        train_loader_tqdm.set_postfix(
            loss=f"{loss.item():.4f}",
            cls_loss=f"{loss_cls.item():.4f}",
            start_loss=f"{loss_start.item():.4f}",
            end_loss=f"{loss_end.item():.4f}",
            acc=f"{(correct / total) * 100:.2f}%"
        )

    # Calculate averages
    num_batches = len(train_loader)
    avg_loss = train_loss / num_batches
    avg_cls_loss = total_loss_cls / num_batches
    avg_start_loss = total_loss_start / num_batches
    avg_end_loss = total_loss_end / num_batches
    accuracy = 100 * correct / total

    return avg_loss, avg_cls_loss, avg_start_loss, avg_end_loss, accuracy


def plot_metrics(train_accuracies, test_accuracies, train_losses,
                train_cls_losses, train_start_losses, train_end_losses,
                test_losses, config, exp_folder_path):
    """
    Plot and save training metrics

    Args:
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
        train_losses: List of training total losses
        train_cls_losses: List of training classification losses
        train_start_losses: List of training gesture start losses
        train_end_losses: List of training gesture end losses
        test_losses: List of test losses
        config: Configuration dictionary
        exp_folder_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config["epochs"] + 1), train_accuracies,
             label="Train Accuracy", color="blue", linewidth=2, marker="o")

    if len(test_accuracies) > 0:
        test_epochs = list(range(config["start_test_epoch"] + 1, config["epochs"] + 1))
        plt.plot(test_epochs, test_accuracies,
                label="Test Accuracy", color="red",
                linestyle="dashed", linewidth=2, marker="s")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config["epochs"] + 1), train_losses,
             label="Train Total Loss", color="black", linewidth=2, marker="o")
    plt.plot(range(1, config["epochs"] + 1), train_cls_losses,
             label="Train Classification Loss", color="blue", linewidth=2, marker="v")
    plt.plot(range(1, config["epochs"] + 1), train_start_losses,
             label="Train Gesture Start Loss", color="green", linewidth=2, marker="s")
    plt.plot(range(1, config["epochs"] + 1), train_end_losses,
             label="Train Gesture End Loss", color="purple", linewidth=2, marker="D")

    if len(test_losses) > 0:
        test_epochs = list(range(config["start_test_epoch"] + 1, config["epochs"] + 1))
        plt.plot(test_epochs, test_losses,
                label="Test Classification Loss", color="red",
                linestyle="dashed", linewidth=2, marker="x")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Trends (Train & Test)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(exp_folder_path,
                                f"shrec22_exp{config['exp_num']}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training plots saved to {plot_filename}")


def main():
    """
    Main training function
    """
    # Print configuration
    print("=" * 60)
    print(f"Experiment Number: {CONFIG['exp_num']}")
    print("=" * 60)

    # Setup environment
    device = setup_environment(CONFIG)

    # Create dataloaders
    train_loader, test_loader, num_classes = create_dataloaders(CONFIG)

    # Create model
    model = SwinWithRegressor(
        num_classes=num_classes,
        pretrained_model=CONFIG["pretrained_model"]
    ).to(device)

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Learning rate scheduler
    num_training_steps = CONFIG["epochs"] * len(train_loader)
    num_warmup_steps = int(num_training_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Setup experiment folder and save config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_folder_path = os.path.join(script_dir, f"exp_{CONFIG['exp_num']}")
    config_filename = save_config(exp_folder_path, CONFIG)

    # Metrics tracking
    train_accuracies, test_accuracies = [], []
    train_losses, train_cls_losses = [], []
    train_start_losses, train_end_losses = [], []
    test_losses = []
    best_accuracy = 0.0

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 60)

        # Train for one epoch
        avg_loss, avg_cls_loss, avg_start_loss, avg_end_loss, accuracy = train_epoch(
            model, train_loader, criterion_cls, criterion_reg,
            optimizer, scheduler, device, CONFIG
        )

        # Record training metrics
        train_accuracies.append(accuracy)
        train_losses.append(avg_loss)
        train_cls_losses.append(avg_cls_loss)
        train_start_losses.append(avg_start_loss)
        train_end_losses.append(avg_end_loss)

        # Save temporary model
        torch.save(model.state_dict(), f"train_tmp_exp{CONFIG['exp_num']}.pth")

        # Evaluate on test set
        if epoch >= CONFIG["start_test_epoch"]:
            test_accuracy, test_loss, _, best_accuracy = evaluate_model(
                model, test_loader, criterion_cls, device, CONFIG,
                epoch, best_accuracy, exp_folder_path, config_filename
            )
            test_accuracies.append(test_accuracy)
            test_losses.append(test_loss)

        print("Training completed for this epoch!")

    # Print final results
    print("\n" + "=" * 60)
    print(f"[FINAL] Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 60 + "\n")

    # Plot and save metrics
    plot_metrics(
        train_accuracies, test_accuracies,
        train_losses, train_cls_losses, train_start_losses, train_end_losses,
        test_losses, CONFIG, exp_folder_path
    )

    print(f"\nAll results saved to: {exp_folder_path}")


if __name__ == "__main__":
    main()

