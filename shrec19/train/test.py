"""
Test/Evaluation utilities for SHREC19
"""
import os
import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(model, test_loader, criterion_cls, device, config, epoch,
                   best_accuracy, exp_folder_path, config_filename):
    """
    Evaluate the model on test set

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion_cls: Classification loss criterion
        device: Device to run evaluation on
        config: Configuration dictionary
        epoch: Current epoch number
        best_accuracy: Best accuracy achieved so far
        exp_folder_path: Path to save experiment results
        config_filename: Path to config file for logging

    Returns:
        test_accuracy: Accuracy on test set
        test_loss: Average test loss
        test_predictions: List of predictions
        best_accuracy: Updated best accuracy
    """
    model.eval()
    test_loss_cls = 0.0
    correct_test = 0
    total_test = 0
    test_predictions = []

    test_loader_tqdm = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for images, labels, _, _, seq in test_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)
            seq = seq.to(device)

            # Forward pass
            outputs_cls, _, _ = model(images, seq)
            loss_cls = criterion_cls(outputs_cls, labels)
            test_loss_cls += loss_cls.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs_cls, 1)
            test_predictions.extend(predicted.cpu().numpy().tolist())
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Update progress bar
            test_loader_tqdm.set_postfix(
                cls_loss=f"{loss_cls.item():.4f}",
                acc=f"{(correct_test / total_test) * 100:.2f}%"
            )

    # Calculate final metrics
    test_accuracy = 100 * correct_test / total_test
    avg_test_loss = test_loss_cls / len(test_loader)

    # Log results
    with open(config_filename, "a") as f:
        f.write(f"Epoch {epoch + 1}: {test_accuracy:.2f}%\n")

    print(f"Epoch {epoch + 1}/{config['epochs']} | "
          f"Test Classification Loss: {avg_test_loss:.4f} | "
          f"Test Accuracy: {test_accuracy:.2f}%")

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        model_filename = os.path.join(exp_folder_path,
                                     f"best_model_exp{config['exp_num']}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # Save predictions
        prediction_file = os.path.join(exp_folder_path,
                                      f"test_predictions_exp{config['exp_num']}.npy")
        np.save(prediction_file, np.array(test_predictions))
        print(f"Test predictions saved to {prediction_file}")

    return test_accuracy, avg_test_loss, test_predictions, best_accuracy

