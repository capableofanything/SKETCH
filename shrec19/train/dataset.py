"""
Custom Dataset class for SHREC19
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted


class CustomImageDataset(Dataset):
    """
    Custom dataset for SHREC19 gesture recognition

    Args:
        image_dir: Directory containing images
        label_path: Path to hard labels (.npy file)
        window_label_path: Path to window labels (.npy file)
        sequence_path: Path to sequence data (.npy file)
        transform: Image transformations to apply
        threshold: Threshold for separating pure/unpure samples
    """

    def __init__(self, image_dir, label_path, window_label_path, sequence_path,
                 transform=None, threshold=1.0):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = natsorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Load labels
        self.hard_labels = np.load(label_path)  # (N,)
        self.window_labels = np.load(window_label_path)  # (N,40)
        self.sequence_labels = np.load(sequence_path).astype(np.float32)

        # Verify data consistency
        assert len(self.image_filenames) == len(self.hard_labels), \
            f"Number of images ({len(self.image_filenames)}) and labels ({len(self.hard_labels)}) do not match!"

        self.num_window_classes = 6  # SHREC19 has 6 window classes (5 gestures + 1 no-gesture)
        self.threshold = threshold

        # Calculate probabilities for each sample
        counts = np.array([
            np.bincount(wl, minlength=self.num_window_classes)
            for wl in self.window_labels
        ])  # shape (N,6)
        probs = counts / counts.sum(axis=1, keepdims=True)  # shape (N,6)

        # Separate indices based on purity threshold
        max_probs = probs.max(axis=1)
        self.unpure_indices = np.where(max_probs < self.threshold)[0]
        self.pure_indices = np.where(max_probs >= self.threshold)[0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get hard label
        hard_label = int(self.hard_labels[idx])

        # Calculate gesture start/end indices
        gesture_start_idx = 0
        window_label = self.window_labels[idx]
        gesture_end_idx = len(window_label) - 1
        lab = window_label[0]

        for i, val in enumerate(window_label):
            if val != lab and val != 5:  # 5 is no-gesture class in SHREC19
                gesture_start_idx = i
            if val != lab and val == 5:
                gesture_end_idx = i - 1
            lab = val

        # Normalize gesture start/end to [0, 1]
        gesture_start = gesture_start_idx / (len(window_label) - 1)
        gesture_end = gesture_end_idx / (len(window_label) - 1)

        # Load and normalize sequence data
        seq = torch.from_numpy(self.sequence_labels[idx])  # [40,3,16], float32

        # Calculate mean and std along frames (axis 0) and joints (axis 2)
        mean = seq.mean(dim=(0, 2), keepdim=True)  # shape [1, 3, 1]
        std = seq.std(dim=(0, 2), keepdim=True)  # shape [1, 3, 1]

        # Normalize
        seq_norm = (seq - mean) / (std + 1e-8)

        return image, hard_label, gesture_start, gesture_end, seq_norm

