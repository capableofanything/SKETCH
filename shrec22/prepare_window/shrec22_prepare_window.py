import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
from tqdm import tqdm
import os.path as opt
from random import randint, shuffle


def _rot_matrix_batch(rot):  # rot: (T,3) in radians
    """
    Compute batch rotation matrices from Euler angles.
    rot[t] = (rx, ry, rz) for frame t.
    Returns R: (T,3,3) with R = Rz @ Ry @ Rx
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # (T,3)

    T = rot.shape[0]
    device = rot.device
    dtype = rot.dtype
    zeros = torch.zeros(T, 1, device=device, dtype=dtype)
    ones = torch.ones(T, 1, device=device, dtype=dtype)

    # Rx
    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # (T,1,3)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], -sin_r[:, 0:1]), dim=-1)  # (T,1,3)
    rx3 = torch.stack((zeros, sin_r[:, 0:1],  cos_r[:, 0:1]), dim=-1)  # (T,1,3)
    Rx = torch.cat((r1, rx2, rx3), dim=1)  # (T,3,3)

    # Ry
    ry1 = torch.stack((cos_r[:, 1:2], zeros, sin_r[:, 1:2]), dim=-1)
    r2  = torch.stack((zeros,          ones,  zeros),        dim=-1)
    ry3 = torch.stack((-sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    Ry = torch.cat((ry1, r2, ry3), dim=1)

    # Rz (standard: [[c,-s,0],[s,c,0],[0,0,1]])
    rz1 = torch.stack((cos_r[:, 2:3], -sin_r[:, 2:3], zeros), dim=-1)
    rz2 = torch.stack((sin_r[:, 2:3],  cos_r[:, 2:3], zeros), dim=-1)
    r3  = torch.stack((zeros, zeros, ones), dim=-1)
    Rz = torch.cat((rz1, rz2, r3), dim=1)

    R = Rz.matmul(Ry).matmul(Rx)  # (T,3,3)
    return R


def random_rot_window(window_numpy, theta=0.3):
    """
    Apply random 3D rotation to all frames in a window.

    Args:
        window_numpy: (w, 3, 26) - frames first, channels=3(x,y,z), joints=26
        theta: Rotation angle range in radians

    Returns:
        Rotated window: (w, 3, 26) numpy array

    All frames receive the same (rx,ry,rz) rotation.
    """
    assert window_numpy.ndim == 3 and window_numpy.shape[1] == 3, "expected (w,3,26)"
    w, C, V = window_numpy.shape
    data_torch = torch.from_numpy(window_numpy.astype(np.float32))  # (w,3,26)

    data_torch = data_torch.contiguous().view(w, C, V)  # (T= w, 3, V)
    # Sample same angle set and replicate to all frames
    rot = torch.empty(3).uniform_(-theta, theta)       # (3,)
    rot = torch.stack([rot]*w, dim=0)                  # (w,3)
    R = _rot_matrix_batch(rot)                         # (w,3,3)

    # Batch matrix multiplication: (w,3,3) @ (w,3,V) -> (w,3,V)
    data_rot = torch.matmul(R, data_torch)
    data_rot = data_rot.view(w, C, V)

    return data_rot.cpu().numpy().astype(np.float64)


class Dataset_shrec22(Dataset):
    """
    SHREC2022 dataset class (window generation + optional augmentation saving)

    Args:
        data_dir: Dataset directory path
        data_set: "train" or "test"
        w: Sliding window size
        stride: Sliding window stride
        aug_versions: Total versions to save per window (1=no augmentation, 3=original+2 augmented)
        theta: Random rotation angle range in radians U(-theta, theta)
    """
    def __init__(self, data_dir: str, data_set: str, w: int, stride: int = 1,
                 aug_versions: int = 1, theta: float = 0.3):
        self.path_to_data = data_dir
        self.w = w
        self.stride = stride
        self.aug_versions = max(1, int(aug_versions))
        self.theta = float(theta)
        self.data_set = data_set.lower()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Variables for storing windows and labels
        self.sequence = []             # Each window: (w, 3, 26)
        self.labels_window = []        # Frame-level labels for each window (w,)
        self.label = []                # Representative label for each window (majority vote)
        self.all_file_poses = []       # Pose per file (each file: (num_frames, 3, 26))
        self.window_source_file_indices = []  # File name from which each window was generated
        self.window_frame_indices = []        # (start, end) frame indices for each window

        # Total 17 labels (including no-gesture)
        self.label_map = [
            "ONE", "TWO", "THREE", "FOUR", "OK", "MENU", "LEFT", "RIGHT",
            "CIRCLE", "V", "CROSS", "GRAB", "PINCH", "DENY", "WAVE", "KNOB",
            "nongesture"
        ]

        # Read annotation file and construct sequence for each file
        with open(opt.join(self.path_to_data, "annotations.txt"), "r") as gt:
            for line in gt.readlines():
                # Example: "1;CROSS;212;253;TWO;309;341;LEFT;426;444\n"
                line = line.strip('\n').split(";")
                if '' in line:
                    line.remove('')
                file_name = line[0]
                line = line[1:]

                # gt_window: Initial label value for all frames in each file (initialized to 16)
                gt_window = np.full(780, 16, dtype=np.int64)
                for index in range(0, len(line), 3):
                    lab = line[index]
                    s = int(line[index + 1])
                    e = int(line[index + 2])
                    gt_window[s: e + 1] = self.label_map.index(lab)

                file_poses = []
                with open(opt.join(self.path_to_data, f"{file_name}.txt"), "r") as fp:
                    for line_idx, line in enumerate(fp.readlines()):
                        # Extract coordinate data: from after first two values to before the last
                        coords = line.split(";")[2:-1]
                        # Reshape to (26,3) then transpose: (3,26)
                        coords = np.reshape(coords, (26, 3)).transpose().astype(np.float64)
                        file_poses.append([coords[0], coords[1], coords[2]])  # (3,26)
                file_poses = np.array(file_poses, dtype=np.float64)  # (num_frames, 3, 26)
                self.all_file_poses.append(file_poses)

                # Apply sliding window to extract sequences
                num_frames = file_poses.shape[0]
                for poses_index in range(0, num_frames - self.w, self.stride):
                    start_idx = poses_index
                    end_idx = poses_index + self.w - 1

                    window = file_poses[poses_index: poses_index + self.w, :, :]  # (w,3,26)
                    self.sequence.append(window)
                    self.window_source_file_indices.append(file_name)

                    label_window = gt_window[poses_index: poses_index + self.w]   # (w,)
                    # Majority vote representative label
                    binc = np.bincount(label_window.astype(np.int64))
                    maj = int(np.argmax(binc))
                    self.label.append(maj)
                    self.labels_window.append(label_window)
                    self.window_frame_indices.append([start_idx, end_idx])

        self.len_data = len(self.sequence)
        self.all_file_poses = np.array(self.all_file_poses, dtype=object)

        # Apply augmentation (before saving): only for train with aug_versions>1
        if self.data_set == "train" and self.aug_versions > 1:
            self._apply_augmentations_in_memory()

        # Save to files
        self.save_windows_and_labels_no_norm(self.data_set, self.stride)

        print("Initialization Complete")

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.label[idx], self.labels_window[idx]

    # ----------------- NEW: augmentation in-memory -----------------
    def _apply_augmentations_in_memory(self):
        """
        Apply random 3D rotations to current windows in self.sequence/label/labels_window.
        Each window is augmented (aug_versions-1) times and added to the lists.
        """
        print(f"[Augment] Applying random 3D rotations: aug_versions={self.aug_versions}, theta={self.theta}")
        base_N = len(self.sequence)
        extra = self.aug_versions - 1

        new_seq = []
        new_lab = []
        new_lab_win = []
        new_src = []
        new_idx = []

        for i in tqdm(range(base_N), desc="Augmenting windows"):
            window = self.sequence[i]              # (w,3,26)
            label_major = self.label[i]            # int
            label_window = self.labels_window[i]   # (w,)
            file_name = self.window_source_file_indices[i]
            start_idx, end_idx = self.window_frame_indices[i]

            # Original is already included, only add augmented versions
            for k in range(extra):
                aug_w = random_rot_window(window, theta=self.theta)  # (w,3,26)
                new_seq.append(aug_w)
                new_lab.append(label_major)
                new_lab_win.append(label_window.copy())
                new_src.append(file_name)
                new_idx.append([start_idx, end_idx])

        if new_seq:
            self.sequence.extend(new_seq)
            self.label.extend(new_lab)
            self.labels_window.extend(new_lab_win)
            self.window_source_file_indices.extend(new_src)
            self.window_frame_indices.extend(new_idx)

        print(f"[Augment] Added {len(new_seq)} augmented windows "
              f"(total windows: {len(self.sequence)})")

    # ----------------- 저장 루틴 -----------------
    def save_windows_and_labels_no_norm(self, data_set, stride):
        """
        Save windows, representative labels, and frame-level labels to .npy files.
        For train with aug_versions>1, add _aug{aug_versions} suffix to filenames.
        """
        aug_suffix = ""
        if self.data_set == "train" and self.aug_versions > 1:
            aug_suffix = f"_aug{self.aug_versions}"

        save_path = os.path.join(self.script_dir, f"{data_set}_sequence_w{self.w}_s{stride}{aug_suffix}.npy")
        labels_save_path = os.path.join(self.script_dir, f"{data_set}_labels_w{self.w}_s{stride}{aug_suffix}.npy")
        labels_window_save_path = os.path.join(self.script_dir, f"{data_set}_labels_window_w{self.w}_s{stride}{aug_suffix}.npy")

        sequence_array = np.array(self.sequence, dtype=np.float64)         # (N, w, 3, 26)
        labels_array = np.array(self.label, dtype=np.int64)                # (N,)
        labels_window_array = np.array(self.labels_window, dtype=np.int64) # (N, w)

        print("Sequence shape:", sequence_array.shape)
        print("Labels shape:", labels_array.shape)
        print("Labels_window shape:", labels_window_array.shape)

        np.save(save_path, sequence_array)
        np.save(labels_save_path, labels_array)
        np.save(labels_window_save_path, labels_window_array)

        print(f"Sequence saved to {save_path}")
        print(f"Labels saved to {labels_save_path}")
        print(f"Labels_window saved to {labels_window_save_path}")

    def _save_window_source_file_indices(self, data_set, stride):
        """
        Save the source file name for each window to a .npy file.
        """
        save_path = os.path.join(self.script_dir, f"{data_set}_window_source_file_indices_s{stride}.npy")
        source_indices_array = np.array(self.window_source_file_indices)
        np.save(save_path, source_indices_array)
        print(f"Window source file indices saved to {save_path}")
        print(f"Shape: {source_indices_array.shape}")

    def _save_window_frame_indices(self, data_set, stride):
        """
        Save the (start, end) frame indices for each window to a .npy file.
        """
        save_path = os.path.join(self.script_dir, f"{data_set}_window_frame_indices_s{stride}.npy")
        frame_indices_array = np.array(self.window_frame_indices, dtype=np.int64)
        np.save(save_path, frame_indices_array)
        print(f"Window frame indices saved to {save_path}")
        print(f"Shape: {frame_indices_array.shape}")


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_train = os.path.join(script_dir, "../SHREC2022/shrec2022_training_set")
    data_dir_test  = os.path.join(script_dir, "../SHREC2022/shrec2022_test_set")

    # train: 3 versions per window (original + 2 augmented), theta=0.3 radians
    dataset = Dataset_shrec22(data_dir=data_dir_train, data_set="train", w=16, stride=1,
                              aug_versions=3, theta=0.3)
    print("Accumulated file poses shape:", np.shape(dataset.all_file_poses))

    # test: no augmentation (aug_versions=1)
    dataset = Dataset_shrec22(data_dir=data_dir_test, data_set="test", w=16, stride=1,
                              aug_versions=1)
    print("Accumulated file poses shape:", np.shape(dataset.all_file_poses))
