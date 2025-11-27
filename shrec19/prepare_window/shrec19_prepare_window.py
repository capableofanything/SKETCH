import numpy as np
import os
import os.path as opt
import torch
from torch.utils.data import Dataset


# -------------------- Rotation utilities (same policy as before) --------------------
def _rot_matrix_batch(rot):  # rot: (T,3) in radians
    """
    Compute batch rotation matrices from Euler angles.
    rot[t] = (rx, ry, rz) for frame t.
    Returns R: (T,3,3) with R = Rz @ Ry @ Rx
    (All frames will receive the same angles, but T axis is maintained for batch multiplication)
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # (T,3)

    T = rot.shape[0]
    device = rot.device
    dtype = rot.dtype
    zeros = torch.zeros(T, 1, device=device, dtype=dtype)
    ones  = torch.ones(T, 1, device=device, dtype=dtype)

    # Rx (standard sign)
    r1  = torch.stack((ones,  zeros,              zeros),              dim=-1)  # (T,1,3)
    rx2 = torch.stack((zeros, cos_r[:, 0:1], -sin_r[:, 0:1]),          dim=-1)
    rx3 = torch.stack((zeros, sin_r[:, 0:1],  cos_r[:, 0:1]),          dim=-1)
    Rx  = torch.cat((r1, rx2, rx3), dim=1)  # (T,3,3)

    # Ry (standard sign)
    ry1 = torch.stack(( cos_r[:, 1:2], zeros,  sin_r[:, 1:2]), dim=-1)
    r2  = torch.stack(( zeros,         ones,   zeros),         dim=-1)
    ry3 = torch.stack((-sin_r[:, 1:2], zeros,  cos_r[:, 1:2]), dim=-1)
    Ry  = torch.cat((ry1, r2, ry3), dim=1)

    # Rz (standard sign: [[c,-s,0],[s,c,0],[0,0,1]])
    rz1 = torch.stack(( cos_r[:, 2:3], -sin_r[:, 2:3], zeros), dim=-1)
    rz2 = torch.stack(( sin_r[:, 2:3],  cos_r[:, 2:3], zeros), dim=-1)
    r3  = torch.stack(( zeros,          zeros,         ones),  dim=-1)
    Rz  = torch.cat((rz1, rz2, r3), dim=1)

    R = Rz.matmul(Ry).matmul(Rx)  # (T,3,3)
    return R


def random_rot_window(window_numpy, theta=0.3):
    """
    Apply random 3D rotation to all frames in a window.

    Args:
        window_numpy: (w, 3, V) - frames first, channels=3(x,y,z), joints=V(16 for SHREC19)
        theta: Rotation angle range in radians

    Returns:
        Rotated window: (w, 3, V) numpy (float64)

    All frames receive the same (rx,ry,rz) rotation.
    """
    assert window_numpy.ndim == 3 and window_numpy.shape[1] == 3, "expected (w,3,V)"
    w, C, V = window_numpy.shape

    data_torch = torch.from_numpy(window_numpy.astype(np.float32))  # (w,3,V)
    # Sample same angle set and replicate to all frames
    rot = torch.empty(3).uniform_(-theta, theta)  # (3,)
    rot = torch.stack([rot] * w, dim=0)           # (w,3)
    R   = _rot_matrix_batch(rot)                  # (w,3,3)

    # Batch matrix multiplication: (w,3,3) @ (w,3,V) -> (w,3,V)
    data_rot = torch.matmul(R, data_torch)
    return data_rot.cpu().numpy().astype(np.float64)


class Dataset_shrec19(Dataset):  # Inherit from Dataset class
    def __init__(
            self,
            data_dir: str,
            data_set: str,
            w: int,
            stride: int = 1,   # Sliding window stride
            aug_versions: int = 1,  # Total versions to save per window for train (1=no augmentation, 3=original+2 augmented)
            theta: float = 0.3       # Random rotation angle range in radians U(-theta, theta)
    ):
        """
        SHREC2019 dataset class (window generation + optional augmentation saving)

        Args:
            data_dir: Dataset directory path (e.g., D:/SHREC2019/Training or /TestLabel)
            data_set: Dataset type (train/test)
            w: Window size (e.g., 40)
            stride: Sliding window step size
            aug_versions: Only for train. Total versions to save per window (including original)
            theta: Random 3D rotation angle range in radians
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_to_data = data_dir
        self.w = w
        self.sequence = []         # (N, w, 3, 16)
        self.labels_window = []    # (N, w)
        self.label = []            # (N,)
        self.stride = stride

        self.aug_versions = max(1, int(aug_versions))
        self.theta = float(theta)
        self.data_set = data_set.lower()

        self.all_file_poses = []   # list: Each file's (frame, 3, 16)
        self.concatenated_poses = None  # All frame data (frame, 3, 16)
        self.window_source_file_indices = []  # Record source filename for each window

        self.label_map = ["X", "O", "V", "^", "[]", "nongesture"]  # Total 6 classes (last is no-gesture)

        if data_set == "train":
            self.name_name = "training"
        elif data_set == "test":
            self.name_name = "test"
        else:
            raise ValueError("data_set must be 'train' or 'test'")

        # 📌 Check annotation file path (search both Training folder and parent SHREC2019 folder)
        annotation_paths = [
            opt.join(self.path_to_data, f"{data_set}_annotation.txt"),
            opt.join(opt.dirname(self.path_to_data), f"{data_set}_annotation.txt")
        ]

        annotation_file = None
        for path in annotation_paths:
            if opt.exists(path):
                annotation_file = path
                break

        if annotation_file is None:
            raise FileNotFoundError(f"❌ Cannot find {data_set}_annotation.txt file. Please check.")

        print(f"📂 Loading annotation file: {annotation_file}")

        # List to record window counts
        summary_lines = []

        # Read annotation file
        with open(annotation_file, "r") as gt:
            for line in gt.readlines():  # Example: "1;X;196;225"
                line = line.strip('\n').split(";")
                file_name = line[0]
                line = line[1:]

                if self.data_set == 'test' and file_name == '17':
                    print()
                    continue

                # Initialize all frame labels in file as no-gesture (5)
                file_labels = np.full(1000, 5, dtype=np.int64)

                # Apply gesture labels
                for index in range(0, len(line), 3):
                    gesture_type = line[index]
                    s = int(line[index + 1])
                    e = int(line[index + 2])
                    file_labels[s:e + 1] = self.label_map.index(gesture_type)

                # Load pose data
                file_poses = []
                with open(opt.join(self.path_to_data, f"{self.name_name}_{file_name}.txt"), "r") as fp:
                    for line_idx, l in enumerate(fp.readlines()):  # line: 145;time;[112 vals...]
                        if l.startswith("-1"):  # Skip start/end frames
                            continue
                        values = l.split(";")[2:]  # Exclude frame_ID, timestamp
                        values = np.array([float(v) for v in values if v.strip()])
                        # Extract only pos (x, y, z) for 16 joints
                        pos_values = values.reshape(-1, 7)[:, :3]  # (16,3)
                        pos_values = pos_values.T  # (3,16)
                        file_poses.append(pos_values)

                file_poses = np.array(file_poses, dtype=np.float64)  # (num_frames, 3, 16)
                print(f"{self.name_name}_{file_name}.txt total frame number:", file_poses.shape)
                summary_lines.append(f"{self.name_name}_{file_name}.txt total frame number: {file_poses.shape}")

                self.all_file_poses.append(file_poses)

                # Generate windows
                window_count = 0
                num_frames = file_poses.shape[0]
                for poses_index in range(0, num_frames - self.w, self.stride):
                    window = file_poses[poses_index: poses_index + self.w, :, :]  # (w,3,16)
                    self.sequence.append(window)
                    self.window_source_file_indices.append(file_name)

                    label_window = file_labels[poses_index: poses_index + self.w]  # (w,)
                    label_count = np.bincount(label_window.astype("int64"), minlength=len(self.label_map))
                    self.label.append(int(np.argmax(label_count)))
                    self.labels_window.append(label_window)

                    window_count += 1

                print(f"{self.name_name}_{file_name}.txt total window number: {window_count}")
                summary_lines.append(f"{self.name_name}_{file_name}.txt total window number:{window_count}")

        self.len_data = len(self.sequence)

        # Convert lists to numpy arrays (original data first)
        print(np.array(self.sequence).shape)  # (N, w, 3, 16)
        self.sequence = np.array(self.sequence, dtype=np.float64)
        self.labels_window = np.array(self.labels_window, dtype=np.int64)
        self.label = np.array(self.label, dtype=np.int64)

        print(f"📂 Data loading complete: sequence={self.sequence.shape}, label={self.label.shape}, labels_window={self.labels_window.shape}")

        # --------- train + aug_versions>1: Apply in-memory augmentation ---------
        if self.data_set == "train" and self.aug_versions > 1:
            self._apply_augmentations_in_memory()

        # Save data
        self.save_windows_and_labels_no_norm(self.data_set, self.stride)

        if self.data_set == "test":
            self.save_summary(self.data_set, summary_lines, self.stride)  # Save window count summary


    # ----------------- augmentation in-memory -----------------
    def _apply_augmentations_in_memory(self):
        """
        Apply random 3D rotations to current windows in self.sequence/label/labels_window.
        Each window is augmented (aug_versions-1) times and added to the arrays.
        """
        print(f"[Augment] Applying random 3D rotations: aug_versions={self.aug_versions}, theta={self.theta}")
        base_N = len(self.sequence)
        extra = self.aug_versions - 1

        if extra <= 0:
            return

        new_seq = []
        new_lab = []
        new_lab_win = []
        new_src = []

        # Original is already included, only add augmented versions
        for i in range(base_N):
            window = self.sequence[i]            # (w,3,16)
            label_major = self.label[i]          # ()
            label_window = self.labels_window[i] # (w,)
            file_name = self.window_source_file_indices[i]

            for _ in range(extra):
                aug_w = random_rot_window(window, theta=self.theta)  # (w,3,16)
                new_seq.append(aug_w)
                new_lab.append(label_major)
                new_lab_win.append(label_window.copy())
                new_src.append(file_name)

        if new_seq:
            self.sequence = np.concatenate([self.sequence, np.array(new_seq, dtype=np.float64)], axis=0)
            self.label = np.concatenate([self.label, np.array(new_lab, dtype=np.int64)], axis=0)
            self.labels_window = np.concatenate([self.labels_window, np.array(new_lab_win, dtype=np.int64)], axis=0)
            self.window_source_file_indices.extend(new_src)

        print(f"[Augment] Added {len(new_seq)} augmented windows (total windows: {len(self.sequence)})")

    # ----------------- Save routine -----------------
    def save_windows_and_labels_no_norm(self, data_set, stride):
        """
        Save the sequence, its labels, and labels_window to .npy files.
        For train with aug_versions>1, add _aug{aug_versions} suffix to filenames.
        """
        aug_suffix = ""
        if self.data_set == "train" and self.aug_versions > 1:
            aug_suffix = f"_aug{self.aug_versions}"

        save_path = os.path.join(self.script_dir, f"{data_set}_sequence_w{self.w}_s{stride}{aug_suffix}.npy")
        labels_save_path = os.path.join(self.script_dir, f"{data_set}_labels_w{self.w}_s{stride}{aug_suffix}.npy")
        labels_window_save_path = os.path.join(self.script_dir, f"{data_set}_labels_window_w{self.w}_s{stride}{aug_suffix}.npy")

        np.save(save_path, self.sequence)        # (N, w, 3, 16)
        np.save(labels_save_path, self.label)    # (N,)
        np.save(labels_window_save_path, self.labels_window)  # (N, w)

        print(f"📂 Data saved: {save_path}, {labels_save_path}, {labels_window_save_path}")

    def save_summary(self, data_set, summary_lines, stride):
        """Save total frame count and window count to a file"""
        summary_path = os.path.join(self.script_dir, f"inform_w{self.w}_s{stride}.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
        print(f"📂 Window count summary saved: {summary_path}")


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_train = os.path.join(script_dir, "../SHREC2019/Training")
    data_dir_test  = os.path.join(script_dir, "../SHREC2019/TestLabel")

    # train: 3 versions per window (original + 2 augmented), theta=0.3 radians
    dataset = Dataset_shrec19(data_dir_train, data_set="train", w=40, stride=1,
                              aug_versions=3, theta=0.3)
    print("📂 Total sequence count:", np.shape(dataset.all_file_poses))

    # test: no augmentation
    dataset = Dataset_shrec19(data_dir_test, data_set="test", w=40, stride=1,
                              aug_versions=1)
    print("📂 Total sequence count:", np.shape(dataset.all_file_poses))
