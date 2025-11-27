import numpy as np
import os
import re  # Regular expression library

script_dir = os.path.dirname(os.path.abspath(__file__))
# Data directory paths
data_dir = os.path.join(script_dir, "../SHREC2019/Training")  # Training folder path
annotation_file = os.path.join(script_dir, "../SHREC2019/train_annotation.txt")  # Annotation save file

# Initialize annotation file (overwrite existing data)
with open(annotation_file, "w") as f:
    f.write("")  # Initialize as empty file

# Find all `.txt` files in Training folder (sort by numeric order)
txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
txt_files = sorted(txt_files, key=lambda x: int(re.search(r'\d+', x).group()))  # Sort by number

# Lists to store pos and quat data
all_pos_list = []
all_quat_list = []

for file_name in txt_files:
    file_path = os.path.join(data_dir, file_name)

    # Extract `gesture_id` from filename (e.g., "training_1.txt" → "1")
    gesture_id_match = re.search(r'\d+', file_name)  # Extract number
    gesture_id = gesture_id_match.group() if gesture_id_match else "0"  # Default to "0" if no number found

    # Lists to store pos (position coordinates) and quat (rotation coordinates)
    pos_list = []
    quat_list = []
    gesture_annotations = []  # List to store gesture start and end

    # Variables to track gesture start and end
    start_frame = None
    gesture_label = None
    prev_frame = None  # Store previous frame
    found_start = False  # Flag to set start_frame on next frame
    first_frame_id = None  # Store first frame_id

    # Read file
    with open(file_path, "r") as f:
        for line in f:
            # Split data by semicolon (;)
            data = line.strip().split(";")

            # frame_ID and timestamp
            frame_id = data[0]
            timestamp = data[1] if len(data) > 1 else ""

            # Store first valid frame_id (determined in first line)
            if first_frame_id is None and frame_id.isdigit():
                first_frame_id = int(frame_id)  # Store first frame ID

            # Find gesture start/end (when frame_id == "-1")
            if frame_id == "-1":
                if gesture_label is None:
                    # First `-1` → Set next frame as `start_frame`
                    gesture_label = data[-1]  # Gesture label (e.g., 'X', 'O', 'V')
                    found_start = True  # Flag to set start_frame on next frame
                else:
                    # Second `-1` → End of gesture
                    if first_frame_id is not None:
                        adjusted_start_frame = int(start_frame) - first_frame_id + 1
                        adjusted_end_frame = int(prev_frame) - first_frame_id + 1
                        gesture_annotations.append(f"{gesture_id};{gesture_label};{adjusted_start_frame};{adjusted_end_frame}")
                    # Reset variables
                    start_frame = None
                    gesture_label = None
                continue  # Skip pos_list and quat_list processing when frame_id == "-1"

            # Set first frame after `-1` as `start_frame`
            if found_start:
                start_frame = frame_id
                found_start = False  # Reset flag after setting once

            prev_frame = frame_id  # Store current frameID as prev_frame

            # Exclude frame_ID and timestamp
            values = data[2:]

            # Data cleaning: Handle empty values
            try:
                values = [float(v) if v.strip() else 0.0 for v in values]  # Convert empty strings to 0.0
            except ValueError:
                print(f"❌ Conversion error (file: {file_name}): {values}")
                continue  # Skip error lines

            # Extract only pos (x, y, z) coordinates (16 joints × 3 coordinates)
            pos_values = [values[i:i+3] for i in range(0, len(values), 7)]  # (16, 3) shape
            pos_list.append(pos_values)

            # Extract only quat (x, y, z, w) coordinates (16 joints × 4 values)
            quat_values = [values[i + 3:i + 7] for i in range(0, len(values), 7)]  # (16, 4) shape
            quat_list.append(quat_values)

    # Add to total data list
    all_pos_list.extend(pos_list)
    all_quat_list.extend(quat_list)

    # Convert to array and check shape
    pos_array = np.array(pos_list).reshape(-1, 16, 3)  # Convert to (num_frames, 16, 3)
    quat_array = np.array(quat_list).reshape(-1, 16, 4)  # Convert to (num_frames, 16, 4)

    print(f"📂 {file_name} processing complete: pos={pos_array.shape}, quat={quat_array.shape}")

    # Save gesture information (append to existing file)
    with open(annotation_file, "a") as f:
        for annotation in gesture_annotations:
            f.write(annotation + "\n")

print(np.array(all_pos_list).shape)
print(np.array(all_quat_list).shape)

print(f"✅ All files processed! Gesture annotations saved to {annotation_file}.")
