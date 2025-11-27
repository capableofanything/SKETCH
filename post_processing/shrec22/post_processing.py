import os
import numpy as np
from matlab import evaluate_and_save_results
from visualize_confusion_matrix import plot_confusion_matrix_from_txt as plot_confusion_matrix_from_txt

def main():
    """
    Main function to process predictions and generate evaluation results.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_num = 4949
    base_model_dir = os.path.join(script_dir ,f"../../shrec22/train/exp_{exp_num}")
    base_dataset_dir = os.path.join(script_dir, "../../shrec22/SHREC2022")

    config = {
        "paths": {
            "base_model_dir": base_model_dir,
            "base_dataset_dir": base_dataset_dir
        },
        "hyperparameters": {
            "window_size": 16,
            "robust_window_size": 18
        }
    }

    final_result_file = reconstruct_and_save(config)
    evaluation_output_file = config["paths"]["evaluation_output_file"]
    ground_truth_file = config["paths"]["ground_truth_file"]
    evaluate_and_save_results(ground_truth_file, final_result_file, evaluation_output_file)

    plot_confusion_matrix_from_txt(evaluation_output_file, label_mapping, os.path.join(config["paths"]["results_folder"],"confusion_matrix.png"))

label_mapping = {
    0: "ONE",
    1: "TWO",
    2: "THREE",
    3: "FOUR",
    4: "OK",
    5: "MENU",
    6: "LEFT",
    7: "RIGHT",
    8: "CIRCLE",
    9: "V",
    10: "CROSS",
    11: "GRAB",
    12: "PINCH",
    13: "DENY",
    14: "WAVE",
    15: "KNOB"
}
NO_ACTION_LABEL = 16


def get_sequence_info(seq_info_folder):
    """
    Read sequence information from text files and return list of (sequence_id, frame_count) tuples.
    """
    sequence_info = []
    files = sorted(
        [f for f in os.listdir(seq_info_folder) if f.endswith('.txt') and os.path.splitext(f)[0].isdigit()],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    for f in files:
        file_path = os.path.join(seq_info_folder, f)
        with open(file_path, 'r') as fp:
            lines = fp.read().strip().splitlines()
        frame_count = len(lines)
        sequence_info.append((os.path.splitext(f)[0], frame_count))
    return sequence_info


def load_window_predictions(pred_file):
    """
    Load window-level predictions from file and return as integer list.
    """
    with open(pred_file, 'r') as f:
        content = f.read().strip()
    preds = [int(x) for x in content.split()]
    return preds


def post_processed_segmentation_result(preds, robust_window_size, label_mapping):
    """
    Extract gesture segments from frame-level predictions using robust window voting.

    Parameters:
        preds (np.ndarray): Padded window prediction array
        robust_window_size (int): Sliding window size for voting
        label_mapping (dict): Class index to string name mapping

    Returns:
        str: Predicted gesture segments in string format
             Format: "LABEL;start;end;delay;" concatenated multiple times
    """
    result = ""
    start = 0
    lab = NO_ACTION_LABEL
    num_class = len(label_mapping)
    queue = [-1] * num_class

    for window_index in range(len(preds)):
        window = preds[window_index: window_index + robust_window_size]

        if window.size == 0:
            break

        bc = np.bincount(window.astype(np.int64), minlength=num_class + 1)
        most_seen_label = int(bc.argmax())

        if most_seen_label != NO_ACTION_LABEL:
            if lab == NO_ACTION_LABEL:
                start = window_index + robust_window_size // 2
            lab = most_seen_label
            queue[lab] += 1

        if most_seen_label == NO_ACTION_LABEL and lab != NO_ACTION_LABEL:
            chosen_label = int(np.argmax(queue))
            result += f"{label_mapping[chosen_label]};" \
                      f"{start};" \
                      f"{window_index + robust_window_size // 2};" \
                      f"{start + robust_window_size // 2};"
            lab = NO_ACTION_LABEL
            queue = [-1] * num_class

    return result


def save_to_file(filepath, lines):
    """
    Save lines to file and print completion message.
    """
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved: {filepath}")


def reconstruct_and_save(config):
    """
    Reconstruct and post-process prediction results using config parameters.
    """
    base_model_dir = config["paths"]["base_model_dir"]
    base_dataset_dir = config["paths"]["base_dataset_dir"]

    seq_info_folder = os.path.join(base_dataset_dir, "test_set")
    window_pred_file = os.path.join(base_model_dir, "predictions.txt")

    create_predictions_txt_if_missing(base_model_dir, window_pred_file)

    results_folder = os.path.join(base_model_dir, "post_process_results")
    ground_truth_file = os.path.join(base_dataset_dir, "test_set", "annotations.txt")
    evaluation_output_file = os.path.join(results_folder, "evaluation_results.txt")

    config["paths"]["seq_info_folder"] = seq_info_folder
    config["paths"]["window_pred_file"] = window_pred_file
    config["paths"]["results_folder"] = results_folder
    config["paths"]["ground_truth_file"] = ground_truth_file
    config["paths"]["evaluation_output_file"] = evaluation_output_file

    window_size = config["hyperparameters"]["window_size"]

    all_window_preds = load_window_predictions(window_pred_file)
    sequence_info = get_sequence_info(seq_info_folder)

    predictions_by_seq = {}
    current_index = 0
    for seq_id, frame_count in sequence_info:
        num_window_preds = frame_count - window_size
        predictions_by_seq[seq_id] = all_window_preds[current_index: current_index + num_window_preds]
        current_index += num_window_preds

    post_processed_results = {}
    final_results = {}

    for seq_id, frame_count in sequence_info:
        window_preds = predictions_by_seq[seq_id]
        half_window = window_size // 2

        padded_preds = np.concatenate((
            np.full(half_window, NO_ACTION_LABEL),
            np.array(window_preds),
            np.full(half_window, NO_ACTION_LABEL)
        ))
        post_proc_seg_str = post_processed_segmentation_result(
            padded_preds,
            config["hyperparameters"]["robust_window_size"],
            label_mapping=label_mapping,
        )
        post_processed_results[seq_id] = post_proc_seg_str
        final_results[seq_id] = post_proc_seg_str

    os.makedirs(results_folder, exist_ok=True)
    final_lines = [f"{seq_id};{seg_str}" for seq_id, seg_str in final_results.items()]
    post_processing_results_dir = os.path.join(results_folder, "post_processing_results.txt")
    save_to_file(post_processing_results_dir, final_lines)
    return post_processing_results_dir

def create_predictions_txt_if_missing(base_model_dir, window_pred_file):
    """
    Create predictions.txt file from .npy file if it doesn't exist.
    """
    if not os.path.exists(window_pred_file):
        npy_files = [f for f in os.listdir(base_model_dir) if f.lower().endswith(".npy")]
        if not npy_files:
            raise FileNotFoundError("predictions.txt file not found and no .npy file found in base_model_dir.")
        npy_file_path = os.path.join(base_model_dir, npy_files[0])
        data = np.load(npy_file_path)
        np.savetxt(window_pred_file, [data], fmt="%s", delimiter=" ")
        print(f"{window_pred_file} file created from {npy_file_path}.")

if __name__ == "__main__":
    main()
