import numpy as np
import os

def evaluate_and_save_results(gt_file_path, result_file_path, output_file_path):
    """
    Evaluate gesture recognition results and save metrics to output file.

    gt_file_path: Ground truth annotation file path.
    result_file_path: Algorithm result file path.
    output_file_path: Output file path for evaluation results (Compact_Results).

    This function reads each line (each sequence) from both files and calculates
    total count, correct detections, missed, FP, and delay for each class in the same way as MATLAB code.
    Finally, it calculates Detection Rate, FP Rate, Jaccard Index, average Delay for each class,
    and overall averages in a 17 x 4 result matrix (Compact_Results) and saves to output file and console.
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    max_dim_ratio = 2
    min_overlap_ratio = 0.5

    class_labels = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU", "LEFT", "RIGHT",
                    "CIRCLE", "V", "CROSS", "GRAB", "PINCH", "DENY", "WAVE", "KNOB"]

    raw_results = {label: [0, 0, 0, 0, 0] for label in class_labels}

    with open(gt_file_path, 'r') as gt_file:
        gt_lines = gt_file.readlines()
    with open(result_file_path, 'r') as res_file:
        result_lines = res_file.readlines()

    for seq_idx in range(len(gt_lines)):
        gt_line = gt_lines[seq_idx].strip()
        result_line = result_lines[seq_idx].strip()

        gt_parts = [x for x in gt_line.split(';') if x != '']
        result_parts = [x for x in result_line.split(';') if x != '']

        for gt_idx in range(1, len(gt_parts), 3):
            if gt_idx + 2 < len(gt_parts):
                gt_label = gt_parts[gt_idx]
                raw_results[gt_label][0] += 1

        for res_idx in range(1, len(result_parts), 4):
            if res_idx + 3 < len(result_parts):
                res_label = result_parts[res_idx]
                res_start = float(result_parts[res_idx + 1])
                res_end = float(result_parts[res_idx + 2])
                delay_value = float(result_parts[res_idx + 3]) - float(result_parts[res_idx + 1])
                match_found = False
                for gt_idx in range(1, len(gt_parts), 3):
                    if gt_idx + 2 < len(gt_parts):
                        gt_label = gt_parts[gt_idx]
                        gt_start = float(gt_parts[gt_idx + 1])
                        gt_end = float(gt_parts[gt_idx + 2])
                        if not match_found:
                            if gt_label == res_label:
                                if (res_end - res_start) <= (gt_end - gt_start) * max_dim_ratio:
                                    overlap = (min(gt_end, res_end) - max(gt_start, res_start)) / (gt_end - gt_start)
                                    if overlap >= min_overlap_ratio:
                                        gt_parts[gt_idx] = '-'
                                        gt_parts[gt_idx + 1] = '0'
                                        gt_parts[gt_idx + 2] = '0'
                                        raw_results[gt_label][1] += 1
                                        raw_results[gt_label][4] += delay_value
                                        match_found = True
                                    else:
                                        if (min(gt_end, res_end) - max(gt_start, res_start)) / (gt_end - gt_start) > 0:
                                            res_label = ''
                if not match_found:
                    if res_label != '':
                        raw_results[res_label][3] += 1

        for gt_idx in range(1, len(gt_parts), 3):
            if gt_idx < len(gt_parts) and gt_parts[gt_idx] != '-':
                raw_results[gt_parts[gt_idx]][2] += 1

    results_array = np.zeros((16, 5))
    for i, label in enumerate(class_labels):
        results_array[i, 0] = raw_results[label][0]
        results_array[i, 1] = raw_results[label][1]
        results_array[i, 2] = raw_results[label][2]
        results_array[i, 3] = raw_results[label][3]
        results_array[i, 4] = raw_results[label][4]

    for i in range(16):
        if results_array[i, 1] > 0:
            results_array[i, 4] = results_array[i, 4] / results_array[i, 1]
        else:
            results_array[i, 4] = np.nan

    compact_results = np.zeros((17, 4))
    for i in range(16):
        if results_array[i, 0] > 0:
            compact_results[i, 0] = results_array[i, 1] / results_array[i, 0]
            compact_results[i, 1] = results_array[i, 3] / results_array[i, 0]
        else:
            compact_results[i, 0] = np.nan
            compact_results[i, 1] = np.nan
        denominator = results_array[i, 1] + results_array[i, 2] + results_array[i, 3]
        if denominator > 0:
            compact_results[i, 2] = results_array[i, 1] / denominator
        else:
            compact_results[i, 2] = np.nan
        compact_results[i, 3] = results_array[i, 4]

    compact_results[16, 0] = np.nanmean(compact_results[0:16, 0])
    compact_results[16, 1] = np.nanmean(compact_results[0:16, 1])
    compact_results[16, 2] = np.nanmean(compact_results[0:16, 2])
    compact_results[16, 3] = np.nanmean(compact_results[0:16, 3])

    output_lines = []
    for i in range(compact_results.shape[0]):
        line = f"{i + 1}\t: " + "\t".join(f"{val:.6f}" for val in compact_results[i, :])
        output_lines.append(line)

    with open(output_file_path, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")

    for line in output_lines:
        print(line)

    print(f"Evaluation results saved to {output_file_path}.")
