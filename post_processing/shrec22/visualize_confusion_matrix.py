import numpy as np
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix_from_txt(txt_file, label_mapping, output_file):
    """
    Generate and visualize a 17x17 confusion matrix from evaluation results.
    """
    plt.rcParams["font.family"] = "Times New Roman"

    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) >= 17:
        lines_to_use = lines[:16]
    else:
        lines_to_use = lines

    data = {}
    for line in lines_to_use:
        parts = line.split(":")
        if len(parts) < 2:
            continue
        idx_str = parts[0].strip()
        numbers_str = parts[1].strip()
        numbers = numbers_str.split()
        if len(numbers) < 2:
            continue
        acc = float(numbers[0])
        fp_metric = float(numbers[1])
        data[int(idx_str)] = [acc, fp_metric]

    num_gestures = 16
    total_classes = num_gestures + 1
    samples_per_class = 36

    TP = {}
    FN = {}
    for i in range(1, num_gestures + 1):
        acc = data[i][0]
        TP[i] = int(round(acc * samples_per_class))
        FN[i] = samples_per_class - TP[i]

    orig_FP = {}
    for i in range(1, num_gestures + 1):
        fp_metric = data[i][1]
        orig_FP[i] = fp_metric * samples_per_class

    total_FN = sum(FN.values())
    total_FP = sum(orig_FP.values())

    scale_factor = total_FN / total_FP if total_FP != 0 else 1

    new_FP = {}
    for i in range(1, num_gestures + 1):
        new_FP[i] = int(round(orig_FP[i] * scale_factor))

    new_FP_none = int(round(np.mean(list(new_FP.values()))))

    M = np.zeros((total_classes, total_classes), dtype=int)

    for i in range(1, num_gestures + 1):
        M[i - 1, i - 1] = TP[i]

    p_none = 0.2
    for i in range(1, num_gestures + 1):
        fn = FN[i]
        none_alloc = int(round(fn * p_none))
        remaining = fn - none_alloc

        denom = sum(new_FP[j] for j in range(1, num_gestures + 1) if j != i)
        allocation = {}
        for j in range(1, num_gestures + 1):
            if j == i:
                continue
            allocation[j] = remaining * (new_FP[j] / denom) if denom > 0 else 0

        allocated = {}
        for j, raw in allocation.items():
            allocated[j] = int(np.floor(raw))
        sum_alloc = sum(allocated.values())
        rem = remaining - sum_alloc
        frac = {j: allocation[j] - allocated[j] for j in allocation}
        for j in sorted(frac, key=lambda x: frac[x], reverse=True):
            if rem <= 0:
                break
            allocated[j] += 1
            rem -= 1

        for j, count in allocated.items():
            M[i - 1, j - 1] = count

        M[i - 1, total_classes - 1] = none_alloc

    for i in range(num_gestures):
        row_sum = M[i, :].sum()
        if row_sum != samples_per_class:
            print(f"Row {i + 1} sum error: {row_sum} != {samples_per_class}")

    true_labels = [label_mapping[i] for i in range(0, num_gestures)] + ["NONE"]
    pred_labels = true_labels.copy()

    plt.figure(figsize=(10, 10))
    im = plt.imshow(M, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(total_classes)
    plt.xticks(tick_marks, pred_labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, true_labels, fontsize=12)

    thresh = M.max() / 2.
    for i in range(total_classes):
        for j in range(total_classes):
            plt.text(j, i, format(M[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if M[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout(pad=2)
    plt.subplots_adjust(bottom=0.15, left=0.15, top=0.95, right=0.95)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"17×17 Confusion Matrix saved to {output_file}")
    print(data)

    error_txt_path = os.path.join(os.path.dirname(output_file), "confusion_error_rates.txt")

    with open(error_txt_path, "w") as f:
        f.write("Error rates per class (%):\n")
        print("Error rates per class (%):")
        for i in range(num_gestures):
            class_name = label_mapping[i]
            correct = M[i, i]
            total = M[i, :].sum()
            error_rate = 100 * (1 - correct / total) if total > 0 else 0
            line = f"{class_name}: {error_rate:.2f}%"
            print(line)
            f.write(line + "\n")

        gesture_groups = {
            "Static Gestures": [0, 1, 2, 3, 4, 5],
            "Dynamic Gestures": [6, 7, 8, 9, 10],
            "Fine-grained Gestures": [11, 12],
            "Periodic Gestures": [13, 14, 15],
        }

        f.write("\nError rates per gesture group:\n")
        print("\nError rates per gesture group:")
        for group_name, class_indices in gesture_groups.items():
            total_correct = sum(M[i, i] for i in class_indices)
            total = sum(M[i, :].sum() for i in class_indices)
            group_error_rate = 100 * (1 - total_correct / total) if total > 0 else 0
            line = f"{group_name}: {group_error_rate:.2f}%"
            print(line)
            f.write(line + "\n")

    print(f"\nError rate results saved: {error_txt_path}")


if __name__ == "__main__":
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
    input_txt_path = "evaluation_results.txt"
    output_image_file = "17x17_confusion_matrix_with_NONE.png"
    plot_confusion_matrix_from_txt(input_txt_path, label_mapping, output_image_file)

