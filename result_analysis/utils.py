from pathlib import Path
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_csv_by_task(folder_path, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    regression_data = defaultdict(list)
    classification_data = defaultdict(list)
    regression_group_data = defaultdict(list)
    classification_group_data = defaultdict(list)
    regression_model_data = defaultdict(list)
    classification_model_data = defaultdict(list)

    folder_path = Path(folder_path)
    for csv_file in folder_path.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    task_type = row.get("task", "").lower()
                    model = row.get("model", "unknown")
                    dataset = row.get("name", "unknown")
                    target = row.get("target", "unknown")
                    key = f"{model}::{dataset}::{target}"
                    group_key = f"{model}::{dataset}"
                    model_key = model

                    pred = float(row["prediction"]) if row.get("prediction") not in [None, "", "null"] else None
                    truth = float(row["truth"]) if row.get("truth") not in [None, "", "null"] else None

                    if pred is None or truth is None:
                        continue

                    if task_type == "regression":
                        regression_data[key].append((truth, pred))
                        regression_group_data[group_key].append((truth, pred))
                        regression_model_data[model_key].append((truth, pred))
                    elif "classification" in task_type:
                        classification_data[key].append((truth, pred))
                        classification_group_data[group_key].append((truth, pred))
                        classification_model_data[model_key].append((truth, pred))

                except Exception as e:
                    print(f"⚠️ Skipping bad row in {csv_file}: {row} ({e})")
                    continue

    def plot_regression(values, save_path, title_suffix=""):
        truths, preds = zip(*values)
        mse = mean_squared_error(truths, preds)
        mae = mean_absolute_error(truths, preds)
        r2 = r2_score(truths, preds)
        count = len(truths)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=truths, y=preds, alpha=0.6, edgecolor='w', linewidth=0.5)
        min_val = min(min(truths), min(preds))
        max_val = max(max(truths), max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predictions", fontsize=12)
        plt.title(f"{title_suffix} MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}, N={count}", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved regression plot: {save_path}")

    def plot_classification(values, save_path, title_suffix=""):
        truths, preds = zip(*values)
        y_numeric = np.array(truths)
        pred_probs = np.array(preds)
        pred_labels = (pred_probs >= 0.5).astype(int)
        correct = y_numeric == pred_labels
        jitter = np.random.normal(0, 0.05, size=len(y_numeric))

        class_0_mask = y_numeric == 0
        class_1_mask = y_numeric == 1
        acc_0 = np.mean(correct[class_0_mask]) if np.any(class_0_mask) else 0.0
        acc_1 = np.mean(correct[class_1_mask]) if np.any(class_1_mask) else 0.0
        n_0 = np.sum(class_0_mask)
        n_1 = np.sum(class_1_mask)

        plt.figure(figsize=(10, 6))
        unique_classes = np.unique(y_numeric)
        if len(unique_classes) == 1:
            plt.scatter(
                x=y_numeric + jitter,
                y=pred_probs,
                c='green' if unique_classes[0] == 1 else 'blue',
                alpha=0.7,
                edgecolors='w'
            )
        else:
            plt.scatter(
                x=y_numeric + jitter,
                y=pred_probs,
                c=correct,
                cmap='coolwarm',
                alpha=0.7,
                edgecolors='w'
            )

        xticks_labels = []
        if n_0 > 0:
            xticks_labels.append(f'Class 0 (Acc: {acc_0:.2f}, N={n_0})')
        if n_1 > 0:
            xticks_labels.append(f'Class 1 (Acc: {acc_1:.2f}, N={n_1})')
        plt.xticks([0, 1][:len(xticks_labels)], xticks_labels)
        plt.ylabel("Prediction Confidence", fontsize=12)
        plt.title(f"{title_suffix} Classification Confidence", fontsize=12)
        plt.colorbar(label="Correct Prediction")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved classification plot: {save_path}")

    for key, values in regression_data.items():
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_regression.png")
        plot_regression(values, save_path, title_suffix=key)

    for key, values in classification_data.items():
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_classification.png")
        plot_classification(values, save_path, title_suffix=key)

    for key, values in regression_group_data.items():
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_regression_all_targets.png")
        plot_regression(values, save_path, title_suffix=key)

    for key, values in classification_group_data.items():
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_classification_all_targets.png")
        plot_classification(values, save_path, title_suffix=key)

    for key, values in regression_model_data.items():
        save_path = os.path.join(save_dir, f"{key}_regression_all_datasets.png")
        plot_regression(values, save_path, title_suffix=key)

    for key, values in classification_model_data.items():
        save_path = os.path.join(save_dir, f"{key}_classification_all_datasets.png")
        plot_classification(values, save_path, title_suffix=key)
