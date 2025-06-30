import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
def plot_regression_scatter(true_values, pred_values):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=true_values, y=pred_values, alpha=0.6, 
        edgecolor='w', linewidth=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', lw=2)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title('Regression Prediction Accuracy', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()
def plot_classification_scatter(true_labels, pred_probs):# 将标签转换为数值
    y_numeric = np.where(true_labels == true_labels[0], 0, 1)  # 假设⼆分类
    pred_labels = np.where(pred_probs >= 0.5, 1, 0)
    correct = y_numeric == pred_labels
    # 添加⽔平抖动
    jitter = np.random.normal(0, 0.05, size=len(y_numeric))
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(x=y_numeric + jitter, 
        y=pred_probs, 
        c=correct,
        cmap='coolwarm',
        alpha=0.7,
        edgecolors='w')
    plt.xticks([0,1], [f'Class {true_labels[0]}', f'Class {true_labels[-1]}'])
    plt.ylabel('Prediction Confidence', fontsize=12)
    plt.title('Classification Confidence Distribution', fontsize=14)
    plt.colorbar(scatter, label='Correct Prediction')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_jsonl_by_task(jsonl_path, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # 用来按 task 类型收集数据
    regression_data = defaultdict(list)
    classification_data = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            task_type = item.get("task", "").lower()
            dataset = item.get("name", "unknown")
            target = item.get("target", "unknown")
            key = f"{dataset}::{target}"

            pred = item.get("prediction")
            truth = item.get("truth")

            if pred is None or truth is None:
                continue

            if task_type == "regression":
                regression_data[key].append((truth, pred))
            elif "classification" in task_type:
                classification_data[key].append((truth, pred))

    # 绘图：回归任务
    for key, values in regression_data.items():
        truths, preds = zip(*values)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=truths, y=preds, alpha=0.6, edgecolor='w', linewidth=0.5)
        min_val = min(min(truths), min(preds))
        max_val = max(max(truths), max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predictions", fontsize=12)
        plt.title(f"Regression Prediction: {key}", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_regression.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved regression plot: {save_path}")

    # 绘图：分类任务
    for key, values in classification_data.items():
        truths, preds = zip(*values)
        y_numeric = np.array(truths)
        pred_probs = np.array(preds)
        pred_labels = (pred_probs >= 0.5).astype(int)
        correct = y_numeric == pred_labels
        jitter = np.random.normal(0, 0.05, size=len(y_numeric))

        plt.figure(figsize=(10,6))
        scatter = plt.scatter(
            x=y_numeric + jitter,
            y=pred_probs,
            c=correct,
            cmap='coolwarm',
            alpha=0.7,
            edgecolors='w'
        )
        plt.xticks([0,1], ['Class 0', 'Class 1'])
        plt.ylabel("Prediction Confidence", fontsize=12)
        plt.title(f"Classification Confidence: {key}", fontsize=14)
        plt.colorbar(scatter, label="Correct Prediction")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{key.replace('::','_')}_classification.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved classification plot: {save_path}")