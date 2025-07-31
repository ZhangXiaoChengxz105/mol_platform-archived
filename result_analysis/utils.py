# ✅ Final complete version of plot_csv_by_task
# Includes classification + regression plotting, metric summaries, NaN filtering, 
# and layout fixes (no overlap with colorbar)

# Final version of plot_csv_by_task with classification and regression support
# This file includes metric annotation for both individual and aggregate tasks.
# Saved automatically by ChatGPT on your request.


import os
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics as skm
import numpy as np
import inspect
from matplotlib.ticker import MaxNLocator
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')




def get_save_path(base_dir, model, dataset, filename, is_model_analysis=False):
    if is_model_analysis:
        subfolder = os.path.join(base_dir, model, "analysis")
    else:
        subfolder = os.path.join(base_dir, model, dataset)
    os.makedirs(subfolder, exist_ok=True)
    return os.path.join(subfolder, filename)

def finalize_plot_classification_with_metrics(
    ax, y_true, y_pred_prob, classification_metrics,
    model_dataset_metrics_classification, model, dataset, target, threshold=0.5
):
    # 转为 numpy 数组，过滤 nan
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred_prob))
    y_true = y_true[valid_mask]
    y_pred_prob = y_pred_prob[valid_mask]

    if len(y_true) == 0:
        ax.text(
            0.01, 0.99, "No valid data",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        return

    y_pred = (y_pred_prob >= threshold).astype(int)

    text_lines = []

    for metric_name in classification_metrics:
        metric_func = getattr(skm, metric_name, None)
        if callable(metric_func):
            try:
                # 判断函数参数，是否有 y_score（概率）参数
                sig = inspect.signature(metric_func)
                if 'y_score' in sig.parameters or 'probas_pred' in sig.parameters:
                    score = metric_func(y_true, y_pred_prob)
                else:
                    score = metric_func(y_true, y_pred)
            except Exception:
                score = float('nan')
            model_dataset_metrics_classification[model][dataset][target][f"Macro {metric_name}"] = score
            text_lines.append(f"{metric_name}: {score:.3f}")
        else:
            text_lines.append(f"{metric_name}: N/A")

    ax.text(
        0.01, 0.99, "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    return model_dataset_metrics_classification

def finalize_plot_macro_metrics(classification_tasks, task_outputs, threshold=0.5):
    import numpy as np
    import sklearn.metrics as skm

    metrics_results = {metric: [] for metric in classification_tasks}

    for y_true_raw, y_pred_prob_raw in task_outputs:
        y_true = np.array(y_true_raw)
        y_pred_prob = np.array(y_pred_prob_raw)

        valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred_prob))
        y_true = y_true[valid_mask]
        y_pred_prob = y_pred_prob[valid_mask]

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            continue

        y_pred = (y_pred_prob >= threshold).astype(int)

        for metric_name in classification_tasks:
            metric_func = getattr(skm, metric_name, None)
            if callable(metric_func):
                try:
                    sig = inspect.signature(metric_func)
                    if 'y_score' in sig.parameters or 'probas_pred' in sig.parameters:
                        score = metric_func(y_true, y_pred_prob)
                    else:
                        score = metric_func(y_true, y_pred)
                except Exception as e:
                    print(f"⚠️ Failed to compute metric {metric_name}: {e}")
                    score = float('nan')
            else:
                print(f"⚠️ Metric {metric_name} not found in sklearn.metrics")
                score = float('nan')

            metrics_results[metric_name].append(score)

    def safe_mean(lst):
        return np.nanmean(lst) if lst else np.nan

    macro_metrics = {}
    for metric_name in classification_tasks:
        pretty_name = metric_name.replace("_score", "").replace("_", " ").capitalize()
        macro_metrics[f"Macro {pretty_name}"] = safe_mean(metrics_results[metric_name])

    return macro_metrics

def plot_macro_metrics_on_ax(ax, metrics_dict):
    text = "\n".join([f"{k.split()[-1]}: {v:.3f}" for k, v in metrics_dict.items()])
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

def update_classification_aggregates(classification_tasks,classification_group_data, classification_model_data,
                                     classification_data, save_dir, get_save_path):
    
    def compute_and_plot_macro(classification_tasks,related_keys, title, save_path, write_plot=True):
        #这是
        task_outputs = [
            (list(zip(*classification_data[k])) if len(classification_data[k]) > 0 else ([], []))
            for k in related_keys
        ]
        metrics = finalize_plot_macro_metrics(classification_tasks,task_outputs)

        # ✅ 写入 metrics dict
        # if dataset == "all":
        #     model_dataset_metrics_dict[model]["all"] = metrics
        # else:
        #     model_dataset_metrics_dict[model][dataset] = metrics

        # ✅ 可选绘图（只有多个任务时才画）
        if not write_plot:
            return

        truths_preds = [pair for k in related_keys for pair in classification_data[k]]
        truths, preds = zip(*truths_preds)
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
            color = 'green' if unique_classes[0] == 1 else 'blue'
            plt.scatter(x=y_numeric + jitter, y=pred_probs, c=color, alpha=0.7, edgecolors='w')
        else:
            plt.scatter(x=y_numeric + jitter, y=pred_probs, c=correct, cmap='coolwarm', alpha=0.7, edgecolors='w')

        xticks_labels = []
        if n_0 > 0:
            xticks_labels.append(f'Class 0 (Acc: {acc_0:.2f}, N={n_0})')
        if n_1 > 0:
            xticks_labels.append(f'Class 1 (Acc: {acc_1:.2f}, N={n_1})')
        plt.xticks([0, 1][:len(xticks_labels)], xticks_labels)
        plt.ylabel("Prediction Confidence", fontsize=12)
        plt.title(f"{title}", fontsize=12)
        plt.colorbar(label="Correct Prediction")
        plt.grid(axis='y', alpha=0.3)
        plot_macro_metrics_on_ax(plt.gca(), metrics)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Saved macro avg plot: {save_path}")

    # ✅ Per-dataset
    for key, values in classification_group_data.items():
        model, dataset = key.split("::")
        related_keys = [k for k in classification_data if k.startswith(f"{model}::{dataset}::")]
        save_path = get_save_path(save_dir, model, dataset, f"{dataset}_classification_all_targets.png")
        do_plot = len(set(k.split("::")[2] for k in related_keys)) > 1
        compute_and_plot_macro(classification_tasks,related_keys, f"{key} (Macro Avg)", save_path, write_plot=do_plot)

    # ✅ Per-model
    for key, values in classification_model_data.items():
        model = key
        related_keys = [k for k in classification_data if k.startswith(f"{model}::")]
        save_path = get_save_path(save_dir, model, None, f"{model}_classification_all_datasets.png", is_model_analysis=True)
        do_plot = len(set(k.split("::")[1] for k in related_keys)) > 1
        compute_and_plot_macro(classification_tasks,related_keys, f"{model} (All Datasets Macro Avg)", save_path, write_plot=do_plot)

    # return model_dataset_metrics_dict

def plot_csv_by_task(folder_path,regression_tasks,classification_tasks, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    print(regression_tasks)
    regression_data = defaultdict(list)
    classification_data = defaultdict(list)
    regression_group_data = defaultdict(list)
    classification_group_data = defaultdict(list)
    regression_model_data = defaultdict(list)
    classification_model_data = defaultdict(list)
    model_dataset_metrics_classification = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    model_dataset_metrics_regression = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


    folder_path = Path(folder_path)
    run_dirs = sorted(
    [d for d in folder_path.iterdir() if d.is_dir() and re.match(r"run\d+", d.name)],
    key=lambda d: int(re.findall(r"\d+", d.name)[0]),
    reverse=True
    )
    print(run_dirs)
    if not run_dirs:
        for csv_file in folder_path.glob("*.csv"):
            if "error" in csv_file.name.lower():
                continue
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
                        elif "classification" in task_type:
                            classification_data[key].append((truth, pred))
                            # classification_group_data[group_key].append((truth, pred))
                            # classification_model_data[model_key].append((truth, pred))
                    except Exception as e:
                        print(f"⚠️ Skipping bad row in {csv_file}: {row} ({e})")
                        continue
    else:
        all_paths =[]
        for run_dir in run_dirs:
            for csv_file in run_dir.glob("*.csv"):
                if "error" in csv_file.name.lower():
                    continue
                if csv_file.name in all_paths:
                    continue
                else:
                    all_paths.append(csv_file.name)
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
                            elif "classification" in task_type:
                                classification_data[key].append((truth, pred))
                                # classification_group_data[group_key].append((truth, pred))
                                # classification_model_data[model_key].append((truth, pred))
                        except Exception as e:
                            print(f"⚠️ Skipping bad row in {csv_file}: {row} ({e})")
                            continue

    for key, values in classification_data.items():
        model, dataset, target = key.split("::")
        save_path = get_save_path(save_dir, model, dataset, f"{dataset}_{target}_classification.png")
        truths, preds = zip(*values)
        y_numeric = np.array(truths)
        pred_probs = np.array(preds)
        
        original_len = len(y_numeric) #new
        valid_mask = (~np.isnan(y_numeric)) & (~np.isnan(pred_probs))
        y_numeric = y_numeric[valid_mask]
        pred_probs = pred_probs[valid_mask]
        nan_ratio = 1.0 - len(y_numeric) / original_len if original_len > 0 else 1.0

        skip_eval = nan_ratio > 0.5  
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
            color = 'green' if unique_classes[0] == 1 else 'blue'
            plt.scatter(x=y_numeric + jitter, y=pred_probs, c=color, alpha=0.7, edgecolors='w')
        else:
            plt.scatter(x=y_numeric + jitter, y=pred_probs, c=correct, cmap='coolwarm', alpha=0.7, edgecolors='w')

        xticks_labels = []
        if n_0 > 0:
            xticks_labels.append(f'Class 0 (Acc: {acc_0:.2f}, N={n_0})')
        if n_1 > 0:
            xticks_labels.append(f'Class 1 (Acc: {acc_1:.2f}, N={n_1})')
        plt.xticks([0, 1][:len(xticks_labels)], xticks_labels)
        plt.ylabel("Prediction Confidence", fontsize=12)
        plt.title(f"{key} Classification Confidence", fontsize=12)
        plt.colorbar(label="Correct Prediction")
        plt.grid(axis='y', alpha=0.3)
        if skip_eval:
            plt.gca().text(
                0.01, 0.95, "Too many invalid data.\nSkipping evaluation.",
                transform=plt.gca().transAxes,
                fontsize=10,
                color='red',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )
        else:
            model_dataset_metrics_classification = finalize_plot_classification_with_metrics(plt.gca(), y_numeric, pred_probs,classification_tasks,model_dataset_metrics_classification,model,dataset,target)
            classification_group_data[f"{model}::{dataset}"].extend(zip(y_numeric, pred_probs))
            classification_model_data[model].extend(zip(y_numeric, pred_probs))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved single-task classification plot: {save_path}")

    
    # ✅ Single regression task plots
    for key, values in regression_data.items():
        model, dataset, target = key.split("::")
        save_path = get_save_path(save_dir, model, dataset, f"{dataset}_{target}_regression.png")
        should_include, model_dataset_metrics_regression = plot_regression_with_metrics(
        regression_metrics =regression_tasks,
        values=values,
        save_path=save_path,
        title_suffix=key,
        model=model,
        dataset=dataset,
        target=target,
        model_dataset_metrics_regression=model_dataset_metrics_regression,
        )        
        if should_include:
            regression_group_data[f"{model}::{dataset}"].extend(values)
            regression_model_data[model].extend(values)


    # ✅ Grouped regression (multi-target per dataset)
    for key, values in regression_group_data.items():
        model, dataset = key.split("::")
        related_keys = [k for k in regression_data if k.startswith(f"{model}::{dataset}::")]
        task_outputs = [
                (list(zip(*regression_data[k])) if len(regression_data[k]) > 0 else ([], []))
                for k in related_keys
                ]
        metrics = finalize_macro_regression_metrics(task_outputs)
        # model_dataset_metrics_regression[model][dataset] = metrics
        if len(related_keys) > 1:
            save_path = get_save_path(save_dir, model, dataset, f"{dataset}_regression_all_targets.png")
            plot_regression_with_metrics(values, save_path, title_suffix=f"{key} (Macro Avg)",regression_metrics =regression_tasks)

    # ✅ Model-level regression (multi-dataset per model)
    for key, values in regression_model_data.items():
        model = key
        related_keys = [k for k in regression_data if k.startswith(f"{key}::")]
        task_outputs = [
                (list(zip(*regression_data[k])) if len(regression_data[k]) > 0 else ([], []))
                for k in related_keys
                ]
        metrics = finalize_macro_regression_metrics(task_outputs)
        # model_dataset_metrics_regression[model]["all"] = metrics
        if len(set(k.split("::")[1] for k in related_keys)) > 1:
            save_path = get_save_path(save_dir, key, None, f"{key}_regression_all_datasets.png", is_model_analysis=True)
            plot_regression_with_metrics(values, save_path, title_suffix=f"{key} (All Datasets Macro Avg)",regression_metrics =regression_tasks)

    update_classification_aggregates(
        classification_tasks,
        classification_group_data=classification_group_data,
        classification_model_data=classification_model_data,
        classification_data=classification_data,
        save_dir=save_dir,
        get_save_path=get_save_path,
    )   
    model_dataset_metrics_classification,model_dataset_metrics_regression = aggregate_metrics(model_dataset_metrics_classification,model_dataset_metrics_regression)
    plot_analysis_metrics_with_values(regression_tasks, classification_tasks,
        model_dataset_metrics_classification=model_dataset_metrics_classification,
        model_dataset_metrics_regression=model_dataset_metrics_regression,
        save_dir=save_dir
    )


# Regression aggregation utilities

def finalize_macro_regression_metrics(task_outputs):
    mses, maes, r2s = [], [], []
    count = 0

    for truths_raw, preds_raw in task_outputs:
        truths = np.array(truths_raw)
        preds = np.array(preds_raw)

        valid_mask = (~np.isnan(truths)) & (~np.isnan(preds))
        truths = truths[valid_mask]
        preds = preds[valid_mask]

        if len(truths) < 2 or len(np.unique(truths)) < 2:
            continue

        try:
            mses.append(mean_squared_error(truths, preds))
            maes.append(mean_absolute_error(truths, preds))
            r2s.append(r2_score(truths, preds))
            count += len(truths)
        except Exception:
            continue

    def safe_mean(arr): return float(np.nanmean(arr)) if arr else float("nan")

    return {
        "Macro MSE": safe_mean(mses),
        "Macro MAE": safe_mean(maes),
        "Macro R2": safe_mean(r2s),
        "Total Samples": count
    }

def plot_macro_regression_on_ax(ax, metrics_dict):
    text = "\n".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()])
    ax.text(
        1.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )


# Regression plot function with metrics

def plot_regression_with_metrics(values, save_path, title_suffix="", model='',dataset ='',target = '',regression_metrics =None,model_dataset_metrics_regression=''):
    
    if not values:
        print(f"⚠️ Empty input to plot: {save_path}")
        return False

    truths, preds = zip(*values)
    truths = np.array(truths)
    preds = np.array(preds)

    original_len = len(truths)
    valid_mask = (~np.isnan(truths)) & (~np.isnan(preds))
    truths = truths[valid_mask]
    preds = preds[valid_mask]

    valid_len = len(truths)
    nan_ratio = 1.0 - valid_len / original_len if original_len > 0 else 1.0
    skip_eval = nan_ratio > 0.5

    plt.figure(figsize=(8, 6))

    if valid_len == 0 or len(np.unique(truths)) < 2:
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.title(f"{title_suffix} Regression", fontsize=12)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📉 No valid data for regression plot: {save_path}")
        return False,model_dataset_metrics_regression

    # 正常散点图 + 拟合线
    sns.scatterplot(x=truths, y=preds, alpha=0.6, edgecolor='w', linewidth=0.5)
    min_val = min(min(truths), min(preds))
    max_val = max(max(truths), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predictions", fontsize=12)
    plt.title(f"{title_suffix} Regression", fontsize=12)
    plt.grid(alpha=0.3)

    if skip_eval:
        plt.gca().text(
            0.01, 0.95, "Too many invalid data.\nSkipping evaluation.",
            transform=plt.gca().transAxes,
            fontsize=10,
            color='red',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"⚠️ Skipped eval due to invalid data: {save_path}")
        return False,model_dataset_metrics_regression

    # ✅ 强制使用外部提供的指标
    if regression_metrics is not None and isinstance(regression_metrics, list):
        import sklearn.metrics as skm
        metrics = {}
        for metric_name in regression_metrics:
            metric_func = getattr(skm, metric_name, None)
            if callable(metric_func):
                try:
                    score = metric_func(truths, preds)
                except Exception as e:
                    print(f"⚠️ Failed to compute metric {metric_name}: {e}")
                    score = float('nan')
                metrics[f"Macro {metric_name}"] = score
            else:
                print(f"⚠️ Metric {metric_name} not found in sklearn.metrics")
                metrics[f"Macro {metric_name}"] = float('nan')

        # 是否写入结果字典，取决于变量是不是空字符串
        if model_dataset_metrics_regression != "":
            no_count = {k: v for k, v in metrics.items()}
            model_dataset_metrics_regression[model][dataset][target] = no_count

    else:
        # 固定指标逻辑
        mse = mean_squared_error(truths, preds)
        mae = mean_absolute_error(truths, preds)
        r2 = r2_score(truths, preds)
        count = len(truths)
        metrics = {
            "Macro MSE": mse,
            "Macro MAE": mae,
            "Macro R2": r2,
            "Total Samples": count
        }
        no_count = {
            "Macro MSE": mse,
            "Macro MAE": mae,
            "Macro R2": r2
        }

        if model_dataset_metrics_regression != "":
            model_dataset_metrics_regression[model][dataset][target] = no_count
            

    # 处理展示文本（根据是否需要显示 Macro 决定）
    display_macro = (model_dataset_metrics_regression == "")

    lines = []
    for k, v in metrics.items():
        if k == "Total Samples":
            continue
        display_name = k if display_macro else k.replace("Macro ", "")
        lines.append(f"{display_name}: {v:.3f}")

    # 样本数单独放末尾
    total_samples = metrics.get("Total Samples", len(truths))
    lines.append(f"N: {total_samples}")

    # 绘图部分
    text = "  ".join(lines[i] + ("\n" if (i + 1) % 2 == 0 else "  ") for i in range(len(lines)))
    plt.gca().text(
        0.01, 0.99, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 Saved regression plot: {save_path}")

    return True, model_dataset_metrics_regression


def aggregate_metrics(model_dataset_metrics_classification, model_dataset_metrics_regression):
    def average_metrics(metric_list):
        return float(np.mean(metric_list)) if metric_list else float('nan')

    def process_dict(input_dict):
        result = defaultdict(lambda: defaultdict(dict))

        for model, dataset_dict in input_dict.items():
            for dataset, target_dict in dataset_dict.items():
                metrics_accumulator = defaultdict(list)
                for target, metrics in target_dict.items():
                    for metric, value in metrics.items():
                        metrics_accumulator[metric].append(value)

                for metric, values in metrics_accumulator.items():
                    result[model][dataset][metric] = average_metrics(values)

        # 添加 model -> 'all' 聚合所有 dataset
        for model, dataset_metrics in result.items():
            all_metrics_accumulator = defaultdict(list)
            for dataset, metrics in dataset_metrics.items():
                for metric, value in metrics.items():
                    all_metrics_accumulator[metric].append(value)
            for metric, values in all_metrics_accumulator.items():
                result[model]["all"][metric] = average_metrics(values)

        return result

    classification_macro = process_dict(model_dataset_metrics_classification)
    regression_macro = process_dict(model_dataset_metrics_regression)

    return classification_macro, regression_macro

def plot_analysis_metrics_with_values(regression_tasks, classification_tasks,
                                      model_dataset_metrics_classification,
                                      model_dataset_metrics_regression,
                                      save_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    def draw_plot(data_dict, metrics_keys, filename_prefix, ylabel):
        os.makedirs(os.path.join(save_dir, "analysis"), exist_ok=True)
        try:
            for dataset in next(iter(data_dict.values())).keys():
                values_per_model = defaultdict(list)

                for model, dataset_dict in data_dict.items():
                    if dataset not in dataset_dict:
                        continue
                    for metric in metrics_keys:
                        val = dataset_dict[dataset].get(metric, np.nan)
                        values_per_model[model].append(float(val))

                if not values_per_model:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))

                models = list(values_per_model.keys())
                x = np.arange(len(metrics_keys))
                total_width = 0.8
                width = total_width / len(models)
                offsets = np.linspace(-total_width / 2 + width / 2, total_width / 2 - width / 2, len(models))

                for i, model in enumerate(models):
                    vals = values_per_model[model]
                    bar = ax.bar(x + offsets[i], vals, width, label=model)
                    for j, v in enumerate(vals):
                        if np.isnan(v):
                            continue
                        ax.text(x[j] + offsets[i], v + 0.01 * max(vals), f"{v:.3f}",
                                ha='center', va='bottom', fontsize=8)

                ax.set_ylabel(ylabel)
                ax.set_title(f"{dataset} - {filename_prefix}")
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_keys, rotation=45)
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                save_path = os.path.join(save_dir, "analysis", f"analysis_{dataset}_{filename_prefix}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
        except Exception as e:
            print(f"❌ No data to plot for {filename_prefix}: {e}")

    # 分类任务：将 skm 函数名包装为 "Macro ..." 格式
    classification_metrics = [f"Macro {m}" for m in classification_tasks]
    draw_plot(model_dataset_metrics_classification, classification_metrics, "classification", "Score")

    # 回归任务：同样用 "Macro ..." 格式
    regression_metrics = [f"Macro {m}" for m in regression_tasks]
    draw_plot(model_dataset_metrics_regression, regression_metrics, "regression", "Value")