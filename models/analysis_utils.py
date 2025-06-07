import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_regression_scatter(true_values, pred_values):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=true_values, y=pred_values, alpha=0.6,
    edgecolor='w', linewidth=0.5)
    plt.plot([min(true_values), max(true_values)],
             [min(true_values), max(true_values)],
             'r--', lw=2)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title('Regression Prediction Accuracy', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

# 将标签转换为数值
def plot_classification_scatter(true_labels, pred_probs):
    # 假设二分类
    y_numeric = np.where(true_labels == true_labels[0], 0, 1) 
    # 添加水平抖动
    jitter = np.random.normal(0, 0.05, size=len(y_numeric))

    plt.figure(figsize=(10,6))
    scatter = plt.scatter(x=y_numeric + jitter,
                          y=pred_probs,
                          c=np.array(true_labels) == np.round(pred_probs),
                          cmap='coolwarm',
                          alpha=0.7,
                          edgecolors='w')
    
    plt.xticks([0,1], [f'Class {true_labels[0]}', f'Class {true_labels[-1]}'])
    plt.ylabel('Prediction Confidence', fontsize=12)
    plt.title('Classification Confidence Distribution', fontsize=14)
    plt.colorbar(scatter, label='Correct Prediction')
    plt.grid(axis='y', alpha=0.3)
    plt.show()