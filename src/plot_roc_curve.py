from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

from set_seed import set_seed
set_seed(42)

def plot_roc_curve(all_probabilities_val, all_ground_truth_val):

    # ROC曲線の表示
    plt.figure(figsize=(8, 6))
    for i in range(len(all_probabilities_val)):
        fold_probabilities = np.array(all_probabilities_val[i])
        fold_ground_truth = np.array(all_ground_truth_val[i])

        # クラス1の確率を取得
        if fold_probabilities.ndim > 1 and fold_probabilities.shape[1] > 1:
            y_prob = fold_probabilities[:, 1]
        else:
            y_prob = fold_probabilities # 確率が1次元の場合

        fpr, tpr, thresholds = roc_curve(fold_ground_truth, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Cross-Validation)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()