import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

from set_seed import set_seed
set_seed(42)

def plot_confusion_matrix(all_ground_truth_val, all_predicted_val):

    # 正解・予測ラベルの１次元のNumpy配列
    all_true_labels = np.concatenate(all_ground_truth_val) # 正解ラベルのリストのリストの、要素のリストを順に結合して、１次元のNumpy配列にする
    all_pred_labels = np.concatenate(all_predicted_val)    # 予測ラベルのリストのリストの、要素のリストを順に結合して、１次元のNumpy配列にする

    # 最終的な混同行列を計算
    final_cm = confusion_matrix(all_true_labels, all_pred_labels) # sklearn.metrics.confusion_matrix()関数　引数：正解ラベル, 予測ラベル

    # 混同行列をヒートマップで表示
    plt.figure(figsize=(4, 3)) # figure()：新しい図の作成　figsize=(　, )：作成する図の(横幅, 縦幅)
    sns.heatmap(
        final_cm,     # 混同行列のデータ
        annot=True,   # セルに数字を表示するか
        fmt='d',      # 数字は整数
        cmap='Blues', # 色は青系
        xticklabels=['Defect', 'Normal'], # x軸のメモリラベル
        yticklabels=['Defect', 'Normal']  # y軸のメモリラベル
        )
    plt.xlabel('Predicted Label') # x軸のラベル
    plt.ylabel('True Label')      # y軸のラベル
    plt.title('Final Confusion Matrix (Aggregated Across All Folds)') # グラフのタイトル
    plt.show() # 表示

    print("\nFinal Confusion Matrix (Aggregated Across All Folds):")
    print(final_cm)

    # 欠陥あり、なし、ごとの評価指標　苦手なクラスはないかの確認
    print("\nClassification Report (Aggregated Across All Folds):")
    print(
        classification_report(             # 分類モデルの性能をクラスごとに評価　sklearn.metricsの関数
        all_true_labels,                   # 正解ラベルのNumpy配列
        all_pred_labels,                   # 予測ラベルのNumpy配列
        target_names=['Defect', 'Normal'], # クラスのラベルの指定
        zero_division=0                    # 分母０対策
        )
    )