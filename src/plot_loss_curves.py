import matplotlib.pyplot as plt
import numpy as np

from set_seed import set_seed
set_seed(42)


def plot_loss_curves(num_epochs, all_loss_train, all_loss_val):


    # 損失のグラフ　各フォールドの学習過程
    plt.figure(figsize=(12, 4)) # figure()：新しい図の作成　figsize=(12, 4)：作成する図の(横幅, 縦幅)
    plt.subplot(1, 2, 1)        # suplot()：サブプロットは図の中に複数のグラフを配置　1, 2：図を縦１つ横２つに分割　1：左側の領域をアクティブにする
    for i in range(len(all_loss_train)):
        plt.plot(               # アクティブなサブプロットにグラフを描画
            range(num_epochs),  # x軸の値を生成　エポック数に対応する０～num_epochs-1の数列
            all_loss_train[i],  # y軸の値を生成　i番目のフォールドの訓練データに対する損失
            label=f"Fold {i+1}" # 描画する線にラベルを設定
            )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss per Fold")

    plt.subplot(1, 2, 2)
    for i in range(len(all_loss_val)):
        plt.plot(
            range(num_epochs),
            all_loss_val[i],
            label=f"Fold {i+1}"
            )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss per Fold")

    plt.tight_layout()
    plt.show()