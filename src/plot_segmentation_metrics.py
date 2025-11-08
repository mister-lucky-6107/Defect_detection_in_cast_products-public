import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation_metrics(num_epochs, all_mIoU_val, all_mDice_val):
    """
    セグメンテーション精度（mIoU、mDice）のエポックごとの推移をプロットする
    
    Parameters:
    -----------
    num_epochs : int
        エポック数
    all_mIoU_val : list of list
        各フォールドのmIoUの履歴 [[fold1のmIoU履歴], [fold2のmIoU履歴], ...]
    all_mDice_val : list of list
        各フォールドのmDiceの履歴 [[fold1のmDice履歴], [fold2のmDice履歴], ...]
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # mIoUのプロット
    for fold_idx, mIoU_history in enumerate(all_mIoU_val):
        epochs = range(1, len(mIoU_history) + 1)
        axes[0].plot(epochs, mIoU_history, label=f'Fold {fold_idx + 1}', marker='o', markersize=3)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('mIoU')
    axes[0].set_title('Mean Intersection over Union (mIoU) per Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # mDiceのプロット
    for fold_idx, mDice_history in enumerate(all_mDice_val):
        epochs = range(1, len(mDice_history) + 1)
        axes[1].plot(epochs, mDice_history, label=f'Fold {fold_idx + 1}', marker='o', markersize=3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mDice')
    axes[1].set_title('Mean Dice Coefficient (mDice) per Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最終エポックの平均値を表示
    final_mIoU = np.mean([fold_mIoU[-1] for fold_mIoU in all_mIoU_val])
    final_mDice = np.mean([fold_mDice[-1] for fold_mDice in all_mDice_val])
    
    print(f"\n=== セグメンテーション精度（最終エポックの平均） ===")
    print(f"Average mIoU: {final_mIoU:.4f}")
    print(f"Average mDice: {final_mDice:.4f}")

