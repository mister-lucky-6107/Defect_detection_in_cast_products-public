#import torch
import numpy as np

def segmentation_metrics(pred, target, num_classes):
    """
    pred: [N, H, W] 予測ラベル（argmax済み）
    target: [N, H, W] 正解ラベル
    num_classes: クラス数
    """
    ious = []
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)

        if (pred_sum + target_sum) == 0:
            dice = float('nan')
        else:
            dice = 2 * intersection / (pred_sum + target_sum)
        dices.append(dice)

    return {
        "mIoU": np.nanmean(ious),
        "mDice": np.nanmean(dices),
        "IoU_per_class": ious,
        "Dice_per_class": dices,
    }

# 例: モデル出力
#pred = torch.randint(0, 3, (4, 256, 256))   # 予測ラベル (batch=4, 3クラス)
#target = torch.randint(0, 3, (4, 256, 256)) # 正解ラベル

#metrics = segmentation_metrics(pred, target, num_classes=3)
#print(metrics)
