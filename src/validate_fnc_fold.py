import numpy as np
import torch

from set_seed import set_seed
set_seed(42)

def validate_fnc_fold(net, val_loader_fold):
    # 検証2（フォールドレベル）
      # フォールドのすべての検証データに対する予測ラベルと正解ラベルの保存
      # フォールドごとの詳細な分析（混同行列の作成など）、フォールドごとの特性の理解、モデルの安定性や汎化性能の分析
    net.eval() # 評価モード

    current_predicted_val = []     # 予測結果の一時保存
    current_ground_truth_val = []  # 正解ラベルの一時保存
    misclassified_in_fold = []     # フォールドで誤分類した画像の情報（パス、予測ラベル、正解ラベル）
    current_probabilities_val = [] # 予測確率の一時保存 # ROC曲線表示用

    img_np = None
    gt_mask_np = None
    pred_mask_np = None

    # withブロック内で勾配計算は無効
    with torch.no_grad(): 

        for input_images, class_labels, seg_mask, path in val_loader_fold:

            input_images, class_labels, seg_mask = input_images.cuda(), class_labels.cuda(), seg_mask.cuda() 
            outputs_class_val_fold, outputs_seg_val_fold = net(input_images)       # outputs_seg_val_fold：(B, 7, H, W)

            probabilities = torch.softmax(outputs_class_val_fold, dim=1) # 各クラスの予測確率を取得 # ROC曲線表示用
            _, predicted = torch.max(outputs_class_val_fold.data, 1)

            # 誤分類があったバッチ内のインデックスを取得
            misclassified_mask = (predicted != class_labels).cpu().numpy()
            misclassified_indices = np.where(misclassified_mask)[0]

            # 誤分類された画像の情報（パス、予測ラベル、正解ラベル）を保存
            for index in misclassified_indices:
                misclassified_info = {
                    'path': path[index],
                    'predicted_label': predicted[index].item(),
                    'ground_truth_label': class_labels[index].item()
                }
                misclassified_in_fold.append(misclassified_info)


            # 誤分類があった場合、その画像を保存
            if img_np is None and len(misclassified_indices) > 0:
                idx = misclassified_indices[0]
                img_np = input_images[idx].detach().cpu().permute(1, 2, 0).numpy()
                pred_seg_label = torch.argmax(outputs_seg_val_fold, dim=1)  # shape: [B, H, W] セグメンテーション出力 → クラスラベル（0〜6）に変換
                gt_mask_np = seg_mask[idx].detach().cpu().numpy()
                pred_mask_np = pred_seg_label[idx].detach().cpu().numpy()

                print(f"可視化用に保存した誤分類画像のパス: {path[idx]}")


            current_predicted_val.extend(predicted.cpu().numpy())         # 予測ラベル
            current_ground_truth_val.extend(class_labels.cpu().numpy())   # 正解ラベル
            current_probabilities_val.extend(probabilities.cpu().numpy()) # 予測確率

    return current_predicted_val, current_ground_truth_val, misclassified_in_fold, current_probabilities_val, img_np, gt_mask_np, pred_mask_np