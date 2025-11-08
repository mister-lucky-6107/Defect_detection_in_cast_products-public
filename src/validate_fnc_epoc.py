import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score

from set_seed import set_seed
set_seed(42)

def validate_fnc_epoc(net, val_loader_fold, segmentation_loss_weight):

    # 検証1（エポックレベル） 評価指標の保存　学習の進行に伴うモデルの性能の変化の追跡
    net.eval()
    loss_val_fold = 0              # 「エポックごとの、検証データに対する損失」を累積させる変数
    correct_val_fold = 0           # 検証データに対する正解数をカウントするための変数
    total_val_fold = 0             # 検証データの総数をカウントするための変数
    all_predicted_val_fold = []    # 検証データに対する予測ラベルを格納するためのリスト　あとでまとめて評価するため
    all_ground_truth_val_fold = [] # 検証データに対する正解ラベルを格納するためのリスト

    class_weights = {0: 51, 1: 85, 2: 55, 3: 218, 4: 1, 5: 8, 6: 18}

    # withブロック内で勾配計算は無効
    with torch.no_grad():

        for j, (input_images, class_labels, seg_mask, _) in enumerate(val_loader_fold):

            input_images, class_labels, seg_mask = input_images.cuda(), class_labels.cuda(), seg_mask.cuda()
            outputs_class_val, outputs_seg_val = net(input_images) 

            # 予測結果のテンソル
            _, predicted_val_fold = torch.max(outputs_class_val.data, 1)

            # 分類損失
            loss_cls = nn.CrossEntropyLoss()(outputs_class_val, class_labels)
            
            # セグメンテーション損失
            weight_tensor = torch.ones_like(seg_mask, dtype=torch.float)
            for label, weight in class_weights.items():
                weight_tensor[seg_mask == label] = weight
            loss_raw = nn.functional.cross_entropy(outputs_seg_val, seg_mask, reduction='none')  # shape: [B, H, W]
            loss_seg = (loss_raw * weight_tensor).mean()

            # 総合損失 セグメンテーション損失は、分類損失に比較して重み付けされる
            loss = loss_cls + segmentation_loss_weight * loss_seg

            # 検証の初めに初期化した変数・リストに加算/追加
            loss_val_fold += loss.item()                               # 損失を累積させる変数に加算　.item()でPyTorchのテンソル型からPythonの数値型に変換
            total_val_fold += class_labels.size(0)                                # バッチのデータの数（正解ラベルのテンソルの最初の次元の要素数）を「検証データの総数をカウントするための変数」に加算
            correct_val_fold += (predicted_val_fold == class_labels).sum().item() # Trueが1として合計されたテンソルからPythonの数値型を取り出し、「検証データに対する正解数をカウントするための変数」に加算
            all_predicted_val_fold.extend(predicted_val_fold.cpu().numpy()) # バッチでの予測値をCPUに移し、Numpy配列にし、その中身を「検証データに対する予測ラベルを格納するためのリスト」に追加
            all_ground_truth_val_fold.extend(class_labels.cpu().numpy())          # バッチでの正解ラベルをCPUに移し、Numpy配列にし、その中身を「検証データに対する正解ラベルを格納するためのリスト」に追加

    # 検証データに対する損失
    loss_val_fold /= j+1

    # エポックでの正解率
    accuracy_val_fold = correct_val_fold / total_val_fold # エポック全体の正解率＝検証データに対する正解数/検証データの総数

    # エポックでの適合率
    precision_val_fold = precision_score(all_ground_truth_val_fold, all_predicted_val_fold, zero_division=0) # sklearnの関数でエポック全体の適合率を計算　zero_divisionは分母0対策

    # エポックでの再現率
    recall_val_fold = recall_score(all_ground_truth_val_fold, all_predicted_val_fold, zero_division=0) # sklearnの関数でエポック全体の再現率を計算　zero_divisionは分母0対策

    return loss_val_fold, accuracy_val_fold, precision_val_fold, recall_val_fold