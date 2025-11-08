import time
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import pickle

from MultiTaskDataset_2 import MultiTaskDataset
from MultiTaskEfficientNet_FPN import MultiTaskEfficientNet_FPN
from train_fnc_2 import train_fnc
from validate_fnc_epoc import validate_fnc_epoc
from validate_fnc_fold import validate_fnc_fold
from set_seed import set_seed
from path_config import MODEL_DIR

set_seed(42)

# 各ワーカーのシード固定　DataLoaderの引数に渡す
def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 乱数生成器
g = torch.Generator()
g.manual_seed(42)

def train_one_fold(all_images, all_labels, all_seg_masks, all_image_paths, notebook_number, n_splits=5, num_epochs=10, batch_size=16, lr=1e-4, weight=2.0, desired_fold=1, segmentation_class_weights={0: 51, 1: 85, 2: 55, 3: 218, 4: 1, 5: 200, 6: 18}): # , weight_foreground, dilation_iter)

    start_time_all = time.time() # 交差検証全体の開始時間の記録

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    
    # フォールドごとの損失と評価指標のログを保存するリスト
    all_loss_train = []         # フォールドごとの、訓練データに対する損失
    all_loss_val = []           # フォールドごとの、検証データに対する損失
    all_accuracy_val = []       # フォールドごとの正解率
    all_precision_val = []      # フォールドごとの適合率
    all_recall_val = []         # フォールドごとの再現率
    all_ground_truth_val = []   # フォールドごとの、各データに対する正解ラベル
    all_predicted_val = []      # フォールドごとの、各データに対する予測ラベル
    all_probabilities_val = []  # ROC曲線表示用
    
    
    for fold, (train_index, val_index) in enumerate(skf.split(all_images, all_labels)):

        if fold != desired_fold:
            continue

        print(f"Fold {fold+1}/{n_splits}")

        start_time_fold = time.time() # 各フォールドの開始時間の記録

        # 学習データと検証データの作成
        train_images_fold, val_images_fold = all_images[torch.tensor(train_index)], all_images[torch.tensor(val_index)]
        train_labels_fold, val_labels_fold = all_labels[torch.tensor(train_index)], all_labels[torch.tensor(val_index)]
        train_seg_masks_fold, val_seg_masks_fold = all_seg_masks[torch.tensor(train_index)], all_seg_masks[torch.tensor(val_index)]
        train_paths_fold, val_paths_fold = all_image_paths[torch.tensor(train_index)], all_image_paths[torch.tensor(val_index)]

        # データセットの作成
        train_dataset_fold = MultiTaskDataset(train_images_fold, train_labels_fold, train_seg_masks_fold, train_paths_fold, is_train = True)
        val_dataset_fold = MultiTaskDataset(val_images_fold, val_labels_fold, val_seg_masks_fold, val_paths_fold, is_train = False)

        # DataLoaderの作成
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g) # gは乱数生成器
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, num_workers=2, worker_init_fn=seed_worker, generator=g)

        # モデルの初期化
        net = MultiTaskEfficientNet_FPN()
        for param in net.parameters():                # モデルのすべてのパラメータの抽出
            param.requires_grad = True
        net.classifier[1] = nn.Linear(1280, 2)        # 最後の全結合層を入れ替え
        net.cuda()

        print('MultiTaskEfficientNet_FPNモデルを初期化しました。')

        # フォールドごとに、オプティマイザの初期化
        optimizer = optim.Adam(net.parameters(), lr=lr)


        # 学習

        train_loss_per_epoch_history = [] # エポックごとの、訓練データに対する損失
        record_loss_val_fold = []        # エポックごとの、検証データに対する損失
        record_accuracy_val_fold = []    # エポックごとの正解率
        record_precision_val_fold = []   # エポックごとの適合率
        record_recall_val_fold = []      # エポックごとの再現率

        for epoch in tqdm(range(num_epochs), desc=f"Fold {fold+1} Epochs"):

            # 学習

            net, loss_train_fold = train_fnc(net, train_loader_fold, optimizer, weight, segmentation_class_weights) # , weight_foreground, dilation_iter)  return  net, train_loss_per_epoch_history

            train_loss_per_epoch_history.append(loss_train_fold) # 「エポックごとの、訓練データに対する損失」のリストに、計算した訓練データの平均損失を追加

            # 検証1（エポックレベル）

            loss_val_fold, accuracy_val_fold, precision_val_fold, recall_val_fold = validate_fnc_epoc(net, val_loader_fold, weight)

            record_loss_val_fold.append(loss_val_fold) # 「エポックごとの、検証データに対する損失」をリストに追加
            record_accuracy_val_fold.append(accuracy_val_fold) # 「エポック全体の正解率」をリストに追加
            record_precision_val_fold.append(precision_val_fold) # 「エポック全体の適合率」をリストに追加
            record_recall_val_fold.append(recall_val_fold) # 「エポック全体の再現率」をリストに追加


            # エポックごとに評価指標を表示
            if epoch % 1 == 0:
                print(f"  Epoch: {epoch+1}/{num_epochs}, Loss_Train: {loss_train_fold:.4f}, Loss_val: {loss_val_fold:.4f}, Accuracy_val: {accuracy_val_fold:.4f}, Precision_val: {precision_val_fold:.4f}, Recall_val: {recall_val_fold:.4f}")

        # フォールドごとの所要時間を計算、表示
        end_time_fold = time.time()
        execution_time_fold = end_time_fold - start_time_fold
        print(f'Fold {fold+1}の学習時間：{execution_time_fold:.2f}秒')

        # 損失と評価指標の履歴について、エポックごとの推移の記録を、フォールドごとに推移を記録するリストに追加する
        all_loss_train.append(train_loss_per_epoch_history)
        all_loss_val.append(record_loss_val_fold)
        all_accuracy_val.append(record_accuracy_val_fold)
        all_precision_val.append(record_precision_val_fold)
        all_recall_val.append(record_recall_val_fold)


        # 検証2（フォールドレベル）
        current_predicted_val, current_ground_truth_val, misclassified_in_fold, current_probabilities_val, img_np, gt_mask_np, pred_mask_np = validate_fnc_fold(net, val_loader_fold)

        # 各フォールドの予測結果と正解ラベル　描画用
        all_predicted_val.append(current_predicted_val)
        all_ground_truth_val.append(current_ground_truth_val)
        all_probabilities_val.append(current_probabilities_val)

        # このフォールドで誤分類された画像情報を表示
        print("\n--- Fold {} の誤分類された画像 ---".format(fold + 1))
        if misclassified_in_fold:
            for info in misclassified_in_fold:
                print(f"  パス: {info['path']}, 予測ラベル: {info['predicted_label']}, 正解ラベル: {info['ground_truth_label']}")
        else:
            print("  誤分類された画像はありませんでした。")


        if img_np is not None:
            # 欠陥領域の可視化（ループの後で1回だけ）
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Input Image")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask_np, cmap='gray')
            plt.title("Ground Truth")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask_np, cmap='gray')
            plt.title("Predicted Mask")

            plt.tight_layout()
            plt.show()

            print(f"[DEBUG] Ground Truth shape: {gt_mask_np.shape}")
            print(f"[DEBUG] Ground Truth max value: {gt_mask_np.max()}")
            plt.imshow(gt_mask_np.squeeze())
            plt.title("GT Mask in Visualization")
            plt.show()

        # フォールドごとの学習後にモデルを保存
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_save_path = f'{MODEL_DIR}/model_fold{fold+1}_nb{notebook_number}.pth'
        torch.save(net.state_dict(), model_save_path)
        print(f"モデルを保存しました: {model_save_path}")

    # 交差検証の時間測定
    end_time_all = time.time()
    execution_time_all = end_time_all - start_time_all
    print(f'\n交差検証ループ全体の実行時間 : {execution_time_all:.2f}秒')

    # 保存用ディレクトリ作成（なければ）
    os.makedirs('outputs/predicted_labels', exist_ok=True)

    # 保存するデータをまとめる
    result_dict = {
        'train_loss_per_epoch_history': train_loss_per_epoch_history,
        'record_loss_val_fold': record_loss_val_fold,
        'record_accuracy_val_fold': record_accuracy_val_fold,
        'record_precision_val_fold': record_precision_val_fold,
        'record_recall_val_fold': record_recall_val_fold,
        'current_predicted_val': current_predicted_val,
        'current_ground_truth_val': current_ground_truth_val,
        'current_probabilities_val': current_probabilities_val
    }

    # pickleで保存
    with open(f'outputs/predicted_labels/results_fold{fold+1}_nb{notebook_number}.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    print(f"学習・検証データを保存しました: outputs/predicted_labels/results_fold{fold+1}_nb{notebook_number}.pkl")

    return train_loss_per_epoch_history, record_loss_val_fold, record_accuracy_val_fold, record_precision_val_fold, record_recall_val_fold, current_predicted_val, current_ground_truth_val, current_probabilities_val