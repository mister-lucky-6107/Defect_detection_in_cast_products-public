
    

    # データローダ
    full_loader = DataLoader(full_dataset,batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)

    # モデルの初期化
    net = MultiTaskEfficientNet_FPN() # EfficientNet_b0モデルのロード（事前学習済みの重みを使用）
    for param in net.parameters():                # モデルのすべてのパラメータの抽出
        param.requires_grad = True               # 全ての層のパラメータを訓練不可に
    net.classifier[1] = nn.Linear(1280, 2)        # 最後の全結合層を入れ替え（デフォルトで訓練可能）
    net.cuda()                                    # モデルをGPUに転送

    # オプティマイザの初期化　オプティマイザは最適化アルゴリズムを使って、損失が最小になるようにパラメータを更新
    optimizer = optim.Adam(net.parameters(), lr=1e-4)



    # 再学習
    """
    
    for epoch in range(num_epochs): # 交差検証と同じエポック数繰り返す
        net.train()                 # モデルを訓練モードに設定　モデルの特定の層が訓練に適した動作をする
        segmentation_class_weights = {0: 51, 1: 85, 2: 55, 3: 218, 4: 1, 5: 8, 6: 18}    # 重みテーブル（欠陥種別ごとの逆面積）
        for input_images, class_labels, seg_mask in full_loader: # DataLoader バッチごとに学習データを供給（バッチごとの画像データと正解ラベルを組み合わせ）
            input_images, class_labels, seg_mask = input_images.cuda(), class_labels.cuda(), seg_mask.cuda()    # 画像データと正解ラベルをGPUに転送

            y_cls, y_seg = net(input_images)                     # 画像データをモデルに入力し、予測値を算出

            # 損失の計算　予測値と正解ラベルを損失関数に入力
            # 分類損失（softmaxクロスエントロピー）　予測値と正解ラベルを損失関数に入力
            loss_cls = nn.CrossEntropyLoss()(y_cls, class_labels)
            # セグメンテーション損失
            weight_tensor = torch.ones_like(seg_mask, dtype=torch.float)
            for label, weight in segmentation_class_weights.items():
                weight_tensor[seg_mask == label] = weight
            loss_raw = nn.functional.cross_entropy(y_seg, seg_mask, reduction='none')
            loss_seg = (loss_raw * weight_tensor).mean()
            # 総合損失（重みを調整可）
            loss = loss_cls + weight * loss_seg

            optimizer.zero_grad()          # オプティマイザのメソッドで、モデルのパラメータに紐づいて保存されている勾配を０に設定する　勾配はオプティマイザには保存されていない
            loss.backward()                # 損失の勾配を計算
            optimizer.step()               # 勾配に基づいて、モデルのパラメータを更新
    """

    # 学習済みのモデルのパラメータを'net.pth'ファイルに保存
    torch.save(           # PyTorchのオブジェクトをメインメモリから補助記憶装置に保存
        net.state_dict(), # モデルのパラメータをPythonの辞書として取得
        'net.pth'         # 保存先のファイル名を取得
        )


from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import os
import numpy as np
import random


from MultiTaskDataset_2 import MultiTaskDataset
from MultiTaskEfficientNet_FPN import MultiTaskEfficientNet_FPN
from train_fnc_2 import train_fnc
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

def train_one_fold(all_images, all_labels, all_seg_masks, all_image_paths, notebook_number,  num_epochs=10, batch_size=16, lr=1e-4, weight=2.0,segmentation_class_weights={0: 51, 1: 85, 2: 55, 3: 218, 4: 1, 5: 200, 6: 18}):
        
        # データセット　train/val分割をやめる
        full_dataset = MultiTaskDataset(all_images, all_labels, all_seg_masks, all_image_paths, is_train = True)

        # DataLoaderの作成
        train_loader_fold = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g) # gは乱数生成器

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

        for epoch in tqdm(range(num_epochs), desc=f"Fold {fold+1} Epochs"):

            # 学習

            net, loss_train_fold = train_fnc(net, train_loader_fold, optimizer, weight, segmentation_class_weights) # return  net, train_loss_per_epoch_history

            train_loss_per_epoch_history.append(loss_train_fold) # 「エポックごとの、訓練データに対する損失」のリストに、計算した訓練データの平均損失を追加




        # フォールドごとの学習後にモデルを保存
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_save_path = f'{MODEL_DIR}/model_fold{fold+1}_nb{notebook_number}.pth'
        torch.save(net.state_dict(), model_save_path)
        print(f"モデルを保存しました: {model_save_path}")