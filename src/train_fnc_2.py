from torch import nn
import torch

#from make_spatial_weight_tensor import make_spatial_weight_tensor
from set_seed import set_seed
set_seed(42)

def train_fnc(net, train_loader_fold, optimizer, segmentation_loss_weight, segmentation_class_weights = {0: 0, 1: 51, 2: 85, 3: 55, 4: 218, 5: 1, 6: 200, 7: 18}): #, weight_foreground=5, dilation_iter=1
    net.train()
    loss_train_fold = 0

    for j, (input_images, class_labels, seg_mask) in enumerate(train_loader_fold):


        #print(f"[DEBUG] GT mask unique values: {seg_mask.unique()}")

        input_images = input_images.cuda()
        class_labels = class_labels.cuda()
        seg_mask = seg_mask.cuda()

        #print("セグメンテーションラベル shape:", seg_mask.shape)

        if seg_mask.dim() == 4 and seg_mask.shape[1] == 1:
            seg_mask = seg_mask.squeeze(1) 

        #print("セグメンテーションラベル 変形後 shape:", seg_mask.shape)

        outputs_class, outputs_seg = net(input_images)


        #print("クラス出力 shape:", outputs_class.shape)
        #print("クラスラベル shape:", class_labels.shape)
        #print("セグメンテーション出力 shape:", outputs_seg.shape)
        


        # 分類損失
        loss_cls = nn.CrossEntropyLoss()(outputs_class, class_labels)

        # セグメンテーション損失
        weight_tensor = torch.ones_like(seg_mask, dtype=torch.float)
        for label, weight in segmentation_class_weights.items():
            weight_tensor[seg_mask == label] = weight
        loss_raw = nn.functional.cross_entropy(outputs_seg, seg_mask, reduction='none')  # shape: [B, H, W]
        loss_seg = (loss_raw * weight_tensor).mean()

        """
        # --- ラベルごとの基本重み ---
        label_weight_tensor = torch.ones_like(seg_mask, dtype=torch.float)
        for label, weight in segmentation_class_weights.items():
            label_weight_tensor[seg_mask == label] = weight

        # --- 空間的強調マップ（欠陥周囲にweightを追加） ---
        spatial_weight_tensor = make_spatial_weight_tensor(seg_mask, weight_foreground, dilation_iter)

        # --- 最終的な重み（ピクセルごと）---
        final_weight_tensor = label_weight_tensor * spatial_weight_tensor  # shape: (B, H, W)

        # --- 損失計算 ---
        loss_raw = nn.functional.cross_entropy(outputs_seg, seg_mask, reduction='none')  # shape: [B, H, W]
        loss_seg = (loss_raw * final_weight_tensor).mean()
        """
        
        # 総合損失
        loss = loss_cls + segmentation_loss_weight * loss_seg

        loss_train_fold += loss.item() # PyTorchのテンソル型からPythonの数値型に変換
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net, loss_train_fold / (j + 1)