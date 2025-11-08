import torch
from torch.utils.data import TensorDataset
#from matplotlib import pyplot as plt

from set_seed import set_seed
set_seed(42)

class MultiTaskDataset(TensorDataset):
    def __init__(self, images, labels, seg_masks, paths, is_train=True):
        self.images = images        # 前処理済み画像データのリスト テンソル形式
        self.labels = labels        # 正解ラベルのリスト テンソル形式
        self.seg_masks = seg_masks  # マルチクラスマスク
        self.paths = paths          # 画像ファイルのパスのリスト 文字列形式
        self.is_train = is_train    # 学習か検証 ブール値

    def __len__(self):          
        return len(self.images) 

    def __getitem__(self, idx):              
        image = self.images[idx]             
        label = self.labels[idx]             
        seg_mask = self.seg_masks[idx]       # shape: [1, H, W]
        if seg_mask.ndim == 3:
            seg_mask = seg_mask.squeeze(0)   # shape: [H, W] に変換
        seg_mask = seg_mask.long()           # クロスエントロピー損失の要件：long型
        seg_mask = torch.clamp(seg_mask, max=7)  # 万が一マスクに異常値が混入していても補正
        #seg_mask[seg_mask > 0] -= 1  # 背景はそのまま0、1〜7 → 0〜6

        #path = self.paths[idx]
        path = str(self.paths[idx])

        #print(f"[DEBUG] mask_path: {path}")
        #print(f"[DEBUG] mask.size: {seg_mask.size}, mask.mode: {seg_mask.mode}")
        #plt.imshow(seg_mask)
        #plt.title("Loaded Mask")
        #plt.show()

        if self.is_train:
            return image, label, seg_mask
        else:
            return image, label, seg_mask, path
