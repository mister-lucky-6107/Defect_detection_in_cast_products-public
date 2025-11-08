import cv2
import numpy as np
import torch

def make_spatial_weight_tensor(seg_mask: torch.Tensor, weight_foreground=2.0, dilation_iter=1):
    """
    欠陥部分を膨張して、空間的な重みマップを作る
    seg_mask: (B, H, W) のラベルマスク (torch.Tensor, CPU or CUDA)
    weight_foreground: 欠陥の重み（背景は1.0）
    dilation_iter: 膨張の回数（1回 = 1px周囲を拡張）
    """
    batch_size, h, w = seg_mask.shape
    weight_tensor = torch.ones_like(seg_mask, dtype=torch.float32)

    for i in range(batch_size):
        # 欠陥領域（背景以外）を抽出して numpy に変換
        mask_np = (seg_mask[i].cpu().numpy() > 0).astype(np.uint8)

        # OpenCVで膨張（欠陥の周囲を拡張）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #(5, 5)より変更
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=dilation_iter)

        # 重みを割り当て（背景は1、欠陥周辺はweight_foreground）
        weight_np = np.ones((h, w), dtype=np.float32)
        weight_np[dilated_mask > 0] = weight_foreground

        weight_tensor[i] = torch.from_numpy(weight_np).to(seg_mask.device)

    return weight_tensor  # shape: (B, H, W)

"""
from PIL import Image, ImageFilter
import numpy as np
import torch

def make_spatial_weight_tensor(seg_mask: torch.Tensor, weight_foreground=10, dilation_iter=1):
    
    #欠陥部分を膨張して、空間的な重みマップを作る（Pillow版）
    #seg_mask: (B, H, W) のラベルマスク (torch.Tensor, CPU or CUDA)
    #weight_foreground: 欠陥の重み（背景は1.0）
    #dilation_iter: 膨張の回数（1回 = 1px周囲を拡張）
    
    batch_size, h, w = seg_mask.shape
    weight_tensor = torch.ones_like(seg_mask, dtype=torch.float32)

    for i in range(batch_size):
        # 欠陥領域（背景以外）を抽出して numpy に変換
        mask_np = (seg_mask[i].cpu().numpy() > 0).astype(np.uint8) * 255

        # Pillow画像に変換
        mask_img = Image.fromarray(mask_np)

        # MaxFilterで膨張処理
        for _ in range(dilation_iter):
            mask_img = mask_img.filter(ImageFilter.MaxFilter(5))  # 5x5カーネル

        dilated_mask = np.array(mask_img) > 0

        # 重みを割り当て（背景は1、欠陥周辺はweight_foreground）
        weight_np = np.ones((h, w), dtype=np.float32)
        weight_np[dilated_mask] = weight_foreground

        weight_tensor[i] = torch.from_numpy(weight_np).to(seg_mask.device)

    return weight_tensor  # shape: (B, H, W)
"""