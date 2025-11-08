import os
from PIL import Image
import numpy as np

from set_seed import set_seed
set_seed(42)

def generate_multiclass_segmentation_masked_images(train_data_dir, seg_mask_dir, seg_masked_dir):
    # ファイル名ベースでマスクと画像を対応させる
    for mask_file in os.listdir(seg_mask_dir):
        if not mask_file.endswith('_mask.png'):
            continue

        base_name = mask_file.replace('_mask.png', '')
        image_path = os.path.join(train_data_dir, base_name + '.jpeg')
        mask_path = os.path.join(seg_mask_dir, mask_file)
        output_path = os.path.join(seg_masked_dir, base_name + '_masked.png')

        if not os.path.exists(image_path):
            print(f"スキップ（元画像が存在しません）：{image_path}")
            continue

        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        mask = Image.open(mask_path).convert('L') # mode='L'（グレースケール）だが値は整数ラベル
        mask_np = np.array(mask)  # dtype: uint8, 値：0〜7（背景含む）

        # 種別ごとの色を定義（背景含む8色）
        colormap = {
            0: [0, 0, 0],        # 背景：黒
            1: [255, 0, 0],      # 種別1：赤
            2: [0, 255, 0],      # 種別2：緑
            3: [0, 0, 255],      # 種別3：青
            4: [255, 255, 0],    # 種別4：黄
            5: [255, 0, 255],    # 種別5：マゼンタ
            6: [0, 255, 255],    # 種別6：シアン
            7: [255, 128, 0],    # 種別7（オレンジ）
        }

        # カラーマスク画像（H, W, 3）を作成
        mask_rgb_np = np.zeros((*mask_np.shape, 3), dtype=np.uint8)

        for class_id, color in colormap.items():
            mask_rgb_np[mask_np == class_id] = color

        # RGB画像と合成（可視化用：マスクを半透明に重ねる）
        blended = (0.6 * image_np + 0.4 * mask_rgb_np).astype(np.uint8)
        blended_img = Image.fromarray(blended)
        blended_img.save(output_path)

    print(f'セグメンテーションマスク適用画像を {len(os.listdir(seg_masked_dir))} 枚出力しました：{seg_masked_dir}')
